# 从零部署与重跑

本项目从用户提供论文开始实现，未读取旧复现项目。服务器上的专用目录是 `/home/campus/zhangweihao/flash-repro-20261015`；本地 SSH 工具只从 Git 忽略的 `服务器.md` 读凭据。

## 环境

- Mac 本地 Python 需 `paramiko`，本次使用 Codex bundled Python3.12。
- 两节点使用 containerd 的 moby namespace。基础镜像须已在服务器；镜像信息见 `docs/provenance.md`。
- 正式运行设 `FLASH_RUNTIME=paper`。`greyhound` 选择独立官方基线容器；不设置会落到早期排障容器，禁止用于正式对照。
- 正式容器暴露 RDMA 设备并安装主机匹配的 verbs 用户库。NCCL 设置 `NCCL_NET=IB`，不允许静默退回 socket。
- `torch` 2.10.0+cu126 / NCCL2.27.5 / Megatron-core0.12.1；原生 torch normalization/attention，无 TE；AdamW，FP32。版本差异不藏在“复现效果”里。

## 在 Mac 准备离线资产

```bash
python -m pip install paramiko
mkdir -p vendor wheelhouse
git clone --depth 1 https://github.com/wutianyuan1/Greyhound.git vendor/Greyhound
git -C vendor/Greyhound rev-parse HEAD  # 本次 d641b3997c17de700e1ef7d37e353418df7e336d
git clone --depth 1 --branch core_v0.12.1 https://github.com/NVIDIA/Megatron-LM.git vendor/MegatronCore
python -m pip download --dest wheelhouse --only-binary=:all: \
  --platform manylinux_2_17_x86_64 --platform manylinux2014_x86_64 \
  --python-version 312 --implementation cp --abi cp312 --no-deps \
  redis==8.1.0 Rbeast==0.1.25 einops==0.8.2
```

仓库自带真实语料 `data/shakespeare.txt` 和来源/SHA256。Megatron tag commit 为 `a845aa7e12b3a117e24c2352b9e3e60bad2e3a17`。若 GreyHound HEAD 已变，应检出上述 commit 后再使用。

## 部署

```bash
python -m scripts.bootstrap --create
python -m scripts.sync --assets
python -m scripts.container 'bash scripts/build.sh'
python -m scripts.prepare_rdma
FLASH_RUNTIME=paper python -m scripts.container \
  'python -m pip install --target /flash/deps --no-index --find-links=/flash/wheelhouse --no-deps redis Rbeast einops'
```

在主节点早期工具容器启动任务独立 Redis（不占用全局业务端口 6379）：

```bash
python -m scripts.container --host 10.10.4.1 \
  'redis-server --bind 10.10.4.1 --port 16379 --protected-mode no --save "" --appendonly no --daemonize yes --pidfile /flash/redis.pid --logfile /flash/redis.log'
```

若 Redis 已运行，复用该任务专属实例；不得停掉它所在的早期容器，否则正式控制器会失去 Redis。

## 一次验证

为每次实验使用新 ID；启动器会拒绝已有 run 目录并检查两台机器 GPU 空闲。禁止同时启动两项独立性能实验。

```bash
FLASH_RUNTIME=paper python -m scripts.experiment trial_baseline \
  --mode baseline --args '--steps 120 --seq 128 --batch 2'
FLASH_RUNTIME=paper python -m scripts.experiment trial_monitor \
  --args '--steps 120 --seq 128 --batch 2'
FLASH_RUNTIME=paper python -m scripts.experiment trial_fault \
  --args '--steps 100 --min-seconds 34 --seq 128 --batch 2 --fault delay --severity .45'
FLASH_RUNTIME=paper python -m scripts.clock_experiment trial_clock600 \
  --mhz 600 --force-profile
```

`--force-profile` 只用于剖析器/信号验证，不能计为自动检测成功。GPU sleep 的名义严重度不会精确等于全局吞吐损失，必须看实测迭代耗时。

控制器配置首次生成于新容器 `~/.robust_training_flash/config.json`；阈值保持论文详细设计的默认值。训练回调启用开关为 `FAILSLOW_AWARE=TRUE`，C++ probe 按 shared memory 控制字段切换 monitor/profile。

## 结果与清理

每个 run 的原始日志、16 rank 迭代序列、Redis telemetry、实际退出码、源码/库哈希写到远端 `results/RUN`，收回本地同名目录。模型 checkpoint 保留在服务器 `.jsonl.pt` 文件，不重复下载。

```bash
FLASH_RUNTIME=paper python -m scripts.launch RUN --collect
python -m scripts.collect_all
```

正常收尾由 `scripts.experiment.finish` 设置 `stop=True`，结束独立控制器。OOM 与 NCCL timeout 预期训练进程非零退出。停机证据采集完成后可终止残留训练 worker，清理动作应写入该 run 的 events，不能当作故障检测时刻。

GPU 降频脚本通过 `finally` 调用 `nvidia-smi -rgc`。若本地进程异常中断，应直接检查两节点当前频率和训练进程，不要在未确认 GPU 空闲时运行下一项实验。

## 首轮历史矩阵与报告重建

首轮历史运行见 `docs/report.md` 和 `results/trials.csv`；审计后的v1以 `docs/v1_report.md`、`docs/audit_test_results.md` 为准。`scripts/suite.py` 及 `scripts/validate_final.py` 内的编号是首轮记录；不要原样再次执行覆盖结果。后续重跑逐项用新编号。

```bash
FLASH_RUNTIME=paper python -m scripts.experiment NEW_PROFILE \
  --force-profile --args '--steps 120 --min-seconds 34 --seq 128 --batch 2'
FLASH_RUNTIME=paper python -m scripts.network_experiment NEW_BACKGROUND
python -m scripts.summarize
python -m scripts.paper_metrics
python -m scripts.write_report
```

`plot_results.py` 使用标准matplotlib，基于固定报告编号重建静态图；本轮在早期工具容器3.8环境运行，结果为 `results/figures/*.png`。脱离当前机器时可在任意有matplotlib的Python环境运行。

DeepSpeed只做独立兼容性检查，主FLASH工作负载仍为Megatron MoE。离线准备DeepSpeed0.19.7源码包以及hjson、py-cpuinfo、pydantic及其依赖、tqdm、msgpack的Linux cp312 wheels；清单与哈希见 `data/offline_assets.json`。传至两节点wheelhouse后，依赖以 `--no-index --no-deps --target /flash/deps` 安装，DeepSpeed本体以 `DS_BUILD_OPS=0 python -m pip install --no-deps --no-build-isolation --target /flash/deps-optional /flash/wheelhouse/deepspeed-0.19.7.tar.gz` 安装，避免联网和JIT自定义优化器。兼容脚本显式加载可选目录，防止共同依赖里的DeepSpeed影响Transformers导入。

```bash
FLASH_RUNTIME=paper python -m scripts.experiment NEW_DS \
  --task deepspeed --args '--steps 24'
```

官方GreyHound构建：`python -m scripts.prepare_greyhound`生成仅硬件/路径差异，`python -m scripts.prepare_greyhound_remote`在独立baseline容器构建并安装。运行入口 `FLASH_RUNTIME=greyhound python -m scripts.greyhound_experiment` 内使用本轮固定编号；再次执行须先改为新编号。无论成功失败，保留原始退出和日志，不将其与不同现代栈的FLASH作无条件对比。

首轮后期 `paper_gloo_*` 曾用每步Gloo结束协调，避免更早的额外NCCL退出communicator污染DP标签。审计后发现该CPU屏障也属额外干扰，v1固定步数已移除全部每步退出通信；仅 `--min-seconds > 0` 仍需该协调。`--stop-backend nccl`只用于明确重现早期条件。首轮四种拓扑与六频率扫描保留为旧fixture证据，审计后的四拓扑另用新ID；不将两代结果混算。

官方baseline复测可使用新ID：

```bash
FLASH_RUNTIME=greyhound python -m scripts.greyhound_experiment \
  --run NEW_GREYHOUND --seed-parallel-states --wait-precheck
```

该入口预填实际TP1/PP1/DP16上游键，避免world_size早于并行状态发布的竞争；先预热惰性communicator，再服务上游503-byte marker验证协议，等待其真正precheck_done，随后才做训练故障注入。没有伪造precheck完成标记，也没有修改上游检测/定位算法。上游缓解器的输出仅保留观察，不作为本轮FLASH恢复能力。


## v1审计后入口说明

默认训练启动器会先协同完成两节点GEMM数值与AllReduce结果预检。固定步数不再执行每步退出通信，建议较长运行使用 `--steps 800 --no-save-checkpoint`，让真实采样窗口有足够迭代；若要检查参数一致性，对应baseline/monitor不要禁存checkpoint。min-seconds仍使用额外Gloo协调，必须记录这一测试条件。

采样完整性以两个节点相同epoch、complete=true、active_scopes_at_finish=0以及lost/errors计数核对，不凭等满20秒认定完成。后台iteration_reporter包含received和last_iteration，应与rank0最终步数一致。incomplete_profile不应计定位成功。SFTP导出先写.part，校验大小稳定后原子替换，失败重试；不能把中断留下的空文件当证据。

官方近似Qwen下载是可恢复资产准备，v1尚未完成完整权重验证或训练。后续获准继续时使用 `python -m scripts.download_qwen_ranges`，固定官方SHA256全部通过后才能上传并运行；不让服务器连接公网。小型转换fixture验证不等于论文13.3B模型已测。
