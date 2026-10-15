# FLASH 小模型审计测试结果

本报告记录 v1 收尾阶段的小模型论文式测试。正式训练运行使用 `FLASH_RUNTIME=paper`，每项均使用新的 `audit_*` run ID。测试覆盖健康强制剖析、自动触发、训练监控 ABBA、故障注入、AllReduce ABBA、共享 GPU 背景通信、600 MHz 多拓扑和真实 OOM/timeout 路径。结果按梯度归一化修复前后分代，不把两代训练数据混合。

## 实现代与复现入口

`post_gradient_fix` 使用以下源码：

| 文件 | SHA-256 |
|---|---|
| `scripts/train_moe.py` | `dc275946b8d22a3d9d3e4318d4b94e511fb3e19c19a0184eb4ebd0e2d573c529` |
| `scripts/gradient_sync.py` | `1b96293e61bfc2d3769873c02e370e9eb9b7edbc91ed775d8ad1b5dd5c39d9b4` |
| `flash/global_controller.py` | `51bf4f4fe532381ecba03ac2a10180e926e171c265fb491d7a917d37ca974286` |

独立 HF/MCore 梯度对照先验证了修复：16 rank、EP4、DP16、EDP4 下，14 个 dense/router/shared/expert 梯度与完整 HF 参考的最大 relative L2 为 `8.51e-7`，专家梯度范数比约为 1。此前 EDP 分母会使专家梯度按 EP 因子放大；旧训练结果保留为 `pre_gradient_fix`，不用于新版训练性能结论。

可重复入口为 [scripts/audit_campaign.py](../scripts/audit_campaign.py)。实验等待逻辑已改为依据控制器设置的实际剖析时长计算有界等待。`scripts/experiment.py`、`scripts/launch.py`、`scripts/audit_campaign.py`、`scripts/export_run.py` 均通过 Black 格式检查和 `py_compile`。

## 健康与自动剖析

| 条件 | Run | 结果 |
|---|---|---|
| 健康 forced，800 steps/rank | `audit_healthy_forced_post_gradient_fix_20261015T163528Z_47b835d1` | 两节点退出 `0/0`，各 800 步，单个后台 writer 收到 800 条，last_iteration=800；两节点 complete，lost/errors/unknown/pending 均为 0。强制剖析完整运行 20.14 秒，没有 failstop。 |
| delay45 自动回归，800 steps/rank | `audit_delay45_auto_post_gradient_fix_20261015T163747Z_2a995db6` | 两节点退出 `0/0`，各 800 步，后台 writer 收到 800 条，last_iteration=800；两节点 complete，lost/errors/unknown/pending 均为 0。第 80 步注入，第 83 步自动剖析，变点后验概率 1.0；rank 8 得分 0.99278。 |

健康强制运行 rank 0 的 `time_ms` 中位数从剖析前 15 步的 `81.259650 ms` 到剖析采样期的 `84.197137 ms`，增加 3.6149%。采样期按 `stats.profile.started_at` 与 `capture_stopped_at` 确定。这是单次运行中的观察，不是严格因果估计。强制诊断给出 rank 5 候选、得分约 0.812。

delay45 的控制器估计吞吐损失 39.56%，基线/退化估计分别为 `83.2056/137.6700 ms`。按注入前 steps 65–79 和注入后 steps 85–99 对比，`time_ms` 中位数为 `83.6531 → 141.5379 ms`，时间增加 69.20%，对应逆时间吞吐损失 40.87%；timestamp 单调时钟差分为 `83.9187 → 141.7777 ms`，时间增加 68.87%，对应损失 40.81%。报告两种口径是为了区分迭代内计算/通信/优化器时间和包含上报及日志的端到端时间。

训练监控 ABBA 运行顺序为 base1、mon1、mon2、base2，均 800 steps/rank，四项两节点均退出 `0/0`。rank 0 `steady_e2e_tokens_per_second` 分别为 `97333.4382、98320.1614、97562.3949、96691.3475`；两个 monitor 对相邻 baseline 的差异约为 `+1.01%` 和 `+0.90%`，按运行波动解释，不宣称加速。base1/mon1 检查点覆盖全部 16 rank、每 rank 4,073,728 个元素，均为有限值，最大绝对差为 0。这四项训练 run 的 PyTorch `peak_allocated/peak_reserved` 为 `209,829,376/236,978,176` 字节。

## 故障强度与瞬态

各条件均为 post-gradient-fix 的新鲜 run。本表只取注入目标rank8：注入前iterations 45–59、后65–79各15条记录的窗口内中位数，非跨rank统计。端到端值为同一rank相邻monotonic时间差，归属后一个iteration。损失使用吞吐的逆时间口径；负损失表示该次运行的后窗口略快，按噪声处理。

| 条件 / run | steps/rank | `time_ms` 前后中位数 | 时间增加 / 吞吐损失 | 单调时钟端到端前后中位数 | 时间增加 / 吞吐损失 | 结果 |
|---|---:|---:|---:|---:|---:|---|
| delay5 / `audit_delay5_post_gradient_fix_20261015T165105Z_66d7d32b` | 300 | 84.798 → 83.145 ms | -1.95% / -1.99% | 85.029 → 83.357 ms | -1.97% / -2.01% | 退出 0/0；未触发剖析，未观察到明显退化 |
| delay25 / `audit_delay25_post_gradient_fix_20261015T165232Z_7f75a06c` | 300 | 83.821 → 99.182 ms | +18.33% / 15.49% | 84.039 → 99.372 ms | +18.24% / 15.43% | 退出 0/0；实测低于 20% 门槛，因此未触发不算漏检 |
| delay65 / `audit_delay65_post_gradient_fix_20261015T165404Z_8c9cf2cd` | 250 | 84.118 → 229.105 ms | +172.36% / 63.28% | 84.330 → 229.273 ms | +171.88% / 63.22% | 退出 0/0；第 63 步自动剖析，完整、lost 0；rank 8 得分 0.9968 |
| delay85 / `audit_delay85_post_gradient_fix_20261015T165554Z_7ab4d13f` | 180 | 83.707 → 550.042 ms | +557.10% / 84.78% | 83.945 → 550.215 ms | +555.45% / 84.74% | 退出 0/0；第 63 步自动剖析，完整、lost 0；rank 8 得分 0.99898 |
| transient 500 ms 单步 / `audit_transient500ms_post_gradient_fix_20261015T165808Z_d933a743` | 200 | 84.182 → 81.822 ms | -2.80% / -2.88% | 84.374 → 82.036 ms | -2.77% / -2.85% | 退出 0/0；未自动剖析，瞬态恢复；failstop 告警时 16 个进程均仍存活 |

delay85 产生 16 条 failstop 告警，但告警时 `process_exists=true` 且 `process_alive=true`。它们是通信静默告警，不应解释为进程死亡。瞬态条件也出现相同存活状态的告警。delay25 的实测损失低于自动门槛；报告采用实测数值，而非按名义故障比例推断全局损失。

## AllReduce 与后台通信

AllReduce ABBA 独立于训练梯度归一化，保留在 `pre_gradient_fix` cohort，不需要重跑。16 rank 使用 100 次 warmup、5 批 × 100 次重复，消息为 4、8、16、32、64、128 MiB。base1、mon1、mon2、base2 逐尺寸对16rank各自的 `mean_ms` 取中位数，monitor平均开销用两次monitor中位数均值相对两次baseline中位数均值计算：：

| 消息 | base1 / mon1 / mon2 / base2（ms） | monitor 平均开销 |
|---:|---|---:|
| 4 MiB | 0.144107 / 0.138037 / 0.147871 / 0.142509 | -0.25% |
| 8 MiB | 0.188417 / 0.172097 / 0.180736 / 0.182347 | -4.84% |
| 16 MiB | 0.266506 / 0.262809 / 0.262767 / 0.264071 | -0.94% |
| 32 MiB | 0.470931 / 0.467670 / 0.464848 / 0.466479 | -0.52% |
| 64 MiB | 0.828141 / 0.834983 / 0.831359 / 0.831692 | +0.39% |
| 128 MiB | 1.549492 / 1.555547 / 1.552648 / 1.558005 | +0.02% |

大消息开销接近零；小消息的负值是测量噪声，不作为加速证据。

训练中的 mild/strong 背景流量是真实 NCCL 通信并共享 GPU，因此测量包含 GPU 争用，不应归因为纯网络故障。时间窗口和实际背景 worker 的 `started_at/finished_at` 对齐，而不是只看启动时声明的迭代号。共同区间为两节点started_at的最大值至finished_at的最小值；基线取此前15秒。各rank按行timestamp筛选，在rank内先取窗口中位数，再对16rank的中位数取中位数。

| 条件 / run | 背景流量 | `time_ms` 前后中位数 | 时间增加 / 吞吐损失 | 单调时钟端到端前后中位数 | 时间增加 / 吞吐损失 | 自动触发 |
|---|---|---|---|---|---|---|
| mild / `audit_background_mild_post_gradient_fix_20261015T171155Z_434df2ff` | 128 MiB，duty 0.25，30 秒，664 calls | 83.32 → 84.54 ms | +1.47% / 1.45% | 83.55 → 84.75 ms | +1.44% / 1.42% | 否；实测低于 20% 门槛 |
| strong / `audit_background_strong_post_gradient_fix_20261015T171422Z_caf37609` | 128 MiB，duty 1.0，30 秒，3467 calls | 82.45 → 96.58 ms | +17.13% / 14.63% | 82.67 → 96.77 ms | +17.06% / 14.57% | 否；实测低于 20% 门槛 |

两项训练 run 均为 800 steps/rank、两节点退出 `0/0`。低于门槛而没有自动剖析与当前阈值相符。

## OOM 和 timeout

- `audit_oom_post_gradient_fix_20261015T170018Z_08e12f16`：rank 8 在注入 step 45 报真实 CUDA `OutOfMemoryError`，尝试分配 158.36 GiB，而单卡为 79.18 GiB。两节点 launcher 均自然退出，代码 `1/1`，没有发送清理信号。注入后 0.5 秒两节点各观察到 8 个训练进程和 8 个 GPU PID；最终进程快照为空。
- `audit_timeout_post_gradient_fix_20261015T170357Z_1e8a1b6b`：两节点日志都记录 NCCL watchdog `Timeout(ms)=90000`，实测约 90073–90094 ms。两端约 156 秒后自然退出，代码 `1/1`，没有发送清理信号；注入后 0.5 秒与 10 秒采集存活状态，最终进程快照为空。

故障发生证据、告警时进程状态和最终自然退出分别记录，不把结束清理时刻当成故障发生时刻。

## 600 MHz 多拓扑强制剖析

所有拓扑均为 400 steps/rank，锁定 node 1 GPU 0 到 600 MHz，并使用强制剖析。基线是锁频前 15 秒；剖析窗口依据 `stats.profile.started_at/capture_stopped_at`。各rank先取窗口样本中位数，再对16rank取中位数；profile窗口使用两节点捕获区间的交集。端到端时间差归属后一个iteration。这些值是观察性窗口对照，不是自动检测召回测试。

| 拓扑 / run | profile | `time_ms` 基线 → profile；时间增加 / 吞吐损失 | 端到端基线 → profile；时间增加 / 吞吐损失 |
|---|---|---|---|
| TP1/PP1/EP8 / `audit_clock600_topology1_post_gradient_fix_20261015T171705Z_199281be` | 完整 19.933 s，lost 0；rank 9 候选 0.549 | 81.10 → 83.73 ms；+3.24% / 3.14% | 81.21 → 83.95 ms；+3.38% / 3.27% |
| TP2/PP1/EP4 / `audit_clock600_topology2_post_gradient_fix_20261015T171844Z_0bb98786` | 完整 20.006 s，lost 0；rank 4 候选 0.424 | 96.06 → 100.52 ms；+4.64% / 4.44% | 96.21 → 100.71 ms；+4.68% / 4.47% |
| TP1/PP2/EP4 / `audit_clock600_topology3_post_gradient_fix_20261015T172027Z_714a8dcb` | 完整 20.048 s，lost 0；rank 0/9 候选 0.704/0.577 | 74.63 → 77.75 ms；+4.18% / 4.01% | 74.77 → 77.93 ms；+4.23% / 4.05% |
| TP2/PP2/EP4 / `audit_clock600_topology4_post_gradient_fix_20261015T172201Z_bab9449e` | 完整 20.051 s，lost 0；rank 4/8 候选 0.614/0.581 | 86.15 → 88.25 ms；+2.43% / 2.38% | 86.23 → 88.37 ms；+2.49% / 2.43% |

四个拓扑的迭代内吞吐损失约2.38%–4.44%，端到端约2.43%–4.47%，不足以据此检验严重退化下的自动召回。每次实验 finally 均复位时钟；最终两节点所有 GPU 的 SM 当前频率均为 345 MHz。

## 清场和范围

本轮 19 个 post-gradient-fix run 的共享内存清理只针对 [run ID 清单](../results/audit_campaign_cleanup_runs.json)：每节点第一批删除 88 个、第二批删除 48 个，共 136 个 buffer。Redis 16379 和 `flash-paper-20261015` 容器保留。FLASH audit 工作进程均停止，控制器已停，锁频复位已确认。

最终状态记录在 [results/v1_server_state.json](../results/v1_server_state.json)。快照显示本轮 audit 工作进程停止；同一时点还检测到未归属的非项目主机任务，因此不宣称服务器可独占。该任务未被处理，公开快照不保留其命令行、路径或参数。大模型权重验证不属于本轮小模型 campaign。

逐 run 的机器可读指标和 run IDs 见 [results/audit_campaign_summary.json](../results/audit_campaign_summary.json)。主报告可引用该文件中的数值和 cohort 区分。

## 数字复算

`python -m scripts.audit_window_metrics`从原始日志复算故障扫表、背景通信与四拓扑的窗口指标，生成`results/v1_window_metrics.json`，包含实际窗口、各rank样本数和未舍入值。背景表在发布复核时统一了取窗方式，替换此前不够明确的临时统计；不影响低于20%门槛的结论。
