# FLASH 第一版复现交接

2026-10-15：本轮已完成设计、实现、两节点16卡开发验证与结果归档。用户目标是严格依中文论文第三章复现FLASH，使用真实小MoE、少次数测试，记录问题；本轮没有做SHIFT、工业界调研或后续算法优化，也没有读取此前复现尝试。用户偏好主窗口GPT6.1 Sol high、子智能体GPT6 Luna Max；本轮未创建子智能体或新聊天。

## 直接阅读

- `docs/report.md`：完整中文报告，结论、设计验收、实际吞吐、故障矩阵、频率/拓扑、官方基线及限制。
- `docs/spec.md`：论文阈值与未定义细节的处理，区分论文歧义和本轮实现错误。
- `docs/runbook.md`：离线部署、使用新RUN编号重跑、结果重建。
- `docs/provenance.md`：模型/数据/软件版本、公开源固定commit与论文差异。
- `results/summary.json`、`trials.csv`、`paper_equations.csv`：从原始证据导出；图表在results/figures。

## 部署位置

本地 `/Users/henry/Desktop/autoflash`。本地Python为 `/Users/henry/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3`，有paramiko/black/clang-format，无torch和matplotlib。

远端新目录 `/home/campus/zhangweihao/flash-repro-20261015`，挂载 `/flash`。凭据只由scripts/remote.py读取被Git忽略的服务器.md，禁止打印/提交。

正式运行必须 `FLASH_RUNTIME=paper`，选择flash-paper-20261015，PyTorch2.10/CUDA12.6/NCCL2.27.5、Megatron-core0.12.1、Rbeast0.1.25，实际NCCL NET/IB。未设该变量会选择早期工具容器，不能用于正式结果。

模型为真实15,083,776参数GPT MoE：4层256hidden、8路由专家top2、1共享专家、SwiGLU/RMSNorm、GQA4、RoPE、expertTP1、All-to-All；FP32/AdamW、真实Shakespeare字节训练，随机初始化。不是通信模拟，不声称预训练模型。TP2时SP开启，原生norm适配经分片/非分片输出与梯度等价测试。

flash-repro-20261015保留任务Redis16379与构建工具，不能在重跑期间停掉其Redis。FLASH_RUNTIME=greyhound选择独立官方基线容器，私有IPC，使用旧PyTorch2.1/NCCL2.19栈；不能与现代主栈无条件比较。

## 最终重点结果

- 最新fixture默认Gloo CPU控制组协调结束，避免额外world NCCL规约污染DP标签；旧条件用--stop-backend nccl可恢复，所有旧结果保留。
- 最新160步Gloo对照：baseline83873.97、monitor83961.42 tokens/s，监控相对约+0.10%，视为短测噪声；16分片参数均finite且最大差0。
- 最新正常forced剖析：无自动profile、无fail-stop，仍有误候选rank6；采样前/期间95.122/94.460ms，变化约-0.70%，不能解释为剖析加速。
- 最新持续退化实测38.1%，自动profile，唯一候选rank8，约0.683s触发、21.227s诊断，errors/lost0。
- 最新600MHz（实际查询600）：损失1.56%，未达20%门槛，无自动profile；forced误候选rank5，未定位rank8。锁频完成后才放行forced。
- 早期辅助NCCL条件：实测27.6/40.8/62.9/84.7%四次自动正确定位rank8；5%名义实际2.8%、25%名义实际16.4%未触发。旧六频率/四拓扑未复现论文定位分数，只作探索记录，不排除fixture影响。
- OOM/timeout真实日志齐全；目标rank8约0.217/0.224s告警，但每次16个停机告警（1TP、15级联FP）。85%持续与单步500ms同样可混淆为停机；没有改原始规则掩盖它。
- 背景通信有效重测：双端3399次128MiB，约30s，correct=true，损失14.3%、无自动/停机告警，同时存在GPU争用，非纯链路故障。
- DeepSpeed0.19.7 stage0 dense兼容性16rank成功，只验证probe/共同回调，未声称DeepSpeed MoE/ZeRO全面验证。
- 官方Greyhound第三次gated运行0/0，实际完成precheck、ACF慢事件上报、profile和主动验证；local报8–15，global判comm。训练sleep在独立验证时不持续，且栈/模型不同，不形成可信同条件精确率比较。首次并行状态race和第二次训练早于precheck均LIMITATION。
- 四种EP/TP/PP/SP真实训练全部0/0；4项算法/norm单测通过；最终probe边界修复后约299万timed、0errors/lost。black/compileall及凭据精确扫描通过。

## 已修复的复现错误与数据边界

PYTHONPATH漏deps导致旧observer未停、陈旧heartbeat采样、重复OOM调度、背景通信退出不同步与barrier设备推断、TP2/SP原生norm缺适配、多stream有效标记边界、辅助NCCL退出规约污染分组，均已修复并记录。旧失败有INVALID/LIMITATION标记，不归因论文。

论文未说明的BOCD/Rbeast、20iter与1–2iter矛盾、CV解释、expertTP/标签选择均公开；不能证明与师兄不可提供的原代码二进制等价。没有做数周零误触、13.3B/Muon/TE同栈、10,000次微基准、纯网络分档或完整同栈Greyhound准确率对照；依用户快速小模型范围缩减，不声称论文全指标已达到。

## 运行与后续

所有GPU实验已结束；两节点GPU空闲、任务锁频已复位，证据results/final_server_state.json。模型checkpoint保留远端各run的train_rank*.jsonl.pt；本地保留检查结果，未重复下载大文件。任务控制器已退出；导出后释放本任务probe共享内存，保留容器/Redis用于重跑。

每次重跑必须新RUN编号，启动器拒绝已有运行且检查GPU空闲。scripts/suite.py、validate_final.py、validate_control.py的固定ID是本轮记录，不能原样覆盖重跑。新启动入口示例：

```bash
FLASH_RUNTIME=paper python -m scripts.experiment NEW_ID --args '--steps 120 --min-seconds 34 --seq 128 --batch 2 --fault delay --severity .45'
```

源码与证据保存本地Git；origin为用户给定auto_flash，未fetch旧代码、未push覆盖远端。下一阶段先讨论报告中的差距和解释，再由用户决定优化范围；本轮不自动开展新目标。
