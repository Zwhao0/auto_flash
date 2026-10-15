# FLASH v1 阶段交接

更新：2026-10-16 Asia/Shanghai。第一版按当前覆盖冻结，发布位置为 [Zwhao0/auto_flash 的 v1](https://github.com/Zwhao0/auto_flash/tree/v1)。**完成发布后暂停；没有用户新的继续指示，不重启训练、下载、调优或工业界调研。**

## 目标与当前判断

用户要求尽量按中文论文第三章复现原始FLASH，允许论文歧义、未完善内容和未达指标，但须保留证据、先修确定的复现问题。没有读取用户指定不看的早先复现尝试。当前主流程已有两节点16卡真实MoE训练、采集、门控与定位验证，可以冻结为可审计的第一版；不能宣称全部论文功能、原13.3B模型或统计指标已复现。

- 当前结论：[v1报告](docs/v1_report.md)、[扩展测试](docs/audit_test_results.md)。
- 忠实性和归因：[fidelity_audit](docs/fidelity_audit.md)、[独立审计](docs/audit_findings.md)、[规格选择](docs/spec.md)。
- 首轮历史：[report](docs/report.md)，保留原始失败，不与修复后数据混算。

## 冻结源码与证据

核心冻结提交`db828f4`；v1发布提交另含最终测试脚本、证据、汇总和文档。修复覆盖group模式/epoch、多stream计时、采集完成竞态、异步迭代上报、聚合/标签、precheck与训练额外退出通信。训练专家梯度分母修为dense DP规模，独立EP4/DP16/EDP4 FP32参考14项最大relative L2为8.51e-7；范围不包含完整大模型或TP2/PP2梯度等价。

- `results/audit_campaign_summary.json`：逐项测试汇总；新运行使用fresh ID和post_gradient_fix标记。
- `results/v1_evidence_check.json`：32个审计目录，已知无效/中止记录单列，PASS仅指档案检查。
- `results/v1_core_source_check.json`：20个修复后运行的核心源码及构建库一致。
- `results/v1_runtime_hashes.json`：两节点源码、Probe/注入器构建库和实际Torch/CUDA/NCCL版本。
- `results/audit_unit_tests_final.log`、`audit_probe_contract.log`：两节点10项CPU回归及实际C++ Probe契约通过。

最终训练ABBA每次800步，baseline/monitor首对16分片最大参数差0；healthy forced仍有rank5候选；45%名义延迟唯一定位rank8，实际吞吐损失约40.81%。85%及500ms瞬态发生存活进程fail-stop告警，按原规则保留。OOM与90秒NCCL timeout均有真实错误且自然非零退出，未人工终止。两档背景通信各800步，实测端到端损失1.42%/14.57%，未达20%门槛。四种600MHz拓扑各400步、forced采集完整，候选[9]/[4]/[0,9]/[4,8]，实测损失2.43%–4.47%；不能当严重全局退化准确率统计。

## 资源状态与安全访问

本项目GPU训练/控制器/下载已停止；任务锁频复位、新增共享内存已清理。容器、Redis及检查点保留，状态以`results/v1_server_state.json`时间点为准。检测到其他主机任务，不属于本项目，未处理；不要据此认定机器可独占。

两节点任务目录`/home/campus/zhangweihao/flash-repro-20261015`，容器内`/flash`。正式运行必须设`FLASH_RUNTIME=paper`（容器`flash-paper-20261015`）；不设置会落到旧工具容器。任务Redis16379位于旧工具容器，保留该服务。凭据只通过`scripts.remote.connect`在内存读取本地被忽略的`服务器.md`，禁止cat/sed整个文件或写入日志。发布前检查包括工作树和Git全部可达blob的凭据扫描。

## Qwen与剩余限制

论文13.3B/A1.4B没有可核验原检查点；官方近似Qwen1.5-MoE-A2.7B是14.3B/A2.7B、60专家top4，不能冒充原模型，不能EP8。完整权重下载已暂停保留断点，0/8完整分片SHA256验证，未上传、未大测；状态见`results/qwen_download_status.json`。只有小型独立HF/MCore前向和梯度验证，见`docs/qwen_validation.md`。不自动恢复下载。

仍有限制：未实现可选NCCL私有ring/tree完整解析与自动降级，未覆盖CommSplit/非阻塞NCCL/CUDA Graph计时；原生MCore模块、自管梯度/AdamW不等价完整Megatron/TE/FlashAttention/Muon训练栈。六频率扫描只有旧fixture历史证据；无数周/万次统计、纯链路故障或同栈Greyhound完整精确率召回率对照。后续完善必须由用户另行启动。

## 后续协作约定

已采纳用户建议于`AGENTS.md`：适合的独立重复工作优先GPT 6 Luna Max子智能体，主窗口验收；一个协调者串行拥有全部16卡实验；上下文较长时先更新本文件，再用新窗口交接。未来开始任何GPU工作先确认外部任务与GPU占用，不能修改无关任务。
