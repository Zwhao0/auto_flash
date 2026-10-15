# FLASH 第三章复现

在两台H800、16 GPU上从零实现论文第三章FLASH：C++ NCCL Probe、POSIX共享内存、两级Python控制器、Rbeast门控、按并行组的两阶段定位和fail-stop检测。以真实15M参数MoE完成前反向训练与故障测试。未读取此前复现尝试。

- [v1报告](docs/v1_report.md)：审计后结论、修复和验证、未完成项与暂停状态。
- [忠实性审计](docs/fidelity_audit.md)：实现问题、环境/模型差异、论文问题分别记录。
- [扩展测试](docs/audit_test_results.md)：较长运行、重复测量与完整性检查。
- [首轮开发记录](docs/report.md)：修复前结果、覆盖情况及未达到的论文指标。
- [Qwen转换验证](docs/qwen_validation.md)：官方近似开源模型与论文模型的差异和真实验证范围。
- [规格与歧义](docs/spec.md)：论文参数、未定义细节的实现选择。
- [部署与重跑](docs/runbook.md)：离线资产、环境、运行命令与清理。
- [模型与来源](docs/provenance.md)：固定版本与论文环境差异。
- [阶段交接](handoff.md)：当前状态与后续入口。
- `results/audit_campaign_summary.json`：最终审计测试汇总；`v1_*check.json`记录发布检查。`results/summary.json`、`trials.csv`、`paper_equations.csv`属于首轮历史汇总，不能与修复后的训练结果混算。各运行目录保留rank日志、控制事件、实际退出码和NCCL日志。

```bash
export FLASH_RUNTIME=paper
python -m scripts.experiment NEW_RUN --args '--steps 100 --seq 128 --batch 2'
python -m scripts.experiment NEW_FAULT --args '--steps 800 --seq 128 --batch 2 --fault delay --severity .45 --fault-step 80'
python -m scripts.verify_evidence --final
python -m scripts.check_core_sources
```

每次使用新RUN编号。凭据只从本地被Git忽略的 `服务器.md` 读取。服务器工作目录为 `/home/campus/zhangweihao/flash-repro-20261015`。正式容器为 `flash-paper-20261015`，须显式设置 `FLASH_RUNTIME=paper`。仅实现FLASH，不涉及SHIFT、工业界调研或后续优化。
