# FLASH 第三章复现

在两台H800、16 GPU上从零实现论文第三章FLASH：C++ NCCL Probe、POSIX共享内存、两级Python控制器、Rbeast门控、按并行组的两阶段定位和fail-stop检测。以真实15M参数MoE完成前反向训练与故障测试。未读取此前复现尝试。

- [实验报告](docs/report.md)：实测结果、覆盖情况及尚未达到的论文指标。
- [规格与歧义](docs/spec.md)：论文参数、未定义细节的实现选择。
- [部署与重跑](docs/runbook.md)：离线资产、环境、运行命令与清理。
- [模型与来源](docs/provenance.md)：固定版本与论文环境差异。
- [阶段交接](handoff.md)：当前状态与后续入口。
- `results/summary.json`、`trials.csv`、`paper_equations.csv`：从原始证据生成的汇总；每个运行目录保留16 rank迭代、控制事件、实际退出码和NCCL日志。

```bash
export FLASH_RUNTIME=paper
python -m scripts.experiment NEW_RUN --args '--steps 100 --seq 128 --batch 2'
python -m scripts.experiment NEW_FAULT --args '--steps 100 --min-seconds 34 --seq 128 --batch 2 --fault delay --severity .45'
python -m scripts.summarize
python -m scripts.paper_metrics
```

每次使用新RUN编号。凭据只从本地被Git忽略的 `服务器.md` 读取。服务器工作目录为 `/home/campus/zhangweihao/flash-repro-20261015`。正式容器为 `flash-paper-20261015`，须显式设置 `FLASH_RUNTIME=paper`。仅实现FLASH，不涉及SHIFT、工业界调研或后续优化。
