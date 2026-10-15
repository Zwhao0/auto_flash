# 官方 Qwen 开源模型导入验证（v1范围）

采用 Qwen 官方 [Qwen1.5-MoE-A2.7B](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B)，revision `1a758c50ecb6350748b9ce0a99d2352fd9fc11c9`。v1收尾时完整权重下载已暂停并保留断点，8个完整分片均尚未完成SHA256校验、未上传服务器，完整模型没有测试。状态见 `results/qwen_download_status.json`。

导入器 `scripts/qwen_support.py` 保留原模型层数、RMSNorm、RoPE、QKV偏置、GQA、路由全专家softmax后topk（不再归一化）、路由/共享专家SwiGLU和共享专家sigmoid门控。每个目标参数必须有来源且形状相同；未映射参数直接报错。QKV按Megatron分组顺序重排，专家按EP rank切分。当前验证转换路径支持TP1/PP1，完整60专家用EP4。

先用独立Transformers实现生成7609088参数随机Qwen2MoE测试检查点（2层、256hidden、8专家top2、共享专家），再在2节点16卡、Megatron EP4加载相同权重。`audit_qwen_conversion2` 的rank0转换验证结果：

| 指标 | 结果 |
|---|---:|
| FP32 logits相对RMS误差 | 2.8613453650905285e-7 |
| logits余弦相似度 | 1.0 |
| 最大绝对误差 | 3.5762786865234375e-7 |
| 8个位置最高token一致率 | 1.0 |

通用CLI元数据的默认layers=4未参与HF构图；实际层数来自检查点config，为2层，导入张量只有layer0/1。

随后完成4步真实前向、反向、梯度规约和AdamW更新。原始结果 `results/audit_qwen_conversion2/train_rank0.jsonl`；两节点执行状态已确认均为0；完整日志已补齐，conversion_checks.json核实全部16rank各完成4步且转换校验通过（首次导出因SSH断连中止，原子下载重试后恢复）。该小模型验证不替代完整BF16检查点的独立logits对照，不代表论文原模型已经复现。

第一次 `audit_qwen_conversion` 因检查点生成受到可选DeepSpeed与Transformers循环导入影响、文件缺失而失败，保留为无效开发运行。DeepSpeed已经从共同依赖目录隔离到 `/flash/deps-optional`，其兼容性专用脚本显式加载，避免无关可选包影响Qwen原实现；未修改Transformers模型计算。

导入路径为完整大模型预留BF16参数与AdamW（状态跟随参数dtype），与论文Muon存在差异。v1未运行该配置，不能推断其性能、显存或收敛质量；如用户以后授权运行，必须另行记录具体序列、微批、各rank参数数、实际峰值显存和校验结果。


## 训练梯度数学验证

后续发现fixture的专家梯度同步分母错误（应同dense除DP，而非除EDP），已修复并加入独立16卡测试 `scripts/validate_qwen_gradients.py`。`audit_gradient_equivalence_20261016_0034` 比较EP4/DP16/EDP4实际训练同步路径和完整HF模型，覆盖embedding、output、router、共享门控与本地两个专家的FC1/FC2，共14项；最大relative L2为8.5068e-7，范数比约1，双端0/0。校验用eval排除训练专有aux loss，验证主语言模型损失的反向与跨EP/DP归一化，不代表完整BF16大权重验证。原始conversion2的4步训练用旧分母，其前向等价结果仍成立，训练数学结论以本次追加梯度检查为准。
