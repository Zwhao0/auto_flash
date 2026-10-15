# FLASH 第三章独立审计记录

范围：对照第三章印刷页23–45、`tmp/pdfs/thesis.txt`、原PDF页面图像，以及当前`csrc/probe.cpp`和`flash/*.py`。本审计未运行GPU或修改核心实现；PDF页码均指论文印刷页，代码行号指本次核对时的工作树。

## 结论

主流程（Rbeast窗口检测与过滤、Profile模式切换、Group级CUDA Event、按通信组和rank两阶段分析、跨标签证据、基于NCCL静默的fail-stop）与论文的明确描述大体一致。审计先发现一处确定的诊断严重级别错误，已由主线程修复；后续训练集成核对又确认专家梯度归一化分母错误，现已修复并通过独立的小模型梯度对照。10项CPU回归通过，实际C++ probe契约测试在两节点通过；修复后健康强制剖析与延迟定位各有一次800步16卡验证。CV分支、完整NCCL拓扑路径和训练框架覆盖范围仍有公开的解释/范围限制，不能描述成逐项等价。

v1核心冻结提交为`db828f4`，发布标签还包含后续测试证据与文档。官方Qwen1.5-MoE-A2.7B完整权重下载现为可续传暂停状态，8个权重SHA256均未核验（0/8）；完整转换、全模型logits等价和大模型测试尚未完成。小型独立FP32 HF/Megatron转换fixture相对误差为2.86e-7，小型独立梯度对照最大relative L2为8.51e-7；这些只证明对应fixture路径，不代表大模型或论文13.3B/A1.4B配置已测试。网络两档和四种并行拓扑的冻结版本补测已完成，结果见`audit_test_results.md`；本审计不宣称论文全矩阵或完整算法已复现。

## 发现与证据

| 分类 | 论文证据 | 当前实现/事实 | 结论与状态 |
|---|---|---|---|
| 已修的实现错误 | §3.2.3（印刷页31）将CRITICAL输出与慢rank及其证据关联 | `flash/global_controller.py:31–41, 79–95`此前将所有diagnosis标为CRITICAL，包括`incomplete_profile`和无候选结果 | 主线程已修复：有suspect的diagnosis和fail-stop为CRITICAL；incomplete为WARNING；完整但无候选为INFO。10项CPU回归通过。修复后健康强制profile仍有候选，因此不能把该次GPU运行说成验证了完整无候选分支。 |
| Fail-stop指标口径 | §3.2.4/§3.3.2（印刷页31、34）规定动态窗口内没有新NCCL调用即告警 | `flash/local_controller.py:124–144`用最后调用时间与`2×rank0最近迭代时长`比较，并上报告警；进程状态和PID作为附加字段 | 规则与论文一致，但它检测的是通信静默，不是根因或实际进程死亡。OOM/timeout后其他rank的级联静默可能同时告警，不能只凭告警数把其余15个全计为健康误报或停机真阳性。 |
| CV分支 | §3.2.3（印刷页31）明确rank级CV<0.1时跳过百分比阈值筛选；§3.3.2（印刷页34）对多通信组标签的CV<0.1只写“跳过筛选” | `flash/analysis.py:77–88`对多组低CV直接跳过该标签；单组则跳过组筛选并进入rank CV判断 | rank级规则符合明确文字。多组低CV后是否继续rank分析没有被论文写清；当前行为是可复核的解释选择，不据此判定确定错误。 |
| 近似模型参数 | 表3-3（印刷页37）给出总13.3B、激活1.4B、36层、hidden 1024、128专家、top-8 | 官方Qwen1.5-MoE-A2.7B索引给`total_size=28,631,568,384`字节；其BF16权重按2字节/参数对应14,315,784,192参数；模型卡标称14.3B总量、2.7B激活，配置为24层、hidden 2048、60专家、top-4 | 审计稿准确地区分它与论文配置。60不能整除EP8，故该近似模型的EP8复现不可直接照搬；选EP4是集成适配，不是论文原模型验证。完整权重下载暂停、0/8权重SHA256已验证；完整转换/训练未完成。 |
| 论文模型表述疑点 | 表3-3后（印刷页37）称“单个专家参数量1.4B”；同表列专家中间维896、hidden1024、36层、128专家 | 按普通SwiGLU三矩阵估算，每层每专家约`3×1024×896=2,752,512`参数，36层同一专家约0.099B；均不是1.4B | 该文字与表内架构数量级不一致，且缺模型config、词表、共享专家宽度和检查点地址，不能严格重放。这里只判定表述不自洽，不替作者推断本意。 |
| 性能开销算术 | §3.3.1（印刷页33）称共享内存写入50–200ns、AllReduce约100us–10ms，并声称写入占比<0.001% | 50ns/100us=0.05%，200ns/100us=0.2%；对完整时长范围极值为0.0005%–0.2% | 论文给出的数值范围不能推出普遍低于0.001%。`docs/fidelity_audit.md`现已按量纲记录。 |
| 图文数值不一致 | 图3-4（印刷页30）顶部示意慢rank纯通信时间为120ms；同页上方正文示例说慢rank测得3ms | 已渲染核对原PDF印刷页30 | 审计稿所列不一致属实，是论文内部示例不一致，不代表算法实现失效。 |
| 集成覆盖边界 | §3.3.1–3.3.2（印刷页33–34）描述完整/简化communicator信息兼容与自动降级 | `csrc/probe.cpp:276–352`拦截当前支持的初始化、集合通信、Send/Recv和Group API；不解析NCCL私有ring/tree，也未实现CommSplit/共享communicator完整路径；CUDA Graph捕获会跳过计时 | 当前实现覆盖范围需作为集成限制；不应宣称对任意NCCL程序或论文完整/简化自动降级都已验证。 |
| 训练集成范围 | §3.3.3及表3-5（印刷页35、37）描述Megatron回调、Qwen训练配置和Muon | 当前fixture以官方Megatron模块构图，但项目自管梯度同步与AdamW；TE/FlashAttention/Muon及论文规模训练未完成 | 属复现集成/模型范围差异，不是硬件差异；小模型验证不可代替论文配置结果。 |

更完整的规则、兼容范围和实验边界见[`fidelity_audit.md`](fidelity_audit.md)与[`spec.md`](spec.md)。

## 后续核对：专家梯度归一化

本项目手动对梯度做sum all-reduce，修复后dense与expert梯度都除以`get_data_parallel_world_size(with_context_parallel=True)`，见 [scripts/gradient_sync.py](../scripts/gradient_sync.py#L8)。这一做法与官方Megatron-Core固定tag [`core_v0.12.1`的distributed_data_parallel.py](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.12.1/megatron/core/distributed/distributed_data_parallel.py#L244-L292)的sum-reduce分支一致：`average_in_collective=False`时dense和expert都按`1/dp_size`缩放，再分别对DP/EDP求和（[上游行271–292](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.12.1/megatron/core/distributed/distributed_data_parallel.py#L271-L292)）。因此，修复前只把expert梯度除以EDP size，不符合该mean-loss目标；在本次EP4、dense DP16、EDP4条件下会令expert梯度放大4倍。若改用`average_in_collective=True`，上游采用的是expert预乘`EDP/DP`再在EDP组内平均；不能把当前手动sum路径的结论直接套到那种集体平均配置。

独立验证记录为`results/audit_gradient_equivalence_20261016_0034`：双节点退出码均为0，rank0将MCore FP32 fixture梯度与同输入、同权重的独立HF fixture模型参考比较14项dense、router、shared-expert和expert梯度，最大relative L2误差`8.51e-7`；记录中dense DP=16、EDP=4，回算旧分母的expert梯度norm ratio约为4。验证代码调用训练实际使用的同步函数，见 [validate_qwen_gradients.py](../scripts/validate_qwen_gradients.py#L17)。

结论限于该小型FP32 fixture和已抽查的14项梯度，不代表完整Qwen权重、Muon或所有参数的长程训练已验证。该数值对照使用TP1/PP1/CP1；已有TP2/PP2训练拓扑并非独立梯度等价测试。归一化轴本身与TP/PP维度正交，但更高TP/PP下的端到端梯度等价仍未由该对照证明。当前训练入口未传`context_parallel_size`，Megatron-Core默认值为1；且`train_moe.py`取得的dense组是默认不含CP的DP组，而修复分母按含CP的DP world size计算。上游dense DDP在启用CP时使用[含CP的DP组](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.12.1/megatron/core/distributed/distributed_data_parallel.py#L279-L285)，所以CP>1不在当前验证范围；若以后启用CP，需让dense同步组也包含CP ranks并另行验证。
