# FLASH 第三章忠实性审计（v1冻结审计与覆盖边界）

本稿核对论文第三章与v1冻结实现（核心提交`db828f4`），区分确定的实现偏差、未验证范围和论文文本问题。首轮`pre_gradient_fix`运行保留为历史记录；修复后测试单独标为`post_gradient_fix`，不混合统计。核心实现已冻结；网络两档和四种并行拓扑的修复后补测已完成。详细运行条目见[测试结果明细](audit_test_results.md)。本报告只给出当前可审计的忠实性结论和覆盖边界，不宣称已逐项等价复现整篇论文。

## 追加确认：专家梯度归一化

审计末期复查训练fixture发现确定数学偏差：专家梯度allreduce后除EDP副本数；但官方Megatron-Core 0.12.1 [`distributed_data_parallel.py:244–276`](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.12.1/megatron/core/distributed/distributed_data_parallel.py#L244-L276)明确dense与expert最终都应除dense DP规模，因为专家梯度已跨EP汇总。已提取实际训练使用的 `scripts/gradient_sync.py` 修正，没有改变FLASH算法。

独立16卡EP4/DP16/EDP4检查`audit_gradient_equivalence_20261016_0034`双端退出0：14项实际dense/router/shared/expert梯度与独立HF fixture参考最大relative L2为8.5068e-7，专家范数比约1；旧分母在该配置会放大4倍（此旧值由线性缩放反推，未伪称重跑旧分母），主EP8配置相应放大8倍。原始`gradient_validation.json`保留。数值对照范围仅FP32小fixture、EP4/DP16/EDP4、TP1/PP1/CP1的rank0局部14项梯度；TP2/PP2尚无独立梯度数值对照。当前CLI固定CP1，不能宣称支持CP；未来启用CP必须同时改用包含CP的dense DP同步组，不能只扩大分母。AllReduce基准不受影响。修复前训练数据标为历史，不与修复后结果合并。

修复后另有两次800步16卡回归：健康强制剖析和rank8持续延迟自动定位。两次都收到800条迭代报告，剖析complete、lost=0、无fail-stop告警；健康强制剖析仍给出rank5单标签候选（score 0.8123），这是强制定位产生的误候选，不是自动门控误触。延迟运行在第80步注入、观测步83触发，唯一候选为rank8（score 0.9928）。四次修复后baseline/monitor训练组成ABBA顺序，每次16 rank均完成800步；首对baseline/monitor的16份参数分片比较均finite且最大差0。该有限ABBA与单次健康/故障回归不构成论文开销上界或总体准确率证明。运行明细见[测试结果明细](audit_test_results.md)。

## 已确认的实现问题与修复

| 论文要求 | 首版问题 | 本轮修改与验证 |
|---|---|---|
| §3.3.1 分组通信只在安全边界切换模式 | 同一group内部读取变化的mode/epoch会使调用分属不同采样阶段 | 外层GroupStart锁存mode/epoch，到GroupEnd完成后更新；嵌套group测试通过 |
| 多CUDA stream分别计时，结束后再等待 | 逐stream“记结束event→同步”会把第一个stream的CPU等待加入其他stream区间 | 先提交全部结束event，再分别同步；直接编译实际probe的mock CUDA契约测试在两节点通过 |
| §3.3.2 等待各节点采集、读取完整剖析 | 全局固定等待与本地稍晚启动存在竞态，可能读到未完成或少尾部的数据 | 相同epoch、两节点complete确认、在途group排空；不完整时输出incomplete_profile而不定位。修复后两次800步profile均complete、lost=0；范围仍限于这些回归 |
| §3.3.3 后台进程上报迭代 | 训练回调同步写Redis，给小模型增加CPU路径干扰 | 独立子进程写Redis，训练端短pipe消息；修复后健康/延迟运行的16 ranks各完成800步，后台收到800条；详项见测试结果明细 |
| 相邻变点去重 | 同一次Rbeast结果中的相邻候选未互相去重 | 按时间依次接纳，CPU回归通过 |
| 按rank聚合全部communicator记录 | 同hash同rank多个handle被最后一个覆盖 | 按记录次数加权聚合，保留hash代际，CPU回归通过 |
| 未匹配并行语义的group作为special | 未知group被静默忽略 | 完整成员集合生成special标签，仍排除已知EP，CPU回归通过 |
| 协同GPU计算与通信precheck后训练 | 首版只单独运行了precheck，启动器未自动设置关卡 | 默认两节点16卡GEMM结果校验和AllReduce校验通过后才启动训练 |
| 按框架真实通信识别并行语义 | 首版额外每步NCCL退出规约污染DP；改Gloo后仍增加人工CPU屏障 | 固定步数完全移除每步退出通信；仅有min-seconds条件才使用Gloo，全量新测试优先固定步数 |
| 仅在定位出慢rank时输出CRITICAL | 所有`diagnosis`（含采集不完整、无候选）曾被标为CRITICAL | 已改为仅fail-stop和有suspect的diagnosis为CRITICAL；incomplete_profile为WARNING，完整但无候选为INFO；10项CPU回归通过。修复后强制健康剖析有suspect，未单独覆盖GPU上的完整无候选分支 |

首轮OOM/timeout各16个停机告警中的15个额外告警，不能直接解释为15个健康rank误报。torchrun/NCCL可能已级联终止这些进程，而首轮没有同时记录存活状态。修复后OOM/timeout测试记录了PID、/proc进程状态和最后调用年龄；这些字段仅作证据，不参与原论文判定，不能掩盖fail-slow与通信沉默的混淆。首轮与修复后告警口径分别报告，不合并计算。

## 尚未完全覆盖的论文实现

- 目前实现论文允许的简化communicator模式：公开NCCL初始化ID哈希和框架成员集合。尚未实现“先解析NCCL私有ring/tree完整结构、失败自动降级”的可选完整路径，不能称为全部功能逐项等价。
- 验证范围为阻塞NCCL、目前训练实际调用的集合通信与Send/Recv。非阻塞ncclInProgress、CommSplit/共享communicator、CUDA Graph捕获等未覆盖，不能泛称所有NCCL使用方式均兼容；捕获跳过有计数。
- 模型使用官方Megatron-core模块，但训练fixture的梯度规约和优化器由本项目管理，未达到论文完整Megatron训练器、重叠调度、TE/FlashAttention/Muon栈。该差异属于复现集成范围，不能笼统归为硬件差异。
- 当前报告两阶段规则采用每标签最大max_avg_latency候选、rank最小均值及CV/.2阈值。论文未明确要求至少两个标签才上报，故不添加此门槛。第一版正常forced误候选rank6本来已有两个标签。

## 环境与模型差异：事实和待验证因果

已知硬件差异：本集群2×8 H800 80GiB，论文H200约141GB；本任务实际IB四个mlx5路径/节点，论文描述8张400G RoCE和另1张IB。软件现为PyTorch2.10/CUDA12.6/NCCL2.27.5，论文2.8a0/12.9/2.26.5；Megatron-core同为0.12.1。原生PyTorch norm/attention和AdamW与论文TE2.4/FlashAttention2.8/Muon不同。

首版真实小MoE约15.08M参数、序列128、微批2，明显不同于论文13.3B、4096序列、微批8、全局4096。小模型可能主要受Python调度、kernel launch和网络延迟影响，锁频对总迭代影响很小，20秒剖析下限对应数百迭代。这是需要大模型对照验证的因果假设，不能单凭型号差异就宣称解释了定位失败。

找到的官方开源近似模型为[Qwen1.5-MoE-A2.7B](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B)，固定revision `1a758c50ecb6350748b9ce0a99d2352fd9fc11c9`，14,315,784,192总参数、官方标称2.7B激活。其24层/2048 hidden/60专家top4与论文36层/1024 hidden/128专家top8不同；60不能被EP8整除，因此该对照用EP4。未找到可核验的论文13.3B/A1.4B同一检查点，不能把近似模型改名为论文模型。

官方Qwen1.5-MoE-A2.7B模型卡与固定revision的本地配置/权重索引已核对。完整权重下载当前为可续传暂停状态：857个范围块中125块完成，官方8个权重SHA256均未验证（0/8），权重未上传、完整模型未运行。该模型的完整训练格式转换和全模型logits等价验证仍未完成。独立的小型FP32 HF/Megatron转换fixture已通过，相对误差2.86e-7；这不代表完整Qwen权重转换通过，更不代表论文13.3B/A1.4B模型已测。该近似模型仅用于系统验证，不用于比较收敛质量。

## 论文自身的问题与可验证局限

1. §3.2称BOCD/在线变点，§3.3实际指定Rbeast；Rbeast公开实现是BEAST贝叶斯集成分解，并非同一Adams–MacKay run-length算法。按指定库复现，不能同时声称两种算法都精确实现。[Rbeast原仓库](https://github.com/zhaokg/Rbeast)。
2. >20%吞吐退化门槛与图3-9“5%退化召回72%”缺少解释。真实5%且没有其他放大或触发条件时，不应通过20%门槛；强制profile只能测试定位，不能算自动检测召回。
3. 详细设计采集 `max(2.2×迭代时间,20秒)`，评估称剖析约1迭代、总2–3迭代；两者非普遍等价，小模型尤其明显。右侧保护区1–2步与其他持续20步表述也不一致，需公开采用的规则。
4. rank级CV分支较明确：§3.2.3（印刷页31）说CV<0.1时跳过离群阈值筛选；多通信组的标签级CV分支在§3.3.2（印刷页34）只写“跳过筛选”，未说明是否继续对该标签做rank分析。当前实现遇到多组CV<0.1时跳过整个标签；这是公开的解释选择，不能据此判为确定实现错误。
5. §3.5.2自己指出绝对通信时长忽略数据量，不同消息大小、通信类型或调用频次混合会影响对比。不同rank到达时间也不一定由计算速度唯一决定，因而“通信最短者就是故障根因”是有条件的启发式。
6. 2×迭代无NCCL调用检测的是通信沉默，不能单独区分OOM根因、等待/级联退出、较长计算段或严重fail-slow。评估必须区分停机进程检测与唯一根因定位。
7. 表3-3缺完整config、词表、共享专家宽度、检查点地址，且“单个专家1.4B”等表述与给定专家矩阵规模不一致；不具备严格同模型重放条件。
8. 图3-4（印刷页30）示意图把慢rank纯通信时间标为120ms，上方正文举例为3ms。§3.3.1（印刷页33）称共享内存写入50–200ns、AllReduce通常100us至10ms，却推出占比<0.001%；相对100us实际为0.05%–0.2%，覆盖两端完整范围则为0.0005%–0.2%。这些文本/算术问题本身不证明算法失效。

## v1冻结时的测试覆盖与统计口径

修复前首轮数据继续保留为历史样本，包括旧健康误候选、旧fail-stop告警、早期AllReduce/训练开销、锁频与并行拓扑探索。它们对应旧源码或旧fixture条件，不与`post_gradient_fix`结果混合，也不能直接用来推断当前冻结版本的效果。

修复后已有的主要验证包括：10项CPU回归通过；实际C++ probe嵌套group/多stream契约在两节点通过；独立FP32 HF/Megatron专家梯度对照通过；两次800步16卡健康强制剖析与持续延迟回归均收齐16-rank日志和800条后台报告，profile complete且lost=0；四次800步baseline/monitor ABBA运行均正常完成，首对checkpoint分片逐项相同。健康强制剖析仍会产生误候选，持续延迟样本能定位rank8；这些是有限观察，不代表一般化准确率或论文开销上界。

冻结版本另完成两档背景NCCL混合争用和四种600MHz并行拓扑，每拓扑400步、强制剖析。四拓扑均完整采集且无丢记录，候选分别为[9]、[4]、[0,9]、[4,8]，目标为rank8，未复现普遍唯一准确定位。首轮六档频率全扫描仍只作历史证据，未在冻结版本完整重跑。论文万次统计、数周零误报、论文规模同模型复现、完整Qwen权重转换/训练和Greyhound全功能逐项等价均未完成。标称注入强度与实测吞吐损失分别报告；强制剖析与自动触发分开；共享GPU背景NCCL同时有算力/显存带宽争用，不能冒充纯链路故障。完整逐运行数据、参数与口径见[测试结果明细](audit_test_results.md)。
