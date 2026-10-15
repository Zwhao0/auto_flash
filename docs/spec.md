# FLASH 第三章复现规格

依据：本地中文论文第三章（印刷页 19–47），GreyHound 官方仓库 https://github.com/wutianyuan1/Greyhound 仅作参照和对照基线。本次不读取旧复现代码或结果。

## 设计与实现验收

|论文位置|约束|实现与验收|
|---|---|---|
|3.2.1、3.3.1|C++ LD_PRELOAD NCCL Probe|真实拦截 NCCL；训练正确性对照；monitor 无 CUDA Event|
|3.3.1|共享内存环形缓冲、写端锁、读端无锁|每进程 POSIX shm，原子提交序号、覆盖计数；Python 消费|
|3.3.1|Group 级 CUDA Event、复用事件|每 group/stream 起止事件，GroupEnd 后完成计时；原通信语义保持|
|3.3.1|简化 communicator 模式|地址、unique-id 哈希、global rank；不逆向 NCCL 内部结构|
|3.2.2|rank0 迭代耗时、Redis 最多 500 条|框架显式迭代回调；初始化上报 parallel states|
|3.2.2、3.3.2|30 点窗口、Rbeast、概率 .8、退化 .2|跳过前 3 次；z-score >3 过滤；恢复比 .75；相邻变点距离 2；右侧保护 2|
|3.2.3|本地 100–500ms 轮询；profile=max(2.2*iter,20s)|0.1s 轮询、严格 20s 下限；Redis 协调两节点|
|3.2.3、3.3.2|按标签选 max_avg_latency 最大组，组内取均值最小 rank|clique/rank CV .1；rank 相对均值低至少 20%；跨标签证据|
|3.2.4|2*iteration 内无新 NCCL 调用则 fail-stop|动态时间窗，无额外存活心跳；记录级联误报|
|3.4.3|吞吐、4–128MiB AllReduce、profile 开销与误触|每项少量开发轮次、预热与测量隔离；保留原始数据|
|3.4.4–3.4.6|降频、背景通信、OOM/timeout、四组并行配置|真实模型训练故障注入；报告 TP/FP/FN、延迟、score；未运行项目明确标注|

## 论文歧义及处理（不修改论文算法以追求指标）

1. **BOCD 与 Rbeast 不等价**：Rbeast 实现 BEAST 的滑窗贝叶斯分段，不是 Adams–MacKay run-length 递推。采用论文指定 Rbeast，并记录此命名问题；缺库时直接报错，不偷偷换算法。
2. **持久性矛盾**：3.2.2 用 recovery_ratio；3.4.3 提及持续 >20 iter；3.4.5 称 1–2 iter。默认采用详细设计中的 recovery_ratio、右侧保护 2；提供可选 min_persistence=20 以测量差异，不能把两种结果混报。
3. **退化量纲**：论文没有给出 degradation 公式。使用吞吐损失 1-baseline_ms/degraded_ms，与 GreyHound 的相对差/max 一致；输出原始均值以便审查。
4. **CV 分支歧义**：3.3.2 多组 CV<.1 时“跳过筛选”含义不明。采用保守解释：不产生该标签候选；单组直接做 rank 分析。与 3.2.3 的 CV 证据不足就跳过一致。
5. **20% 阈值与 5% 召回**：严格门控很可能不触发 5% 持续退化；如观察到漏报，报告为论文阈值的预期限制，不能关闭门控后计作端到端成功。
6. **Fail-stop 级联**：同步阻塞会让健康 rank 也停止 NCCL 调用。严格实现无调用判定，报告所有告警，不用注入目标泄露给检测器。
7. **小模型/硬件差异**：用户指定 H800、小型真实 MoE；论文为 H200、13.3B。绝不比较绝对 TFLOP/s，也不能把小样本通过当作论文准确率的统计复现。
8. **无侵入口径**：probe 透明，但迭代与并行组需要框架回调，不宣称零改动；监控集成行数单独报告。
9. **Pre-check**：论文提及但未给完整实现，实施独立计算/通信预检，输出环境与基线，不添入运行时诊断规则。
10. **耗时精度**：Group 内多 stream 必须分别计时，不能把一条 stream 的时间写到其他 stream；同 stream 的 group 操作共享时间是论文明确选择。
11. **模型通信栈差异（第一版）**：GreyHound 官方所附 Megatron 分支的 MoE dispatcher 使用 AllGather/ReduceScatter 而非现代 All-to-All。它保留真实的 EP、top-k 路由、专家梯度同步，但不能用来声称验证了论文讨论的 All-to-All 噪声。需将该项明确标为未覆盖，后续优先切换论文版本 Megatron-core0.12.1 完成剩余验收，而不是掩盖差异。
12. **早期环境排查记录**：第一轮 GreyHound 镜像容器未映射 RDMA 设备，AllReduce 微基准暴露 4MiB 约 5ms、128MiB 约 113ms 的异常；此前 early 结果只用于功能验证，不能作网络性能结论。正式实验改用具有 /dev/infiniband 的独立容器，并检查实际 NCCL transport 日志。
13. **修复控制器采样陈旧问题**：早期 `paper_profile_healthy` 在 drain、聚合、Redis 上报之前读取 heartbeat，再在耗时处理后比较当前时间，可能把控制器自身处理时间算成 rank 的静默。现改为判定前重新读取共享内存 heartbeat；这是复现实现错误修复，不属于论文算法优化。早期该次 fail-stop 告警不可用于认定论文缺陷；修复后需复测。
14. **无故障强制剖析不是自动误触率**：即使 forced-profile 产生可疑 rank，也应单独作为定位器在正常负载下的离群观察；只对自动门控触发的告警计算端到端慢故障误报。无调用 fail-stop 告警另行统计。
15. **实验框架辅助通信**：为让 `--min-seconds` 的两节点训练一致结束，fixture 每步额外执行一次全局 MIN AllReduce；该 communicator 与 dense-DP 成员集合相同，简化拓扑匹配可能赋相同标签。原始数据保留 communicator hash；clique 比较可能同时包含不同消息规模，此因素须在评价 Eq.(3-1) 时说明。诊断器没有读取注入目标。
16. **GPU sleep 名义强度与全局损失不同**：额外等待可先消耗目标 rank 原本的同步等待空隙，故名义25%不等于端到端25%；用实测前后迭代中位数报告实际吞吐损失。不得据此错误归因“25%故障漏检”。
17. **停机实验退出**：真实 OOM / NCCL watchdog timeout 的检测与原训练进程长时间清理分开记录；结束残留进程不计作检测行为，不实现恢复。现代 Megatron subgroup timeout 显式设90s，避免其默认30min拖慢开发测试。
18. **混合并行标签**：PP>1 时 DP/TP 标签按 pipeline stage 分开（如 ppstage0_dp），避免不同阶段的消息模式被直接比较；expert_dp 作为独立标签，EP通信只记录不参与定位。expertTP显式1，其 singleton 不提供定位证据；第三章未给 expertTP，需披露这一选择，不能保证论文图3-10的“5组证据”计数。
19. **SP归一化适配**：Megatron-core0.12.1的无TE/Apex原生WrappedTorchNorm禁止SP。实现每token原生RMSNorm，sequence_parallel权重梯度按TP求和，通过分片/非分片forward及梯度等价测试。TP2两拓扑真实训练成功；这是框架兼容修复，不改变FLASH定位算法。
20. **物理锁频与采样边界**：早期频率组与四拓扑运行按iteration60放行forced，锁频完成事件可能晚约0.8s；该20秒窗口含转换区间，仅作探索数据。新的clock入口在锁频完成后放行，已保留旧数据与实际频率查询，不算forced延迟为自动检测时延。1965请求实际采样1830、1600请求1605，不能把请求值当测量值。
21. **背景通信fixture设备映射**：2rank均使用本地GPU0，必须向init_process_group和barrier显式声明device0，避免rank1被推断到GPU1并建立不同communicator。有效重测为paper_background_devicefixed，两端3399次128MiB，30秒，correct=true；先前失败并非FLASH算法结果。
22. **多stream记录标记**：CUDA group记录完成后将valid flag改为1；遍历其他stream时必须同时检查未处理标记，避免CUDA特殊stream handle与valid flag数值重合导致重复赋值。final probe加此边界修复并重新构建验证；主实验未观察到该特殊handle情形，未据此改变旧结果。
23. **去除fixture辅助NCCL干扰**：最终发现退出用world MIN AllReduce可被简化成员标签当作DP组，影响组内/组间分数。新默认使用独立Gloo CPU控制组，模型NCCL调用保持不变。新对照paper_gloo_*覆盖baseline/monitor参数一致性、healthy forced、持续delay与完成锁频后forced采样；其他旧扫描不重新标成同条件验证。属于测试方法修正，不是修改FLASH算法。新旧绝对吞吐不能混比，因为CPU控制成本不同。
