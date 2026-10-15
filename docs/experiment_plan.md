# 少量开发实验计划

所有正式实验均两台 H800、16 rank，Megatron-core0.12.1，真实 top-2/8 专家和 1 共享专家 GPT MoE，EP All-to-All，固定种子和真实字节语料。一次/条件，误差只描述观察值，不给统计准确率保证。

1. 正确性：相同模型和数据，baseline/monitor 120步，16 分片 checkpoint 对比。
2. 无侵入：统计训练框架的两个回调；PyTorch collective 和 Megatron 集成；DeepSpeed 如未实际运行则明确写未测。
3. 监控吞吐：预热5步，用完整连续稳态墙钟时间统计 tokens/s，包含上报与循环控制开销。
4. AllReduce：4/8/16/32/64/128MiB；20 次预热，100次测量，逐次检查规约数值；保留16rank分布。
5. 剖析：正常训练 forced-profile，20秒下限；检查采样计数、丢失、CUDA错误、无故障候选、分段吞吐；forced 不计入自动检测成功。
6. 定位：5/25/45/65/85% 名义退化，GPU sleep 校准，比较实际迭代损失；真实 GPU 锁频1965/1600/1200/900/600/345MHz；记录逐组与逐rank分数、TP/FP/FN。
7. 网络：独立跨节点 NCCL 背景流量；结果标注同时存在的 GPU 争用，不能当作纯链路故障。
8. 误触：正常样本、单步瞬时迟延；持久性阈值保留论文默认。
9. 停机：实际 GPU 分配 OOM；NCCL调用阻塞/timeout；分别记录注入目标和全部告警，保留级联FP。
10. 混合并行：EP8/TP1/PP1、EP4/TP2/PP1、EP4/TP1/PP2、EP4/TP2/PP2；expertTP1、SP=TP；600MHz按论文，强制profile仅补充定位信号验证。
11. GreyHound：编译官方baseline并尝试运行；仅允许机械硬件配置和集成回调适配。失败保留日志，不能以简化近似或论文数字替代实测对比。

环境差异：PyTorch2.10/CUDA12.6/NCCL2.27.5、native torch attention/norm、AdamW，不同于论文2.8a0/12.9/2.26.5/TE2.4/Muon。目标是原FLASH机制在用户指定小模型上的复现，非13.3B训练数值复现。
