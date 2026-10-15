# 来源、模型与环境

本项目只针对用户提供的 `中文论文.pdf` 第三章，印刷页19–47。从空目录开始编写；未读取用户此前的复现代码、日志或结果。复用了两节点已有的基础容器镜像和系统依赖，没有复用旧系统实现。

|项目|本次正式实验|论文|
|---|---|---|
|GPU|2节点×8 H800 80GB，NV8|2节点×8 H200 PCIe 140/141GB|
|网络|每节点4条mlx5路径，NCCL日志确认NET/IB，RoCE|8×ConnectX-7 400G和1×ConnectX-6|
|驱动|580.178.04|570.86.15|
|PyTorch/CUDA/NCCL|2.10.0+cu126 /12.6 /2.27.5|2.8.0a0 /12.9 /2.26.5|
|Megatron-core|0.12.1官方tag|0.12.1|
|Rbeast|0.1.25|指定Rbeast，未明确版本|
|模型|真实15,083,776参数GPT MoE|Qwen 13B A1.4B，13.3B参数|
|数值/算子/优化器|FP32，原生Torch attention/RMSNorm，AdamW|TE2.4、FlashAttention2.8、Muon|
|数据|真实Shakespeare字节next-token，随机初始化|finedweb-edu-350B-qwen|

小模型为4层、hidden256、8个attention heads、GQA4、8路由专家top2、1共享专家、SwiGLU、RMSNorm、RoPE；micro batch2、sequence128、2个microbatches。主配置EP8/TP1/PP1，expertTP明确1，原生All-to-All专家dispatch。它具有真实可训练权重、token-dependent router、前反向和专家梯度同步，不是通信模拟，也不声称是预训练检查点。TP2时SP开启；无TE/Apex的原生RMSNorm适配及权重梯度规约经过分片/非分片等价性测试。

公共来源及固定版本：

- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM/tree/core_v0.12.1)：commit `a845aa7e12b3a117e24c2352b9e3e60bad2e3a17`，正式训练框架。
- [GreyHound官方源码](https://github.com/wutianyuan1/Greyhound)：commit `d641b3997c17de700e1ef7d37e353418df7e336d`，仅作公开参考和独立基线尝试。硬件/路径改动保存在 `patches/greyhound-hardware.patch`。
- [Rbeast](https://github.com/zhaokg/Rbeast)：按论文指定库执行滑窗贝叶斯分段，不宣称等价于run-length BOCD。
- [Shakespeare数据](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)：离线内容和SHA256见 `data/manifest.json`。
- [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)：0.19.7，只用于独立stage0框架兼容性smoke，不能替代主MoE实验。

正式容器 `flash-paper-20261015` 使用已有镜像 `registry.reinfra.cn/reinfra/campus/docker-images/pytorch-runtime:2.10.0-cu126-r1`。`flash-repro-20261015` 使用 `tianyuanwu/greyhound:ae`，只用于早期排障、构建工具和任务Redis。独立官方基线容器 `flash-greyhound-20261015` 使用后者，PyTorch2.1.0a0、CUDA12.2、NCCL2.19.3、Rbeast0.1.19及上游旧Megatron；不能直接作同栈性能优劣结论。

远端专用目录：`/home/campus/zhangweihao/flash-repro-20261015`，挂载 `/flash`。Redis使用任务端口16379和 `flash:RUN:` 命名空间。官方基线只在独立容器内使用私有IPC及临时端口6379。

NCCL与probe同时预加载，probe在前，避免PyTorch局部动态库加载使RTLD_NEXT无法解析。实验的monitor/baseline使用相同NCCL库。新运行保存源码与二进制SHA256；早期缺该文件的运行仍保留真实日志，不伪造追溯元数据。
