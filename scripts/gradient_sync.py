"""Match Megatron-core DDP's mean-loss gradient scaling for dense and experts."""

import torch
import torch.distributed as dist
from megatron.core import parallel_state as ps


def sync_gradients(params, group):
    have = [p for p in params if p.grad is not None]
    if not have:
        return
    flat = torch.cat([p.grad.reshape(-1) for p in have])
    dist.all_reduce(flat, group=group)
    # Expert gradients already sum token contributions routed across EP.
    # Both buffers require final 1/dense_DP scaling, not 1/expert_DP.
    # Upstream core_v0.12.1 distributed_data_parallel.py:244-276.
    flat.div_(ps.get_data_parallel_world_size(with_context_parallel=True))
    offset = 0
    for param in have:
        param.grad.copy_(flat[offset : offset + param.numel()].view_as(param))
        offset += param.numel()
