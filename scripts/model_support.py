"""Framework-only native RMSNorm adapter when Transformer Engine/Apex is absent."""

import torch
from torch import nn
from torch.nn import functional as F


class SequenceParallelRMSNorm(nn.Module):
    def __init__(self, config, hidden_size, eps=1e-5, **kwargs):
        super().__init__()
        if config.normalization != "RMSNorm" or config.layernorm_zero_centered_gamma:
            raise ValueError("The fixture adapter supports ordinary RMSNorm only")
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=config.params_dtype))
        self.weight.sequence_parallel = config.sequence_parallel
        self.eps = eps
        self.hidden_size = hidden_size

    def forward(self, x):
        # Sequence partitioning leaves each token's hidden dimension intact.
        # Weight gradients are summed across TP in train_moe.sync's SP pass.
        return F.rms_norm(x, (self.hidden_size,), self.weight, self.eps)


def install_sequence_parallel_norm():
    from megatron.core.models.gpt import gpt_layer_specs
    from megatron.core.transformer import transformer_block

    gpt_layer_specs.WrappedTorchNorm = SequenceParallelRMSNorm
    gpt_layer_specs.LNImpl = SequenceParallelRMSNorm
    transformer_block.LayerNormImpl = SequenceParallelRMSNorm
