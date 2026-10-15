import unittest
from types import SimpleNamespace
import torch
from scripts.model_support import SequenceParallelRMSNorm


@unittest.skipUnless(
    hasattr(torch.nn.functional, "rms_norm"), "requires native Torch RMSNorm"
)
class NormEquivalence(unittest.TestCase):
    def test_sequence_shards_equal_dense_forward_and_gradients(self):
        torch.manual_seed(42)
        config = SimpleNamespace(
            normalization="RMSNorm",
            layernorm_zero_centered_gamma=False,
            params_dtype=torch.float32,
            sequence_parallel=True,
        )
        x = torch.randn(6, 2, 8, requires_grad=True)
        weight = torch.randn(8, requires_grad=True)
        reference = torch.nn.functional.rms_norm(x, (8,), weight, 1e-5)
        reference.square().sum().backward()
        outputs = []
        gradients = []
        weight_gradients = []
        for shard in x.detach().chunk(2, dim=0):
            shard = shard.clone().requires_grad_()
            module = SequenceParallelRMSNorm(config, 8)
            module.weight.data.copy_(weight.detach())
            out = module(shard)
            out.square().sum().backward()
            outputs.append(out.detach())
            gradients.append(shard.grad)
            weight_gradients.append(module.weight.grad)
            self.assertTrue(module.weight.sequence_parallel)
        torch.testing.assert_close(torch.cat(outputs), reference.detach())
        torch.testing.assert_close(torch.cat(gradients), x.grad)
        torch.testing.assert_close(sum(weight_gradients), weight.grad)


if __name__ == "__main__":
    unittest.main()
