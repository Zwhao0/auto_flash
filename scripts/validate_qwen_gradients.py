"""Independent full-HF vs native MCore EP gradient check with identical inputs."""

import datetime
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from megatron.core import parallel_state as ps, tensor_parallel
from scripts.gradient_sync import sync_gradients
from scripts.qwen_support import build, load

rank = int(os.environ["RANK"])
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
torch.set_num_threads(2)
torch.manual_seed(1234)
dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=180))
ps.initialize_model_parallel(
    expert_model_parallel_size=4, expert_tensor_parallel_size=1
)
tensor_parallel.model_parallel_cuda_manual_seed(1234)
path = "/flash/models/qwen_conversion_fixture"
model, hf = build(path, 128, 1, 1, 4, torch.float32)
load(model, path, hf)
model.eval()  # Disable auxiliary training-only router losses for the independent comparison.
tokens = (torch.arange(64, device="cuda", dtype=torch.long) + 10).reshape(8, 8)
labels = tokens.roll(-1, dims=1)
pos = torch.arange(8, device="cuda").expand_as(tokens)
mask = torch.triu(torch.ones(1, 1, 8, 8, device="cuda", dtype=torch.bool), diagonal=1)
logits = model(tokens, pos, mask)
torch.nn.functional.cross_entropy(
    logits.reshape(-1, hf["vocab_size"]), labels.reshape(-1)
).backward()
dense = [p for p in model.parameters() if getattr(p, "allreduce", True)]
expert = [p for p in model.parameters() if not getattr(p, "allreduce", True)]
# The actual training synchronization code, not a duplicate reference formula.
sync_gradients(dense, ps.get_data_parallel_group())
sync_gradients(expert, ps.get_expert_data_parallel_group())
summary = {}
if rank == 0:
    from transformers import Qwen2MoeForCausalLM

    reference = (
        Qwen2MoeForCausalLM.from_pretrained(
            path,
            local_files_only=True,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        )
        .cuda()
        .eval()
    )
    expected = reference(tokens).logits
    torch.nn.functional.cross_entropy(
        expected.reshape(-1, hf["vocab_size"]), labels.reshape(-1)
    ).backward()
    params = dict(model.named_parameters())
    checks = []

    def check(name, expected):
        actual = params[name].grad
        if actual is None or expected is None:
            raise RuntimeError("missing gradient " + name)
        relative = (
            (actual - expected).square().sum().sqrt()
            / expected.square().sum().sqrt().clamp_min(1e-20)
        ).item()
        norm_ratio = (actual.norm() / expected.norm().clamp_min(1e-20)).item()
        checks.append(
            dict(
                name=name,
                relative_l2=relative,
                norm_ratio=norm_ratio,
                old_normalization_norm_ratio=(
                    norm_ratio
                    * ps.get_data_parallel_world_size()
                    / ps.get_expert_data_parallel_world_size()
                    if ".experts.local_experts." in name
                    else norm_ratio
                ),
                passed=relative < 0.0002,
            )
        )

    check("embedding.word_embeddings.weight", reference.model.embed_tokens.weight.grad)
    check("output_layer.weight", reference.lm_head.weight.grad)
    for layer in range(2):
        source = reference.model.layers[layer]
        check(f"decoder.layers.{layer}.mlp.router.weight", source.mlp.gate.weight.grad)
        check(
            f"decoder.layers.{layer}.mlp.shared_experts.gate_weight",
            source.mlp.shared_expert_gate.weight.grad,
        )
        for expert_id in range(2):
            e = source.mlp.experts[expert_id]
            check(
                f"decoder.layers.{layer}.mlp.experts.local_experts.{expert_id}.linear_fc1.weight",
                torch.cat([e.gate_proj.weight.grad, e.up_proj.weight.grad], dim=0),
            )
            check(
                f"decoder.layers.{layer}.mlp.experts.local_experts.{expert_id}.linear_fc2.weight",
                e.down_proj.weight.grad,
            )
    summary = dict(
        rank=rank,
        world_size=dist.get_world_size(),
        ep=4,
        dense_dp=ps.get_data_parallel_world_size(),
        expert_dp=ps.get_expert_data_parallel_world_size(),
        checks=checks,
        passed=all(c["passed"] for c in checks),
    )
    out = (
        Path("/flash/results") / os.environ["FLASH_RUN_ID"] / "gradient_validation.json"
    )
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
flag = [summary]
dist.broadcast_object_list(flag, src=0)
dist.barrier()
dist.destroy_process_group()
if not flag[0]["passed"]:
    raise RuntimeError("HF/MCore gradient equivalence failed")
