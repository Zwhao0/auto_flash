"""Load official Qwen2-MoE weights into native Megatron EP modules."""

import json
from pathlib import Path
import re
import torch
from megatron.core import parallel_state as ps
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.transformer.transformer_config import TransformerConfig


def build(checkpoint, seq, tp, pp, ep, dtype):
    if tp != 1 or pp != 1:
        raise ValueError("Verified Qwen importer currently requires TP1/PP1")
    hf = json.loads((Path(checkpoint) / "config.json").read_text())
    if hf["num_experts"] % ep:
        raise ValueError("Qwen experts must divide EP exactly")
    cfg = TransformerConfig(
        num_layers=hf["num_hidden_layers"],
        hidden_size=hf["hidden_size"],
        num_attention_heads=hf["num_attention_heads"],
        num_query_groups=hf["num_key_value_heads"],
        ffn_hidden_size=hf["moe_intermediate_size"],
        num_moe_experts=hf["num_experts"],
        moe_router_topk=hf["num_experts_per_tok"],
        moe_aux_loss_coeff=hf.get("router_aux_loss_coef", 0.001),
        moe_router_pre_softmax=not hf.get("norm_topk_prob", False),
        moe_router_load_balancing_type="aux_loss",
        moe_router_dtype=None,
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        expert_model_parallel_size=ep,
        expert_tensor_parallel_size=1,
        moe_token_dispatcher_type="alltoall",
        moe_shared_expert_intermediate_size=hf["shared_expert_intermediate_size"],
        hidden_dropout=0.0,
        attention_dropout=0.0,
        params_dtype=dtype,
        pipeline_dtype=dtype,
        bf16=dtype == torch.bfloat16,
        normalization="RMSNorm",
        layernorm_epsilon=hf["rms_norm_eps"],
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        add_bias_linear=False,
        add_qkv_bias=True,
        gradient_accumulation_fusion=False,
        use_cpu_initialization=False,
        moe_permute_fusion=False,
    )
    spec = get_gpt_layer_local_spec(
        num_experts=hf["num_experts"], normalization="RMSNorm"
    )
    spec.submodules.mlp.submodules.shared_experts.params["gate"] = True
    model = (
        GPTModel(
            cfg,
            spec,
            vocab_size=hf["vocab_size"],
            max_sequence_length=seq,
            share_embeddings_and_output_weights=False,
            position_embedding_type="rope",
            rotary_base=hf["rope_theta"],
        )
        .cuda()
        .to(dtype)
    )
    return model, hf


class Checkpoint:
    def __init__(self, path):
        from safetensors import safe_open

        self.path = Path(path)
        index = self.path / "model.safetensors.index.json"
        if index.exists():
            self.index = json.loads(index.read_text())["weight_map"]
        else:
            with safe_open(
                str(self.path / "model.safetensors"), framework="pt", device="cpu"
            ) as f:
                self.index = {k: "model.safetensors" for k in f.keys()}
        self.files = {}

    def get(self, key):
        from safetensors import safe_open

        filename = self.index[key]
        if filename not in self.files:
            self.files[filename] = safe_open(
                str(self.path / filename), framework="pt", device="cpu"
            )
        return self.files[filename].get_tensor(key)


def load(model, path, hf):
    checkpoint = Checkpoint(path)
    report = []
    ep = ps.get_expert_model_parallel_rank()
    local = hf["num_experts"] // ps.get_expert_model_parallel_world_size()
    hd = hf["hidden_size"] // hf["num_attention_heads"]
    ng = hf["num_key_value_heads"]
    nq = hf["num_attention_heads"] // ng

    def qkv(prefix, suffix):
        q = checkpoint.get(prefix + "self_attn.q_proj." + suffix)
        k = checkpoint.get(prefix + "self_attn.k_proj." + suffix)
        v = checkpoint.get(prefix + "self_attn.v_proj." + suffix)
        tail = q.shape[1:]
        return torch.cat(
            [
                q.reshape(ng, nq * hd, *tail),
                k.reshape(ng, hd, *tail),
                v.reshape(ng, hd, *tail),
            ],
            dim=1,
        ).reshape(-1, *tail)

    for name, param in model.named_parameters():
        if name == "embedding.word_embeddings.weight":
            value = checkpoint.get("model.embed_tokens.weight")
        elif name == "output_layer.weight":
            value = checkpoint.get("lm_head.weight")
        elif name == "decoder.final_layernorm.weight":
            value = checkpoint.get("model.norm.weight")
        else:
            match = re.fullmatch(r"decoder.layers.(\d+).(.+)", name)
            if not match:
                raise KeyError("Unmapped Megatron tensor " + name)
            layer, field = match.groups()
            prefix = f"model.layers.{layer}."
            simple = {
                "input_layernorm.weight": "input_layernorm.weight",
                "pre_mlp_layernorm.weight": "post_attention_layernorm.weight",
                "self_attention.linear_proj.weight": "self_attn.o_proj.weight",
                "mlp.router.weight": "mlp.gate.weight",
                "mlp.shared_experts.gate_weight": "mlp.shared_expert_gate.weight",
            }
            if field in simple:
                value = checkpoint.get(prefix + simple[field])
            elif field.startswith("self_attention.linear_qkv."):
                value = qkv(prefix, field.rsplit(".", 1)[1])
            else:
                expert = re.fullmatch(
                    r"mlp.experts.local_experts.(\d+).linear_fc([12]).weight", field
                )
                shared = re.fullmatch(
                    r"mlp.shared_experts.linear_fc([12]).weight", field
                )
                if expert:
                    i, fc = expert.groups()
                    source = prefix + f"mlp.experts.{ep*local+int(i)}."
                elif shared:
                    fc = shared.group(1)
                    source = prefix + "mlp.shared_expert."
                else:
                    raise KeyError("Unmapped Megatron tensor " + name)
                value = (
                    torch.cat(
                        [
                            checkpoint.get(source + "gate_proj.weight"),
                            checkpoint.get(source + "up_proj.weight"),
                        ],
                        dim=0,
                    )
                    if fc == "1"
                    else checkpoint.get(source + "down_proj.weight")
                )
        if value.shape != param.shape:
            raise ValueError(f"{name} shape mismatch {value.shape} {param.shape}")
        with torch.no_grad():
            param.copy_(value)
        report.append(dict(name=name, shape=list(param.shape), numel=param.numel()))
    return dict(
        loaded_tensors=len(report),
        loaded_parameters=sum(r["numel"] for r in report),
        tensors=report,
    )


def validate(model, path, hf, dtype):
    """All EP participants run MCore; rank 0 independently runs full HF weights."""
    import torch.distributed as dist
    from transformers import Qwen2MoeForCausalLM

    tokens = (torch.arange(8, device="cuda", dtype=torch.long) + 10).unsqueeze(0)
    pos = torch.arange(8, device="cuda").unsqueeze(0)
    mask = torch.triu(
        torch.ones(1, 1, 8, 8, device="cuda", dtype=torch.bool), diagonal=1
    )
    model.eval()
    with torch.no_grad():
        actual = model(tokens, pos, mask).float()
    result = {}
    if dist.get_rank() == 0:
        reference = (
            Qwen2MoeForCausalLM.from_pretrained(
                path,
                torch_dtype=dtype,
                local_files_only=True,
                attn_implementation="eager",
            )
            .cuda()
            .eval()
        )
        with torch.no_grad():
            expected = reference(tokens).logits.float()
        delta = actual - expected
        relative = (
            delta.square().mean().sqrt() / expected.square().mean().sqrt()
        ).item()
        cosine = torch.nn.functional.cosine_similarity(
            actual.flatten(), expected.flatten(), dim=0
        ).item()
        maxdiff = delta.abs().max().item()
        result = dict(
            relative_rms=relative,
            cosine=cosine,
            max_absolute=maxdiff,
            top1_agreement=(actual.argmax(-1) == expected.argmax(-1))
            .float()
            .mean()
            .item(),
            hf_parameters=sum(p.numel() for p in reference.parameters()),
            dtype=str(dtype),
            passed=(
                relative < (0.03 if dtype == torch.bfloat16 else 0.0001)
                and cosine > 0.999
            ),
        )
        print("QWEN_CONVERSION_VALIDATION " + json.dumps(result), flush=True)
        del reference, expected, delta
        torch.cuda.empty_cache()
    payload = [result]
    dist.broadcast_object_list(payload, src=0)
    if not payload[0]["passed"]:
        raise RuntimeError(
            "Qwen imported logits fail independent HF comparison: " + str(payload[0])
        )
    model.train()
    return payload[0]
