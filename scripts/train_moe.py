"""Actual Megatron GPT-MoE training (not a synthetic collective benchmark)."""

import os, time, json, argparse, pathlib, datetime, inspect
import torch
import torch.distributed as dist
from megatron.core import parallel_state as ps, tensor_parallel
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from flash.integration import report_iteration, report_parallel_states
from flash.store import Store


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--pp", type=int, default=1)
    p.add_argument("--ep", type=int, default=8)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--min-seconds", type=float, default=0)
    p.add_argument("--seq", type=int, default=128)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--microbatches", type=int, default=2)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument(
        "--fault", choices=["none", "delay", "oom", "timeout"], default="none"
    )
    p.add_argument("--fault-rank", type=int, default=8)
    p.add_argument("--fault-step", type=int, default=45)
    p.add_argument("--delay-ms", type=float, default=100)
    p.add_argument("--severity", type=float, default=None)
    p.add_argument(
        "--fault-placement",
        choices=["before-step", "before-sync"],
        default="before-sync",
    )
    p.add_argument("--fault-duration", type=int, default=0)
    p.add_argument("--output", required=True)
    p.add_argument("--stop-backend", choices=["gloo", "nccl"], default="gloo")
    a = p.parse_args()
    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=90))
    # Disable the official GreyHound fork's duplicate Redis logger (it polls port 6379 forever).
    ps.log_model_parallel_to_redis = lambda *args, **kwargs: None
    modern = (
        "expert_tensor_parallel_size"
        in inspect.signature(ps.initialize_model_parallel).parameters
    )
    ps.initialize_model_parallel(
        tensor_model_parallel_size=a.tp,
        pipeline_model_parallel_size=a.pp,
        expert_model_parallel_size=a.ep,
        **(
            {"expert_tensor_parallel_size": 1, "distributed_timeout_minutes": 1.5}
            if modern
            else {}
        ),
    )
    tensor_parallel.model_parallel_cuda_manual_seed(a.seed)
    # The fixture's stopping protocol must not create NCCL cliques that the
    # paper's member-set-based labeling could mistake for model DP traffic.
    stop_group = (
        dist.new_group(backend="gloo", timeout=datetime.timedelta(seconds=90))
        if a.stop_backend == "gloo"
        else None
    )
    if modern and a.tp > 1:
        from scripts.model_support import install_sequence_parallel_norm

        install_sequence_parallel_norm()
    extra = (
        {
            "moe_token_dispatcher_type": "alltoall",
            "expert_tensor_parallel_size": 1,
            "use_cpu_initialization": False,
            "moe_permute_fusion": False,
            "add_bias_linear": False,
            "gated_linear_unit": True,
            "activation_func": torch.nn.functional.silu,
            "normalization": "RMSNorm",
            "moe_shared_expert_intermediate_size": a.hidden * 2,
            "num_query_groups": 4,
        }
        if modern
        else {}
    )
    cfg = TransformerConfig(
        num_layers=a.layers,
        hidden_size=a.hidden,
        num_attention_heads=8,
        ffn_hidden_size=a.hidden * 2,
        num_moe_experts=8,
        moe_router_topk=2,
        moe_aux_loss_coeff=0.01,
        tensor_model_parallel_size=a.tp,
        pipeline_model_parallel_size=a.pp,
        expert_model_parallel_size=a.ep,
        sequence_parallel=a.tp > 1,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        params_dtype=torch.float32,
        pipeline_dtype=torch.float32,
        gradient_accumulation_fusion=False,
        async_tensor_model_parallel_allreduce=a.tp == 1,
        **extra,
    )
    model = GPTModel(
        cfg,
        get_gpt_layer_local_spec(
            num_experts=8, **({"normalization": "RMSNorm"} if modern else {})
        ),
        vocab_size=256,
        max_sequence_length=a.seq,
        pre_process=ps.is_pipeline_first_stage(),
        post_process=ps.is_pipeline_last_stage(),
        share_embeddings_and_output_weights=False,
        position_embedding_type="rope",
    ).cuda()
    dense = [p for p in model.parameters() if getattr(p, "allreduce", True)]
    expert = [p for p in model.parameters() if not getattr(p, "allreduce", True)]
    dp = ps.get_data_parallel_group()
    edp = (
        ps.get_expert_data_parallel_group()
        if modern
        else ps.get_data_modulo_expert_parallel_group()
    )
    tp = ps.get_tensor_model_parallel_group()
    # Equal parameters in each replica group, including checkpoint-free random initialization.
    for params, group in [(dense, dp), (expert, edp)]:
        source = dist.get_process_group_ranks(group)[0]
        for param in params:
            dist.broadcast(param.data, source, group=group)
    groups = []
    for tag, group in [
        ("dp", dp),
        ("expert_dp", edp),
        ("tp", tp),
        ("pp", ps.get_pipeline_model_parallel_group()),
        (
            "ep" if modern else "tp_ep",
            (
                ps.get_expert_model_parallel_group()
                if modern
                else ps.get_tensor_and_expert_parallel_group()
            ),
        ),
    ]:
        if a.pp > 1 and tag in ["dp", "tp"]:
            tag = f"ppstage{ps.get_pipeline_model_parallel_rank()}_{tag}"
        groups.append(dict(tag=tag, ranks=dist.get_process_group_ranks(group)))
    if modern:
        groups.append(
            dict(
                tag="expert_tp",
                ranks=dist.get_process_group_ranks(
                    ps.get_expert_tensor_parallel_group()
                ),
            )
        )
    report_parallel_states(groups)
    if os.getenv("FLASH_GREYHOUND") == "TRUE":
        import redis

        db = redis.Redis(host=os.environ["MASTER_ADDR"], port=6379)
        for tag, group in [
            ("tp", tp),
            ("dp", dp),
            ("pp", ps.get_pipeline_model_parallel_group()),
        ]:
            db.set(
                f"{rank}_{tag}",
                "_".join(str(r) for r in dist.get_process_group_ranks(group)),
            )
    store = Store()
    store.set("world_size", dist.get_world_size())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    data = torch.tensor(
        list(pathlib.Path("/flash/data/shakespeare.txt").read_bytes()),
        dtype=torch.long,
        device="cuda",
    )
    dp_rank = ps.get_data_parallel_rank()
    offset = [dp_rank * a.seq * a.batch]

    def forward_step(iterator, module):
        idx = (torch.arange(a.batch * (a.seq + 1), device="cuda") + offset[0]) % len(
            data
        )
        offset[0] += a.batch * a.seq
        b = data[idx].reshape(a.batch, a.seq + 1)
        tokens = b[:, :-1].contiguous()
        labels = b[:, 1:].contiguous()
        pos = torch.arange(a.seq, device="cuda").expand_as(tokens)
        mask = torch.triu(
            torch.ones((1, 1, a.seq, a.seq), device="cuda", dtype=torch.bool),
            diagonal=1,
        )
        out = module(tokens, pos, mask, labels=labels)

        def loss_func(losses):
            loss = losses.float().mean()
            return loss, {"lm_loss": loss.detach().clone()}

        return out, loss_func

    def sync(params, group):
        have = [p for p in params if p.grad is not None]
        if not have:
            return
        flat = torch.cat([p.grad.reshape(-1) for p in have])
        dist.all_reduce(flat, group=group)
        flat.div_(dist.get_world_size(group))
        offset = 0
        for p in have:
            p.grad.copy_(flat[offset : offset + p.numel()].view_as(p))
            offset += p.numel()

    effective_delay_ms = a.delay_ms
    fb = get_forward_backward_func()
    path = pathlib.Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    calibration_cycles_per_ms = None
    if rank == a.fault_rank and a.fault == "delay":
        e1 = torch.cuda.Event(enable_timing=True)
        e2 = torch.cuda.Event(enable_timing=True)
        e1.record()
        torch.cuda._sleep(10000000)
        e2.record()
        e2.synchronize()
        calibration_cycles_per_ms = 10000000 / e1.elapsed_time(e2)
    total = sum(p.numel() for p in model.parameters())
    metadata = dict(
        kind="metadata",
        rank=rank,
        config=vars(a),
        local_parameters=total,
        dense_parameters=sum(p.numel() for p in dense),
        expert_parameters=sum(p.numel() for p in expert),
        torch=torch.__version__,
        groups=groups,
        sleep_cycles_per_ms=calibration_cycles_per_ms,
    )
    with path.open("w") as f:
        f.write(json.dumps(metadata) + "\n")
    dist.barrier()
    torch.cuda.synchronize()
    begin = time.monotonic()
    step = 0
    duration = []
    steady_start = None
    steady_steps = 0
    while True:
        step += 1
        if step == 10 and os.getenv("FLASH_GREYHOUND_WAIT_PRECHECK") == "TRUE":
            # Warm all lazy communicators, then service the upstream validation
            # protocol before the fault experiment. Do not skip its pre-check.
            marker = torch.zeros(503, dtype=torch.int8, device="cuda")
            deadline = time.monotonic() + 180
            while True:
                dist.all_reduce(marker)
                ready = torch.tensor(
                    [int(db.get("precheck_done") == b"1")], device="cuda"
                )
                dist.all_reduce(ready, op=dist.ReduceOp.MIN)
                if ready.item():
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError("Official GreyHound pre-check did not finish")
                time.sleep(0.1)
        t = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        if step == 6:
            steady_start = time.monotonic()
        if step >= 6:
            steady_steps += 1

        def inject():
            nonlocal effective_delay_ms
            if a.severity is not None and step == a.fault_step:
                effective_delay_ms = (
                    (sum(duration[-10:]) / len(duration[-10:]))
                    * a.severity
                    / (1 - a.severity)
                )
            elif a.severity is None:
                effective_delay_ms = a.delay_ms
            if step == a.fault_step:
                store.event(
                    "injection",
                    rank=rank,
                    step=step,
                    fault=a.fault,
                    delay_ms=effective_delay_ms,
                    severity=a.severity,
                    placement=a.fault_placement,
                )
            if a.fault == "delay":
                torch.cuda._sleep(int(effective_delay_ms * calibration_cycles_per_ms))
            elif a.fault == "oom":
                torch.empty(
                    (2 * torch.cuda.get_device_properties(local).total_memory,),
                    device="cuda",
                    dtype=torch.uint8,
                )
            elif a.fault == "timeout":
                os.environ["FLASH_FAULT_TIMEOUT_ARM"] = "1"

        fault_active = (
            a.fault != "none"
            and rank == a.fault_rank
            and step >= a.fault_step
            and (not a.fault_duration or step < a.fault_step + a.fault_duration)
        )
        if fault_active and a.fault_placement == "before-step":
            inject()
        losses = fb(
            forward_step_func=forward_step,
            data_iterator=None,
            model=model,
            num_microbatches=max(a.microbatches, a.pp),
            seq_length=a.seq,
            micro_batch_size=a.batch,
            forward_only=False,
        )
        if a.tp > 1:
            for param in model.parameters():
                if (
                    getattr(param, "sequence_parallel", False)
                    and param.grad is not None
                ):
                    dist.all_reduce(param.grad, group=tp)
        if fault_active and a.fault_placement == "before-sync":
            inject()
        sync(dense, dp)
        sync(expert, edp)
        optimizer.step()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t) * 1000
        duration.append(ms)
        report_iteration(step, ms)
        loss = (
            float(torch.stack([d["lm_loss"] for d in losses]).mean())
            if losses
            else None
        )
        row = dict(
            kind="iteration",
            rank=rank,
            iteration=step,
            time_ms=ms,
            loss=loss,
            tokens_per_second=a.batch
            * a.seq
            * max(a.microbatches, a.pp)
            * ps.get_data_parallel_world_size()
            / (ms / 1000),
        )
        with path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        if rank == 0 and (step <= 3 or step % 10 == 0):
            print(json.dumps(row), flush=True)
        # A globally agreed stop prevents PP stages leaving at different iterations.
        stop = torch.tensor(
            [int(step >= a.steps and time.monotonic() - begin >= a.min_seconds)],
            device="cpu" if a.stop_backend == "gloo" else "cuda",
        )
        dist.all_reduce(stop, op=dist.ReduceOp.MIN, group=stop_group)
        if os.getenv("FLASH_GREYHOUND") == "TRUE":
            dist.all_reduce(torch.zeros(503, dtype=torch.int8, device="cuda"))
        if stop.item():
            break
    elapsed = time.monotonic() - steady_start if steady_start else None
    if rank == 0:
        store.set("training_done", True)
    with path.open("a") as f:
        f.write(
            json.dumps(
                dict(
                    kind="summary",
                    rank=rank,
                    steps=step,
                    steady_steps=steady_steps,
                    steady_wall_seconds=elapsed,
                    steady_e2e_tokens_per_second=(
                        (
                            steady_steps
                            * a.batch
                            * a.seq
                            * max(a.microbatches, a.pp)
                            * ps.get_data_parallel_world_size()
                            / elapsed
                        )
                        if elapsed
                        else None
                    ),
                )
            )
            + "\n"
        )
    # Preserve a compact real model checkpoint for independent correctness inspection.
    torch.save(
        {"model": model.state_dict(), "config": vars(a), "step": step},
        str(path) + ".pt",
    )
    dist.barrier()
    if rank == 0:
        store.set("training_done", True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
