"""Small real GPU compute and communication pre-check before fault campaigns."""

import torch, torch.distributed as dist, os, time, json, pathlib, datetime

rank = int(os.environ["RANK"])
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=90))
a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
b = torch.randn_like(a)
for _ in range(5):
    a @ b
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(20):
    a @ b
end.record()
end.synchronize()
gemm_ms = start.elapsed_time(end) / 20
computed = (a @ b)[:4, :4].float().cpu()
reference = a[:4].float().cpu() @ b[:, :4].float().cpu()
gemm_correct = bool(torch.allclose(computed, reference, rtol=0.02, atol=0.5))
x = torch.ones(1024 * 1024, device="cuda")
dist.all_reduce(x)
torch.cuda.synchronize()
correct = bool(torch.all(x == dist.get_world_size()).item())
row = dict(
    rank=rank,
    gemm_ms=gemm_ms,
    bf16_tflops=2 * 2048**3 / (gemm_ms / 1000) / 1e12,
    allreduce_correct=correct,
    gemm_correct=gemm_correct,
    device=torch.cuda.get_device_name(),
    world_size=dist.get_world_size(),
)
path = pathlib.Path("results") / os.environ["FLASH_RUN_ID"]
path.mkdir(parents=True, exist_ok=True)
(path / f"precheck_rank{rank}.json").write_text(json.dumps(row, indent=2))
dist.barrier()
dist.destroy_process_group()
if not correct or not gemm_correct:
    raise RuntimeError("precheck GPU compute/communication result mismatch")
