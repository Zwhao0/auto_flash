"""Paper-style batch timing; fewer batches than the publication for development."""

import os, time, json, argparse, datetime
from pathlib import Path
import torch, torch.distributed as dist

p = argparse.ArgumentParser()
p.add_argument("--output", required=True)
p.add_argument("--repeats", type=int, default=100)
p.add_argument("--warmup", type=int, default=20)
p.add_argument("--batches", type=int, default=3)
a = p.parse_args()
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=90))
rank = dist.get_rank()
rows = []
for mib in [4, 8, 16, 32, 64, 128]:
    x = torch.zeros(mib * 1024 * 1024 // 4, device="cuda")
    for _ in range(a.warmup):
        dist.all_reduce(x)
    times = []
    for _ in range(a.batches):
        # A nonzero correctness check separately from the sustained timed batch.
        x.fill_(1)
        dist.all_reduce(x)
        torch.cuda.synchronize()
        if not torch.all(x == dist.get_world_size()).item():
            raise RuntimeError("AllReduce result corrupted")
        x.zero_()
        dist.barrier()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(a.repeats):
            dist.all_reduce(x)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / a.repeats)
        if not torch.all(x == 0).item():
            raise RuntimeError("AllReduce batch result corrupted")
    row = dict(
        rank=rank,
        mib=mib,
        repeats=a.repeats,
        batches=a.batches,
        warmup=a.warmup,
        mean_ms=sum(times) / len(times),
        samples_ms=times,
    )
    rows.append(row)
    if rank == 0:
        print(json.dumps(row), flush=True)
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
Path(a.output).write_text(json.dumps(rows))
dist.barrier()
dist.destroy_process_group()
