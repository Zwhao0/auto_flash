"""Framework compatibility only: actual DeepSpeed stage-0 dense next-byte training."""

import argparse
import datetime
import json
import os
import pathlib
import time
import torch
import torch.distributed as dist
import deepspeed
from flash.integration import report_iteration, report_parallel_states
from flash.store import Store

p = argparse.ArgumentParser()
p.add_argument("--output", required=True)
p.add_argument("--steps", type=int, default=24)
a = p.parse_args()
rank = int(os.environ["RANK"])
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
torch.set_num_threads(2)
torch.manual_seed(1234)
dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=90))
model = torch.nn.Sequential(
    torch.nn.Embedding(256, 64), torch.nn.Linear(64, 256)
).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
engine, _, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    dist_init_required=False,
    config={
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 1,
        "zero_optimization": {"stage": 0},
        "steps_per_print": 1000000,
    },
)
report_parallel_states([dict(tag="dp", ranks=list(range(dist.get_world_size())))])
data = torch.tensor(
    list(pathlib.Path("data/shakespeare.txt").read_bytes()),
    device="cuda",
    dtype=torch.long,
)
out = pathlib.Path(a.output)
with out.open("w") as f:
    f.write(
        json.dumps(
            dict(
                kind="framework",
                framework="deepspeed",
                version=deepspeed.__version__,
                rank=rank,
                scope="stage0_dense_compatibility_only",
            )
        )
        + "\n"
    )
    for step in range(a.steps):
        start = (rank * 997 + step * 256) % (len(data) - 257)
        x = data[start : start + 256].view(2, 128)
        y = data[start + 1 : start + 257].view(-1)
        t = time.perf_counter()
        loss = torch.nn.functional.cross_entropy(engine(x).view(-1, 256), y)
        engine.backward(loss)
        engine.step()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t) * 1000
        report_iteration(step, ms)
        assert torch.isfinite(loss)
        f.write(
            json.dumps(
                dict(kind="iteration", iteration=step, loss=loss.item(), time_ms=ms)
            )
            + "\n"
        )
    f.write(json.dumps(dict(kind="summary", steps=a.steps, finite=True)) + "\n")
if rank == 0:
    Store().set("training_done", True)
dist.destroy_process_group()
