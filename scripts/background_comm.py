"""Independent real NCCL traffic across the same two servers (two ranks)."""

import os, time, argparse, json, pathlib, datetime
import torch, torch.distributed as dist
from flash.store import Store

p = argparse.ArgumentParser()
p.add_argument("--seconds", type=float, default=30)
p.add_argument("--mib", type=int, default=128)
p.add_argument("--duty", type=float, default=1)
p.add_argument("--output", required=True)
p.add_argument("--coordinated", action="store_true")
a = p.parse_args()
torch.cuda.set_device(0)
dist.init_process_group(
    "nccl", timeout=datetime.timedelta(seconds=90), device_id=torch.device("cuda", 0)
)
x = torch.zeros(a.mib * 1024 * 1024 // 4, device="cuda")
dist.barrier(device_ids=[0])
dist.all_reduce(x)
torch.cuda.synchronize()
if a.coordinated:
    store = Store()
    pathlib.Path(a.output + ".ready").write_text("ready\n")
    deadline = time.monotonic() + 60
    while not store.get("background_go", False):
        if time.monotonic() > deadline:
            raise TimeoutError("Background traffic was not released")
        time.sleep(0.05)
started_at = time.time()
begin = time.monotonic()
calls = 0
while True:
    t = time.monotonic()
    dist.all_reduce(x)
    torch.cuda.synchronize()
    calls += 1
    if a.duty < 1:
        time.sleep((time.monotonic() - t) * (1 - a.duty) / a.duty)
    stop = torch.tensor([int(time.monotonic() - begin >= a.seconds)], device="cuda")
    dist.all_reduce(stop, op=dist.ReduceOp.MAX)
    if stop.item():
        break
pathlib.Path(a.output).write_text(
    json.dumps(
        dict(
            calls=calls,
            seconds=time.monotonic() - begin,
            mib=a.mib,
            world_size=dist.get_world_size(),
            started_at=started_at,
            finished_at=time.time(),
            correct=bool(torch.count_nonzero(x).item() == 0),
        )
    )
)
dist.destroy_process_group()
