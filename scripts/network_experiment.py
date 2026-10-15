"""Background communication test; labels shared GPU contention explicitly."""

import time, json, pathlib, argparse, concurrent.futures
from scripts.launch import launch, start
from scripts.experiment import finish, redis_command
from scripts.remote import ROOT

p = argparse.ArgumentParser()
p.add_argument("run")
p.add_argument("--mib", type=int, default=128)
p.add_argument("--duty", type=float, default=1)
a = p.parse_args()
launch(a.run, args="--steps 100 --min-seconds 40 --seq 128 --batch 2")
starttime = time.monotonic()
while time.monotonic() - starttime < 100:
    _, _, out = redis_command(
        a.run, "v=s.items('iter_time_0');print(v[-1].split(':')[0] if v else -1)"
    )
    try:
        step = int(out.strip())
    except ValueError:
        step = -1
    if step >= 45:
        break
    time.sleep(0.2)
if step < 45:
    raise RuntimeError("training did not reach injection boundary")
# The training uses all GPUs; traffic generator shares GPU0 on both nodes.
base = f"/flash/results/{a.run}"
env = "export PYTHONPATH=/flash:/flash/deps; export NCCL_SOCKET_IFNAME=bond0 NCCL_NET=IB NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3; "


def bg(pair):
    node, host = pair
    return start(
        host,
        env
        + f"export FLASH_RUN_ID={a.run}; torchrun --nnodes=2 --nproc_per_node=1 --node_rank={node} --master_addr=10.10.4.1 --master_port=29588 scripts/background_comm.py --coordinated --mib {a.mib} --duty {a.duty} --seconds 30 --output {base}/background_node{node}.json",
        base + f"/background_node{node}.log",
    )


with concurrent.futures.ThreadPoolExecutor() as pool:
    for r in pool.map(bg, enumerate(["10.10.4.1", "10.10.4.2"])):
        print(r)
from scripts.container import execute

for _ in range(60):
    if all(
        execute(h, f"test -f {base}/background_node{n}.json.ready")[1] == 0
        for n, h in enumerate(["10.10.4.1", "10.10.4.2"])
    ):
        break
    time.sleep(0.2)
else:
    raise RuntimeError("Background NCCL initialization did not complete")
_, _, observed = redis_command(
    a.run, "v=s.items('iter_time_0'); print(v[-1].split(':')[0])"
)
step = int(observed.strip())
redis_command(a.run, "s.set('background_go',True)")
redis_command(
    a.run,
    f"s.event('injection',fault='background_nccl',ranks=[0,8],mib={a.mib},duty={a.duty},observed_iteration={step},gpu_contention=True)",
)
finish(a.run, "monitor")

background_complete = False
for _ in range(60):
    from scripts.container import execute

    if all(
        execute(h, f"test -f {base}/background_node{n}.json")[1] == 0
        for n, h in enumerate(["10.10.4.1", "10.10.4.2"])
    ):
        background_complete = True
        break
    time.sleep(1)
if not background_complete:
    raise RuntimeError("Background traffic did not complete on both ranks")
from scripts.launch import collect

collect(a.run)
