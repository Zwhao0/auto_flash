"""Run the independent HF/Megatron gradient validation on idle 16 GPUs."""

import argparse
import json
import shlex
from concurrent.futures import ThreadPoolExecutor
from scripts.container import execute
from scripts.launch import collect, start
from scripts.experiment import wait_run

p = argparse.ArgumentParser()
p.add_argument("run")
a = p.parse_args()
base = "/flash/results/" + a.run
env = (
    f"export FLASH_RUN_ID={shlex.quote(a.run)}; export PYTHONPATH=/flash:/flash/deps:/flash/vendor/MegatronCore; "
    "export OMP_NUM_THREADS=2 CUDA_DEVICE_MAX_CONNECTIONS=1 NCCL_SOCKET_IFNAME=bond0 GLOO_SOCKET_IFNAME=bond0 NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3 NCCL_NET=IB NCCL_DEBUG=WARN; "
)
for host in ["10.10.4.1", "10.10.4.2"]:
    idle = execute(host, "python scripts/assert_idle.py")
    if idle[1]:
        raise RuntimeError(idle[2])
init = execute(
    "10.10.4.1",
    env + "python scripts/init_run.py " + shlex.quote(a.run) + " baseline gradient ''",
)
if init[1]:
    raise RuntimeError(init[2])


def go(pair):
    node, host = pair
    command = (
        env
        + f"torchrun --nnodes=2 --nproc_per_node=8 --node_rank={node} --master_addr=10.10.4.1 --master_port=29597 scripts/validate_qwen_gradients.py; code=$?; echo $code > {base}/exit{node}"
    )
    return start(host, command, base + f"/torchrun{node}.log")


with ThreadPoolExecutor() as pool:
    for result in pool.map(go, enumerate(["10.10.4.1", "10.10.4.2"])):
        print(result, flush=True)
exits = wait_run(a.run, 300)
path = collect(a.run)
(path / "execution.json").write_text(
    json.dumps(
        dict(run=a.run, exits=exits, kind="independent_gradient_equivalence"), indent=2
    )
)
print(json.dumps(dict(run=a.run, exits=exits)), flush=True)
if exits != ["0", "0"]:
    raise SystemExit(1)
