"""Run the official GreyHound implementation, preserving its actual failures."""

import os, time, pathlib, json, concurrent.futures, argparse
from scripts.launch import start, collect
from scripts.experiment import wait_run
from scripts.container import execute
from scripts.remote import ROOT

p = argparse.ArgumentParser()
p.add_argument("--run", default="greyhound_official16")
p.add_argument("--seed-parallel-states", action="store_true")
p.add_argument("--wait-precheck", action="store_true")
a = p.parse_args()
run = a.run
base = "/flash/results/" + run
for host in ["10.10.4.1", "10.10.4.2"]:
    idle = execute(host, "python scripts/assert_idle.py")
    if idle[1]:
        raise RuntimeError(idle[2])
    if (
        execute(host, f"test -f {base}/torchrun{0 if host.endswith('.1') else 1}.log")[
            1
        ]
        == 0
    ):
        raise RuntimeError("Existing official-baseline run; use a new run ID")
# Private IPC in the new baseline container prevents use of any previous ncclRecord.
# Its Redis instance is created on an unused port; FLASH uses independent port 16379.
rc = execute("10.10.4.1", "redis-cli -h 10.10.4.1 -p 6379 ping")[1]
if rc == 0:
    raise RuntimeError(
        "port 6379 is already occupied; do not overwrite another Redis instance"
    )
execute(
    "10.10.4.1",
    f'mkdir -p {base}; redis-server --bind 10.10.4.1 --port 6379 --protected-mode no --save "" --appendonly no --daemonize yes --pidfile {base}/redis.pid --logfile {base}/redis.log',
)
if a.seed_parallel_states:
    # Upstream starts its analyzer on world_size before all ranks publish groups.
    # Pre-publish the exact TP1/PP1/DP16 fixture topology using its existing keys.
    result = execute(
        "10.10.4.1",
        "python - <<'END'\nimport redis\nr=redis.Redis(host='10.10.4.1',port=6379)\nfor rank in range(16):\n r.set(f'{rank}_tp',str(rank))\n r.set(f'{rank}_pp',str(rank))\n r.set(f'{rank}_dp','_'.join(map(str,range(16))))\nr.set('world_size',16)\nEND",
    )
    if result[1]:
        raise RuntimeError(result[2])


def go(pair):
    node, host = pair
    env = f"export PYTHONPATH=/flash:/flash/vendor/Greyhound/Megatron-LM; export FLASH_GREYHOUND=TRUE FLASH_MODE=baseline FLASH_TASK=train FLASH_RUN_ID={run} FLASH_REDIS_HOST=10.10.4.1 FLASH_REDIS_PORT=16379; export CONTROL_PLANE_WHL_PATH=/flash/vendor/Greyhound/detector/dist/control_plane-1.0-py3-none-any.whl NCCLPROBE_LOG_PATH={base} LOCAL_CONTROLLER_LOG_PATH={base} GLOBAL_CONTROLLER_LOG_PATH={base}; export OMP_NUM_THREADS=2 CUDA_DEVICE_MAX_CONNECTIONS=1 NCCL_SOCKET_IFNAME=bond0 GLOO_SOCKET_IFNAME=bond0 NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3 NCCL_NET=IB REDIS_PORT=6379; "
    cmd = (
        env
        + ("export FLASH_GREYHOUND_WAIT_PRECHECK=TRUE; " if a.wait_precheck else "")
        + f"torchrun --nnodes=2 --nproc_per_node=8 --node_rank={node} --master_addr=10.10.4.1 --master_port=29591 --no_python bash scripts/run_rank.sh --steps {320 if a.wait_precheck else 180} --min-seconds 40 --seq 128 --batch 2 --fault delay --severity .45 --fault-step 70; echo $? > {base}/exit{node}"
    )
    return start(host, cmd, base + f"/torchrun{node}.log")


with concurrent.futures.ThreadPoolExecutor() as pool:
    for r in pool.map(go, enumerate(["10.10.4.1", "10.10.4.2"])):
        print(r)
try:
    exits = wait_run(run, 300 if a.wait_precheck else 180)
    print(exits)
except TimeoutError as e:
    exits = ["timeout", "timeout"]
    print(str(e), flush=True)
finally:
    # The dedicated baseline containers contain only this experiment and its controllers.
    for host in ["10.10.4.1", "10.10.4.2"]:
        execute(
            host,
            "python - <<'END'\nimport os,signal,subprocess\nfor l in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines():\n p=l.strip().split(None,1)\n if len(p)==2 and (p[1].startswith('python scripts/train_moe.py') or p[1].startswith('/usr/bin/python /usr/local/bin/torchrun') or p[1].startswith('/usr/bin/python /usr/local/bin/global_controller') or p[1].startswith('/usr/bin/python /usr/local/bin/local_controller')):\n  try:os.kill(int(p[0]),signal.SIGTERM)\n  except ProcessLookupError:pass\nEND",
        )
    path = collect(run)
    (path / "execution.json").write_text(
        json.dumps(
            dict(
                run=run,
                exits=exits,
                source="official",
                config_adaptations="patches/greyhound-hardware.patch",
                topology_preseed=a.seed_parallel_states,
                wait_precheck=a.wait_precheck,
            ),
            indent=2,
        )
    )
    execute("10.10.4.1", "redis-cli -h 10.10.4.1 -p 6379 shutdown nosave")
