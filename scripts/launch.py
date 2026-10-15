"""Launch a fresh two-node run and its actual FLASH controllers."""

import argparse, time, shlex, json, pathlib, sys, concurrent.futures, os
from scripts.container import execute
from scripts.remote import ROOT, connect


def start(host, command, log):
    cmd = (
        "mkdir -p "
        + shlex.quote(str(pathlib.PurePosixPath(log).parent))
        + "; nohup bash -lc "
        + shlex.quote(command)
        + " >"
        + shlex.quote(log)
        + " 2>&1 < /dev/null & echo $!"
    )
    return execute(host, cmd)


def launch(run, mode="monitor", task="train", args="", port=29541, force=False):
    base = "/flash/results/" + run
    paper = os.getenv("FLASH_RUNTIME") == "paper"
    mcore = (
        "/flash/vendor/MegatronCore" if paper else "/flash/vendor/Greyhound/Megatron-LM"
    )
    nccl = (
        "/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2"
        if paper
        else "/usr/lib/x86_64-linux-gnu/libnccl.so.2"
    )
    transport = "NCCL_NET=IB NCCL_DEBUG=INFO" if paper else "NCCL_DEBUG=WARN"
    cuda = (
        "/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib"
        if paper
        else "/usr/local/cuda/lib64"
    )
    env = f"export FLASH_RUN_ID={shlex.quote(run)} FLASH_REDIS_HOST=10.10.4.1 FLASH_REDIS_PORT=16379; export PYTHONPATH=/flash:/flash/deps:{mcore}; export FLASH_NCCL_LIB={nccl} LD_LIBRARY_PATH={cuda}:$LD_LIBRARY_PATH; export OMP_NUM_THREADS=2 CUDA_DEVICE_MAX_CONNECTIONS=1 NCCL_SOCKET_IFNAME=bond0 GLOO_SOCKET_IFNAME=bond0 NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3 {transport} TORCH_NCCL_ASYNC_ERROR_HANDLING=1; "
    for host in ["10.10.4.1", "10.10.4.2"]:
        idle = execute(host, "python scripts/assert_idle.py")
        if idle[1]:
            raise RuntimeError(idle[2])
    init = execute(
        "10.10.4.1",
        env
        + "python scripts/init_run.py "
        + shlex.quote(run)
        + " "
        + shlex.quote(mode)
        + " "
        + shlex.quote(task)
        + " "
        + shlex.quote(args),
    )
    if init[1]:
        raise RuntimeError(init[2])
    # Controllers are launched independently of torchrun so crashes cannot kill monitoring.
    if mode != "baseline":
        for node, host in enumerate(["10.10.4.1", "10.10.4.2"]):
            print(
                start(
                    host,
                    env + f"python -m flash.local_controller --run {run} --node {node}",
                    base + "/local" + str(node) + ".log",
                )
            )
        print(
            start(
                "10.10.4.1",
                env
                + f"python -m flash.global_controller --run {run} --output {base}/controller.jsonl"
                + (" --force-profile" if force else ""),
                base + "/global.log",
            )
        )

    def train(pair):
        node, host = pair
        faultenv = (
            "export FLASH_FAULT_INJECT=TRUE; " if "--fault timeout" in args else ""
        )
        cmd = (
            env
            + faultenv
            + f"export FLASH_MODE={mode} FLASH_TASK={task}; torchrun --nnodes=2 --nproc_per_node=8 --node_rank={node} --master_addr=10.10.4.1 --master_port={port} --no_python bash scripts/run_rank.sh {args}; code=$?; echo $code > {base}/exit{node}"
        )
        return start(host, cmd, base + "/torchrun" + str(node) + ".log")

    with concurrent.futures.ThreadPoolExecutor() as pool:
        for r in pool.map(train, enumerate(["10.10.4.1", "10.10.4.2"])):
            print(r)
    return base


def collect(run):
    dest = ROOT / "results" / run
    dest.mkdir(parents=True, exist_ok=True)
    for node, host in enumerate(["10.10.4.1", "10.10.4.2"]):
        c = connect(host)
        try:
            s = c.open_sftp()
            remote = "/home/campus/zhangweihao/flash-repro-20261015/results/" + run
            for f in s.listdir_attr(remote):
                if f.filename.endswith(".pt"):
                    continue
                s.get(remote + "/" + f.filename, str(dest / f.filename))
        finally:
            c.close()
    return dest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--mode", default="monitor", choices=["baseline", "monitor"])
    p.add_argument(
        "--task",
        default="train",
        choices=["train", "allreduce", "precheck", "deepspeed"],
    )
    p.add_argument("--args", default="")
    p.add_argument("--port", type=int, default=29541)
    p.add_argument("--force-profile", action="store_true")
    p.add_argument("--collect", action="store_true")
    a = p.parse_args()
    if a.collect:
        print(collect(a.run))
    else:
        launch(a.run, a.mode, a.task, a.args, a.port, a.force_profile)
