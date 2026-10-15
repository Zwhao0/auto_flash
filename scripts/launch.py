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


def launch(
    run, mode="monitor", task="train", args="", port=29541, force=False, precheck=True
):
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
    if task == "train" and precheck:

        def check(pair):
            node, host = pair
            cmd = (
                env
                + f"export FLASH_MODE=baseline FLASH_TASK=precheck; torchrun --nnodes=2 --nproc_per_node=8 --node_rank={node} --master_addr=10.10.4.1 --master_port={port+1} --no_python bash scripts/run_rank.sh; echo $? > {base}/precheck_exit{node}"
            )
            return start(host, cmd, base + f"/precheck_torchrun{node}.log")

        with concurrent.futures.ThreadPoolExecutor() as pool:
            for result in pool.map(check, enumerate(["10.10.4.1", "10.10.4.2"])):
                if result[1]:
                    raise RuntimeError(result[2])
        deadline = time.monotonic() + 150
        while True:
            codes = [
                execute(host, f"cat {base}/precheck_exit{node} 2>/dev/null")
                for node, host in enumerate(["10.10.4.1", "10.10.4.2"])
            ]
            if all(code[1] == 0 for code in codes):
                if any(code[2].strip() != "0" for code in codes):
                    raise RuntimeError(
                        "Coordinated GPU pre-check failed; training was not started"
                    )
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Coordinated pre-check did not finish")
            time.sleep(0.5)
    # Controllers are launched independently of torchrun so crashes cannot kill monitoring.
    if mode != "baseline":
        for node, host in enumerate(["10.10.4.1", "10.10.4.2"]):
            print(
                start(
                    host,
                    env
                    + f"export FLASH_NODE_ADDR={host}; python -m flash.local_controller --run {run} --node {node}",
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
        remote = "/home/campus/zhangweihao/flash-repro-20261015/results/" + run
        client = sftp = None
        failures = 0

        def open_sftp():
            nonlocal client, sftp
            if sftp is not None:
                try:
                    sftp.close()
                except OSError:
                    pass
            if client is not None:
                client.close()
            client = connect(host)
            sftp = client.open_sftp()

        try:
            for attempt in range(3):
                try:
                    open_sftp()
                    files = sftp.listdir_attr(remote)
                    break
                except Exception:
                    failures += 1
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)

            for f in files:
                if f.filename.endswith(".pt"):
                    continue
                source = remote + "/" + f.filename
                target = dest / f.filename
                partial = target.with_name(target.name + ".part")
                for attempt in range(3):
                    try:
                        before = sftp.stat(source).st_size
                        sftp.get(source, str(partial), prefetch=False)
                        after = sftp.stat(source).st_size
                        if before != after or partial.stat().st_size != after:
                            raise IOError("remote file changed during transfer")
                        os.replace(partial, target)
                        break
                    except Exception:
                        failures += 1
                        partial.unlink(missing_ok=True)
                        if attempt == 2:
                            raise
                        time.sleep(attempt + 1)
                        open_sftp()
        finally:
            if sftp is not None:
                try:
                    sftp.close()
                except OSError:
                    pass
            if client is not None:
                client.close()
        if failures:
            print(
                json.dumps(
                    dict(run=run, node=node, host=host, transfer_retries=failures)
                ),
                flush=True,
            )
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
