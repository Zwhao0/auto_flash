"""Fresh-ID, sequential FLASH paper-style audit campaign.

Run with the repository Python and FLASH_RUNTIME=paper. Each invocation writes
every completed run to results/audit_campaign_progress.json; run names are
generated at launch time and are never reused.
"""

import argparse
import json
import os
import pathlib
import time
import uuid
from datetime import datetime, timezone

os.environ["FLASH_RUNTIME"] = "paper"
COHORT = os.getenv("FLASH_AUDIT_COHORT", "")

from scripts.container import execute
from scripts.experiment import finish, redis_command
from scripts.launch import collect, launch, start
from scripts.remote import ROOT, sudo_run

PROGRESS = ROOT / "results" / "audit_campaign_progress.json"
BASE_TRAIN = "--seq 128 --batch 2"


def fresh_id(label):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cohort = f"_{COHORT}" if COHORT else ""
    run = f"audit_{label}{cohort}_{stamp}_{uuid.uuid4().hex[:8]}"
    if (ROOT / "results" / run).exists():
        raise RuntimeError(f"refusing existing local run directory: {run}")
    return run


def append_record(row):
    rows = json.loads(PROGRESS.read_text()) if PROGRESS.exists() else []
    rows.append(row)
    PROGRESS.write_text(json.dumps(rows, indent=2, ensure_ascii=False))


def run_job(label, mode="monitor", task="train", args="", force=False, timeout=300):
    run = fresh_id(label)
    row = dict(
        label=label,
        run=run,
        cohort=COHORT or None,
        mode=mode,
        task=task,
        args=args,
        force=force,
    )
    print(json.dumps(dict(event="start", **row), ensure_ascii=False), flush=True)
    try:
        launch(run, mode=mode, task=task, args=args, force=force)
        row["exits"] = finish(run, mode, timeout=timeout)
        row["status"] = "completed"
    except Exception as exc:
        row.update(status="error", error=type(exc).__name__ + ": " + str(exc))
        append_record(row)
        raise
    append_record(row)
    print(json.dumps(dict(event="complete", **row), ensure_ascii=False), flush=True)
    return row


def run_regression():
    run_job(
        "healthy_forced",
        args=BASE_TRAIN + " --steps 800 --no-save-checkpoint",
        force=True,
    )
    run_job(
        "delay45_auto",
        args=BASE_TRAIN
        + " --steps 800 --fault delay --severity 0.45 --fault-step 80 --no-save-checkpoint",
        timeout=360,
    )


def run_training_abba():
    abba = []
    for index, (label, mode) in enumerate(
        [
            ("train_base1", "baseline"),
            ("train_mon1", "monitor"),
            ("train_mon2", "monitor"),
            ("train_base2", "baseline"),
        ]
    ):
        checkpoint = index < 2
        args = BASE_TRAIN + " --steps 800"
        if not checkpoint:
            args += " --no-save-checkpoint"
        abba.append(run_job(label, mode=mode, args=args, timeout=360))
    compare_checkpoints(abba[0]["run"], abba[1]["run"])


def run_measure():
    for label, mode in [
        ("allreduce_base1", "baseline"),
        ("allreduce_mon1", "monitor"),
        ("allreduce_mon2", "monitor"),
        ("allreduce_base2", "baseline"),
    ]:
        run_job(
            label,
            mode=mode,
            task="allreduce",
            args="--warmup 100 --batches 5 --repeats 100",
        )

    run_training_abba()


def compare_checkpoints(baseline, observed):
    checks = []
    for node, host in enumerate(["10.10.4.1", "10.10.4.2"]):
        command = (
            "cd /flash && PYTHONPATH=/flash:/flash/deps:/flash/vendor/MegatronCore "
            f"python scripts/check_model.py {baseline} {observed} --node {node}"
        )
        result = execute(host, command, timeout=300)
        checks.append(dict(host=host, exit=result[1], output=result[2]))
        if result[1]:
            raise RuntimeError(f"checkpoint comparison failed on node {node}")
    collect(observed)
    append_record(
        dict(
            kind="checkpoint_comparison",
            baseline=baseline,
            observed=observed,
            checks=checks,
        )
    )


def run_faults():
    cases = [
        ("delay5", 0.05, 300),
        ("delay25", 0.25, 300),
        ("delay65", 0.65, 250),
        ("delay85", 0.85, 180),
    ]
    for label, severity, steps in cases:
        run_job(
            label,
            args=BASE_TRAIN
            + f" --steps {steps} --fault delay --severity {severity} --fault-step 60 --no-save-checkpoint",
            timeout=480,
        )
    run_job(
        "transient500ms",
        args=BASE_TRAIN
        + " --steps 200 --fault delay --delay-ms 500 --fault-duration 1 --fault-step 60 --no-save-checkpoint",
        timeout=360,
    )


def run_stop_faults():
    for fault in ["oom", "timeout"]:
        run = fresh_id(fault)
        args = (
            BASE_TRAIN
            + f" --steps 120 --fault {fault} --fault-step 45 --no-save-checkpoint"
        )
        row = dict(
            label=fault,
            run=run,
            cohort=COHORT or None,
            mode="monitor",
            task="train",
            args=args,
            force=False,
        )
        print(json.dumps(dict(event="start", **row), ensure_ascii=False), flush=True)
        try:
            launch(run, mode="monitor", args=args)
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                _, _, out = redis_command(
                    run,
                    "import json; print(json.dumps([e for e in s.items('events') "
                    "if e.get('kind') == 'injection']))",
                )
                events = json.loads(out.strip().splitlines()[-1])
                if events:
                    row["injection"] = events[-1]
                    break
                time.sleep(0.25)
            else:
                raise TimeoutError(f"{fault} injection event was not recorded")

            def process_snapshot(label):
                processes = []
                for node, host in enumerate(["10.10.4.1", "10.10.4.2"]):
                    command = (
                        "ps -eo pid,stat,comm,args | grep -E '[t]orchrun|[t]rain_moe.py' || true; "
                        "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader"
                    )
                    _, rc, output = execute(host, command, timeout=30)
                    processes.append(dict(node=node, host=host, exit=rc, output=output))
                snapshot = dict(label=label, time=time.time(), nodes=processes)
                row.setdefault("liveness", []).append(snapshot)
                path = ROOT / "results" / run
                path.mkdir(parents=True, exist_ok=True)
                (path / "liveness.json").write_text(
                    json.dumps(row["liveness"], indent=2)
                )

            time.sleep(0.5)
            process_snapshot("after_injection_0.5s")
            if fault == "timeout":
                time.sleep(10)
                process_snapshot("after_injection_10s")

            row["exits"] = finish(run, "monitor", timeout=300)
            process_snapshot("after_training_exit")
            row["status"] = "completed"
        except Exception as exc:
            row.update(status="error", error=type(exc).__name__ + ": " + str(exc))
            append_record(row)
            raise
        append_record(row)
        print(json.dumps(dict(event="complete", **row), ensure_ascii=False), flush=True)


def wait_iteration(run, target, timeout=120):
    start_time = time.monotonic()
    while time.monotonic() - start_time < timeout:
        _, _, output = redis_command(
            run,
            "v=s.items('iter_time_0'); print(v[-1].split(':')[0] if v else -1)",
        )
        try:
            step = int(output.strip())
        except ValueError:
            step = -1
        if step >= target:
            return step
        time.sleep(0.5)
    raise TimeoutError(f"run {run} did not reach iteration {target}")


def run_background(label, duty):
    run = fresh_id(label)
    args = BASE_TRAIN + " --steps 800 --no-save-checkpoint"
    print(json.dumps(dict(event="start", label=label, run=run, duty=duty)), flush=True)
    launch(run, mode="monitor", args=args)
    step = wait_iteration(run, 80)
    remote_base = f"/flash/results/{run}"
    env = (
        "export PYTHONPATH=/flash:/flash/deps; "
        "export NCCL_SOCKET_IFNAME=bond0 NCCL_NET=IB "
        "NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3; "
    )
    for node, host in enumerate(["10.10.4.1", "10.10.4.2"]):
        command = (
            env
            + f"export FLASH_RUN_ID={run}; torchrun --nnodes=2 --nproc_per_node=1 "
            + f"--node_rank={node} --master_addr=10.10.4.1 --master_port=29588 "
            + "scripts/background_comm.py --coordinated --mib 128 "
            + f"--duty {duty} --seconds 30 --output {remote_base}/background_node{node}.json"
        )
        result = start(host, command, remote_base + f"/background_node{node}.log")
        if result[1]:
            raise RuntimeError(
                f"background rank launch failed on node {node}: {result[2]}"
            )
    ready = False
    start_time = time.monotonic()
    while time.monotonic() - start_time < 90:
        checks = [
            execute(host, f"test -f {remote_base}/background_node{node}.json.ready")
            for node, host in enumerate(["10.10.4.1", "10.10.4.2"])
        ]
        if all(check[1] == 0 for check in checks):
            ready = True
            break
        time.sleep(0.5)
    if not ready:
        raise TimeoutError("background NCCL ranks did not both become ready")
    redis_command(run, "s.set('background_go',True)")
    redis_command(
        run,
        f"s.event('injection',fault='background_nccl',ranks=[0,8],mib=128,duty={duty},observed_iteration={step},gpu_contention=True)",
    )
    exits = finish(run, "monitor", timeout=360)
    complete = False
    start_time = time.monotonic()
    while time.monotonic() - start_time < 90:
        checks = [
            execute(host, f"test -f {remote_base}/background_node{node}.json")
            for node, host in enumerate(["10.10.4.1", "10.10.4.2"])
        ]
        if all(check[1] == 0 for check in checks):
            complete = True
            break
        time.sleep(1)
    collect(run)
    row = dict(
        label=label,
        run=run,
        mode="monitor",
        args=args,
        exits=exits,
        duty=duty,
        gpu_contention=True,
        background_complete=complete,
    )
    append_record(row)
    if not complete:
        raise TimeoutError("background NCCL traffic did not finish on both nodes")
    print(json.dumps(dict(event="complete", **row)), flush=True)


def run_clock_topology(label, mhz, tp, pp, ep, steps=400):
    run = fresh_id(label)
    path = ROOT / "results" / run
    path.mkdir(parents=True, exist_ok=True)
    clock_events = []

    def clock(command):
        host, rc, output = sudo_run("10.10.4.2", command)
        clock_events.append(
            dict(time=time.time(), command=command, exit=rc, output=output)
        )
        (path / "clock_injection.json").write_text(json.dumps(clock_events, indent=2))
        if rc:
            raise RuntimeError(f"clock operation failed: {command}")
        return output

    args = (
        BASE_TRAIN
        + f" --steps {steps} --tp {tp} --pp {pp} --ep {ep} --no-save-checkpoint"
    )
    row = dict(
        label=label,
        run=run,
        mode="monitor",
        task="train",
        args=args,
        force=True,
        mhz=mhz,
        tp=tp,
        pp=pp,
        ep=ep,
    )
    try:
        clock(
            "nvidia-smi -i 0 --query-gpu=name,clocks.current.sm,clocks.max.sm,power.limit --format=csv"
        )
        launch(run, mode="monitor", args=args, force=True)
        redis_command(run, "s.set('force_after',1000000000)")
        observed = wait_iteration(run, 45, timeout=120)
        clock(f"nvidia-smi -i 0 -lgc {mhz},{mhz}")
        redis_command(
            run,
            f"s.event('injection',fault='gpu_clock',rank=8,mhz={mhz},observed_iteration={observed})",
        )
        redis_command(run, "s.set('force_after',0)")
        time.sleep(1)
        clock(
            "nvidia-smi -i 0 --query-gpu=clocks.current.sm,clocks.current.memory,utilization.gpu --format=csv"
        )
        row["exits"] = finish(run, "monitor", timeout=360)
        row["status"] = "completed"
    except Exception as exc:
        row.update(status="error", error=type(exc).__name__ + ": " + str(exc))
        raise
    finally:
        try:
            sudo_run("10.10.4.2", "nvidia-smi -i 0 -rgc")
            clock(
                "nvidia-smi -i 0 --query-gpu=clocks.current.sm,clocks.max.sm --format=csv"
            )
        finally:
            pass
    append_record(row)
    print(json.dumps(dict(event="complete", **row), ensure_ascii=False), flush=True)


def run_topologies():
    for index, (tp, pp, ep) in enumerate([(1, 1, 8), (2, 1, 4), (1, 2, 4), (2, 2, 4)]):
        run_clock_topology(f"clock600_topology{index + 1}", 600, tp, pp, ep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        required=True,
        choices=[
            "regression",
            "measure",
            "training",
            "faults",
            "stop",
            "network",
            "topology",
            "all",
        ],
    )
    args = parser.parse_args()
    phases = (
        ["regression", "measure", "faults", "stop", "network", "topology"]
        if args.phase == "all"
        else [args.phase]
    )
    actions = {
        "regression": run_regression,
        "measure": run_measure,
        "training": run_training_abba,
        "faults": run_faults,
        "stop": run_stop_faults,
        "network": lambda: (
            run_background("background_mild", 0.25),
            run_background("background_strong", 1.0),
        ),
        "topology": run_topologies,
    }
    for phase in phases:
        print(json.dumps(dict(event="phase_start", phase=phase)), flush=True)
        actions[phase]()
        print(json.dumps(dict(event="phase_complete", phase=phase)), flush=True)


if __name__ == "__main__":
    main()
