"""Small development campaign; every condition once; never tunes detector to ground truth."""

import os, subprocess, sys, time, json, pathlib
from scripts.launch import launch
from scripts.experiment import finish
from scripts.remote import ROOT


def job(run, mode="monitor", task="train", args="", force=False):
    print("START", run, flush=True)
    launch(run, mode, task, args, force=force)
    return finish(run, mode)


def invoke(module, args):
    print("START", module, args, flush=True)
    return subprocess.run([sys.executable, "-u", "-m", module] + args).returncode


phase = sys.argv[1] if len(sys.argv) > 1 else "faults"
out = []
if phase == "measure":
    out.append(("precheck", job("paper_precheck", mode="baseline", task="precheck")))
    for run, mode in [
        ("paper_batch_base1", "baseline"),
        ("paper_batch_mon1", "monitor"),
        ("paper_batch_mon2", "monitor"),
        ("paper_batch_base2", "baseline"),
    ]:
        out.append(
            (
                run,
                job(
                    run,
                    mode=mode,
                    task="allreduce",
                    args="--repeats 100 --batches 3 --warmup 20",
                ),
            )
        )
    out.append(
        (
            "paper_profile_healthy",
            job(
                "paper_profile_healthy",
                args="--steps 120 --min-seconds 34 --seq 128 --batch 2",
                force=True,
            ),
        )
    )
    out.append(
        (
            "paper_monitor16b",
            job("paper_monitor16b", args="--steps 120 --seq 128 --batch 2"),
        )
    )
    out.append(
        (
            "paper_baseline16b",
            job(
                "paper_baseline16b",
                mode="baseline",
                args="--steps 120 --seq 128 --batch 2",
            ),
        )
    )
elif phase == "faults":
    for severity in [0.05, 0.25, 0.65, 0.85]:
        run = "paper_delay" + str(int(severity * 100))
        out.append(
            (
                run,
                job(
                    run,
                    args=f"--steps 100 --min-seconds {12 if severity==.05 else 34} --seq 128 --batch 2 --fault delay --severity {severity}",
                ),
            )
        )
    out.append(
        (
            "paper_transient",
            job(
                "paper_transient",
                args="--steps 120 --min-seconds 12 --seq 128 --batch 2 --fault delay --delay-ms 500 --fault-duration 1",
            ),
        )
    )
    for fault in ["oom", "timeout"]:
        out.append(
            (
                "paper_" + fault,
                job(
                    "paper_" + fault,
                    args=f"--steps 120 --seq 128 --batch 2 --fault {fault}",
                ),
            )
        )
    out.append(
        ("paper_background", invoke("scripts.network_experiment", ["paper_background"]))
    )
elif phase == "clocks":
    for tp, pp, ep in [(1, 1, 8), (2, 1, 4), (1, 2, 4), (2, 2, 4)]:
        run = f"paper_clock600_ep{ep}_tp{tp}_pp{pp}"
        out.append(
            (
                run,
                invoke(
                    "scripts.clock_experiment",
                    [
                        run,
                        "--mhz",
                        "600",
                        "--tp",
                        str(tp),
                        "--pp",
                        str(pp),
                        "--ep",
                        str(ep),
                        "--force-profile",
                    ],
                ),
            )
        )
    for mhz in [1965, 1600, 1200, 900, 345]:
        run = f"paper_clock{mhz}"
        out.append(
            (
                run,
                invoke(
                    "scripts.clock_experiment",
                    [run, "--mhz", str(mhz), "--force-profile"],
                ),
            )
        )
else:
    raise ValueError(phase)
(ROOT / "results" / f"suite_{phase}.json").write_text(json.dumps(out, indent=2))
print("PHASE COMPLETE", phase, out, flush=True)
