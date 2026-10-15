"""One final development pass after fixture repairs; preserves earlier trials."""

import json
import subprocess
import sys
from scripts.remote import ROOT
from scripts.launch import launch
from scripts.experiment import finish


def job(run, mode="monitor", args="", force=False):
    print("START", run, flush=True)
    launch(run, mode=mode, args=args, force=force)
    return finish(run, mode)


if __name__ == "__main__":
    status = []
    for pp in [1, 2]:
        run = f"paper_clock600_ep4_tp2_pp{pp}_fixed"
        rc = subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                "scripts.clock_experiment",
                run,
                "--mhz",
                "600",
                "--tp",
                "2",
                "--pp",
                str(pp),
                "--ep",
                "4",
                "--force-profile",
            ]
        ).returncode
        status.append((run, rc))
    for run, mode, args, force in [
        ("paper_baseline_final", "baseline", "--steps 160 --seq 128 --batch 2", False),
        ("paper_monitor_final", "monitor", "--steps 160 --seq 128 --batch 2", False),
        (
            "paper_profile_healthy_fixed",
            "monitor",
            "--steps 120 --min-seconds 34 --seq 128 --batch 2",
            True,
        ),
        (
            "paper_delay_boundary",
            "monitor",
            "--steps 120 --min-seconds 34 --seq 128 --batch 2 --fault delay --delay-ms 42",
            False,
        ),
    ]:
        status.append((run, job(run, mode, args, force)))
    status.append(
        (
            "paper_background_final",
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "scripts.network_experiment",
                    "paper_background_final",
                ]
            ).returncode,
        )
    )
    (ROOT / "results" / "suite_final.json").write_text(json.dumps(status, indent=2))
    print("FINAL PASS COMPLETE", status, flush=True)
