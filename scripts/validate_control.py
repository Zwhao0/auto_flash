"""Final controls after removing fixture-only NCCL stopping traffic."""

import json
import subprocess
import sys
from scripts.launch import launch
from scripts.experiment import finish
from scripts.remote import ROOT

if __name__ == "__main__":
    status = []
    for run, mode, args, force in [
        ("paper_gloo_baseline", "baseline", "--steps 160 --seq 128 --batch 2", False),
        ("paper_gloo_monitor", "monitor", "--steps 160 --seq 128 --batch 2", False),
        (
            "paper_gloo_healthy",
            "monitor",
            "--steps 120 --min-seconds 34 --seq 128 --batch 2",
            True,
        ),
        (
            "paper_gloo_delay45",
            "monitor",
            "--steps 120 --min-seconds 34 --seq 128 --batch 2 --fault delay --severity .45",
            False,
        ),
    ]:
        print("START", run, flush=True)
        launch(run, mode=mode, args=args, force=force)
        status.append((run, finish(run, mode)))
    run = "paper_gloo_clock600"
    rc = subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "scripts.clock_experiment",
            run,
            "--mhz",
            "600",
            "--force-profile",
        ]
    ).returncode
    status.append((run, rc))
    (ROOT / "results" / "suite_control.json").write_text(json.dumps(status, indent=2))
    print("CONTROL PASS COMPLETE", status, flush=True)
