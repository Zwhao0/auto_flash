"""Release check of archived run evidence, separate from detector accuracy."""

import json
import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
INVALID = {
    "audit_qwen_conversion": "Development failure: missing checkpoint after optional dependency import failed"
}
parser = argparse.ArgumentParser()
parser.add_argument(
    "--final", action="store_true", help="Reject unfinalized run directories"
)
options = parser.parse_args()
rows = []
for directory in sorted(RESULTS.glob("audit_*")):
    if not directory.is_dir():
        continue
    if not (directory / "execution.json").exists():
        status_path = directory / "run_status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        if status.get("status") == "ABORTED_BEFORE_TRAINING" and not list(
            directory.glob("train_rank*.jsonl")
        ):
            rows.append(
                dict(
                    run=directory.name,
                    status="ABORTED_BEFORE_TRAINING",
                    reason=status.get("reason"),
                )
            )
        else:
            rows.append(dict(run=directory.name, status="UNFINALIZED"))
        continue
    if directory.name in INVALID:
        rows.append(
            dict(run=directory.name, status="INVALID", reason=INVALID[directory.name])
        )
        continue
    issues = []
    execution = json.loads((directory / "execution.json").read_text())
    metadata = json.loads((directory / "run_metadata.json").read_text())
    args = metadata.get("args", "")
    fault = (
        "oom"
        if "--fault oom" in args
        else ("timeout" if "--fault timeout" in args else None)
    )
    exits = execution["exits"]
    if fault:
        if not all(str(e) != "0" for e in exits):
            issues.append("fault campaign did not exit nonzero on both nodes")
    elif exits != ["0", "0"]:
        issues.append("unexpected training/benchmark exit")
    if metadata.get("task") == "train":
        for rank in range(16):
            path = directory / f"train_rank{rank}.jsonl"
            if not path.exists():
                issues.append(f"missing rank {rank} training log")
                continue
            try:
                lines = [json.loads(line) for line in path.read_text().splitlines()]
            except (ValueError, OSError):
                issues.append(f"invalid rank {rank} training log")
                continue
            iterations = [r for r in lines if r.get("kind") == "iteration"]
            if any(
                r.get("loss") is not None and not math.isfinite(r["loss"])
                for r in iterations
            ):
                issues.append(f"nonfinite rank {rank} loss")
            if not fault:
                summaries = [r for r in lines if r.get("kind") == "summary"]
                if not summaries or summaries[-1]["steps"] != len(iterations):
                    issues.append(f"incomplete rank {rank} iterations/summary")
        precheck = list(directory.glob("precheck_rank*.json"))
        if len(precheck) != 16:
            issues.append("missing coordinated 16 rank precheck")
        elif any(
            not all(
                json.loads(p.read_text()).get(k)
                for k in ["gemm_correct", "allreduce_correct"]
            )
            for p in precheck
        ):
            issues.append("failed compute or communication precheck")
    if metadata.get("task") == "allreduce":
        for rank in range(16):
            path = directory / f"allreduce_rank{rank}.json"
            if not path.exists():
                issues.append(f"missing rank {rank} AllReduce results")
            elif [r["mib"] for r in json.loads(path.read_text())] != [
                4,
                8,
                16,
                32,
                64,
                128,
            ]:
                issues.append(f"incomplete rank {rank} AllReduce size matrix")
    if fault:
        logs = "\n".join(
            p.read_text(errors="replace") for p in directory.glob("torchrun*.log")
        )
        if fault == "oom" and "OutOfMemoryError" not in logs:
            issues.append("missing actual CUDA OOM evidence")
        if fault == "timeout" and "Timeout(ms)=90000" not in logs:
            issues.append("missing actual NCCL watchdog timeout evidence")
    if metadata.get("mode") != "baseline":
        telemetry_path = directory / "telemetry.json"
        if not telemetry_path.exists():
            issues.append("missing telemetry")
        else:
            telemetry = json.loads(telemetry_path.read_text())
            headers = []
            for node in ["0", "1"]:
                blob = telemetry.get("stats:" + node, {})
                headers += blob.get("headers", [])
                if blob.get("lost", 0):
                    issues.append("probe record loss node " + node)
            if len(headers) != 16:
                issues.append("not all 16 probe headers archived")
            if any(h[8] for h in headers):
                issues.append("probe error counter nonzero")
            if not fault and metadata.get("task") == "train":
                rank0 = [
                    json.loads(line)
                    for line in (directory / "train_rank0.jsonl")
                    .read_text()
                    .splitlines()
                ]
                steps = rank0[-1].get("steps")
                writer = telemetry.get("iteration_reporter") or {}
                if (
                    writer.get("received") != steps
                    or writer.get("last_iteration") != steps
                ):
                    issues.append("background iteration report count mismatch")
    rows.append(
        dict(
            run=directory.name,
            status="PASS" if not issues else "CHECK_FAILED",
            exits=exits,
            expected_fault=fault,
            training_cohort=(
                (
                    "post_gradient_fix"
                    if "scripts/gradient_sync.py" in metadata.get("source_hashes", {})
                    else "pre_gradient_fix"
                )
                if metadata.get("task") == "train"
                else "not_applicable"
            ),
            issues=issues,
        )
    )
output = dict(
    scope="Audit runs with execution markers; PASS checks evidence integrity, not claimed accuracy",
    runs=rows,
)
(RESULTS / "v1_evidence_check.json").write_text(json.dumps(output, indent=2))
print(
    json.dumps(
        dict(
            checked=len(rows),
            invalid=sum(r["status"] == "INVALID" for r in rows),
            failed=sum(r["status"] == "CHECK_FAILED" for r in rows),
        )
    )
)
if any(
    r["status"] == "CHECK_FAILED" or (options.final and r["status"] == "UNFINALIZED")
    for r in rows
):
    raise SystemExit(1)
