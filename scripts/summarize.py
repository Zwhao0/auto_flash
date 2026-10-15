"""Build machine-readable summaries from saved evidence, never paper target numbers."""

import csv, json, pathlib, statistics, math

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def rows(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text().splitlines():
        if line.strip():
            result.append(json.loads(line))
    return result


def median(values):
    return statistics.median(values) if values else None


trials = []
clique_rows = []
rank_rows = []
for directory in sorted(p for p in RESULTS.iterdir() if p.is_dir()):
    execution = read_json(directory / "execution.json")
    telemetry = read_json(directory / "telemetry.json", {})
    if not execution:
        continue
    training = rows(directory / "train_rank0.jsonl")
    iterations = [r for r in training if r.get("kind") == "iteration"]
    meta = next((r for r in training if r.get("kind") == "metadata"), {})
    summary = next((r for r in training if r.get("kind") == "summary"), {})
    events = telemetry.get("events", rows(directory / "controller.jsonl"))
    injections = [
        e for e in events if e.get("kind") == "injection" and e.get("fault") != "none"
    ]
    profiles = [e for e in events if e.get("kind") == "profile_start"]
    diagnoses = [e for e in events if e.get("kind") == "diagnosis"]
    stops = telemetry.get(
        "failstop_ranks", [e for e in events if e.get("kind") == "failstop"]
    )
    target = injections[0].get("rank") if injections else None
    if target is None and (directory / "clock_injection.json").exists():
        target = 8
    predicted = sorted(
        set(v["rank"] for d in diagnoses for v in d["result"]["suspects"])
    )
    fault = meta.get("config", {}).get("fault", "none")
    faultstep = meta.get("config", {}).get("fault_step", 45)
    if (directory / "clock_injection.json").exists():
        fault = "gpu_clock"
        faultstep = next(
            (
                e.get("observed_iteration", 45)
                for e in injections
                if e.get("fault") == "gpu_clock"
            ),
            45,
        )
    if any(e.get("fault") == "background_nccl" for e in injections):
        fault = "background_nccl"
        faultstep = next(
            e.get("observed_iteration", 45)
            for e in injections
            if e.get("fault") == "background_nccl"
        )
    before = [
        r["time_ms"] for r in iterations if faultstep - 15 <= r["iteration"] < faultstep
    ]
    after = [
        r["time_ms"]
        for r in iterations
        if faultstep + 5 <= r["iteration"] < faultstep + 20
    ]
    base, post = median(before), median(after)
    actual_loss = 1 - base / post if base and post else None
    headers = [
        h
        for node in ["0", "1"]
        for h in (telemetry.get("stats:" + node) or {}).get("headers", [])
    ]
    row = dict(
        run=directory.name,
        exits=execution.get("exits"),
        invalid=(directory / "INVALID.md").exists()
        or (directory / "LIMITATION.md").exists(),
        fault=fault,
        nominal_severity=meta.get("config", {}).get("severity"),
        target_rank=target,
        predicted_ranks=predicted,
        automatic_profiles=sum(
            not e.get("trigger", {}).get("forced", False) for e in profiles
        ),
        forced_profiles=sum(
            bool(e.get("trigger", {}).get("forced", False)) for e in profiles
        ),
        failstop_ranks=sorted(set(e["rank"] for e in stops)),
        baseline_iter_median_ms=base,
        post_iter_median_ms=post,
        actual_throughput_loss=actual_loss,
        steady_tokens_per_second=summary.get("steady_e2e_tokens_per_second"),
        steps=summary.get("steps", iterations[-1]["iteration"] if iterations else 0),
        nccl_events=sum(h[5] for h in headers),
        timed_events=sum(h[7] for h in headers),
        probe_errors=sum(h[8] for h in headers),
        capture_skipped=sum(h[13] for h in headers),
        lost=sum(
            (telemetry.get("stats:" + n) or {}).get("lost", 0) for n in ["0", "1"]
        ),
        nodes_observed=sum(bool(telemetry.get("stats:" + n)) for n in ["0", "1"]),
        config=meta.get("config", {}),
        fixture_stop_backend=meta.get("config", {}).get(
            "stop_backend", "nccl" if iterations else None
        ),
        diagnoses=diagnoses,
    )
    if target is not None:
        row.update(
            tp=int(target in predicted),
            fp=len([r for r in predicted if r != target]),
            fn=int(target not in predicted),
        )
    if injections:
        if diagnoses:
            row["localization_seconds"] = diagnoses[0]["time"] - injections[0]["time"]
        automatic = [
            e for e in profiles if not e.get("trigger", {}).get("forced", False)
        ]
        if automatic:
            row["detection_seconds"] = automatic[0]["time"] - injections[0]["time"]
        root_stops = [e for e in stops if e["rank"] == target]
        if root_stops:
            row["failstop_detection_seconds"] = (
                min(e["time"] for e in root_stops) - injections[0]["time"]
            )
    for d in diagnoses:
        for g in d["result"]["clique_scores"]:
            clique_rows.append(dict(run=directory.name, **g))
        for candidate in d["result"]["suspects"]:
            for evidence in candidate["evidence"]:
                rank_rows.append(
                    dict(run=directory.name, rank=candidate["rank"], **evidence)
                )
    trials.append(row)
(RESULTS / "summary.json").write_text(json.dumps(trials, indent=2, ensure_ascii=False))
for name, data in [
    ("trials", trials),
    ("clique_scores", clique_rows),
    ("rank_scores", rank_rows),
]:
    fields = sorted(set().union(*(r.keys() for r in data))) if data else []
    with (RESULTS / (name + ".csv")).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in data:
            writer.writerow(
                {
                    k: (
                        json.dumps(v, ensure_ascii=False)
                        if isinstance(v, (list, dict))
                        else v
                    )
                    for k, v in r.items()
                }
            )
print(
    "Summarized",
    len(trials),
    "runs;",
    len(clique_rows),
    "clique rows;",
    len(rank_rows),
    "rank evidence rows",
)
