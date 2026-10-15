"""Recompute v1 observation windows from raw rank logs, without GPU work."""

import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def unique(pattern):
    matches = list(RESULTS.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one run for {pattern}, got {len(matches)}")
    return matches[0]


def window_metrics(directory, ranks, windows, axis):
    measurements = []
    for rank in ranks:
        rows = {
            r["iteration"]: r
            for r in map(
                json.loads,
                (directory / f"train_rank{rank}.jsonl").read_text().splitlines(),
            )
            if r.get("kind") == "iteration"
        }
        rank_windows = []
        for lo, hi in windows:
            selected = [r for r in rows.values() if lo <= r[axis] < hi]
            delta = [
                1000 * (r["monotonic"] - rows[r["iteration"] - 1]["monotonic"])
                for r in selected
                if r["iteration"] - 1 in rows
            ]
            rank_windows.append(
                dict(
                    samples=len(selected),
                    e2e_samples=len(delta),
                    time_ms=median(r["time_ms"] for r in selected),
                    e2e_ms=median(delta),
                )
            )
        measurements.append(dict(rank=rank, windows=rank_windows))
    aggregate = {
        metric: [median(r["windows"][i][metric] for r in measurements) for i in [0, 1]]
        for metric in ["time_ms", "e2e_ms"]
    }
    return dict(
        run=directory.name,
        axis=axis,
        windows=windows,
        aggregation="window median within rank, then median across selected ranks",
        per_rank=measurements,
        medians=aggregate,
        throughput_loss_percent={
            k: 100 * (1 - v[0] / v[1]) for k, v in aggregate.items()
        },
        time_increase_percent={
            k: 100 * (v[1] / v[0] - 1) for k, v in aggregate.items()
        },
    )


def main():
    result = dict(
        scope="Observation windows, not randomized causal estimates or detector recall",
        fault_sweep={},
        background={},
        clock={},
    )
    for label in ["delay5", "delay25", "delay65", "delay85", "transient500ms"]:
        directory = unique(f"audit_{label}_post_gradient_fix_*")
        result["fault_sweep"][label] = window_metrics(
            directory, [8], [(45, 60), (65, 80)], "iteration"
        )
    for label in ["mild", "strong"]:
        directory = unique(f"audit_background_{label}_post_gradient_fix_*")
        workers = [
            json.loads(p.read_text()) for p in directory.glob("background_node*.json")
        ]
        begin = max(w["started_at"] for w in workers)
        end = min(w["finished_at"] for w in workers)
        result["background"][label] = window_metrics(
            directory, range(16), [(begin - 15, begin), (begin, end)], "timestamp"
        )
    for index in range(1, 5):
        directory = unique(f"audit_clock600_topology{index}_post_gradient_fix_*")
        events = json.loads((directory / "clock_injection.json").read_text())
        locked = next(e["time"] for e in events if "-lgc" in e["command"])
        telemetry = json.loads((directory / "telemetry.json").read_text())
        profiles = [telemetry[f"stats:{n}"]["profile"] for n in range(2)]
        begin = max(p["started_at"] for p in profiles)
        end = min(p["capture_stopped_at"] for p in profiles)
        result["clock"][str(index)] = window_metrics(
            directory, range(16), [(locked - 15, locked), (begin, end)], "timestamp"
        )
    (RESULTS / "v1_window_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    for category in ["fault_sweep", "background", "clock"]:
        for name, row in result[category].items():
            print(category, name, row["medians"], row["throughput_loss_percent"])


if __name__ == "__main__":
    main()
