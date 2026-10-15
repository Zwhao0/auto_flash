"""Static research figures from saved evidence; run with matplotlib installed."""

import json
import pathlib
import csv
import statistics
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = pathlib.Path(__file__).resolve().parents[1] / "results"
dest = root / "figures"
dest.mkdir(exist_ok=True)
rows = {r["run"]: r for r in json.loads((root / "summary.json").read_text())}
plt.rcParams.update(
    {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
)
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
prefix = "paper_gloo" if "paper_gloo_monitor" in rows else "paper"
pair = (
    [("paper_gloo_baseline", "Baseline"), ("paper_gloo_monitor", "FLASH monitor")]
    if prefix == "paper_gloo"
    else [
        ("paper_baseline_final", "Baseline"),
        ("paper_monitor_final", "FLASH monitor"),
    ]
)
for run, label in pair:
    samples = [
        json.loads(s)
        for s in (root / run / "train_rank0.jsonl").read_text().splitlines()
    ]
    samples = [s for s in samples if s["kind"] == "iteration" and s["iteration"] >= 6]
    axes[0].plot(
        [s["iteration"] for s in samples],
        [s["tokens_per_second"] / 1000 for s in samples],
        label=label,
        alpha=0.75,
        lw=1,
    )
axes[0].set(
    xlabel="Iteration (warmup 1-5 excluded)",
    ylabel="Iteration throughput (k tokens/s)",
    title=(
        "16 H800, real 15M MoE; Gloo fixture control"
        if prefix == "paper_gloo"
        else "16 H800, real 15M MoE"
    ),
)
axes[0].legend()
sizes = [4, 8, 16, 32, 64, 128]
for base, mon, label in [
    ("paper_batch_base1", "paper_batch_mon1", "Pair 1"),
    ("paper_batch_base2", "paper_batch_mon2", "Pair 2"),
]:
    b = json.loads((root / base / "allreduce_rank0.json").read_text())
    m = json.loads((root / mon / "allreduce_rank0.json").read_text())
    axes[1].plot(
        sizes,
        [100 * (v["mean_ms"] / u["mean_ms"] - 1) for u, v in zip(b, m)],
        marker="o",
        label=label,
    )
axes[1].axhline(0.8, ls="--", color="gray", label="Paper 0.8% bound")
axes[1].set(
    xlabel="AllReduce payload (MiB)",
    ylabel="Monitor latency overhead (%)",
    title="Rank 0, 3 batches x 100 calls per size",
    xscale="log",
    xticks=sizes,
)
axes[1].set_xticklabels(sizes)
axes[1].legend()
fig.savefig(dest / "performance.png", dpi=180)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
for run, label in [
    ("paper_delay45", "40% measured loss"),
    ("paper_delay65", "63% measured loss"),
    ("paper_delay85", "85% measured loss"),
]:
    samples = [
        json.loads(s)
        for s in (root / run / "train_rank0.jsonl").read_text().splitlines()
    ]
    samples = [
        s for s in samples if s["kind"] == "iteration" and 25 <= s["iteration"] <= 80
    ]
    axes[0].plot(
        [s["iteration"] for s in samples], [s["time_ms"] for s in samples], label=label
    )
axes[0].axvline(45, color="gray", ls="--")
axes[0].set(
    xlabel="Iteration",
    ylabel="Iteration time (ms)",
    title="Sustained GPU delay (target rank 8)",
)
axes[0].legend()
data = list(csv.DictReader((root / "paper_equations.csv").open()))
freqs = [1965, 1600, 1200, 900, 600, 345]
values = []
for f in freqs:
    run = "paper_clock600_ep8_tp1_pp1" if f == 600 else f"paper_clock{f}"
    values.append(
        [float(r["eq32"]) for r in data if r["run"] == run and r["rank"] == "8"]
    )
axes[1].boxplot(values, labels=[str(f) for f in freqs])
axes[1].axhline(0, color="gray", ls="--")
axes[1].set(
    xlabel="Requested node-2 GPU 0 clock (MHz); see actual clocks in logs",
    ylabel="Target rank score, equation (3-2)",
    title="Earlier fixture: auxiliary NCCL; forced profiles",
)
fig.savefig(dest / "faults.png", dpi=180)
plt.close(fig)
print(dest)
