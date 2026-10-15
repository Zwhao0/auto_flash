"""Offline paper equations (3-1)/(3-2), including scores rejected by the detector.

Ground truth is used only in this offline evaluator, never in FLASH controllers.
Unweighted rank means implement the paper's unspecified clique-mean convention.
"""

import csv
import json
import statistics
from collections import defaultdict
from scripts.remote import ROOT

out = []
for directory in sorted((ROOT / "results").glob("paper_*")):
    path = directory / "telemetry.json"
    if not path.exists():
        continue
    data = json.loads(path.read_text())
    cliques = defaultdict(dict)
    for node in ["0", "1"]:
        for s in (data.get("stats:" + node) or {}).get("stats", []):
            if s["count"] and s["mean_us"] > 0:
                cliques[str(s["hash"])][s["rank"]] = s
    specs = set()
    for r in range(16):
        for g in data.get("parallel_states_" + str(r)) or []:
            if g["tag"] not in ["ep", "tp_ep"] and len(g["ranks"]) > 1:
                specs.add((g["tag"], tuple(sorted(g["ranks"]))))
    bytag = defaultdict(list)
    for h, ranks in cliques.items():
        if len(ranks) != next(iter(ranks.values()))["size"]:
            continue
        for tag, members in specs:
            if tuple(sorted(ranks)) == members:
                mean = statistics.mean(s["mean_us"] for s in ranks.values())
                bytag[tag].append((h, ranks, mean))
    for tag, groups in bytag.items():
        mu = statistics.mean(g[2] for g in groups)
        for h, ranks, mean in groups:
            for rank, s in ranks.items():
                out.append(
                    dict(
                        run=directory.name,
                        tag=tag,
                        hash=h,
                        rank=rank,
                        members=sorted(ranks),
                        clique_mean_us=mean,
                        rank_mean_us=s["mean_us"],
                        eq31=(mean - mu) / mu,
                        eq32=(mean - s["mean_us"]) / mean,
                    )
                )
dest = ROOT / "results" / "paper_equations.csv"
with dest.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(out[0]) if out else [])
    writer.writeheader()
    writer.writerows(out)
print("Saved", len(out), "paper equation rows", dest)
