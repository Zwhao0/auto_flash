"""Validate all local shards for finite parameters and baseline/probe equality."""

import torch, json, argparse, pathlib, os

p = argparse.ArgumentParser()
p.add_argument("baseline")
p.add_argument("observed")
p.add_argument("--node", type=int, required=True)
a = p.parse_args()
rows = []
for rank in range(a.node * 8, (a.node + 1) * 8):

    def load(run):
        return torch.load(
            f"results/{run}/train_rank{rank}.jsonl.pt", map_location="cpu"
        )["model"]

    x = load(a.baseline)
    y = load(a.observed)
    worst = 0.0
    finite = True
    elements = 0
    for k, v in x.items():
        if not isinstance(v, torch.Tensor):
            continue
        finite = finite and bool(torch.isfinite(y[k]).all())
        worst = max(worst, float((v - y[k]).abs().max()))
        elements += v.numel()
    rows.append(
        dict(rank=rank, finite=finite, elements=elements, max_abs_difference=worst)
    )
path = pathlib.Path("results") / a.observed / f"checkpoint_check_node{a.node}.json"
path.write_text(json.dumps(rows, indent=2))
print(json.dumps(rows))
if not all(r["finite"] and r["max_abs_difference"] < 1e-5 for r in rows):
    raise SystemExit(1)
