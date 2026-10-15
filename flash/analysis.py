"""Thesis §§3.2.3, 3.3.2 two-stage localization; no ground-truth input."""

from collections import defaultdict
import numpy as np


def cv(xs):
    xs = np.asarray(xs, dtype=float)
    return float(xs.std() / xs.mean()) if len(xs) and xs.mean() > 0 else 0.0


def analyze(stats, group_specs, cv_threshold=0.1, percent=0.2):
    cliques = defaultdict(lambda: defaultdict(list))
    for s in stats:
        if s["count"] and s["mean_us"] > 0:
            cliques[str(s["hash"])][int(s["rank"])].append(s)
    tags = defaultdict(list)
    incomplete = []
    for h, byrank in cliques.items():
        members = set(byrank)
        sizes = {int(s["size"]) for values in byrank.values() for s in values}
        if len(sizes) != 1 or len(members) != next(iter(sizes)):
            incomplete.append(
                dict(hash=h, ranks=sorted(members), expected_sizes=sorted(sizes))
            )
            continue
        # Preserve every communicator for a rank instead of keeping the last
        # handle with the same id_hash. Means are weighted by recorded samples.
        values = {
            r: sum(s["mean_us"] * s["count"] for s in rows)
            / sum(s["count"] for s in rows)
            for r, rows in byrank.items()
        }
        max_avg = max(s["mean_us"] for rows in byrank.values() for s in rows)
        matching = [
            g for g in group_specs if set(g["ranks"]) == members and len(members) > 1
        ]
        eligible = [g for g in matching if g["tag"] not in ["ep", "tp_ep"]]
        if not matching and len(members) > 1:
            eligible = [
                dict(
                    tag="special_" + "_".join(map(str, sorted(members))),
                    ranks=sorted(members),
                )
            ]
        for spec in eligible:
            if (
                set(spec["ranks"]) == members
                and len(members) > 1
                and spec["tag"] not in ["ep", "tp_ep"]
            ):
                tags[spec["tag"]].append(
                    dict(
                        hash=h,
                        ranks=sorted(members),
                        values=values,
                        max_avg_latency=max_avg,
                    )
                )
    evidence = defaultdict(list)
    clique_scores = []
    for tag, groups in tags.items():
        # Communicators with identical rank sets but distinct hashes remain distinct cliques.
        avg = float(np.mean([g["max_avg_latency"] for g in groups]))
        spread = cv([g["max_avg_latency"] for g in groups])
        for g in groups:
            clique_scores.append(
                dict(
                    tag=tag,
                    hash=g["hash"],
                    ranks=g["ranks"],
                    score=(g["max_avg_latency"] - avg) / avg,
                    max_avg_latency=g["max_avg_latency"],
                    cv=spread,
                )
            )
        if len(groups) > 1 and spread < cv_threshold:
            continue
        g = max(groups, key=lambda g: g["max_avg_latency"])
        values = list(g["values"].values())
        mean = float(np.mean(values))
        rank_cv = cv(values)
        if rank_cv < cv_threshold:
            continue
        rank = min(g["values"], key=g["values"].get)
        score = (mean - g["values"][rank]) / mean
        if score < percent:
            continue
        evidence[rank].append(
            dict(
                tag=tag,
                hash=g["hash"],
                clique=g["ranks"],
                mean_us=float(g["values"][rank]),
                clique_mean_us=mean,
                cv=rank_cv,
                score=score,
            )
        )
    suspects = [
        dict(
            rank=r,
            tags=len(set(v["tag"] for v in es)),
            score=sum(e["score"] for e in es),
            evidence=es,
        )
        for r, es in evidence.items()
    ]
    suspects.sort(key=lambda s: (-s["tags"], -s["score"], s["rank"]))
    return dict(
        suspects=suspects,
        clique_scores=clique_scores,
        incomplete=incomplete,
        complete_cliques=len(cliques) - len(incomplete),
    )
