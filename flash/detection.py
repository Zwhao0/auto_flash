"""Paper-selected Rbeast sliding-window detector, with explicit ambiguities."""

import math
import numpy as np


class Detector:
    def __init__(
        self,
        window=30,
        prob=0.8,
        degradation=0.2,
        recovery=0.75,
        guard=2,
        min_persistence=0,
    ):
        self.window = window
        self.prob = prob
        self.degradation = degradation
        self.recovery = recovery
        self.guard = guard
        self.min_persistence = min_persistence
        self.seen = []

    def update(self, rows):
        rows = [(int(i), float(t)) for i, t in rows if int(i) > 3][-self.window :]
        if len(rows) < self.window:
            return []
        ids = np.array([r[0] for r in rows])
        x = np.array([r[1] for r in rows])
        sd = x.std()
        if sd > 0:
            keep = np.abs((x - x.mean()) / sd) <= 3
            ids = ids[keep]
            x = x[keep]
        if len(x) < 8 or np.std(x) < 1e-9:
            return []
        import Rbeast as rb

        result = rb.beast(
            x,
            season="none",
            print_options=False,
            print_progress=False,
            quiet=True,
            hasOutlier=True,
            tcp_minmax=[0, 3],
            torder_minmax=[0, 0],
            tseg_min=2,
            tseg_leftmargin=2,
            tseg_rightmargin=self.guard,
            mcmc_seed=12345,
            mcmc_samples=3000,
            mcmc_burnin=500,
            mcmc_chainNumber=3,
        )
        if not hasattr(result, "trend"):
            raise RuntimeError("Rbeast did not return trend")
        candidates = []
        for cp, p in zip(
            np.asarray(result.trend.cp).ravel(), np.asarray(result.trend.cpPr).ravel()
        ):
            if (
                not math.isfinite(float(cp))
                or not math.isfinite(float(p))
                or p < self.prob
            ):
                continue
            # Rbeast time starts at 1 by default; convert to zero-based first post-change index.
            c = int(round(float(cp))) - 1
            if c < 1 or c > len(x) - self.guard - 1:
                continue
            left = x[:c]
            right = x[c:]
            baseline = float(left.mean())
            degraded = float(right.mean())
            recent = float(right[1:].mean())
            loss = 1 - baseline / degraded if degraded > 0 else 0
            ratio = (
                (degraded - recent) / (degraded - baseline)
                if degraded != baseline
                else 1
            )
            if (
                loss <= self.degradation
                or ratio >= self.recovery
                or len(right) < self.min_persistence
            ):
                continue
            absolute = int(ids[c])
            if any(abs(absolute - old) < 2 for old in self.seen):
                continue
            candidates.append(
                dict(
                    iteration=absolute,
                    observed_iteration=int(ids[-1]),
                    prob=float(p),
                    baseline_ms=baseline,
                    degraded_ms=degraded,
                    throughput_loss=loss,
                    recovery_ratio=ratio,
                )
            )
        if candidates:
            candidates.sort(key=lambda c: c["iteration"])
            accepted = []
            for candidate in candidates:
                if any(abs(candidate["iteration"] - old) < 2 for old in self.seen):
                    continue
                accepted.append(candidate)
                self.seen.append(candidate["iteration"])
            candidates = accepted
        return candidates
