import time, json, argparse, os, socket
from collections import defaultdict, Counter
import numpy as np
from .store import Store
from .shm import Reader, discover


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--node", required=True)
    a = p.parse_args()
    s = Store(a.run)
    readers = {}
    epoch = -1
    mode = 0
    acc = defaultdict(list)
    operations = defaultdict(Counter)
    reported = set()
    last_seen = {}
    start = time.monotonic()
    while not s.get("stop", False):
        ctrl = s.get("control", dict(mode=0, epoch=0))
        new = (ctrl["epoch"], ctrl["mode"]) != (epoch, mode)
        if new and ctrl["mode"] == 1:
            acc.clear()
        epoch, mode = ctrl["epoch"], ctrl["mode"]
        for path in discover(a.run):
            key = str(path)
            if key not in readers:
                try:
                    readers[key] = Reader(path)
                except (ValueError, OSError):
                    continue
            reader = readers[key]
            reader.mode(mode, epoch)
            meta = reader.metadata()
            h = reader.header()
            rank = int(h[9])
            for r in reader.records():
                operations[rank][str(r[4])] += 1
                if r[8] == epoch and r[7] > 0 and r[2] in meta and r[9] == 1:
                    acc[(rank, r[2], r[3])].append(r[7] / 1000.0)
            last_seen[rank] = h[4] / 1e9
        if new:
            s.set("ack:" + a.node, dict(epoch=epoch, mode=mode, time=time.time()))
        stats = []
        for (rank, comm, chash), vals in acc.items():
            md = next(
                (
                    reader.metadata().get(comm)
                    for reader in readers.values()
                    if int(reader.header()[9]) == rank
                ),
                None,
            )
            if md:
                stats.append(
                    dict(
                        **md,
                        count=len(vals),
                        mean_us=float(np.mean(vals)),
                        min_us=float(np.min(vals)),
                        max_us=float(np.max(vals)),
                        std_us=float(np.std(vals)),
                    )
                )
        s.set(
            "stats:" + a.node,
            dict(
                epoch=epoch,
                mode=mode,
                stats=stats,
                operations={str(k): dict(v) for k, v in operations.items()},
                lost=sum(r.lost for r in readers.values()),
                headers=[list(r.header()) for r in readers.values()],
                time=time.time(),
            ),
        )
        it = s.items("iter_time_0")
        latest = float(it[-1].split(":")[1]) / 1000.0 if it else None
        # Arm after a completed iteration. Normal job completion is signalled by the launcher.
        if latest is not None and not s.get("training_done", False):
            window = s.config["failstop_multiplier"] * latest
            # Refresh after draining/aggregation: stale snapshots can falsely report live ranks.
            last_seen = {
                int(reader.header()[9]): reader.header()[4] / 1e9
                for reader in readers.values()
            }
            checked_at = time.monotonic()
            for rank, ts in last_seen.items():
                if checked_at - ts > window and rank not in reported:
                    reported.add(rank)
                    s.push(
                        "failstop_ranks",
                        dict(
                            rank=rank,
                            node=a.node,
                            window=window,
                            last_call_age=checked_at - ts,
                            time=time.time(),
                        ),
                        1000,
                    )
        time.sleep(s.config["poll_seconds"])
    for r in readers.values():
        r.close()


if __name__ == "__main__":
    main()
