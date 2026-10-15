import time, argparse, os, socket
from collections import defaultdict, Counter
import numpy as np
from .store import Store
from .shm import Reader, discover
from .profile import process_state
from .driver_status import snapshot


def aggregate(acc, metadata):
    result = []
    for key, vals in acc.items():
        md = metadata[key]
        result.append(
            dict(
                **md,
                count=len(vals),
                mean_us=float(np.mean(vals)),
                min_us=float(np.min(vals)),
                max_us=float(np.max(vals)),
                std_us=float(np.std(vals)),
            )
        )
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--node", required=True)
    a = p.parse_args()
    s = Store(a.run)
    readers, metadata = {}, {}
    epoch, mode = 0, 0
    acc = defaultdict(list)
    operations = defaultdict(Counter)
    reported = set()
    active = None
    stats = []
    complete = False
    unknown = 0
    profile_info = {}
    while not s.get("stop", False):
        now = time.monotonic()
        ctrl = s.get("control", dict(mode=0, epoch=0))
        if ctrl["mode"] == 1 and ctrl["epoch"] != epoch:
            epoch, mode = ctrl["epoch"], 1
            acc.clear()
            stats, complete, unknown = [], False, 0
            its = s.items("iter_time_0")
            latest = float(its[-1].split(":")[1]) / 1000 if its else 0
            seconds = max(
                latest * s.config["profile_multiplier"], s.config["profile_min_seconds"]
            )
            # The request freezes one budget; both nodes acknowledge their own start.
            seconds = float(ctrl.get("seconds", seconds))
            active = dict(
                deadline=now + seconds,
                drain_deadline=None,
                quiet=0,
                drain_seconds=max(latest * s.config["profile_multiplier"], 2.0),
            )
            profile_info = dict(started_at=time.time(), requested_seconds=seconds)
        if active and mode == 1 and now >= active["deadline"]:
            mode = 0
            active["drain_deadline"] = now + active["drain_seconds"]
            profile_info["capture_stopped_at"] = time.time()
        for path in discover(a.run):
            if str(path) not in readers:
                try:
                    readers[str(path)] = Reader(path)
                except (ValueError, OSError):
                    continue
        added = 0
        for reader in readers.values():
            reader.mode(mode, epoch)
            h = reader.header()
            rank = int(h[9])
            md = reader.metadata_by_identity()
            for (comm, chash), value in md.items():
                metadata[(rank, comm, chash)] = value
            for r in reader.records():
                operations[rank][str(r[4])] += 1
                if active and r[8] == epoch and r[7] > 0 and r[9] == 1:
                    key = (rank, r[2], r[3])
                    if key in metadata:
                        acc[key].append(r[7] / 1000.0)
                        added += 1
                    else:
                        unknown += 1
        if active and mode == 0:
            pending = sum(r.header()[15] for r in readers.values())
            active["quiet"] = active["quiet"] + 1 if pending == 0 and added == 0 else 0
            if active["quiet"] >= 2 or now >= active["drain_deadline"]:
                complete = active["quiet"] >= 2 and unknown == 0
                stats = aggregate(acc, metadata)
                profile_info.update(
                    finished_at=time.time(),
                    active_scopes_at_finish=pending,
                    unknown_records=unknown,
                    driver_status=snapshot(),
                )
                active = None
        headers = [list(r.header()) for r in readers.values()]
        s.set("ack:" + a.node, dict(epoch=epoch, mode=mode, time=time.time()))
        s.set(
            "stats:" + a.node,
            dict(
                epoch=epoch,
                mode=mode,
                node=a.node,
                host=os.getenv("FLASH_NODE_ADDR", socket.gethostname()),
                complete=complete,
                stats=stats,
                operations={str(k): dict(v) for k, v in operations.items()},
                lost=sum(r.lost for r in readers.values()),
                headers=headers,
                profile=profile_info,
                time=time.time(),
            ),
        )
        its = s.items("iter_time_0")
        latest = float(its[-1].split(":")[1]) / 1000 if its else None
        if latest is not None and not s.get("training_done", False):
            window = s.config["failstop_multiplier"] * latest
            checked_at = time.monotonic()
            for reader in readers.values():
                h = reader.header()
                rank, ts = int(h[9]), h[4] / 1e9
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
                            pid=int(h[10]),
                            **process_state(int(h[10])),
                        ),
                        1000,
                    )
        time.sleep(s.config["poll_seconds"])
    for reader in readers.values():
        reader.close()


if __name__ == "__main__":
    main()
