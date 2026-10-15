import argparse, time, json, pathlib
from .store import Store
from .detection import Detector
from .analysis import analyze


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--force-profile", action="store_true")
    p.add_argument("--min-persistence", type=int, default=0)
    a = p.parse_args()
    s = Store(a.run)
    cfg = s.config
    d = Detector(
        window=cfg["window"],
        prob=cfg["thresh_prob"],
        degradation=cfg["degradation_thresh"],
        recovery=cfg["recovery_ratio"],
        guard=cfg["right_guard"],
        min_persistence=a.min_persistence or cfg["min_persistence"],
    )
    epoch = 0
    last = -1
    active = None
    done = set()
    s.set("control", dict(mode=0, epoch=0))
    path = pathlib.Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)

    def emit(kind, **kw):
        row = dict(kw, kind=kind, time=time.time())
        print(json.dumps(row), flush=True)
        with path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        s.event(kind, **kw)

    while not s.get("stop", False):
        its = [tuple(v.split(":")) for v in s.items("iter_time_0")]
        latest = int(its[-1][0]) if its else -1
        fs = s.items("failstop_ranks")
        for e in fs:
            key = (e["node"], e["rank"])
            if key not in done:
                emit("failstop", **e)
                done.add(key)
        if active:
            if time.monotonic() > active["deadline"]:
                blobs = [s.get("stats:" + n, {}) for n in ["0", "1"]]
                valid = [b for b in blobs if b.get("epoch") == epoch]
                specs = {}
                for r in range(int(s.get("world_size", 16))):
                    for g in s.get("parallel_states_" + str(r), []):
                        specs[(g["tag"], tuple(g["ranks"]))] = g
                result = analyze(
                    [st for b in valid for st in b.get("stats", [])],
                    list(specs.values()),
                    cv_threshold=cfg["cv_threshold"],
                    percent=cfg["rank_threshold"],
                )
                emit(
                    "diagnosis",
                    epoch=epoch,
                    trigger=active["trigger"],
                    profile_seconds=active["seconds"],
                    nodes_received=len(valid),
                    lost=sum(b.get("lost", 0) for b in valid),
                    result=result,
                )
                s.set("control", dict(mode=0, epoch=epoch))
                active = None
        elif latest != last and its:
            last = latest
            triggers = d.update(its)
            if (
                a.force_profile
                and epoch == 0
                and latest >= int(s.get("force_after", 35))
            ):
                triggers = [dict(forced=True, observed_iteration=latest)]
            if triggers:
                epoch += 1
                secs = max(
                    float(its[-1][1]) / 1000.0 * cfg["profile_multiplier"],
                    cfg["profile_min_seconds"],
                )
                trigger = triggers[0]
                active = dict(
                    deadline=time.monotonic() + secs + 0.5,
                    seconds=secs,
                    trigger=trigger,
                )
                s.set("control", dict(mode=1, epoch=epoch))
                emit("profile_start", epoch=epoch, seconds=secs, trigger=trigger)
        time.sleep(cfg["poll_seconds"])
    emit("controller_stopped", epochs=epoch)


if __name__ == "__main__":
    main()
