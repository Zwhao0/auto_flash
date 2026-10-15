import argparse, time, json, pathlib
from .store import Store
from .detection import Detector
from .analysis import analyze
from .profile import complete_nodes


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
    epoch, last, active = 0, -1, None
    done = set()
    s.set("control", dict(mode=0, epoch=0))
    path = pathlib.Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)

    def emit(kind, **kw):
        row = dict(kw, kind=kind, time=time.time())
        if kind == "failstop" or (
            kind == "diagnosis" and kw.get("result", {}).get("suspects")
        ):
            row["level"] = "CRITICAL"
        elif kind == "diagnosis":
            row["level"] = (
                "WARNING"
                if kw.get("result", {}).get("status") == "incomplete_profile"
                else "INFO"
            )
        print(json.dumps(row), flush=True)
        with path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        s.event(kind, **kw)

    while not s.get("stop", False):
        its = [tuple(v.split(":")) for v in s.items("iter_time_0")]
        latest = int(its[-1][0]) if its else -1
        for event in s.items("failstop_ranks"):
            key = (event["node"], event["rank"])
            if key not in done:
                emit("failstop", **event)
                done.add(key)
        if active:
            blobs = [s.get("stats:" + node, {}) for node in ["0", "1"]]
            valid, complete = complete_nodes(blobs, epoch)
            expired = time.monotonic() > active["deadline"]
            if complete or expired:
                if complete:
                    specs = {}
                    for rank in range(int(s.get("world_size", 16))):
                        for g in s.get("parallel_states_" + str(rank), []):
                            specs[(g["tag"], tuple(g["ranks"]))] = g
                    result = analyze(
                        [st for b in valid for st in b.get("stats", [])],
                        list(specs.values()),
                        cv_threshold=cfg["cv_threshold"],
                        percent=cfg["rank_threshold"],
                    )
                    result["status"] = "complete"
                    locations = {
                        int(h[9]): dict(node=b["node"], host=b["host"], pid=int(h[10]))
                        for b in valid
                        for h in b.get("headers", [])
                    }
                    for candidate in result["suspects"]:
                        candidate["location"] = locations.get(candidate["rank"])
                else:
                    result = dict(
                        status="incomplete_profile",
                        suspects=[],
                        clique_scores=[],
                        incomplete=[],
                        complete_cliques=0,
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
                iteration_seconds = float(its[-1][1]) / 1000
                seconds = max(
                    iteration_seconds * cfg["profile_multiplier"],
                    cfg["profile_min_seconds"],
                )
                # Leave a bounded tail for in-flight groups and each node's final drain.
                deadline = (
                    time.monotonic()
                    + seconds
                    + max(iteration_seconds * cfg["profile_multiplier"], 2)
                    + 3
                )
                active = dict(deadline=deadline, seconds=seconds, trigger=triggers[0])
                s.set("control", dict(mode=1, epoch=epoch, seconds=seconds))
                emit("profile_start", epoch=epoch, seconds=seconds, trigger=triggers[0])
        time.sleep(cfg["poll_seconds"])
    emit("controller_stopped", epochs=epoch)


if __name__ == "__main__":
    main()
