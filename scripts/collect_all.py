"""Refresh complete runs, avoiding repeated checkpoint and static log downloads."""

import pathlib, concurrent.futures, argparse, stat
from scripts.remote import connect, ROOT

p = argparse.ArgumentParser()
p.add_argument("--prefix", default="paper_")
a = p.parse_args()


def fetch(pair):
    node, host = pair
    c = connect(host)
    count = 0
    try:
        s = c.open_sftp()
        base = "/home/campus/zhangweihao/flash-repro-20261015/results"
        for entry in s.listdir_attr(base):
            run = entry.filename
            if not run.startswith(a.prefix) or not stat.S_ISDIR(entry.st_mode):
                continue
            dest = ROOT / "results" / run
            dest.mkdir(parents=True, exist_ok=True)
            for f in s.listdir_attr(base + "/" + run):
                if f.filename.endswith(".pt"):
                    continue
                out = dest / f.filename
                if (
                    out.exists()
                    and out.stat().st_size == f.st_size
                    and f.filename
                    not in [
                        "telemetry.json",
                        "controller.jsonl",
                        "global.log",
                        "local0.log",
                        "local1.log",
                    ]
                ):
                    continue
                s.get(base + "/" + run + "/" + f.filename, str(out))
                count += 1
    finally:
        c.close()
    return host, count


with concurrent.futures.ThreadPoolExecutor() as pool:
    for r in pool.map(fetch, enumerate(["10.10.4.1", "10.10.4.2"])):
        print(r)
