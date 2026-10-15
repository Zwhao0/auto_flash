"""Release only this task's exported, completed per-run probe buffers."""

import argparse
import json
import pathlib
import subprocess
import time
from flash.store import Store

p = argparse.ArgumentParser()
p.add_argument("runs_json", help="JSON list of this task's completed run IDs")
a = p.parse_args()
runs = json.loads(pathlib.Path(a.runs_json).read_text())
active = subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
).strip()
if active:
    raise RuntimeError("GPU workers are still active; refuse to clean probe buffers")
expected = set()
for run in runs:
    store = Store(run)
    store.set("stop", True)
    for node in ["0", "1"]:
        for header in (store.get("stats:" + node) or {}).get("headers", []):
            expected.add(f"flash_{run}_{int(header[9])}_{int(header[10])}")
time.sleep(0.5)
removed = []
for path in pathlib.Path("/dev/shm").glob("flash_*"):
    if path.name in expected:
        path.unlink()
        removed.append(path.name)
print(json.dumps(dict(removed_buffers=len(removed), runs=len(runs))))
