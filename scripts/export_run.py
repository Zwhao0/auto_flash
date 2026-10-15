import argparse, json, pathlib
from flash.store import Store

p = argparse.ArgumentParser()
p.add_argument("run")
a = p.parse_args()
s = Store(a.run)
keys = [
    "run_metadata",
    "control",
    "world_size",
    "training_done",
    "stats:0",
    "stats:1",
    "ack:0",
    "ack:1",
] + [f"parallel_states_{r}" for r in range(int(s.get("world_size", 16)))]
import hashlib, sys, importlib.metadata

result = {k: s.get(k) for k in keys}
result["source_hashes"] = {
    str(p): hashlib.sha256(p.read_bytes()).hexdigest()
    for d in ["flash", "csrc", "scripts"]
    for p in pathlib.Path(d).glob("*")
    if p.is_file()
}
result["runtime"] = {"python": sys.version}
try:
    import torch

    result["runtime"].update(
        torch=torch.__version__, cuda=torch.version.cuda, nccl=torch.cuda.nccl.version()
    )
except ImportError:
    pass
result.update({k: s.items(k) for k in ["iter_time_0", "events", "failstop_ranks"]})
p = pathlib.Path("results") / a.run / "telemetry.json"
p.write_text(json.dumps(result, indent=2))
