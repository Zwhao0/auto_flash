import json, pathlib, hashlib, sys, time, os
from flash.store import Store

run, mode, task, args = sys.argv[1:5]
s = Store(run)
if (pathlib.Path("results") / run / "torchrun0.log").exists():
    raise RuntimeError("existing run directory; use a new run ID")
meta = dict(
    run=run,
    time=time.time(),
    mode=mode,
    task=task,
    args=args,
    config=s.config,
    source_hashes={
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for d in ["flash", "csrc", "scripts"]
        for p in pathlib.Path(d).glob("*")
        if p.is_file()
    },
    libraries={
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in pathlib.Path("build").glob("libnccl*.so")
    },
    runtime=os.getenv("FLASH_RUNTIME_LABEL"),
)
if not s.r.set(s.k("run_metadata"), json.dumps(meta), nx=True):
    raise RuntimeError("run ID already exists; use a new run ID")
p = pathlib.Path("results") / run
p.mkdir(parents=True, exist_ok=True)
(p / "run_metadata.json").write_text(json.dumps(meta, indent=2))
