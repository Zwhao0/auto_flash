"""Read-only source/library identity snapshot from the two paper containers."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex

os.environ["FLASH_RUNTIME"] = "paper"
from scripts.container import execute

ROOT = Path(__file__).resolve().parents[1]
REMOTE = """
import hashlib,json,pathlib,torch
root=pathlib.Path('/flash')
paths=list(root.glob('flash/*.py'))+list(root.glob('csrc/*.cpp'))
paths += [root/'scripts'/name for name in ['train_moe.py','model_support.py','gradient_sync.py','qwen_support.py']]
paths += list((root/'build').glob('*.so'))
print(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,nccl=torch.cuda.nccl.version(),sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})))
"""


def snapshot(host):
    _, rc, output = execute(host, "python -c " + shlex.quote(REMOTE), timeout=60)
    if rc:
        raise RuntimeError(f"snapshot failed on {host}: {output}")
    return dict(host=host, **json.loads(output))


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=2) as pool:
        nodes = list(pool.map(snapshot, ["10.10.4.1", "10.10.4.2"]))
    same = nodes[0]["sha256"] == nodes[1]["sha256"]
    current = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in nodes[0]["sha256"]
        if not name.startswith("build/")
    }
    matches = all(
        all(node["sha256"][name] == digest for name, digest in current.items())
        for node in nodes
    )
    result = dict(
        captured_utc=datetime.now(timezone.utc).isoformat(),
        scope="Read-only runtime and source identity; no GPU workload",
        nodes=nodes,
        nodes_identical=same,
        local_sources_match=matches,
    )
    (ROOT / "results/v1_runtime_hashes.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(dict(nodes_identical=same, local_sources_match=matches)))
    if not same or not matches:
        raise SystemExit(1)
