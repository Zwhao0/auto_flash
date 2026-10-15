"""Verify archived post-fix runs used the committed FLASH/training core."""

import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = "db828f4"
paths = [*ROOT.glob("flash/*.py"), *ROOT.glob("csrc/*.cpp")]
paths += [
    ROOT / "scripts" / name
    for name in [
        "train_moe.py",
        "model_support.py",
        "gradient_sync.py",
        "qwen_support.py",
    ]
]
expected = {}
for path in paths:
    name = str(path.relative_to(ROOT))
    content = subprocess.check_output(["git", "show", REFERENCE + ":" + name], cwd=ROOT)
    expected[name] = hashlib.sha256(content).hexdigest()
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected[name]:
        raise RuntimeError("working core differs from committed reference: " + name)
checks = []
runtime = json.loads((ROOT / "results/v1_runtime_hashes.json").read_text())
if not runtime["nodes_identical"] or not runtime["local_sources_match"]:
    raise RuntimeError("two-node runtime identity check did not pass")
libraries = {
    name: digest
    for name, digest in runtime["nodes"][0]["sha256"].items()
    if name.startswith("build/")
}
if not libraries:
    raise RuntimeError("runtime snapshot did not include probe/injector libraries")
for path in sorted((ROOT / "results").glob("audit_*/run_metadata.json")):
    metadata = json.loads(path.read_text())
    observed = metadata.get("source_hashes", {})
    observed_libraries = metadata.get("libraries", {})
    if "scripts/gradient_sync.py" not in observed:
        continue
    mismatch = [
        name for name, digest in expected.items() if observed.get(name) != digest
    ]
    mismatch += [
        name
        for name, digest in libraries.items()
        if observed_libraries.get(name) != digest
    ]
    checks.append(
        dict(
            run=path.parent.name,
            passed=not mismatch,
            mismatches=mismatch,
            libraries={name: observed_libraries.get(name) for name in libraries},
        )
    )
result = dict(
    reference_commit=subprocess.check_output(
        ["git", "rev-parse", REFERENCE], cwd=ROOT, text=True
    ).strip(),
    scope="FLASH/training core and probe/injector libraries; orchestration and documentation hashes may vary",
    expected_source_hashes=expected,
    expected_runtime_libraries=libraries,
    runs=checks,
)
(ROOT / "results/v1_core_source_check.json").write_text(json.dumps(result, indent=2))
print(
    "CORE SOURCE CHECK",
    len(checks),
    "runs;",
    sum(not c["passed"] for c in checks),
    "mismatches",
)
if any(not c["passed"] for c in checks):
    raise SystemExit(1)
