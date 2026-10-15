"""Prevent accidental overlap between independent performance/fault runs."""

import subprocess

rows = subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
    text=True,
).strip()
if rows:
    raise RuntimeError(
        "GPU processes still active; wait for the current run to finish: " + rows
    )
