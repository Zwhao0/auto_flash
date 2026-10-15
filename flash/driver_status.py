"""Record NVML-backed driver state without feeding it into FLASH's decisions."""

import csv
import io
import subprocess


def snapshot():
    fields = "index,uuid,name,driver_version,pstate,clocks.sm,utilization.gpu,memory.used,memory.total,temperature.gpu"
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode:
            return dict(backend="nvidia-smi/NVML", error=result.stderr.strip())
        rows = [
            dict(zip(fields.split(","), [v.strip() for v in row]))
            for row in csv.reader(io.StringIO(result.stdout))
        ]
        return dict(backend="nvidia-smi/NVML", devices=rows)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(backend="nvidia-smi/NVML", error=str(exc))
