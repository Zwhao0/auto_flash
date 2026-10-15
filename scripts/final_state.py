"""Snapshot task runtimes and host GPU state after all experiments finish."""

import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from scripts.remote import sudo_run, ROOT


def inspect(host):
    # Reset only the physical GPU manipulated by our clock fixture.
    command = "nvidia-smi --query-compute-apps=pid,process_name --format=csv\nnvidia-smi --query-gpu=index,name,clocks.current.sm,clocks.max.sm,utilization.gpu --format=csv\n"
    if host == "10.10.4.2":
        command = "nvidia-smi -i 0 -rgc\n" + command
    h, rc, output = sudo_run(host, command)
    return dict(host=h, exit=rc, output=output)


if __name__ == "__main__":
    with ThreadPoolExecutor() as pool:
        state = list(pool.map(inspect, ["10.10.4.1", "10.10.4.2"]))
    (ROOT / "results" / "final_server_state.json").write_text(
        json.dumps(state, indent=2)
    )
    print(json.dumps(state))
