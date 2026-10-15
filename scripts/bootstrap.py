"""Prepare the two already-offline servers using only task-owned paths/images.
Run on the Mac after fetching the pinned sources/wheels (see docs/runbook.md).
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from scripts.remote import sudo_run

p = argparse.ArgumentParser()
p.add_argument(
    "--create", action="store_true", help="Create only missing task containers"
)
a = p.parse_args()
TASK = "/home/campus/zhangweihao/flash-repro-20261015"
IMAGES = {
    "flash-repro-20261015": "tianyuanwu/greyhound:ae",
    "flash-paper-20261015": "registry.reinfra.cn/reinfra/campus/docker-images/pytorch-runtime:2.10.0-cu126-r1",
    "flash-greyhound-20261015": "tianyuanwu/greyhound:ae",
}


def prepare(host):
    commands = [f"mkdir -p {TASK}", f"chown test:test {TASK}"]
    for name, image in IMAGES.items():
        if a.create:
            ipc = "--shm-size 1g" if "greyhound" in name else "--ipc host"
            commands.append(
                f"if ! nerdctl --namespace moby inspect {name} >/dev/null 2>&1; then nerdctl --namespace moby run -d --name {name} --privileged --pull never --net host {ipc} --gpus all --ulimit memlock=-1:-1 -v {TASK}:/flash -w /flash {image} sleep infinity; fi"
            )
        else:
            commands.append(
                f'nerdctl --namespace moby inspect {name} --format "{{{{.State.Status}}}}"'
            )
    return sudo_run(host, "\n".join(commands))


with ThreadPoolExecutor() as pool:
    for result in pool.map(prepare, ["10.10.4.1", "10.10.4.2"]):
        print(result)
