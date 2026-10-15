import argparse, shlex, os
from concurrent.futures import ThreadPoolExecutor
from scripts.remote import sudo_run

NAME = {"paper": "flash-paper-20261015", "greyhound": "flash-greyhound-20261015"}.get(
    os.getenv("FLASH_RUNTIME"), "flash-repro-20261015"
)
BASE = "nerdctl --namespace moby exec " + NAME + " "


def execute(host, command, timeout=120):
    return sudo_run(host, BASE + "bash -lc " + shlex.quote(command), timeout)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command")
    p.add_argument("--host", default="both")
    p.add_argument("--timeout", type=int, default=120)
    a = p.parse_args()
    hs = ["10.10.4.1", "10.10.4.2"] if a.host == "both" else [a.host]
    with ThreadPoolExecutor() as pool:
        for h, rc, s in pool.map(lambda h: execute(h, a.command, a.timeout), hs):
            print(f"HOST {h} EXIT {rc}\n{s}")
