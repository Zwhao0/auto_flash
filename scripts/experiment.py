"""Bounded, sequential 16-GPU experiments; preserves all failures and raw output."""

import time, json, pathlib, argparse, os
from scripts.launch import launch, collect
from scripts.container import execute
from scripts.remote import ROOT


def redis_command(run, code):
    import shlex

    env = "/flash:/flash/deps" if os.getenv("FLASH_RUNTIME") == "paper" else "/flash"
    result = execute(
        "10.10.4.1",
        "FLASH_RUN_ID="
        + shlex.quote(run)
        + " PYTHONPATH="
        + env
        + " python -c "
        + shlex.quote("from flash.store import Store; s=Store(); " + code),
    )
    if result[1]:
        raise RuntimeError(result[2])
    return result


def wait_run(run, timeout=300):
    start = time.monotonic()
    last = 0
    while time.monotonic() - start < timeout:
        exits = []
        for node, host in enumerate(["10.10.4.1", "10.10.4.2"]):
            _, rc, out = execute(
                host, f"cat /flash/results/{run}/exit{node} 2>/dev/null"
            )
            exits.append(out.strip() if rc == 0 else None)
        if all(e is not None for e in exits):
            return exits
        if time.monotonic() - start > last + 25:
            last = time.monotonic() - start
            print(
                json.dumps(dict(run=run, elapsed=round(last), exits=exits)), flush=True
            )
        time.sleep(1)
    raise TimeoutError(f"run {run} exceeded {timeout}s")


def finish(run, mode, timeout=300):
    exits = wait_run(run, timeout)
    if mode != "baseline":
        # Allow final profile report to complete if still active; never pretend partial samples are complete.
        for _ in range(25):
            _, _, out = redis_command(run, "print(s.get('control'))")
            if "'mode': 1" not in out:
                break
            time.sleep(1)
        redis_command(run, "s.set('stop',True)")
        time.sleep(0.3)
    env = "/flash:/flash/deps" if os.getenv("FLASH_RUNTIME") == "paper" else "/flash"
    exported = execute(
        "10.10.4.1", f"PYTHONPATH={env} python scripts/export_run.py {run}"
    )
    if exported[1]:
        raise RuntimeError(exported[2])
    path = collect(run)
    (path / "execution.json").write_text(
        json.dumps(dict(run=run, exits=exits), indent=2)
    )
    print(json.dumps(dict(run=run, exits=exits, path=str(path))), flush=True)
    return exits


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--mode", default="monitor")
    p.add_argument("--task", default="train")
    p.add_argument("--args", default="")
    p.add_argument("--force-profile", action="store_true")
    p.add_argument("--timeout", type=int, default=300)
    a = p.parse_args()
    launch(a.run, a.mode, a.task, a.args, force=a.force_profile)
    finish(a.run, a.mode, a.timeout)
