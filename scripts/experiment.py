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
        # Wait for the actual requested profile budget and local drain, then
        # require several unchanged idle samples before stopping controllers.
        # This avoids truncating profiles longer than the old fixed 25s wait.
        started = time.monotonic()
        active_deadline = None
        overall_deadline = started + 300
        previous = None
        stable = 0
        while time.monotonic() < overall_deadline:
            _, _, out = redis_command(
                run,
                "import json; c=s.get('control',{}); "
                "e=s.items('events'); f=s.items('failstop_ranks'); "
                "v=s.items('iter_time_0'); "
                "print(json.dumps({'control':c,'profiles':sum(x.get('kind')=='profile_start' for x in e),"
                "'diagnoses':sum(x.get('kind')=='diagnosis' for x in e),"
                "'failstops':len(f),'latest':v[-1] if v else None}))",
            )
            try:
                state = json.loads(out.strip().splitlines()[-1])
            except (IndexError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"could not read controller state for {run}: {out}"
                ) from exc
            control = state.get("control") or {}
            if control.get("mode") == 1:
                stable = 0
                if active_deadline is None:
                    seconds = max(float(control.get("seconds", 20)), 1.0)
                    active_deadline = time.monotonic() + 2 * seconds + 10
                if time.monotonic() > active_deadline:
                    raise TimeoutError(
                        f"profile for {run} stayed active beyond its budget; leaving evidence intact"
                    )
            else:
                signature = (
                    control.get("epoch"),
                    state.get("profiles"),
                    state.get("diagnoses"),
                    state.get("failstops"),
                    state.get("latest"),
                )
                stable = stable + 1 if signature == previous else 0
                previous = signature
                if stable >= 3:
                    break
                active_deadline = None
            time.sleep(0.5)
        else:
            raise TimeoutError(
                f"controller state for {run} did not settle; leaving evidence intact"
            )
        redis_command(run, "s.set('stop',True)")
        time.sleep(0.5)
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
