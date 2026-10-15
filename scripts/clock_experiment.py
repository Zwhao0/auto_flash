"""Physical clock injection with unconditional restoration on exit."""

import argparse, time, json, pathlib
from scripts.launch import launch
from scripts.experiment import finish, redis_command
from scripts.remote import sudo_run, ROOT
from scripts.container import execute

p = argparse.ArgumentParser()
p.add_argument("run")
p.add_argument("--mhz", type=int, required=True)
p.add_argument("--tp", type=int, default=1)
p.add_argument("--pp", type=int, default=1)
p.add_argument("--ep", type=int, default=8)
p.add_argument("--force-profile", action="store_true")
a = p.parse_args()
path = ROOT / "results" / a.run
path.mkdir(parents=True, exist_ok=True)
events = []


def hw(cmd):
    h, rc, out = sudo_run("10.10.4.2", cmd)
    events.append(dict(time=time.time(), command=cmd, exit=rc, output=out))
    (path / "clock_injection.json").write_text(json.dumps(events, indent=2))
    return rc, out


try:
    hw(
        "nvidia-smi -i 0 --query-gpu=name,clocks.current.sm,clocks.max.sm,power.limit --format=csv"
    )
    launch(
        a.run,
        args=f"--steps 100 --min-seconds 34 --seq 128 --batch 2 --tp {a.tp} --pp {a.pp} --ep {a.ep}",
        force=a.force_profile,
    )
    # Hold forced profiling until the physical clock command has completed.
    redis_command(a.run, "s.set('force_after',1000000000)")
    start = time.monotonic()
    while time.monotonic() - start < 100:
        _, _, out = redis_command(
            a.run, "v=s.items('iter_time_0'); print(v[-1].split(':')[0] if v else -1)"
        )
        try:
            step = int(out.strip())
        except ValueError:
            step = -1
        if step >= 45:
            break
        if execute("10.10.4.1", f"test -f {"/flash/results/"+a.run+"/exit0"}")[1] == 0:
            raise RuntimeError(
                "training exited before clock injection; inspect torchrun log"
            )
        time.sleep(0.2)
    if step < 45:
        raise RuntimeError("training did not reach clock injection step")
    rc, out = hw(f"nvidia-smi -i 0 -lgc {a.mhz},{a.mhz}")
    if rc:
        raise RuntimeError("GPU clock lock failed")
    redis_command(
        a.run,
        f"s.event('injection',fault='gpu_clock',rank=8,mhz={a.mhz},observed_iteration={step})",
    )
    redis_command(a.run, "s.set('force_after',0)")
    time.sleep(1)
    hw(
        "nvidia-smi -i 0 --query-gpu=clocks.current.sm,clocks.current.memory,utilization.gpu --format=csv"
    )
    finish(a.run, "monitor")
finally:
    hw("nvidia-smi -i 0 -rgc")
    hw("nvidia-smi -i 0 --query-gpu=clocks.current.sm,clocks.max.sm --format=csv")
