"""The two framework callbacks required by thesis §3.3.3."""

import atexit
import json
import os
import select
import subprocess
import sys
from .store import Store

_store = None
_writer = None


def store():
    global _store
    if _store is None:
        _store = Store()
    return _store


def enabled():
    return os.getenv("FAILSLOW_AWARE", "FALSE").upper() == "TRUE"


def start_writer():
    global _writer
    if _writer is not None:
        return
    env = dict(os.environ)
    env.pop("LD_PRELOAD", None)
    _writer = subprocess.Popen(
        [sys.executable, "-m", "flash.iteration_writer"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        env=env,
    )
    if (
        not select.select([_writer.stdout], [], [], 10)[0]
        or _writer.stdout.readline().strip() != b"READY"
    ):
        _writer.terminate()
        raise RuntimeError("FLASH iteration writer did not initialize")
    os.set_blocking(_writer.stdin.fileno(), False)


def close_iteration_reporter():
    global _writer
    if _writer is None:
        return
    writer, _writer = _writer, None
    writer.stdin.close()
    if writer.wait(timeout=10):
        raise RuntimeError("FLASH iteration reporting failed")
    writer.stdout.close()


def report_iteration(iteration, time_ms):
    if enabled() and int(os.getenv("RANK", "0")) == 0:
        if os.getenv("FLASH_ITERATION_ASYNC", "TRUE").upper() != "TRUE":
            store().push("iter_time_0", f"{iteration}:{time_ms:.6f}")
            return
        start_writer()
        data = (json.dumps([iteration, time_ms]) + "\n").encode()
        if _writer.poll() is not None:
            raise RuntimeError("FLASH iteration writer exited")
        # Each short line fits PIPE_BUF; backpressure is an explicit failure,
        # never silent loss or synchronous Redis work in the training process.
        try:
            if os.write(_writer.stdin.fileno(), data) != len(data):
                raise RuntimeError("partial FLASH iteration report")
        except BlockingIOError as exc:
            raise RuntimeError("FLASH iteration reporting pipe is full") from exc


def report_parallel_states(groups):
    if enabled():
        store().set("parallel_states_" + os.getenv("RANK", "0"), groups)
        if (
            int(os.getenv("RANK", "0")) == 0
            and os.getenv("FLASH_ITERATION_ASYNC", "TRUE").upper() == "TRUE"
        ):
            start_writer()


atexit.register(close_iteration_reporter)
