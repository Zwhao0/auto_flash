"""The two framework callbacks required by thesis §3.3.3."""

import os
from .store import Store

_store = None


def store():
    global _store
    if _store is None:
        _store = Store()
    return _store


def report_iteration(iteration, time_ms):
    if (
        os.getenv("FAILSLOW_AWARE", "FALSE").upper() == "TRUE"
        and int(os.getenv("RANK", "0")) == 0
    ):
        store().push("iter_time_0", f"{iteration}:{time_ms:.6f}")


def report_parallel_states(groups):
    if os.getenv("FAILSLOW_AWARE", "FALSE").upper() == "TRUE":
        store().set("parallel_states_" + os.getenv("RANK", "0"), groups)
