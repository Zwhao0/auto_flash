"""Small background process for the paper's rank-0 iteration reporting path."""

import json
import sys
from .store import Store

s = Store()
print("READY", flush=True)
count, last = 0, None
for line in sys.stdin:
    iteration, time_ms = json.loads(line)
    s.push("iter_time_0", f"{iteration}:{time_ms:.6f}")
    count += 1
    last = iteration
s.set(
    "iteration_reporter",
    dict(backend="background_process", received=count, last_iteration=last),
)
