"""Explicit paper defaults; env vars take precedence over the generated config."""

import os, json, pathlib

DEFAULTS = dict(
    redis_host="10.10.4.1",
    redis_port=16379,
    window=30,
    thresh_prob=0.8,
    degradation_thresh=0.2,
    recovery_ratio=0.75,
    right_guard=2,
    min_persistence=0,
    poll_seconds=0.1,
    profile_multiplier=2.2,
    profile_min_seconds=20.0,
    cv_threshold=0.1,
    rank_threshold=0.2,
    failstop_multiplier=2.0,
)


def load():
    path = pathlib.Path(
        os.getenv(
            "FLASH_CONFIG",
            str(pathlib.Path.home() / ".robust_training_flash" / "config.json"),
        )
    )
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x") as f:
                json.dump(DEFAULTS, f, indent=2)
        except FileExistsError:
            pass
    cfg = dict(DEFAULTS)
    cfg.update(json.loads(path.read_text()))
    return cfg
