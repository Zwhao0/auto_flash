"""Sync only this fresh reproduction; never reads old server projects."""

from pathlib import Path
import concurrent.futures, tarfile, sys
from scripts.remote import connect, run, ROOT

REMOTE = "/home/campus/zhangweihao/flash-repro-20261015"


def sync(host):
    c = connect(host)
    try:
        s = c.open_sftp()
        s.put(str(ROOT / "tmp/source.tar.gz"), REMOTE + "/source.tar.gz")
        s.close()
    finally:
        c.close()
    return run(host, f"cd {REMOTE} && tar -xzf source.tar.gz")


if __name__ == "__main__":
    paths = [
        "data",
        "flash",
        "csrc",
        "scripts",
        "tests",
        "docs",
        "README.md",
        "pyproject.toml",
        "requirements-paper.txt",
        "handoff.md",
        "patches",
    ]
    if "--assets" in sys.argv:
        paths += ["vendor/Greyhound", "vendor/MegatronCore", "wheelhouse"]
    with tarfile.open(ROOT / "tmp/source.tar.gz", "w:gz") as t:
        for f in paths:
            if (ROOT / f).exists():
                t.add(
                    ROOT / f,
                    arcname=f,
                    filter=lambda i: (
                        None if "/.git" in i.name or "__pycache__" in i.name else i
                    ),
                )
    with concurrent.futures.ThreadPoolExecutor() as p:
        for r in p.map(sync, ["10.10.4.1", "10.10.4.2"]):
            print(r)
