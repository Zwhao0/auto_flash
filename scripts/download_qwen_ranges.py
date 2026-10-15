"""Parallel mirror transport; identity is pinned by official LFS SHA256."""

import concurrent.futures as cf
import hashlib
import json
import os
import pathlib
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEST = ROOT / "models/Qwen1.5-MoE-A2.7B"
manifest = json.loads((DEST / "source_manifest.json").read_text())
BASE = "https://modelscope.cn/models/qwen/Qwen1.5-MoE-A2.7B/resolve/master/"
CHUNK = 32 * 1024 * 1024
parts = DEST / ".chunks"
parts.mkdir(exist_ok=True)


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def git_blob_digest(path):
    size = path.stat().st_size
    h = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as f:
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def fetch(task):
    name, start, end = task
    path = parts / f"{name}.{start}"
    if path.exists() and path.stat().st_size == end - start + 1:
        return 0
    transferred = 0
    for attempt in range(8):
        try:
            temporary = path.with_suffix(path.suffix + ".tmp")
            offset = temporary.stat().st_size if temporary.exists() else 0
            if offset == end - start + 1:
                temporary.replace(path)
                return transferred
            resume = start + offset
            req = urllib.request.Request(
                BASE + name + f"?flash_offset={resume}",
                headers={"Range": f"bytes={resume}-{end}"},
            )
            with urllib.request.urlopen(req, timeout=90) as response:
                if (
                    response.status != 206
                    or response.headers.get("Content-Range", "").split("/")[0]
                    != f"bytes {resume}-{end}"
                ):
                    raise RuntimeError("mirror did not honor exact byte range")
                with temporary.open("ab" if offset else "wb") as f:
                    while b := response.read(1024 * 1024):
                        f.write(b)
                        transferred += len(b)
            tmp = path.with_suffix(path.suffix + ".tmp")
            if tmp.stat().st_size != end - start + 1:
                raise RuntimeError("incomplete range")
            tmp.replace(path)
            return transferred
        except Exception as exc:
            print("RETRY", name, start, attempt + 1, type(exc).__name__, flush=True)
            time.sleep(min(10, attempt + 1))
    raise RuntimeError(f"failed {name} offset {start}")


def main():
    files = [f for f in manifest["files"] if f["rfilename"].endswith(".safetensors")]
    tasks = []
    pending = []
    for item in files:
        path = DEST / item["rfilename"]
        if (
            path.exists()
            and path.stat().st_size == item["size"]
            and digest(path) == item["lfs"]["sha256"]
        ):
            print("VERIFIED", path.name, flush=True)
            continue
        pending.append(item)
        tasks += [
            (item["rfilename"], s, min(item["size"] - 1, s + CHUNK - 1))
            for s in range(0, item["size"], CHUNK)
        ]
    began = time.monotonic()
    workers = int(os.getenv("QWEN_DOWNLOAD_WORKERS", "16"))
    print("RANGE_PLAN", len(tasks), "workers", workers, flush=True)
    transferred_bytes = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        for i, downloaded in enumerate(pool.map(fetch, tasks), 1):
            transferred_bytes += downloaded
            if i % 16 == 0:
                print(
                    "CHUNKS",
                    i,
                    "/",
                    len(tasks),
                    "MiB",
                    round(transferred_bytes / 1024 / 1024),
                    "seconds",
                    round(time.monotonic() - began),
                    "MiB/s",
                    round(
                        transferred_bytes
                        / max(1, time.monotonic() - began)
                        / 1024
                        / 1024,
                        2,
                    ),
                    flush=True,
                )
    for item in pending:
        path = DEST / item["rfilename"]
        with path.with_suffix(".assembling").open("wb") as out:
            for start in range(0, item["size"], CHUNK):
                with (parts / f"{path.name}.{start}").open("rb") as source:
                    while b := source.read(8 * 1024 * 1024):
                        out.write(b)
        assembled = path.with_suffix(".assembling")
        if digest(assembled) != item["lfs"]["sha256"]:
            raise RuntimeError("official SHA256 mismatch: " + path.name)
        assembled.replace(path)
        print("VERIFIED", path.name, item["lfs"]["sha256"], flush=True)
    # All metadata uses the fixed official revision and its Git blob identity.
    verified_metadata = []
    for item in manifest["files"]:
        name = item["rfilename"]
        if name.endswith(".safetensors"):
            continue
        path = DEST / name
        if (
            path.exists()
            and path.stat().st_size == item["size"]
            and git_blob_digest(path) == item["blobId"]
        ):
            print("VERIFIED_METADATA", name, item["blobId"], flush=True)
            verified_metadata.append(item)
            continue
        url = f"https://huggingface.co/{manifest['repository']}/resolve/{manifest['revision']}/{name}"
        with urllib.request.urlopen(url, timeout=90) as r:
            data = r.read()
        if (
            len(data) != item["size"]
            or hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            != item["blobId"]
        ):
            raise RuntimeError("official metadata identity mismatch: " + name)
        temp = path.with_suffix(path.suffix + ".assembling")
        temp.write_bytes(data)
        temp.replace(path)
        print("VERIFIED_METADATA", name, item["blobId"], flush=True)
        verified_metadata.append(item)
    verified = DEST / "verified.json"
    verified_tmp = DEST / "verified.json.tmp"
    verified_tmp.write_text(
        json.dumps(
            dict(
                repository=manifest["repository"],
                revision=manifest["revision"],
                transport="ModelScope byte ranges",
                files=[
                    dict(name=f["rfilename"], size=f["size"], sha256=f["lfs"]["sha256"])
                    for f in files
                ],
                metadata_files=[
                    dict(name=f["rfilename"], size=f["size"], git_blob_sha1=f["blobId"])
                    for f in verified_metadata
                ],
            ),
            indent=2,
        )
    )
    verified_tmp.replace(verified)
    print("CHECKPOINT VERIFIED", flush=True)


if __name__ == "__main__":
    main()
