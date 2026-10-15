"""Download the pinned official, non-gated Qwen checkpoint on the online Mac."""

import concurrent.futures
import hashlib
import json
import pathlib
import time
import urllib.request

REPO = "Qwen/Qwen1.5-MoE-A2.7B"
REV = "1a758c50ecb6350748b9ce0a99d2352fd9fc11c9"
ROOT = pathlib.Path(__file__).resolve().parents[1]
DEST = ROOT / "models" / "Qwen1.5-MoE-A2.7B"
DEST.mkdir(parents=True, exist_ok=True)
api = json.load(
    urllib.request.urlopen(
        f"https://huggingface.co/api/models/{REPO}/revision/{REV}?blobs=true",
        timeout=30,
    )
)
files = [
    x
    for x in api["siblings"]
    if x["rfilename"].endswith((".safetensors", ".json", ".txt"))
    or x["rfilename"] in ["merges.txt", "README.md", "LICENSE"]
]
manifest = dict(repository=REPO, revision=REV, files=files)
(DEST / "source_manifest.json").write_text(json.dumps(manifest, indent=2))


def download(item):
    name = item["rfilename"]
    path = DEST / name
    expected = item.get("size")
    checksum = (item.get("lfs") or {}).get("sha256")
    if path.exists() and path.stat().st_size == expected:
        if (
            not checksum
            or hashlib.file_digest(path.open("rb"), "sha256").hexdigest() == checksum
        ):
            print("CACHED", name, flush=True)
            return
    part = path.with_name(path.name + ".part")
    began = time.monotonic()
    for attempt in range(5):
        size = part.stat().st_size if part.exists() else 0
        request = urllib.request.Request(
            f"https://huggingface.co/{REPO}/resolve/{REV}/{name}",
            headers={"Range": f"bytes={size}-"} if size else {},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if size and response.status != 206:
                    size = 0
                with part.open("ab" if size else "wb") as out:
                    while chunk := response.read(1024 * 1024):
                        out.write(chunk)
            if expected and part.stat().st_size != expected:
                raise RuntimeError("checkpoint shard size mismatch")
            if checksum:
                with part.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != checksum:
                        raise RuntimeError("checkpoint shard SHA256 mismatch")
            part.replace(path)
            print(
                "DONE",
                name,
                path.stat().st_size,
                "seconds",
                round(time.monotonic() - began, 1),
                flush=True,
            )
            return
        except Exception as exc:
            print("RETRY", name, attempt + 1, type(exc).__name__, flush=True)
            if attempt == 4:
                raise
            time.sleep(2)


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(download, files))
print("OFFICIAL CHECKPOINT COMPLETE", flush=True)
