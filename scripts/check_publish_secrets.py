"""Reject the local server credential in publishable files or Git blob history."""

import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
credential_file = ROOT / "服务器.md"
secret = re.search(r"密码\s+(\S+)", credential_file.read_text()).group(1).encode()
files = subprocess.check_output(
    ["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=ROOT
).split(b"\0")
findings = []
checked = 0
for raw in set(files):
    if not raw:
        continue
    path = ROOT / raw.decode()
    if path.is_file():
        checked += 1
        if secret in path.read_bytes():
            findings.append(str(path.relative_to(ROOT)))
objects = subprocess.check_output(
    ["git", "rev-list", "--objects", "--all"], cwd=ROOT
).splitlines()
blobs = 0
object_ids = [line.split(b" ", 1)[0] for line in objects]
packed = subprocess.check_output(
    ["git", "cat-file", "--batch"],
    cwd=ROOT,
    input=b"\n".join(object_ids) + b"\n",
)
offset = 0
for _ in object_ids:
    line_end = packed.index(b"\n", offset)
    oid, kind, size = packed[offset:line_end].split()
    size = int(size)
    offset = line_end + 1
    data = packed[offset : offset + size]
    offset += size + 1
    if kind == b"blob":
        blobs += 1
        if secret in data:
            findings.append("historical blob " + oid.decode())
if findings:
    print("REJECT: credential matches in paths/objects:", findings)
    raise SystemExit(1)
print(
    f"PASS credential absent from {checked} publishable files and {blobs} historical Git blobs"
)
