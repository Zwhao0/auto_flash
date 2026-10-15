from concurrent.futures import ThreadPoolExecutor
from scripts.remote import connect
from scripts.container import execute
from scripts.sync import REMOTE
from pathlib import Path

names = ["transformers", "tokenizers", "safetensors", "regex", "huggingface_hub"]
files = [
    p
    for p in Path("wheelhouse").glob("*.whl")
    if any(p.name.startswith(n + "-") for n in names)
]


def install(host):
    c = connect(host)
    try:
        s = c.open_sftp()
        for p in files:
            s.put(str(p), REMOTE + "/wheelhouse/" + p.name)
        s.close()
    finally:
        c.close()
    command = (
        "python -m pip install --no-index --no-deps --target /flash/deps "
        + " ".join("/flash/wheelhouse/" + p.name for p in files)
    )
    return execute(host, command, 180)


with ThreadPoolExecutor() as pool:
    for r in pool.map(install, ["10.10.4.1", "10.10.4.2"]):
        print(r)
