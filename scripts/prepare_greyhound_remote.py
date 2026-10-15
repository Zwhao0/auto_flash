from scripts.container import execute
from scripts.remote import connect, ROOT
from concurrent.futures import ThreadPoolExecutor

# Only the four known adaptation files are sent, with the exact patch archived locally.
files = [
    "detector/config.hpp",
    "detector/global_status.cpp",
    "detector/control_plane/local_controller.py",
    "detector/control_plane/config.json",
]


def prepare(host):
    c = connect(host)
    try:
        s = c.open_sftp()
        for f in files:
            s.put(
                str(ROOT / "vendor/Greyhound" / f),
                "/home/campus/zhangweihao/flash-repro-20261015/vendor/Greyhound/" + f,
            )
    finally:
        c.close()
    return execute(
        host,
        "set -e\nmkdir -p /flash/build/greyhound\n"
        "cmake -S /flash/vendor/Greyhound/detector -B /flash/build/greyhound > /flash/build/greyhound/configure.log 2>&1\n"
        "cd /flash/vendor/Greyhound/detector\npython setup.py bdist_wheel > /flash/build/greyhound/wheel.log 2>&1\n"
        "python -m pip install --no-deps --force-reinstall dist/control_plane-1.0-py3-none-any.whl > /flash/build/greyhound/install.log 2>&1\n"
        "cmake --build /flash/build/greyhound --target ncclprobe -j8 > /flash/build/greyhound/rebuild.log 2>&1\ntail -5 /flash/build/greyhound/rebuild.log",
        180,
    )


with ThreadPoolExecutor() as pool:
    for r in pool.map(prepare, ["10.10.4.1", "10.10.4.2"]):
        print(r)
