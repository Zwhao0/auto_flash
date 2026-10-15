"""Fresh official source with only hardware/path build adaptations."""

from pathlib import Path
import subprocess, json

root = Path(__file__).resolve().parents[1]
vendor = root / "vendor/Greyhound"
d = vendor / "detector"
for name, old, new in [
    ("config.hpp", "#define MAX_DEVS 4", "#define MAX_DEVS 8"),
    ("global_status.cpp", "new NcclTopoConnection(4)", "new NcclTopoConnection(16)"),
    (
        "control_plane/local_controller.py",
        "/workspace/Greyhound/detector/control_plane/config.json",
        "/flash/vendor/Greyhound/detector/control_plane/config.json",
    ),
]:
    p = d / name
    s = p.read_text()
    p.write_text(s.replace(old, new))
p = d / "control_plane/config.json"
j = json.loads(p.read_text())
j["MAX_DEVS"] = 8
p.write_text(json.dumps(j))
(root / "patches/greyhound-hardware.patch").write_text(
    subprocess.check_output(["git", "-C", str(vendor), "diff"], text=True)
)
