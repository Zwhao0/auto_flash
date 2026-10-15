"""Expose host RDMA userspace libraries inside the isolated paper runtime."""

from scripts.remote import sudo_run
from concurrent.futures import ThreadPoolExecutor

CMD = """
set -e
root=/home/campus/zhangweihao/flash-repro-20261015/rdma
mkdir -p "$root/providers"
for lib in libibverbs.so.1 libmlx5.so.1 libnl-3.so.200 libnl-route-3.so.200; do cp -L "/usr/lib/x86_64-linux-gnu/$lib" "$root/$lib"; done
cp -L /usr/lib/x86_64-linux-gnu/libibverbs/libmlx5-rdmav57.so "$root/providers/"
nerdctl --namespace moby exec flash-paper-20261015 bash -lc 'mkdir -p /usr/lib/x86_64-linux-gnu/libibverbs /etc/libibverbs.d; cp /flash/rdma/lib*.so.* /usr/lib/x86_64-linux-gnu/; cp /flash/rdma/providers/* /usr/lib/x86_64-linux-gnu/libibverbs/; echo "driver mlx5" > /etc/libibverbs.d/mlx5.driver; ldconfig; ldconfig -p | grep ibverbs'
"""
with ThreadPoolExecutor() as pool:
    for r in pool.map(lambda h: sudo_run(h, CMD), ["10.10.4.1", "10.10.4.2"]):
        print(r)
