import importlib.util, sys, os, pathlib, torch

print(sys.version)
print(torch.__version__, torch.version.cuda, torch.cuda.nccl.version())
print(
    {
        m: bool(importlib.util.find_spec(m))
        for m in [
            "numpy",
            "redis",
            "Rbeast",
            "einops",
            "yaml",
            "packaging",
            "transformer_engine",
        ]
    }
)
print(torch.__path__)
for p in pathlib.Path("/").glob("opt/*"):
    print(p)
