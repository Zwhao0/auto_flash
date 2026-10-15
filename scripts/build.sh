#!/usr/bin/env bash
set -euo pipefail
mkdir -p build
g++ -std=c++17 -O3 -DNDEBUG -shared -fPIC -pthread -I/usr/local/cuda/include -I/usr/include csrc/probe.cpp -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64 -lcudart -ldl -o build/libncclprobe.so.tmp
mv build/libncclprobe.so.tmp build/libncclprobe.so
g++ -std=c++17 -O2 -shared -fPIC -I/usr/local/cuda/include csrc/fault.cpp -ldl -o build/libncclfault.so.tmp
mv build/libncclfault.so.tmp build/libncclfault.so
