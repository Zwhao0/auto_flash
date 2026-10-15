#!/usr/bin/env bash
set -euo pipefail
mkdir -p build/contract
g++ -std=c++17 -shared -fPIC -I/usr/local/cuda/include tests/probe_mock.cpp -o build/contract/libmock.so
g++ -std=c++17 -shared -fPIC -pthread -I/usr/local/cuda/include csrc/probe.cpp -ldl -o build/contract/libprobe.so
g++ -std=c++17 -I/usr/local/cuda/include tests/probe_contract.cpp -Lbuild/contract -Wl,-rpath,/flash/build/contract -lprobe -lmock -o build/contract/probe_contract
build/contract/probe_contract
