#!/usr/bin/env bash
set -euo pipefail
rank="$RANK"
export LD_PRELOAD="${FLASH_NCCL_LIB:-/usr/lib/x86_64-linux-gnu/libnccl.so.2}"
if [[ "$FLASH_MODE" != baseline ]]; then export LD_PRELOAD="/flash/build/libncclprobe.so:${FLASH_NCCL_LIB:-/usr/lib/x86_64-linux-gnu/libnccl.so.2}"; export FAILSLOW_AWARE=TRUE; fi
if [[ "${FLASH_GREYHOUND:-FALSE}" == TRUE ]]; then export LD_PRELOAD="/flash/build/greyhound/libncclprobe.so:${FLASH_NCCL_LIB:-/usr/lib/x86_64-linux-gnu/libnccl.so.2}"; fi
if [[ "${FLASH_FAULT_INJECT:-FALSE}" == TRUE ]]; then export LD_PRELOAD="/flash/build/libncclfault.so:$LD_PRELOAD"; fi
if [[ "$FLASH_TASK" == train ]]; then
 exec python scripts/train_moe.py "$@" --output "/flash/results/$FLASH_RUN_ID/train_rank${rank}.jsonl"
elif [[ "$FLASH_TASK" == precheck ]]; then
 exec python scripts/precheck.py "$@"
elif [[ "$FLASH_TASK" == deepspeed ]]; then
 exec python scripts/deepspeed_smoke.py "$@" --output "/flash/results/$FLASH_RUN_ID/deepspeed_rank${rank}.jsonl"
else
 exec python scripts/allreduce_bench.py "$@" --output "/flash/results/$FLASH_RUN_ID/allreduce_rank${rank}.json"
fi
