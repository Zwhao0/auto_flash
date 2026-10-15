// Test-only fault injector. FLASH detector never reads these environment
// variables.
#include <atomic>
#include <cstdlib>
#include <dlfcn.h>
#include <nccl.h>
#include <time.h>
extern "C" ncclResult_t ncclAllReduce(const void *a, void *b, size_t n,
                                      ncclDataType_t t, ncclRedOp_t op,
                                      ncclComm_t comm, cudaStream_t stream) {
  static auto real = reinterpret_cast<decltype(&ncclAllReduce)>(
      dlsym(RTLD_NEXT, "ncclAllReduce"));
  static std::atomic<bool> done{false};
  if (getenv("FLASH_FAULT_TIMEOUT_ARM") && !done.exchange(true)) {
    timespec delay{110, 0};
    while (nanosleep(&delay, &delay) && delay.tv_sec > 0) {
    }
  }
  return real(a, b, n, t, op, comm, stream);
}
