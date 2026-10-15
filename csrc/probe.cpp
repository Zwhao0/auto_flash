#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <mutex>
#include <nccl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace {
constexpr uint64_t MAGIC = 0x464c415348303031ULL, CAP = 65536, NCOMM = 512;
struct Comm {
  uint64_t ptr, hash, rank, local, size, active, reserved[2];
};
struct Record {
  uint64_t seq, ns, comm, hash, op, bytes, group, duration, epoch, flags;
};
struct Shared {
  std::atomic<uint64_t> h[16];
  Comm comms[NCOMM];
  Record ring[CAP];
};
static Shared *shm = nullptr;
static std::once_flag once;
static std::mutex lock;
static std::unordered_map<uint64_t, Comm> comms;
static std::atomic<uint64_t> groups{0};
uint64_t now() {
  timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return uint64_t(t.tv_sec) * 1000000000ULL + t.tv_nsec;
}
uint64_t hash_bytes(const void *p, size_t n) {
  uint64_t h = 14695981039346656037ULL;
  auto b = (const unsigned char *)p;
  for (size_t i = 0; i < n; ++i) {
    h ^= b[i];
    h *= 1099511628211ULL;
  }
  return h;
}
template <class T> T sym(const char *n) {
  auto p = dlsym(RTLD_NEXT, n);
  if (!p) {
    fprintf(stderr, "FLASH unresolved %s: %s\n", n, dlerror());
    abort();
  }
  return reinterpret_cast<T>(p);
}
void init() {
  std::call_once(once, [] {
    const char *run = getenv("FLASH_RUN_ID"), *r = getenv("RANK");
    if (!run || !r)
      return;
    char name[200];
    snprintf(name, sizeof(name), "/flash_%s_%s_%d", run, r, getpid());
    int fd = shm_open(name, O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd < 0) {
      perror("FLASH shm_open");
      return;
    }
    if (ftruncate(fd, sizeof(Shared))) {
      close(fd);
      return;
    }
    void *p = mmap(nullptr, sizeof(Shared), PROT_READ | PROT_WRITE, MAP_SHARED,
                   fd, 0);
    close(fd);
    if (p == MAP_FAILED)
      return;
    shm = (Shared *)p;
    memset(shm, 0, sizeof(Shared));
    shm->h[9] = strtoull(r, nullptr, 10);
    shm->h[10] = getpid();
    shm->h[14] = sizeof(Shared);
    shm->h[4] = now();
    shm->h[12] = 1;
    shm->h[0].store(MAGIC, std::memory_order_release);
  });
}
void err() {
  if (shm)
    shm->h[8].fetch_add(1, std::memory_order_relaxed);
}
void addcomm(ncclComm_t c, uint64_t hash, int rank, int size) {
  if (!shm || !c)
    return;
  std::lock_guard<std::mutex> g(lock);
  Comm x{uint64_t(c), hash,  shm->h[9].load(), uint64_t(rank), uint64_t(size),
         1,           {0, 0}};
  comms[x.ptr] = x;
  auto n = shm->h[11].load();
  if (n < NCOMM) {
    shm->comms[n] = x;
    shm->h[11].store(n + 1, std::memory_order_release);
  } else
    err();
}
size_t typesize(ncclDataType_t t) {
  switch (t) {
  case ncclInt8:
  case ncclUint8:
    return 1;
  case ncclFloat16:
  case ncclBfloat16:
    return 2;
  case ncclInt32:
  case ncclUint32:
  case ncclFloat32:
    return 4;
  default:
    return 8;
  }
}
void emit(Record r) {
  if (!shm)
    return;
  std::lock_guard<std::mutex> g(lock);
  uint64_t n = shm->h[3].load(std::memory_order_relaxed);
  r.seq = n + 1;
  auto &slot = shm->ring[n % CAP];
  __atomic_store_n(&slot.seq, 0, __ATOMIC_RELEASE);
  memcpy(reinterpret_cast<char *>(&slot) + 8, reinterpret_cast<char *>(&r) + 8,
         sizeof(r) - 8);
  __atomic_store_n(&slot.seq, r.seq, __ATOMIC_RELEASE);
  shm->h[3].store(n + 1, std::memory_order_release);
  if (r.duration)
    shm->h[7].fetch_add(1, std::memory_order_relaxed);
}
struct Timer {
  cudaEvent_t a = nullptr, b = nullptr;
  cudaStream_t stream;
  bool active = false;
  Timer(cudaStream_t s) : stream(s) {
    if (cudaEventCreate(&a) != cudaSuccess ||
        cudaEventCreate(&b) != cudaSuccess) {
      err();
      return;
    }
    active = true;
  }
  ~Timer() {
    if (a)
      cudaEventDestroy(a);
    if (b)
      cudaEventDestroy(b);
  }
};
thread_local std::unordered_map<uint64_t, Timer *> pool;
thread_local int depth = 0;
thread_local uint64_t gid = 0;
thread_local std::vector<Record> pending;
thread_local std::unordered_map<uint64_t, Timer *> current;
Timer *start(cudaStream_t s) {
  auto k = uint64_t(s);
  if (current.count(k))
    return current[k];
  cudaStreamCaptureStatus st;
  if (cudaStreamIsCapturing(s, &st) != cudaSuccess ||
      st != cudaStreamCaptureStatusNone) {
    if (shm)
      shm->h[13]++;
    return nullptr;
  }
  auto it = pool.find(k);
  Timer *t;
  if (it == pool.end()) {
    t = new Timer(s);
    pool[k] = t;
  } else
    t = it->second;
  if (!t->active || cudaEventRecord(t->a, s) != cudaSuccess) {
    err();
    return nullptr;
  }
  current[k] = t;
  return t;
}
void finish() {
  for (auto &v : current) {
    Timer *t = v.second;
    float ms = 0;
    bool ok = cudaEventRecord(t->b, t->stream) == cudaSuccess &&
              cudaEventSynchronize(t->b) == cudaSuccess &&
              cudaEventElapsedTime(&ms, t->a, t->b) == cudaSuccess;
    if (!ok)
      err();
    for (auto &r : pending)
      if (!r.seq && r.flags == v.first) {
        r.duration = ok ? uint64_t(ms * 1e6) : 0;
        r.flags = ok ? 1 : 2;
        emit(r);
        r.seq = 1;
      }
  }
  for (auto &r : pending)
    if (!r.seq) {
      r.flags = 2;
      emit(r);
    }
  pending.clear();
  current.clear();
}
template <class F>
ncclResult_t call(F f, ncclComm_t c, size_t n, ncclDataType_t dt,
                  cudaStream_t stream, uint64_t op) {
  init();
  if (!shm)
    return f();
  uint64_t ns = now();
  shm->h[4].store(ns, std::memory_order_relaxed);
  shm->h[5].fetch_add(1, std::memory_order_relaxed);
  uint64_t hash = 0;
  {
    std::lock_guard<std::mutex> g(lock);
    auto it = comms.find(uint64_t(c));
    if (it != comms.end())
      hash = it->second.hash;
  }
  bool prof = shm->h[1].load(std::memory_order_acquire) == 1;
  Record r{0,
           ns,
           uint64_t(c),
           hash,
           op,
           n * typesize(dt),
           depth ? gid : ++groups,
           0,
           shm->h[2].load(),
           uint64_t(stream)};
  if (prof)
    start(stream);
  auto ret = f();
  if (ret != ncclSuccess)
    err();
  if (prof) {
    pending.push_back(r);
    if (!depth)
      finish();
  } else {
    r.flags = 0;
    emit(r);
  }
  return ret;
}
} // namespace
extern "C" {
ncclResult_t ncclCommInitRank(ncclComm_t *c, int n, ncclUniqueId id, int r) {
  init();
  static auto f = sym<decltype(&ncclCommInitRank)>("ncclCommInitRank");
  auto ret = f(c, n, id, r);
  if (ret == ncclSuccess)
    addcomm(*c, hash_bytes(id.internal, NCCL_UNIQUE_ID_BYTES), r, n);
  return ret;
}
ncclResult_t ncclCommInitRankConfig(ncclComm_t *c, int n, ncclUniqueId id,
                                    int r, ncclConfig_t *cfg) {
  init();
  static auto f =
      sym<decltype(&ncclCommInitRankConfig)>("ncclCommInitRankConfig");
  auto ret = f(c, n, id, r, cfg);
  if (ret == ncclSuccess)
    addcomm(*c, hash_bytes(id.internal, NCCL_UNIQUE_ID_BYTES), r, n);
  return ret;
}
ncclResult_t ncclGroupStart() {
  init();
  static auto f = sym<decltype(&ncclGroupStart)>("ncclGroupStart");
  auto ret = f();
  if (ret == ncclSuccess) {
    if (depth++ == 0)
      gid = ++groups;
  }
  return ret;
}
ncclResult_t ncclGroupEnd() {
  static auto f = sym<decltype(&ncclGroupEnd)>("ncclGroupEnd");
  auto ret = f();
  if (depth > 0 && --depth == 0)
    finish();
  return ret;
}
ncclResult_t ncclAllReduce(const void *a, void *b, size_t n, ncclDataType_t t,
                           ncclRedOp_t o, ncclComm_t c, cudaStream_t s) {
  static auto f = sym<decltype(&ncclAllReduce)>("ncclAllReduce");
  return call([&] { return f(a, b, n, t, o, c, s); }, c, n, t, s, 1);
}
ncclResult_t ncclBroadcast(const void *a, void *b, size_t n, ncclDataType_t t,
                           int root, ncclComm_t c, cudaStream_t s) {
  static auto f = sym<decltype(&ncclBroadcast)>("ncclBroadcast");
  return call([&] { return f(a, b, n, t, root, c, s); }, c, n, t, s, 2);
}
ncclResult_t ncclReduce(const void *a, void *b, size_t n, ncclDataType_t t,
                        ncclRedOp_t o, int root, ncclComm_t c, cudaStream_t s) {
  static auto f = sym<decltype(&ncclReduce)>("ncclReduce");
  return call([&] { return f(a, b, n, t, o, root, c, s); }, c, n, t, s, 3);
}
ncclResult_t ncclAllGather(const void *a, void *b, size_t n, ncclDataType_t t,
                           ncclComm_t c, cudaStream_t s) {
  static auto f = sym<decltype(&ncclAllGather)>("ncclAllGather");
  return call([&] { return f(a, b, n, t, c, s); }, c, n, t, s, 4);
}
ncclResult_t ncclReduceScatter(const void *a, void *b, size_t n,
                               ncclDataType_t t, ncclRedOp_t o, ncclComm_t c,
                               cudaStream_t s) {
  static auto f = sym<decltype(&ncclReduceScatter)>("ncclReduceScatter");
  return call([&] { return f(a, b, n, t, o, c, s); }, c, n, t, s, 5);
}
ncclResult_t ncclSend(const void *a, size_t n, ncclDataType_t t, int peer,
                      ncclComm_t c, cudaStream_t s) {
  static auto f = sym<decltype(&ncclSend)>("ncclSend");
  return call([&] { return f(a, n, t, peer, c, s); }, c, n, t, s, 6);
}
ncclResult_t ncclRecv(void *a, size_t n, ncclDataType_t t, int peer,
                      ncclComm_t c, cudaStream_t s) {
  static auto f = sym<decltype(&ncclRecv)>("ncclRecv");
  return call([&] { return f(a, n, t, peer, c, s); }, c, n, t, s, 7);
}
}
