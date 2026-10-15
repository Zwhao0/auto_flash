// CPU-only contract fixture for the actual interceptor: no GPU work is simulated
// in performance experiments. This library checks CUDA event submission ordering.
#include <cuda_runtime_api.h>
#include <nccl.h>
#include <cassert>
#include <cstdint>
#include <vector>
struct E { bool recorded=false; };
static int records=0, waits=0;
extern "C" {
int mockRecords(){return records;}
int mockWaits(){return waits;}
void mockReset(){records=waits=0;}
cudaError_t cudaEventCreate(cudaEvent_t* e){*e=(cudaEvent_t)new E;return cudaSuccess;}
cudaError_t cudaEventDestroy(cudaEvent_t e){delete (E*)e;return cudaSuccess;}
cudaError_t cudaEventRecord(cudaEvent_t e,cudaStream_t){((E*)e)->recorded=true;records++;return cudaSuccess;}
cudaError_t cudaEventSynchronize(cudaEvent_t e){assert(((E*)e)->recorded);assert(records==4);waits++;return cudaSuccess;}
cudaError_t cudaEventElapsedTime(float* ms,cudaEvent_t,cudaEvent_t){*ms=2.5;return cudaSuccess;}
cudaError_t cudaStreamIsCapturing(cudaStream_t,cudaStreamCaptureStatus* s){*s=cudaStreamCaptureStatusNone;return cudaSuccess;}
ncclResult_t ncclGroupStart(){return ncclSuccess;}
ncclResult_t ncclGroupEnd(){return ncclSuccess;}
ncclResult_t ncclCommInitRank(ncclComm_t* c,int,ncclUniqueId,int){*c=(ncclComm_t)0x100;return ncclSuccess;}
ncclResult_t ncclAllReduce(const void*,void*,size_t,ncclDataType_t,ncclRedOp_t,ncclComm_t,cudaStream_t){return ncclSuccess;}
}
