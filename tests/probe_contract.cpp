#include <nccl.h>
#include <cuda_runtime_api.h>
#include <atomic>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
extern "C" int mockRecords();
extern "C" int mockWaits();
extern "C" void mockReset();
struct Record {uint64_t seq,ns,comm,hash,op,bytes,group,duration,epoch,flags;};
int main(){
 setenv("FLASH_RUN_ID","contract",1);setenv("RANK","0",1);
 ncclUniqueId id{};ncclComm_t c;assert(ncclCommInitRank(&c,2,id,0)==ncclSuccess);
 char name[128];snprintf(name,sizeof(name),"/flash_contract_0_%d",getpid());
 int fd=shm_open(name,O_RDWR,0600);assert(fd>=0);
 auto h=(std::atomic<uint64_t>*)mmap(nullptr,128+512*64+65536*80,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
 assert(h[12]==2);
 h[6]=(7<<1)|1;
 ncclGroupStart();ncclGroupStart();
 ncclAllReduce(nullptr,nullptr,10,ncclFloat,ncclSum,c,(cudaStream_t)1);
 h[6]=(8<<1); // Controller changes mode during the group. Both calls keep epoch 7.
 ncclAllReduce(nullptr,nullptr,20,ncclFloat,ncclSum,c,(cudaStream_t)2);
 ncclGroupEnd();assert(mockWaits()==0);assert(h[15]==1);
 ncclGroupEnd();assert(mockWaits()==2);assert(h[15]==0);assert(h[3]==2);
 auto r=(Record*)((char*)h+128+512*64);
 assert(r[0].epoch==7 && r[1].epoch==7);assert(r[0].duration==2500000 && r[1].duration==2500000);
 assert(r[0].group==r[1].group);assert(r[0].flags==1 && r[1].flags==1);
 mockReset();
 ncclGroupStart();ncclAllReduce(nullptr,nullptr,1,ncclFloat,ncclSum,c,(cudaStream_t)1);
 h[6]=(9<<1)|1;
 ncclAllReduce(nullptr,nullptr,1,ncclFloat,ncclSum,c,(cudaStream_t)2);ncclGroupEnd();
 assert(mockRecords()==0);assert(h[3]==4);assert(r[2].epoch==8 && r[3].epoch==8);
 assert(r[2].duration==0 && r[3].duration==0);
 munmap(h,128+512*64+65536*80);close(fd);shm_unlink(name);
 puts("PASS actual probe nested-group epoch latch and all-stop-events-before-wait contract");
}
