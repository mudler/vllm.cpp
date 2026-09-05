// strix:gpu0 (gfx1151) managed-memory residency staircase.
// Mirrors vt::rocm::RocmBackend::Alloc's managed branch (rocm_backend.hip:162)
// on an integrated device: hipMallocManaged(..., hipMemAttachGlobal).
// Question it answers: can this box hold GLM-5.3-Flash UD-Q2_K_XL (101.2535 GiB)
// as a keep-quant resident set at all?
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>

#define CK(x) do { hipError_t e=(x); if(e!=hipSuccess){ \
  printf("    HIP_FAIL %s -> %s\n",#x,hipGetErrorString(e)); fflush(stdout); goto done; } } while(0)

__global__ void touch(uint8_t* p, size_t n, size_t stride) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t off = i * stride;
  if (off < n) p[off] = (uint8_t)(i & 0xFF);
}

int main() {
  const size_t GiB = 1ull<<30;
  const size_t BLOCK = 2ull*GiB;      // 2 GiB per allocation (finer ceiling, 2nd sample)
  const size_t CEIL  = 112ull*GiB;    // hard stop
  const size_t PAGE  = 2ull<<20;      // touch 1 byte per 2 MiB

  int dev=0; hipDeviceProp_t pr;
  if (hipGetDeviceProperties(&pr, dev)!=hipSuccess){ printf("no device\n"); return 1; }
  printf("device: %s  gcnArch=%s  integrated=%d  managed=%d  concurrentManaged=%d\n",
         pr.name, pr.gcnArchName, pr.integrated, pr.managedMemory, pr.concurrentManagedAccess);
  size_t f0,t0; hipMemGetInfo(&f0,&t0);
  printf("hipMemGetInfo: free=%.3f GiB total=%.3f GiB\n",(double)f0/GiB,(double)t0/GiB);
  printf("TARGET: GLM-5.3-Flash UD-Q2_K_XL text shards = 101.2535 GiB\n\n");
  fflush(stdout);

  {
  std::vector<uint8_t*> blocks;
  size_t total = 0;
  while (total + BLOCK <= CEIL) {
    uint8_t* p=nullptr;
    hipError_t e = hipMallocManaged((void**)&p, BLOCK, hipMemAttachGlobal);
    if (e != hipSuccess) {
      printf("ALLOC_FAIL at total=%.3f GiB (+2 GiB): %s\n",(double)total/GiB,hipGetErrorString(e));
      fflush(stdout); break;
    }
    // GPU-touch one byte per 2 MiB -> forces physical backing, proves GPU reach
    size_t nt = BLOCK/PAGE;
    hipLaunchKernelGGL(touch, dim3((nt+255)/256), dim3(256), 0, 0, p, BLOCK, PAGE);
    e = hipDeviceSynchronize();
    if (e != hipSuccess) {
      printf("TOUCH_FAIL at total=%.3f GiB: %s\n",(double)total/GiB,hipGetErrorString(e));
      fflush(stdout); hipFree(p); break;
    }
    // host-dereference check (this is what the CPU reference tier does)
    volatile uint8_t v = p[BLOCK-PAGE];
    (void)v;
    blocks.push_back(p);
    total += BLOCK;
    size_t fr,tt; hipMemGetInfo(&fr,&tt);
    printf("  OK resident=%7.3f GiB   hipFree=%7.3f GiB%s\n",
           (double)total/GiB,(double)fr/GiB,
           (total>=(size_t)(101.2535*GiB) ? "   <== ARTIFACT FITS" : ""));
    fflush(stdout);
  }
  printf("\nMAX_MANAGED_RESIDENT = %.3f GiB  (artifact needs 101.2535 GiB)\n",(double)total/GiB);
  printf("VERDICT: %s\n", total >= (size_t)(101.2535*(double)GiB) ? "FITS" : "DOES NOT FIT");
  fflush(stdout);
  for (auto* p : blocks) hipFree(p);
  }
done:
  return 0;
}
