// Repack correctness harness: run the vendored C++ MarlinRepack* on the raw
// modelopt fp4 experts dumped by repack_golden.py and diff bit-for-bit vs
// vLLM's marlin-repacked golden (weight + S0E5M3 scales + global scale).
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "vt/cuda/marlin_repack.h"

#define CK(x) do{ cudaError_t e=(x); if(e){fprintf(stderr,"CUDA %s: %s\n",#x,cudaGetErrorString(e));exit(1);} }while(0)

static std::vector<uint8_t> readf(const std::string& p) {
  FILE* f = fopen(p.c_str(), "rb"); if (!f) { fprintf(stderr, "open %s\n", p.c_str()); exit(1); }
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> b(n); if (fread(b.data(), 1, n, f) != (size_t)n) exit(1); fclose(f); return b;
}
static void* dev(const std::vector<uint8_t>& h) {
  void* d; CK(cudaMalloc(&d, h.size())); CK(cudaMemcpy(d, h.data(), h.size(), cudaMemcpyHostToDevice)); return d;
}

int main() {
  const std::string D = "/tmp/repack_dump";
  const int E = 8, K = 2048, N = 512;
  auto raw_packed = readf(D + "/raw_packed.bin");   // [E, N, K/2] u8
  auto raw_scale = readf(D + "/raw_scale.bin");      // [E, N, K/16] u8
  auto raw_scale2 = readf(D + "/raw_scale2.bin");    // [E] f32
  auto gold_weight = readf(D + "/gold_weight.bin");  // [E, K/16, N*2] i32
  auto gold_scale = readf(D + "/gold_scale.bin");    // [E, K/16, N] u8
  auto gold_global = readf(D + "/gold_global.bin");  // [E] f32

  const size_t wbytes_e = (size_t)(K / 16) * (N * 2) * sizeof(uint32_t);
  const size_t sbytes_e = (size_t)(K / 16) * N;
  const size_t pbytes_e = (size_t)N * (K / 2);
  const size_t scbytes_e = (size_t)N * (K / 16);

  // combined_scale_factor across all E experts (host)
  std::vector<const uint8_t*> bufs; std::vector<size_t> lens;
  for (int e = 0; e < E; ++e) { bufs.push_back(raw_scale.data() + e * scbytes_e); lens.push_back(scbytes_e); }
  const float comb_sf = vt::cuda::MarlinNvfp4CombinedScaleFactor(bufs, lens);
  printf("C++ comb_sf = %g\n", comb_sf);

  int device = 0; CK(cudaSetDevice(device));
  uint8_t* d_packed = (uint8_t*)dev(raw_packed);
  uint8_t* d_scale = (uint8_t*)dev(raw_scale);
  uint32_t* d_w; CK(cudaMalloc(&d_w, (size_t)E * wbytes_e));
  uint8_t* d_s; CK(cudaMalloc(&d_s, (size_t)E * sbytes_e));

  for (int e = 0; e < E; ++e) {
    vt::cuda::MarlinRepackExpertWeight(0, device, d_w + (size_t)e * (wbytes_e / 4),
                                       d_packed + e * pbytes_e, K, N);
    vt::cuda::MarlinProcessExpertScales(0, d_scale + e * scbytes_e, d_s + e * sbytes_e, K, N, comb_sf);
  }
  CK(cudaDeviceSynchronize());

  std::vector<uint8_t> my_w((size_t)E * wbytes_e), my_s((size_t)E * sbytes_e);
  CK(cudaMemcpy(my_w.data(), d_w, my_w.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(my_s.data(), d_s, my_s.size(), cudaMemcpyDeviceToHost));

  // diff weight
  size_t wdiff = 0; for (size_t i = 0; i < my_w.size(); ++i) if (my_w[i] != gold_weight[i]) wdiff++;
  // diff scale
  size_t sdiff = 0; for (size_t i = 0; i < my_s.size(); ++i) if (my_s[i] != gold_scale[i]) sdiff++;
  // global
  float gmax = 0; float* gg = (float*)gold_global.data(); float* rs2 = (float*)raw_scale2.data();
  for (int e = 0; e < E; ++e) {
    float mine = vt::cuda::MarlinNvfp4ProcessGlobalScale(rs2[e], comb_sf);
    float rel = std::fabs(mine - gg[e]) / std::fabs(gg[e]);
    if (rel > gmax) gmax = rel;
  }

  printf("weight bytes=%zu diff=%zu  scale bytes=%zu diff=%zu  global max_rel=%.3e\n",
         my_w.size(), wdiff, my_s.size(), sdiff, gmax);
  bool ok = (wdiff == 0) && (sdiff == 0) && (gmax == 0.0f);
  printf("VERDICT: %s\n", ok ? "BIT-EXACT" : "MISMATCH");
  return ok ? 0 : 1;
}
