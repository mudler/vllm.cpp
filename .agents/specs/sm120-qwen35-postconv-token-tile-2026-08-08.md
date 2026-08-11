# sm_120 Qwen3.5 fused post-conv token tile — structured spike

**Rows:** `KERNEL-SSM-MAMBA`, feeding `ROAD-V1-C2-LOCAL-BF16`.
**Issue:** [#206](https://github.com/mudler/vllm.cpp/issues/206) (landed by PR [#155](https://github.com/mudler/vllm.cpp/pull/155)).
**Hardware/workload:** RTX 5070 Ti (`sm_120`), Qwen3.5-4B plain BF16,
128 ShareGPT requests, 128 output tokens, concurrency 32,
`max_num_batched_tokens=2048`, 1,280 KV blocks, greedy. **Lifecycle:**
IMPLEMENTED, byte-exact and locally faster, but opt-in pending repeated and
release-model gates.

## Measured selection

The exact-chunks branch at `c3bb0f39a` was profiled under one `/tmp/gpu` lock,
the 22/25 GiB user-systemd limits, and `--cuda-graph-trace=node`. The accepted
1,280-block workload reproduces the prior post-conv baseline and emits the same
token-file SHA-256 as the exact-chunks rebenchmark:

| Arm | Calls | Total GPU time | Mean call | Total throughput |
|---|---:|---:|---:|---:|
| fast megablock | 1,728 | **228.150171 ms** | 132.031 us | 6,742.52 tok/s |
| per-V-head split | 1,728 | **448.364941 ms** | 259.471 us | 6,677.11 tok/s |

Both token files hash to
`83fcdc45f79ddb06a634c7d7d95eba3384543b3cd781a45a8db1fc4e2a453545`.
The split is therefore falsified on this target: it makes the selected kernel
**1.965x slower** and the enclosing run 0.97% slower. The sealed pinned-vLLM
same-tool trace has 1,923 `_fused_post_conv_kernel` calls, 108.034870 ms total,
and 56.180 us mean. The selected metric is **total GPU time for the fused GDN
post-conv family on the exact c32 workload**; the current total gap is 2.112x
and the per-call gap is 2.350x.

An earlier diagnostic accidentally omitted `--num-blocks 1280`, producing more
scheduler waves (2,376 calls). It independently found split 427.764060 ms versus
fast 215.064822 ms, but its absolute and enclosing numbers are VOID for parity.
It is retained only as a second falsification of the split hypothesis.

## Whole-chain cause and upstream contract

Pinned vLLM uses one Triton program over a **16-token tile and one Q/K or V
head**: grid `(ceil(L, 16), H + HV)`, four warps and two stages
(`${VLLM_SOURCE}/vllm/third_party/flash_linear_attention/ops/fused_gdn_prefill_post_conv.py:57-74,208-245`).
Q/K values remain in the program while each token's float32 square sum is
reduced and the normalized values are stored (`:76-107`). V copy and gating use
the same token tile (`:108-149`). The upstream executable specification sweeps
the 35B, 397B and small shapes, `L={1,16,128,512,2048}`, and BF16 correctness
(`${VLLM_SOURCE}/tests/kernels/test_fused_gdn_post_conv.py:60-141`), plus sanity
and L=0 (`:144-208`).

Our fast kernel launches one `(token, Q/K-head)` block and one whole-value
megablock per token (`src/vt/cuda/cuda_gdn.cu:1271-1333,1434-1449`). Its Q and K
paths each use a 128-thread shared-memory reduction with eight block barriers and
then **reload** the activation for the normalized store. The rejected split only
changes the V grid; it leaves those one-token Q/K reductions intact
(`src/vt/cuda/cuda_gdn.cu:1335-1406`). The trace and source therefore select the
missing 16-token Q/K tile, register reuse and warp reductions—not another V-only
grid change—as the next discriminator.

## Port and first implementation

Add a CUDA token-tile kernel for the production `Dk==Dv==128` shape, behind
`VT_GDN_POSTCONV_TOKEN_TILE` and default OFF while it is evaluated:

1. Launch `grid=(ceil(T,16), Hk+Hv)` with 128 threads (four warps), matching the
   upstream work partition.
2. Assign each warp tokens `warp, warp+4, warp+8, warp+12` within the tile.
   For a Q/K head, each lane owns features `lane+{0,32,64,96}`. Keep Q and K in
   registers, reduce their float32 square sums with warp shuffles, then store
   without rereading `conv` and without shared-memory barriers.
3. For a V head, use the same warp/token mapping; each lane copies four BF16/F32
   elements and lane zero computes the existing softplus/sigmoid gate. Preserve
   the current local softplus arithmetic and tensor strides so this experiment
   changes scheduling and data reuse, not the operation contract.
4. Keep `VT_GDN_POSTCONV_SPLIT` and `VT_GDN_POSTCONV_FAST` as independent
   rollback controls. Unsupported dimensions remain on the current dispatch.

This is a CUDA spelling of vLLM's Triton schedule, not an unrelated sm_120
algorithm. It deliberately does not claim bit identity in advance: the warp
reduction groups the same float32 terms differently from the current 128-lane
tree. The upstream tolerance contract must pass, and the project-level cached
model tokens must remain exact before any default flip.

No GEMM claim is made. If a later profile selects a GEMM, it separately owes the
four-axis same-tool invocation proof (C/output dtype, compute/scale type, entry
point/algo policy, and resolved template dtype).

## Tests and acceptance

RED-first coverage must add:

1. a portable flag-predicate test proving the tile is opt-in and `0` rolls back;
2. the upstream shape/length sweep for BF16, including the partial last tile,
   `T=0`, `T=1`, and strided packed-BA gate inputs;
3. CUDA tile versus current fast output checks: V/g/beta exact, Q/K within the
   upstream `1e-2` BF16 tolerance, finite outputs and unit norms;
4. a structural mutant that restores the one-token grid fails the tile launch
   contract test;
5. full `test_ops_gdn` and cached Qwen3.5-4B 3/3·1672, followed by production
   tile-OFF/ON token-file identity.

The first performance decision is the same-binary graph-node micro-metric.
The tile must improve 228.150171 ms outside run noise and move toward the sealed
108.034870 ms total. An enclosing profile must not regress total/output
throughput, TTFT, TPOT/ITL, E2E or peak VRAM. Default ON additionally requires
token-exactness, repeated A/B evidence, and the unavailable 27B/35B gate-model
gates before claiming shared release coverage; the local 4B result is not
extrapolated.

## Evidence and rollback

- fast trace: `/tmp/qwen35-postconv-split-c3bb-fast-nb1280.nsys-rep`, SHA-256
  `9e430b659dd30950436b7cde57dc85248269febb059d95341b7782c8e1fa5e0a`
- split trace: `/tmp/qwen35-postconv-split-c3bb-split-nb1280.nsys-rep`, SHA-256
  `855665ea3b042f28022e5d5ccdc32a4a85ccac04d6d153dad31ff12428e1823a`
- pinned-vLLM trace: `/tmp/qwen35-async-3f35356e0-vllm.nsys-rep`
- diagnostic no-`num-blocks` traces:
  `/tmp/qwen35-postconv-split-c3bb-{fast,split}.nsys-rep`

Rollback is `VT_GDN_POSTCONV_TOKEN_TILE=0` (or unset while the experiment is
opt-in), which retains the current default fast megablock without changing any
loader, scheduler or model route.

## Implementation outcome

`GdnPostConvTokenTileKernel` now implements the specified four-warp,
16-token/per-head work partition for `Dk==Dv==128`. Q/K activations remain in
registers through the reduction and normalized store; V/g/beta share the same
tile. The dispatch is opt-in through `VT_GDN_POSTCONV_TOKEN_TILE=1`, the slower
explicit split retains priority, and every unsupported shape stays on the
existing fast megablock.

The first warp reduction was a useful negative result. Sequentially summing the
four lane-owned squares before the 32-lane shuffle retained the kernel speedup
but changed production tokens: fast/tile token SHA-256
`83fcdc45...453545`/`1d496ff0...b9756`. The accepted implementation reproduces
the existing 128-lane tree exactly: first `(i+i+64)`, then the two resident
partials `(i+i+32)`, followed by shuffle offsets `16,8,4,2,1`. This restores
byte identity while keeping the values in registers.

Final same-binary graph-node traces have 1,728 calls per arm:

| Arm | Total GPU time | Mean call | Total throughput | TTFT | TPOT / ITL | E2E |
|---|---:|---:|---:|---:|---:|---:|
| fast megablock | 227.887066 ms | 131.879 us | 6,734.82 tok/s | 1,024.14 ms | 35.01 ms | 5,469.87 ms |
| token tile | **122.587027 ms** | **70.942 us** | **6,770.62 tok/s** | **1,015.43 ms** | **34.85 ms** | **5,440.81 ms** |

The tile is **1.859x faster** at the selected kernel and improves the enclosing
profile on every observed axis: total/output throughput +0.532%, TTFT -0.850%,
TPOT -0.457%, E2E -0.531%. It closes the same-tool vLLM gap from 2.112x to
**1.135x** (122.587027/108.034870 ms). Both final token files have SHA-256
`83fcdc45f79ddb06a634c7d7d95eba3384543b3cd781a45a8db1fc4e2a453545`.

Tests are stronger than the upstream BF16 tolerance: portable flag/grid 6/6,
CUDA GDN 67/67 and 4,384 assertions over partial/exact tiles, production and
small shapes, packed BA views with non-zero offsets/wider row strides, exact
q/k/v/g/beta bytes, finiteness and unit norms; cached Qwen3.5-4B 3/3 and 1,672
assertions. `T=0` is pinned by the portable no-work grid contract because local
kernel tensor descriptors require positive dimensions; the runtime sweep covers
`T={1,16,17,128,512,2048}`.

Final evidence:

- fast trace `/tmp/qwen35-postconv-tile-exact-fast.nsys-rep`, SHA-256
  `c75e2cb27797827b3d25d40204d745b3dfa36c4be8bd17a8464229c1b70bcadc`;
- tile trace `/tmp/qwen35-postconv-tile-exact-tile.nsys-rep`, SHA-256
  `a0eb1808e216a39d1350deabe7722e1b9081940183c512cefdb10afd2706f418`;
- rejected arithmetic traces `/tmp/qwen35-postconv-tile-wip-{fast,tile}.nsys-rep`,
  SHA-256 `c2d8872e...b774` / `1c42d489...d66b`.

Default ON remains deliberately unclaimed. One local 4B profile does not close
the required repeated A/B or the unavailable Qwen3.6-27B/35B release-model
correctness/performance gates.
