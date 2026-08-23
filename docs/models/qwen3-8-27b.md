# Qwen3.8 27B

Qwen3.8 27B is a 27B dense model. It runs through the shared paths, so
[the quickstart](../QUICKSTART.md) and [the usage guide](../USAGE.md) cover
starting a server and sending a request.

This page carries what is specific to its quantized checkpoints. Two of them
need it: a third-party mixed-precision set whose FP8 half is refused, and the
first-party block-wise FP8 set that runs on CPU but not yet on a GPU.

## Which arm to use on a GPU today

| Arm | State |
|---|---|
| BF16 | Runs |
| Per-tensor FP8 | Runs |
| NVFP4 | Runs |
| GGUF k-quants | Runs |
| Block-wise FP8 | **CPU only.** The CUDA kernel is built and matches the CPU reference on seven shapes, and no token gate has run through it |
| The FP8 group of `unsloth/Qwen3.8-27B-NVFP4` | **Refused by name at load** |

To run this model on a GPU with a recorded correctness result today, use a
per-tensor FP8, BF16, NVFP4, or GGUF checkpoint of it.

## The Unsloth mixed FP8 and NVFP4 checkpoint

`unsloth/Qwen3.8-27B-NVFP4` is a third-party mixed-precision checkpoint. Its
repository name says NVFP4, while `quantization_config.format` says
`mixed-precision`. Use revision `7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108`.

The set contains a 22,568,192,096-byte `model.safetensors` backbone and an
849,400,392-byte BF16 `model_mtp.safetensors` drafter. The complete set is
23,417,592,488 bytes. The checkpoint registry records the locally computed
SHA-256 for the quantized backbone.

The backbone index contains 1,968 names:

| Scheme | Modules | Tensors | Covers |
|---|---:|---:|---|
| `group_1`, NVFP4 W4A4 with group size 16 | 168 | 672 | MLP projections on layers 0 through 55 |
| `group_0`, FP8 W8A8 | 233 | 466 | Attention, GDN, `lm_head`, and MLP projections on layers 56 through 63 |
| Configuration ignore list | 317 | 475 | GDN low-rank projections, norms, vision blocks, merger, and MTP head |
| No quantization target | 267 | 323 | Norms, `conv1d`, embeddings, and position data |
| `kv_cache_scheme` scales | 16 | 32 | `k_scale` and `v_scale` on full-attention layers |

**The NVFP4 modules load. The engine refuses the FP8 group before it reads a
weight.** That group requires per-output-channel weight scales and dynamic
per-token activation quantization, and this build implements neither. The engine
also refuses `kv_cache_scheme`, because it does not consume the checkpoint's K
and V scales.

Run the real-checkpoint manifest gate with:

```sh
VLLM_CPP_QWEN38_27B_NVFP4_DIR=/path/to/qwen3.8-27b-nvfp4 \
  ./build/tests/test_qwen38_27b_nvfp4_arm
```

The FP8 tower, the quantized KV cache, the resident-byte assertion, and the
token gates are still owed under
[#821](https://github.com/mudler/vllm.cpp/issues/821).

## Block-wise FP8

Block-wise FP8, also called fine-grained FP8, keeps one scale for each 128x128
block of a weight rather than one scale for the whole weight. A block-wise
checkpoint declares `quantization_config.weight_block_size` in its `config.json`
and stores its scales under `weight_scale_inv` rather than under `weight_scale`.

`Qwen/Qwen3.8-27B-FP8` is such a checkpoint. At revision
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` it declares `weight_block_size`
`[128, 128]` with `activation_scheme` `dynamic`, and it stores
`self_attn.q_proj.weight` as `F8_E4M3` `[12288, 5120]` beside
`self_attn.q_proj.weight_scale_inv` as `BF16` `[96, 40]`.

### What runs on CPU

That checkpoint runs on a CPU queue. Ten projections of the Qwen3.5 dense model
quantize their activation per token per 128-wide group, then run a block-scaled
GEMM whose scales apply in the mainloop, once per K-block, into an F32
accumulator. The ten are `q_proj`, `k_proj`, `v_proj`, `o_proj`, the Gated
DeltaNet `in_proj_qkv`, `in_proj_z`, and `out_proj`, and the MLP's `gate_proj`,
`up_proj`, and `down_proj`. Each emits BF16, which is the model dtype and what
vLLM emits at the same sites.

Those ten projections are seven GEMMs, because `gate_proj` and `up_proj` run as
one and `q_proj`, `k_proj`, and `v_proj` run as one. They are the same two merged
linears vLLM builds. A block scale belongs to a 128-row band, so the shards'
scale grids concatenate exactly and the merged GEMM is byte-identical to the
separate ones.

The `gate_proj` and `up_proj` merge always runs. The Q, K, and V merge runs only
when the fused attention preamble is available to read its row-strided output
views, which is the default. `VT_FUSE_ATTN_PREAMBLE=0` turns that consumer off,
and then those three run as three separate block GEMMs and the ten projections
are nine GEMMs. The result is the same either way.

The merge needs each projection in a group except the last to be a multiple of
128 rows wide, which is what vLLM requires of the same checkpoints. A checkpoint
that breaks the rule is refused by name, and the message says which projection
and how wide it is, rather than quietly running different arithmetic:

```text
block-wise FP8 merged 'qkv_proj': shard 'k_proj' has out_features 64, which is
not a multiple of the quantization block's n 128. Only the LAST shard of a
merged block-quant linear may be ragged
```

What exists on CPU is a correctness reference. It makes no speed claim, and no
token-exact comparison against vLLM on this checkpoint has been recorded.

### On a device with no block-scaled GEMM

The model refuses while it is being prepared, before the first forward and
before any CUDA graph is captured:

```text
block-wise (fine-grained) 128x128 FP8 weights LOADED for
model.layers.0.self_attn.q_proj and there is no block-wise FP8 GEMM on device
'cuda'. The linear method and the dense forward wiring are implemented and the
CPU reference GEMM executes them, so this checkpoint runs on CPU today
```

### The CUDA kernel, and the shapes it refuses

A CUDA kernel exists for the sm_120a and sm_121a architectures. It is the
block-scaled CUTLASS GEMM vLLM dispatches on those devices, with scales applied
in the mainloop. Continuous integration compiles it for both architectures, and
a build for either one registers the kernel. It is shape-restricted, and it
matches the CPU reference on the seven GB10 shapes that have been run.

The first GB10 run exposed the shape boundary. vLLM's ported M=32, N=576,
K=7168 case was refused by CUTLASS at `can_implement` before launch.

That is a **shape restriction, not a defect in this tree**. On sm120 the CUTLASS
block-wise collective serves only an N and a K that are whole multiples of 128.
It requires complete scale blocks and full tiles in K where its sm90 counterpart
requires neither, and 576 is `4*128 + 64`. A coarser floor sits under that one
and is asked first where it applies: `K % 16` and `N % 16`, the FP8 operand
alignment, which is the line vLLM draws before rerouting such a shape to a
Triton kernel this build does not have. Four shape classes are refused in all,
two of them at 16 by vLLM's authority and two at 128 by the sm120 collective's.

This arm refuses every one of the four **by name**, before it allocates
anything:

```text
matmul_fp8_block_scaled: no CUDA kernel for this shape. N is 576, which leaves a
remainder of 64 modulo 128, and the sm120 blockwise collective wants COMPLETE
SCALE BLOCKS [...] so N must be a multiple of 128
```

The message names the dimension, its value, the granularity, the CUTLASS line it
comes from, and that the sm90 collective has no such limit. It replaces
`cutlass Invalid status`, which named none of those.

**One real capability gap follows, and it is not repairable here.**
DeepSeek-V3's `kv_a_proj_with_mqa` is exactly N=576, which is why vLLM chose
that shape for its own test, so on an sm120 device this arm cannot serve it at
all. Any block-wise FP8 checkpoint whose projections are not all a multiple of
128 wide is affected the same way. The CPU reference arm runs every one of these
shapes. `Qwen/Qwen3.8-27B-FP8` is not affected, because its ten projections are
all round.

Seven distinct shapes match the CPU reference. Six cover M from 1 to 512 across
all three tile configurations, and one runs vLLM's fixture at N=512, the nearest
supported width to 576. The run reported 5 cases and 136 assertions with no
failures and no portable-fallback line. Unsupported shapes returned their named
refusal.

This is not a model gate and not a speed result. No token-exact comparison
against vLLM has run through this arm, and the device run recorded neither
controlled clocks nor contention. It proves only the shapes that were run.
[#1437](https://github.com/mudler/vllm.cpp/issues/1437) records both runs,
milestone M5 of [#1189](https://github.com/mudler/vllm.cpp/issues/1189) owns the
kernel, and [#1166](https://github.com/mudler/vllm.cpp/issues/1166) is the
original report.

### Two configurations refused at load

No build here implements either, and both messages name the key and the value
your `config.json` declares:

- an `activation_scheme` other than `dynamic`
- a `weight_block_size` other than `[128, 128]`

### One lever that is incompatible

`VT_KV_CACHE_F32=1` selects an F32 paged KV cache while `v_proj` keeps emitting
BF16. The KV write requires both to share one dtype, so it refuses. That affects
every BF16 arm rather than this one, and it is tracked as
[#1249](https://github.com/mudler/vllm.cpp/issues/1249). Leave the lever unset,
which is the default.
