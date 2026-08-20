# Qwen3.8 27B

Use this page for Qwen3.8 27B checkpoints, commands, supported arms, and current limitations.

## Block-wise FP8 runs on CPU, and its CUDA kernel is built but unverified

Block-wise FP8, also called fine-grained FP8, keeps one scale for each 128x128
block of a weight rather than one scale for the whole weight. A block-wise
checkpoint declares `quantization_config.weight_block_size` in its
`config.json` and stores its scales under `weight_scale_inv` rather than under
`weight_scale`.

`Qwen/Qwen3.8-27B-FP8` is such a checkpoint. At revision
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` it declares `weight_block_size`
`[128, 128]` with `activation_scheme` `dynamic`, and it stores
`self_attn.q_proj.weight` as `F8_E4M3` `[12288, 5120]` beside
`self_attn.q_proj.weight_scale_inv` as `BF16` `[96, 40]`.

That checkpoint now RUNS on a CPU queue. Ten projections of the Qwen3.5 dense
model — `q_proj`, `k_proj`, `v_proj`, `o_proj`, the Gated-DeltaNet
`in_proj_qkv`, `in_proj_z` and `out_proj`, and the MLP's `gate_proj`, `up_proj`
and `down_proj` — quantize their activation per token per 128-wide group and
then run a block-scaled GEMM whose scales apply in the mainloop, once per
K-block, into an F32 accumulator. Each of the ten emits BF16, which is the
model dtype and what vLLM emits at the same sites.

Those ten projections are seven GEMMs, because `gate_proj` and `up_proj` run as
one and `q_proj`, `k_proj` and `v_proj` run as one — the same two merged linears
vLLM builds. A block scale belongs to a 128-row band, so the shards' scale grids
concatenate exactly and the merged GEMM is byte-identical to the separate ones.

The `gate_proj`/`up_proj` merge always runs. The Q/K/V merge runs only when the
fused attention preamble is available to read its row-strided output views,
which is the default. `VT_FUSE_ATTN_PREAMBLE=0` turns that consumer off, and
then `q_proj`, `k_proj` and `v_proj` run as three separate block GEMMs and the
ten projections are nine GEMMs. The result is the same either way.

That merge needs each projection in a group except the last to be a multiple of
128 rows wide, which is what vLLM requires of the same checkpoints. A checkpoint
that breaks the rule is refused by name, and the message says which projection
and how wide it is, rather than quietly running a different arithmetic:

```text
block-wise FP8 merged 'qkv_proj': shard 'k_proj' has out_features 64, which is
not a multiple of the quantization block's n 128. Only the LAST shard of a
merged block-quant linear may be ragged
```

On a device with no block-scaled GEMM the model refuses while it is being
prepared, before the first forward and before any CUDA graph is captured:

```text
block-wise (fine-grained) 128x128 FP8 weights LOADED for
model.layers.0.self_attn.q_proj and there is no block-wise FP8 GEMM on device
'cuda'. The linear method and the dense forward wiring are implemented and the
CPU reference GEMM executes them, so this checkpoint runs on CPU today
```

What exists on CPU is a correctness reference. It makes no speed claim, and no
token-exact comparison against vLLM on this checkpoint has been recorded.

A CUDA kernel now exists for the sm_120a and sm_121a architectures, and it is
**run, and shape-restricted: unproven on every shape it can serve**. It is the
block-scaled CUTLASS GEMM vLLM itself dispatches on those devices, ported whole,
with the scales applied in the mainloop; it is compiled by continuous
integration for both architectures and registered, so a build for one of them no
longer refuses the checkpoint at prepare time. On
2026-08-20 `test_ops_matmul_fp8_block_cuda` was run on a GB10 (compute
capability 12.1) for the first time, and vLLM's own ported case -- M=32, N=576,
K=7168 -- threw `cutlass Invalid status` before any kernel launched, because
CUTLASS refused the configuration at `can_implement`.

Which shapes are affected is now isolated, and the answer is a **shape
restriction, not a bug in this tree**: on sm120 the CUTLASS block-wise
collective serves only an N and a K that are whole multiples of 128. It requires
complete scale blocks and full tiles in K, its sm90 counterpart requires
neither, and 576 is `4*128 + 64`. A coarser floor sits under that one and is
asked first where it applies -- `K % 16` and `N % 16`, the fp8 operand
alignment, which is the line vLLM draws before rerouting such a shape to a
Triton kernel this build does not have -- so four shape classes are refused in
all, two of them at 16 by vLLM's authority and two at 128 by the sm120
collective's. This arm refuses every one of the four **by name**, before it
allocates anything:

```text
matmul_fp8_block_scaled: no CUDA kernel for this shape. N is 576, which leaves a
remainder of 64 modulo 128, and the sm120 blockwise collective wants COMPLETE
SCALE BLOCKS [...] so N must be a multiple of 128
```

The message names the dimension, its value, the granularity, the CUTLASS line it
comes from, and that the sm90 collective has no such limit. It replaces
`cutlass Invalid status`, which named none of those.

**One real capability gap follows, and it is not repairable here.** DeepSeek-V3's
`kv_a_proj_with_mqa` is exactly N=576 -- which is why vLLM chose that shape for
its own test -- so on an sm120 device this arm cannot serve it at all. Any
block-wise FP8 checkpoint whose projections are not all a multiple of 128 wide is
affected the same way. The CPU reference arm runs every one of these shapes.
`Qwen/Qwen3.8-27B-FP8`, the checkpoint above, is not affected: its ten
projections are all round.

None of that makes the arm proven. **No shape has yet had its output compared
against the CPU reference on any device**, there is still no token-exact
comparison against vLLM and no throughput number. What changed is what a user is
told when the shape cannot be served, not whether the shapes that can be served
produce the right numbers.
[#1437](https://github.com/mudler/vllm.cpp/issues/1437) records the run,
milestone M5 of [#1189](https://github.com/mudler/vllm.cpp/issues/1189) owns the
kernel and the repair it now owes, and
[#1166](https://github.com/mudler/vllm.cpp/issues/1166) is the original report.

One lever is incompatible with this arm. `VT_KV_CACHE_F32=1` selects an F32
paged KV cache while `v_proj` keeps emitting BF16, and the KV write requires
both to share one dtype, so it refuses. That affects every BF16 arm rather than
this one; it is tracked as
[#1249](https://github.com/mudler/vllm.cpp/issues/1249). Leave the lever unset,
which is the default.

Two block-wise configurations are refused earlier, at load, because no build
here implements them: an `activation_scheme` other than `dynamic`, and a
`weight_block_size` other than `[128, 128]`. Both messages name the key and the
value your `config.json` declares.

To run this model on a GPU with a recorded correctness result today, use a
per-tensor FP8, BF16, NVFP4, or GGUF checkpoint of it.


## Running inference (CLI)

`vllm-cli` runs a one-shot completion through the C ABI. Source:
[`examples/cli/main.cpp`](../../examples/cli/main.cpp).

```sh
build/examples/vllm-cli \
  --model /path/to/Qwen3.6-27B \
  --prompt "The capital of France is" \
  --max-tokens 64
```

| Flag | Default | Meaning |
|---|---|---|
| `--model <dir>` | (required) | Model directory (config.json + tokenizer.json + safetensors) |
| `--prompt "<text>"` | (required) | Prompt text |
| `--tokenizer-config <path>` | (none) | Override `tokenizer_config.json` |
| `--max-tokens N` | `16` | Max tokens to generate |
| `--temperature T` | `0.0` | Sampling temperature (`<= 0` means greedy) |
| `--top-p P` | `1.0` | Nucleus cutoff |
| `--top-k K` | `0` | Top-k (`0` means all) |
| `--seed S` | (unset) | RNG seed (enables seeded sampling) |
| `--stream` | off | Stream token deltas to stdout |
| `--speculative-config '<json>'` | (unset) | Speculative decoding, same JSON as vLLM's flag. Every key is checked and none is dropped: an unknown or misspelled name is refused at startup by name, and a real vLLM key this engine does not implement is refused as such ([#1160](https://github.com/mudler/vllm.cpp/issues/1160)). See [docs/SPECULATIVE-DECODING.md](../SPECULATIVE-DECODING.md) |
| `--offload-config '<json>'` | (unset) | Weight placement, the same JSON document `vllm-server` takes and the same C ABI field. Both halves: vLLM's mirrored `uva`/`prefetch` device-to-host weight offload, and vllm.cpp's `vllm_cpp` key for the host-to-disk residency tier that makes a checkpoint larger than host RAM loadable. An unknown key at any level of the document is refused at startup by name. Added by [#1135](https://github.com/mudler/vllm.cpp/issues/1135); see [Streaming routed experts from disk](#streaming-routed-experts-from-disk-capacity-mode) |
| `--max-num-seqs N` | engine default (32) | Max concurrent sequences. Under speculative decoding on a GDN model the recurrent state is `max-num-seqs x (k+1)` per slot, so this is the knob to lower when a run is refused for state budget |
| `--repeat N` | `1` | Load once, then run N blocking completions. Use it to read a warm decode tok/s without paying model load each time. Not supported with `--stream`, which falls back to 1 |
| `-h`, `--help` | | Print usage and exit |

`--model` resolves a Qwen3.5-family checkpoint's backbone under EITHER weight
namespace. The multimodal wrappers (`Qwen3_5ForConditionalGeneration`,
`Qwen3_5MoeForConditionalGeneration`) publish the text backbone nested under
`model.language_model.`; the text-only arms (`Qwen3_5ForCausalLM`,
`Qwen3_5MoeForCausalLM`) publish it flat under `model.`. The loader decides which
ONCE per checkpoint from the shard index, and REFUSES a checkpoint that carries
backbone tensors under both rather than binding half the model from each.

**Resolving the namespace is not the same as loading the checkpoint, and the
MoE and dense arms differ.** The dense loader routes each projection to BF16,
FP8 or NVFP4 by tensor presence, so a flat bf16 `Qwen3_5ForCausalLM` checkpoint
is expected to load. The **MoE** loader reads two ROUTED-EXPERT layouts and
decides between them ONCE per checkpoint from the shard index: per-expert NVFP4
(`experts.<e>.<proj>.weight` U8 + `.weight_scale` + `.weight_scale_2`, what an
NVFP4 requant ships) and the 3-D stacked BF16
`experts.{gate_up_proj,down_proj}` the published repos (`Qwen/Qwen3.8-2.4T-A95B`,
`Qwen/Qwen3.6-35B-A3B`) ship. A checkpoint carrying BOTH spellings under its
backbone is refused rather than half-bound.

**Outside the routed experts the MoE arm routes by tensor presence too.** The GDN
tower (`linear_attn.{in_proj_qkv,in_proj_z,out_proj}`) and the attention tower
(`self_attn.{q,k,v,o}_proj`) read BF16 or per-tensor FP8; the shared expert
(`mlp.shared_expert.{gate,up,down}_proj`) and `lm_head` read BF16 or NVFP4. Each
of the four is resolved ONCE per checkpoint, and a component whose own
projections disagree — layer 0's `q_proj` BF16 beside layer 4's F8_E4M3 — is
refused naming both sides rather than bound half from each. Different components
MAY disagree with each other: a `modelopt_mixed` checkpoint really does ship an
FP8 tower beside an NVFP4 MLP, and the dense arm reads exactly that.

**Which code runs an FP8 projection is no longer a Qwen3.5 detail.** The
per-tensor FP8 W8A8 residency and GEMM entry points live in
`include/vllm/model_executor/models/dense_fp8_gemm.h`, with the scheme policy in
`include/vllm/model_executor/layers/quantization/fp8.h`, so any model binds them
through `layers::MakeLinearMethod(bf16_weight, fp8_weight)` — the same shape the
NVFP4 W4A16 seam already had. The bound method exposes two arms: `Apply`, which
quantizes the activation itself with the checkpoint's `input_scale`, and
`ApplyPreQuantized`, which takes an activation a preceding fused epilogue already
quantized and runs only the GEMM. Nothing about running Qwen3.5 changes: the
levers (`VT_DENSE_NATIVE`, `VT_DENSE_CUBLASLT_FP8`) keep their names and
defaults, and the path stays CUDA-only.

Still OWED for the MoE arm, and refused BY NAME rather than discovered as a dtype
complaint: an NVFP4 attention or GDN tower, an FP8 shared expert, an FP8
`lm_head`, a per-expert-but-unquantized routed layout, and a non-BF16 stacked
expert tensor.

**The MoE arm's VISION TOWER.** `LoadQwen3_5Moe` reads the text backbone only.
`Qwen/Qwen3.6-35B-A3B` ships 333 `model.visual.*` tensors alongside it, and until
issue #891 they were dropped without a word — the load succeeded and produced a
text-only model. `LoadQwen3_5MoeVision` now reads them, through the SAME
`LoadQwen3VLVisionWeights` the dense `Qwen3_5ForConditionalGeneration` arm uses,
with the tower geometry from the checkpoint's `vision_config` (depth 27, hidden
1152, 16 heads, intermediate 4304, patch 16, spatial merge 2, EMPTY
`deepstack_visual_indexes`) and `out_hidden_size` taken from the text hidden size
because the merger writes into the text residual stream. A checkpoint carrying NO
`model.visual.*` tensor is REFUSED naming them, rather than quietly loading a
model that answers image prompts from text alone — `nvidia/Qwen3.6-35B-A3B-NVFP4`
declares `vision_config` and ships no `visual.*` weights, and is exactly that
case.

**What is and is not proven about a published bf16 MoE repo.** Every arm is
byte-exact on synthetic fixtures, and the real published `Qwen/Qwen3.6-35B-A3B`
and `Qwen/Qwen3.8-2.4T-A95B` indices satisfy the load plan completely — every
name, dtype and enforced shape the reader asks for
(`tests/vllm/models/test_qwen3_8_text_only.cpp`). That reads NO weight byte and
is NOT a token claim: a wrong dtype path or a missing dequant produces wrong
logits rather than an error, so only a token-exact gate closes it. No text-only
Qwen3.5 checkpoint has been RUN here — see [STATUS.md](../STATUS.md) for the owed
run gates.

GGUF and safetensors mapped-payload paths, plus safetensors index paths, use the
host's native filesystem encoding, including Unicode paths on Windows. Native
Windows release artifacts are not published yet; they will remain unavailable
until the `v0.0.3-pre.1` prerelease build and publication gates succeed.

Two more example binaries ship alongside it:

- `vllm-bench` ([`examples/bench/main.cpp`](../../examples/bench/main.cpp)), a
  throughput/latency harness taking `--model`, `--dataset-path`,
  `--num-prompts`, `--input-len`, `--output-len`, `--concurrency`,
  `--max-num-batched-tokens`, and `--num-blocks`. It pretokenizes before timing
  and atomically publishes each concurrency wave. Set
  `VT_BENCH_PRETOKENIZE=0` for the timed-string rollback; the report names the
  resolved mode.
- `tokenize` ([`examples/tokenize/main.cpp`](../../examples/tokenize/main.cpp)), a
  tokenizer smoke tool taking `<tokenizer.json | model.gguf> <corpus.txt>`.
  GGUF `tokenizer.ggml.pre` names accepted: `qwen35`, `qwen2`, `llama-bpe`,
  `gpt-4o` / `llama4` / `kanana2` / `talkie` (the GPT-4o / o200k family),
  `joyai-llm`, `deepseek-llm`, `deepseek-v3`, `laguna`. Any other name is
  refused by name rather than aliased onto a near-miss regex.
