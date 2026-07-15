# Benchmark workload-equivalence audit — 2026-07-15

**Kind:** grounded read-only audit of the binding 27B online gate (`3f256ab`)
configuration equivalence between the vllm.cpp arm and the vLLM v0.25.0
(`702f481`) arm. Sources: the immutable evidence root
`~/work/vllm.cpp-online-gate/evidence/3f256abdbb558e162bf8a2196284deb119648560`
(recorded server/client commands, server logs, corpus manifests), local source
at the gate SHA, pinned vLLM at `e24d1b24` and the oracle venv's `702f481`.
Method: recorded-command extraction, effective-default resolution in both
engines, adversarial adjudication. An intermediate lane's headline claim was
REFUTED during verification (see §SSM dtype); this file records the corrected
final verdict.

## Verdict

The comparison is apples-to-apples on every workload parameter. Batch cap,
token budget, context, sampling, corpus, client, and cache dtypes match; the
one material engine difference is vLLM's Inductor prefill fusion +
piecewise cudagraphs — which is vLLM's production config and therefore the
correct denominator, not a benchmark error.

## Matched (verified on both arms)

- `max_num_seqs=32` — explicit flag both arms (ours default 8, vLLM default
  128 — neither in play); binding concurrency cap both sides.
- `max_num_batched_tokens=2048`, chunked prefill enabled both; the 1024-token
  prompts fit one chunk on both.
- Prefix caching OFF both (flagged; vLLM default would be ON).
- `max_model_len=262144` both (from `max_position_embeddings`).
- Attention KV dtype BF16 both (ours `kv_dtype` default with `VT_KV_CACHE_F32`
  unset; vLLM `kv_cache_dtype=auto`→bf16).
- GDN conv state BF16 both; **GDN SSM/temporal state FP32 both** (see below).
- NVFP4 = CUTLASS FP4 W4A4 with 64 autotuned tactics both (ours
  `cuda_matmul_nvfp4_cutlass.cu`; vLLM `FlashInferCutlassNvFp4LinearKernel`).
- Full attention = FlashAttention-2 bf16 head_dim 256 both.
- Decode CUDA graphs cover {1,2,4,8,16,32} both; `enforce_eager=False` both
  (production-graphed denominator per protocol).
- Client: the SAME pinned `vllm bench serve` binary; verbatim c16 commands
  differ in exactly ONE token (`--result-dir …/ours` vs `…/vllm`); dataset is
  the same physical file (`corpus/27/vllm/c16-r1.jsonl`, sha256 `8ce95b53…`
  both arms; canonical partition `66f9539d…`, seed 0, target_input_len 1024,
  output_len 128); `--temperature 0 --ignore-eos --seed 0
  --custom-output-len 128 --num-warmups=concurrency --request-rate inf
  --skip-chat-template --disable-shuffle`; greedy overrides the model
  generation_config (temp 1.0/top-k 20/top-p 0.95) on both.
- Capacity non-binding both: zero preempt/recompute/swap in either arm's logs;
  vLLM KV usage peaked <18% at c16; both arms cap at `max_num_seqs=32`.

## SSM dtype: MATCH (both FP32) — an intermediate claim refuted

An audit lane claimed vLLM allocates the GDN temporal state in BF16
(`mamba_ssm_cache_dtype='auto'`→model dtype) and never reads the HF field. The
finisher refuted this at source: vLLM's model-specific hook
`Qwen3_5ForConditionalGenerationConfig.verify_and_update_config`
(`vllm/model_executor/models/config.py:744-767` at `702f481`, mapped via
`MODELS_CONFIG_MAP:837`, invoked unconditionally from
`vllm/config/vllm.py:876,1942-1944`) reads
`text_config.mamba_ssm_dtype="float32"` from the checkpoint and sets
`cache_config.mamba_ssm_cache_dtype="float32"` BEFORE the state-dtype
calculator (`qwen3_5.py:513-521` → `mamba_utils.py:83-96`) resolves
(conv=BF16, temporal=FP32). The success path logs nothing, which is why the
server log is silent. Ours resolves identically
(`model_registry.cpp:311-337` → `ResolveMambaSsmCacheDType`,
`qwen3_5.cpp:198-208`). Consequences: our record
(`specs/gdn-packed-decode.md` "BF16 conv + FP32 SSM from nested
`mamba_ssm_dtype=float32`") was correct; there is NO relative state-bandwidth
handicap; `VT_GDN_STATE_BF16` stays a diagnostic rollback and must never
become the default (it would break token-exactness against the FP32 vLLM
oracle).

## Real difference (the parity front, not a config error)

- Prefill/mixed-batch execution: ours EAGER hand-written vt/CUDA AOT kernels
  (`runner.cpp:612-627` at the gate SHA) vs vLLM `VLLM_COMPILE(3)` Inductor
  whole-graph + PIECEWISE cudagraph with `pass_config.fuse_act_quant=True`
  (`fuse_norm_quant=False`). Affects prefill/TTFT-side axes in vLLM's favor;
  it is vLLM's production config — the protocol-correct denominator.

## Method differences (non-binding here; recorded for pinning)

- KV sizing method: ours fixed `--num-blocks 4736` (block_size 32; 151,552
  attention-token slots, FA and GDN groups allocated separately) vs vLLM
  `gpu_memory_utilization=0.6` → 43.4 GiB → 703,445 tokens at hybrid-aligned
  block_size 784 (one unified page across groups). ~4.6× capacity asymmetry,
  irrelevant to the ≈36,864-token working set.
- Ours' `FullAttentionSpec` uses kF32 for block ACCOUNTING only
  (`model_registry.cpp:329`); with explicit `--num-blocks` the runtime KV
  tensor is bf16 (no real inflation).
- vLLM additionally captures decode-graph size 24 (unused by the grid).
- GDN kernel provider: ours AOT CUDA vs vLLM Triton/FLA — parity by design.

## Recommendations for the next authorized exact-grid rerun

1. Keep FP32 SSM as the default (mirror-correct); `VT_GDN_STATE_BF16` stays
   diagnostic-only.
2. Pass `--mamba-ssm-cache-dtype float32` explicitly on the vLLM arm (or dump
   the resolved cache config) so the resolved dtype is IN the evidence and no
   future audit needs source inference.
3. Cite vLLM claims against the actually-run SHA `702f481` (the oracle records
   `client_contract_source_commit`).
4. The parity effort's one true front on this surface remains vLLM's
   prefill fusion; state dtypes and capacity methods are settled.
5. Optional cleanliness (no expected throughput change at this workload): pin
   our `--num-blocks` to vLLM's derived token capacity.

## Audit provenance note

The workflow's two extraction lanes and its adjudicator died at the
structured-output retry cap (large per-parameter array schemas; the known
stub/retry-cap failure mode). The surviving resolver lane recovered the
recorded commands itself; a plain-text finisher then verified the
contradictions at source and produced this corrected verdict. All lane
transcripts are retained in the session workflow directory (`wf_c9f50e31`).
