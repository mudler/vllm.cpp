# Nemotron 3.5 Lightning

Nemotron-3.5-Lightning-30B-A3B is a hybrid architecture. `NemotronHForCausalLM`
runs 6 grouped-query attention layers over a paged KV cache and 23 Mamba2 layers
over a recurrent convolution and SSM state, with mixture-of-experts blocks
between them.

**No token gate result exists for this checkpoint yet.** The example below is the
vehicle for it and the golden is committed, but the run is pending. Nothing about
the released checkpoint's output is claimed here until it is green.

## Run it

`examples/nemotron_h_gen`, installed as `nemotron-h-gen`, drives the model
through the public C ABI and nothing else, using `vllm_engine_load` and
`vllm_complete_tokens`, against the committed oracle golden:

```sh
nemotron-h-gen --model "$CHECKPOINT_ROOT/nemotron-3.5-lightning-30b-nvfp4" \
               --golden tests/parity/goldens/nemotron_35_lightning_greedy/oracle.json
```

`--golden-info` parses the golden and prints its geometry without loading a
model, which is how you check the battery's shape before spending a 20.1 GiB
load. `--load-only` stops after `vllm_engine_load`.

## The checkpoint

| field | value |
|---|---|
| repo | [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) — first party |
| revision | `29f2d1746d8f41e316523194b19018707749b1b1` |
| staged as | `$CHECKPOINT_ROOT/nemotron-3.5-lightning-30b-nvfp4` (a `hf download --local-dir` tree) |
| on-disk total | 21 583 809 748 bytes (20.1 GiB) |
| weights | `model-000{01..52}-of-00052.safetensors` + `model.safetensors.index.json` |
| quantization | `config.json` (1 337 760 B) + `hf_quant_config.json` (928 085 B), the `modelopt_mixed` layout |
| tokenizer | `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `chat_template.jinja` |
| sha256 (first shard) | `672c8bda10fdec0256e0819e112d2aa3a936cc3e5d311a05fd3ff773ca9a44b9` for `model-00001-of-00052.safetensors` (743 427 168 B) |

**A repo id alone is not a pin** — checkpoints get re-quantized in place under an
unchanged name — so the revision is recorded, and it was verified rather than
copied: the first shard on the gate host hashes to the value above, which is
that revision's own LFS record for the file
(`.cache/huggingface/download/model-00001-of-00052.safetensors.metadata`, whose
sidecar names commit `29f2d174`). `tests/parity/hf_snapshot.h` resolves the
directory and refuses a tree staged at any other revision, so
`VT_NEMOTRON35_SNAPSHOT` is left UNSET for a gate run: setting it takes the
explicit-directory escape, which is deliberately not revision-checked.

    hf download nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
      --revision 29f2d1746d8f41e316523194b19018707749b1b1 \
      --local-dir "$CHECKPOINT_ROOT/nemotron-3.5-lightning-30b-nvfp4"

## The arms, and what each one costs you today

The loader materializes all 18 487 tensors in the memory format the checkpoint
ships them in, so nothing is silently widened at load. What differs between arms
is **where the arithmetic happens**, and that is not something a token
comparison can see, so it is written down here instead.

| arm | state |
|---|---|
| bf16 layers, norms, the 6 GQA attention blocks | **device** |
| MoE experts, NVFP4 W4A16 g16 | **device** (Marlin arena) |
| FP8 W8A8 static Mamba2 input projections | **host** — the device arm is owed, [#940](https://github.com/mudler/vllm.cpp/issues/940) |
| `lm_head`, NVFP4 W4A16 g16 | **device** on the paged forward (A2-Q2b, [#810](https://github.com/mudler/vllm.cpp/issues/810)) — it runs through the shared NVFP4 W4A16 route `dense_nvfp4::MatmulNvfp4W4A16D`, so `NemotronHPagedForward` returns device-resident logits to the on-GPU sampler instead of a host projection. **The device arm's own numeric gate has NOT run yet** (it needs a `dgx:gpu0` window); until it does, treat this row as "implemented and unmeasured", which is what the spec's `## Now` records. The host projection is retained and is what serves a build without the Marlin NVFP4 GEMM, a non-NVFP4 `lm_head`, an explicit `VT_NVFP4_MARLIN=0`, and the numeric gate's reference side. `VT_NVFP4_MARLIN=0` is the same-binary A/B escape hatch for the Marlin NVFP4 W4A16 GEMM, and it now routes this arm to the host projection in EVERY build: the model's eligibility test and the shared dispatcher call one predicate, `dense_nvfp4::MarlinW4A16Selects`, rather than restate its clauses, and that predicate asks the op/provider table whether the Marlin arm is realized for the device instead of asking the preprocessor |

And the arms that are **refused by name** rather than substituted:

| arm | the refusal |
|---|---|
| GGUF k-quants / i-quants | not ported. A GGUF path is refused at load naming `.agents/specs/nemotron-h-model.md` §5b W7, because silently dequantizing to a supported path is exactly what a token gate cannot see |
| the MTP draft head | deferred by name at load (W5) |
| batched decode (`num_reqs > 1`) | refused at the forward. One request's KV pages and one request's recurrent state are carried per step; a multi-request step would be decoded as ONE concatenated causal sequence and would return plausible wrong tokens instead of failing. Owed to A2-B, [#810](https://github.com/mudler/vllm.cpp/issues/810) |

### What the host arms cost, per decode token

Counted on the real checkpoint at the single dequant seam, one decode step
(T=1, `top_k` 6, 23 MoE layers), because "where the arithmetic happens" is only
half the answer — the other half is how much of it there is:

| group | elements re-expanded per token | share |
|---|---|---|
| routed experts (6 of 128, x 23 layers) | 1 376 944 128 | 44.7% |
| shared experts (x 23) | 458 981 376 | 14.9% |
| Mamba2 FP8 `in_proj` + `out_proj` (x 23) | 890 265 600 | 28.9% |
| `lm_head` | 352 321 536 | 11.4% |
| **total** | **3 078 512 640 (6.157 GB at bf16)** | |

Only the groups marked **host** above are actually paid at run time. With the
MoE arm on the device that is 1 242 587 136 elements (2.485 GB) per token, and
with `lm_head` on the device too it is the Mamba2 FP8 pair alone. `lm_head` is
the largest SINGLE re-expansion in the model by 12.7x — one call, a 704.6 MB
transient bf16 buffer — which is why it matters more than its 11.4% share
suggests on a unified-memory box that reboots rather than OOM-kills.

## What has not been measured

**No token gate result exists for this checkpoint yet.** The example above is the
vehicle for it and the golden is committed, but the run itself is pending. The
current state is recorded in [`docs/BENCHMARKS.md`](../BENCHMARKS.md) rather than
left as silence, and nothing about the released checkpoint's output is claimed
here until it is green.

The `lm_head` device arm's own numeric gate has also not run. It needs a
`dgx:gpu0` window, and until then that row is implemented and unmeasured.
