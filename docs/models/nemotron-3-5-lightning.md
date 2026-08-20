# Nemotron 3.5 Lightning

Use this page for Nemotron 3.5 Lightning checkpoints, commands, supported arms, and current limitations.

## Nemotron-3.5-Lightning-30B: the exact weights, and which arms run

`NemotronHForCausalLM` is a hybrid: 6 GQA attention layers over a paged KV cache
and 23 Mamba2 layers over a recurrent conv/SSM state, with MoE blocks between
them. `examples/nemotron_h_gen` (`nemotron-h-gen`) drives it through the public
C ABI and nothing else — `vllm_engine_load` + `vllm_complete_tokens` — against
the committed oracle golden:

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
| `lm_head`, NVFP4 W4A16 g16 | **host** — it refuses a non-CPU queue by name, so the forward's last step is a host projection and the model still returns host logits. Owed to A2-Q2b, [#810](https://github.com/mudler/vllm.cpp/issues/810) |

And the arms that are **refused by name** rather than substituted:

| arm | the refusal |
|---|---|
| GGUF k-quants / i-quants | not ported. A GGUF path is refused at load naming `.agents/specs/nemotron-h-model.md` §5b W7, because silently dequantizing to a supported path is exactly what a token gate cannot see |
| the MTP draft head | deferred by name at load (W5) |
| batched decode (`num_reqs > 1`) | refused at the forward. One request's KV pages and one request's recurrent state are carried per step; a multi-request step would be decoded as ONE concatenated causal sequence and would return plausible wrong tokens instead of failing. Owed to A2-B, [#810](https://github.com/mudler/vllm.cpp/issues/810) |

## What has NOT been measured

**No token gate result exists for this checkpoint yet.** The example above is the
vehicle for it and the golden is committed, but the run itself is pending; the
current state is recorded in `docs/BENCHMARKS.md` rather than left as silence,
and nothing about the released checkpoint's output is claimed here until it is
green.
