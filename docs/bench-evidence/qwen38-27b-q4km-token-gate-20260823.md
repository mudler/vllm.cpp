# Qwen3.8-27B Q4_K_M vs llama.cpp `b10451`: the token gate FAILS, 5 of 6 prompts

W3 of row `QUANT-QWEN38-27B-GGUF-ARM`, for
[#821](https://github.com/mudler/vllm.cpp/issues/821). The declared gate is
[`qwen38-27b-quant-arms.md`](../../.agents/specs/qwen38-27b-quant-arms.md)
§Gates: greedy, identical artifact, identical prompts, identical token counts,
identical sampling, with llama.cpp as this arm's oracle because vLLM has no
in-tree GGUF reader at its pin (#979).

**Result: `TOKEN_GATE=FAIL`.** The tokenizer is exact on 6 of 6 prompts. The
generation diverges on 5 of 6. **No speed axis was run and none may be quoted**,
because `AGENTS.md` §Gates admits a performance result only after that arm's
declared token gate passes.

## What ran

Two `rc run` jobs on `thor:gpu0`, worker `rc-worker-kk96r`, 2026-08-23:

| `rc` job | Output tag | Purpose | Window (UTC) |
|---|---|---|---|
| `64f66cda-48be-445a-85d1-49bd689306f6` | `rc-worker-kk96r-20260823T081621Z` | build both engines, run the gate | 08:16:21 to 08:55:32 |
| `8e0d8e54-594f-45d2-bf94-1270401bab49` | `margin-rc-worker-kk96r-20260823T085816Z` | the top-2 margin DIAGNOSIS below | 08:58:16 to 09:00:40 |
| `0aba5d29-5b8b-4bdd-b5d6-f8fc9b5d8d1e` | — | remove the worker-local tree, 17 G reclaimed | 09:0x |

An earlier submission, `3dfaf454-d979-4667-8374-526abe3e77c0`, refused at step 0
and produced no measurement: `/usr/bin/time` is not installed on this worker, and
the job asserts its peak-RSS wrapper works before trusting anything it wraps. The
replacement wrapper is `job/runmem.py`, which reads
`getrusage(RUSAGE_CHILDREN).ru_maxrss` and needs nothing installed.

Evidence and job scripts are on the share at `/mnt/nas_share/rc/qwen38w3/`,
which the worker sees as `/workspace/qwen38w3/`: `job/` holds every script,
`out/<tag>/` holds every log and every raw per-leg file.

Nothing reached the box by `ssh`. `rc devices` reported `thor:gpu0` `ready`
again after the last job, so the resource was verified returned rather than
assumed, and the worker-local tree was removed (17 G reclaimed, `CLEAN=YES`).

### Why `thor:gpu0`

`dgx:gpu0` was held by another session at claim time and queueing behind it is
correct rather than contending with it. `thor:gpu0` is the host on which #857
demonstrated this oracle at this pin, its `/workspace` is the same NAS folder
holding the staged artifact, and 14 cores with 122 GB and 30.7 G of zram leave
real headroom over a 15.93 GiB model: `MemAvailable` was asserted at or above
40 GiB before every model run and read 117-118 GiB each time. Neither engine ran
concurrently with the other.

## Measured identity

Asserted inside the job, which fails on a mismatch.

```text
llama_pin                  = 10bf611e533d81f739128304991c5e133c6aebd8  (tag b10451)
llama_src_head             = 10bf611e533d81f739128304991c5e133c6aebd8
llama_src_porcelain        = EMPTY, before AND after the harness was built
llama_completion_sha256    = 61eda64699bfb3e2ef41c7977a07d167fc5597527fbd38e43d873c813390cbd6
llama_bench_sha256         = b11010fbe21a137a588dd26df6d37d0125ac45852607e61e001b5928ab2df549
oracle_tokens_sha256       = ca57056234449124f11360aef13c90749d46b4f6028aa8e52936606e924c7414
oracle_margin_sha256       = 31a307583fca7f6aad59fa9434ed62e8c94d96a63994ed68deb4daf51b7bb40c
vllmcpp_base_sha           = ff8f728071bd57bf70841ca56d289b5e09cabf00
vllm_cli_sha256            = f2ba2e2199f5b769a049f2ee3d7e62baf98835f05c130c77b672e1e41c3eba6a
vllm_bench_sha256          = e8f5ad7f20a2487c607a3f2d36653388fe95d7f5c9b0ba255f71fd5c6ebf898a
tokenize_sha256            = 04e6d8176c9fa8cc5720820b7cbb81ef9e4e76398ed680fe5c9ceb1b92fc9e75
gguf_sha256                = 7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169
gguf_size                  = 17106775008
host                       = rc-worker-kk96r (thor:gpu0), 14 cores, aarch64
```

**The llama.cpp binary hashes differ from the ones #857 recorded, and that is
expected rather than a discrepancy.** `GGML_NATIVE=ON` tunes the binary to the
box and this tree's own builds are not byte-reproducible, so a binary sha
identifies a BUILD and never a TREE. What identifies the oracle is the source
pin plus the empty porcelain, both asserted above, plus the control run below.

### The artifact, re-verified rather than trusted

Parsed independently of #857 from the file's own header on the devbox before the
job: GGUF v3, `general.architecture = qwen35`, 866 tensors, 51 kv, alignment 32,
header end 10,996,700, data start 10,996,704, **computed data end
17,106,775,008 == file size**. Type histogram F32 456 / Q4_K 294 / Q6_K 67 /
Q5_K 48 / Q8_0 1. `blk.64` is 15 tensors totalling **289,527,808 bytes**. Every
one of those figures re-derives what the spec and #857 recorded.

A CIFS mmap is not a run surface, so the job copied the artifact to the worker's
local disk and re-hashed it there. The hash chain therefore covers the exact
bytes both engines read.

## The `blk.64` asymmetry, and how it was handled

`b10451` loads 851 of the 866 tensors and ignores all 15 of `blk.64`, four of
them the `nextn.*` MTP head. The job re-observed this rather than citing it: the
stock control run emitted exactly **15** `unused tensor blk.64.*` warnings,
listed in `stock_control.stderr`.

**Our arm ran with MTP OFF, so the two engines decoded the same 851 tensors.**
`vllm-cli` and `vllm-bench` were given no `--speculative-config`, and
`src/vllm/entrypoints/model_loader.cpp` calls `AttachMtpDraftWeights` /
`LoadQwen3_5MTPFromGguf` only when `speculative_config.has_value() && method ==
"mtp"`, so `blk.64` is never read on this path. The trunk is
`block_count - nextn_predict_layers` = 65 - 1 = 64 layers on both sides. The
comparison is therefore matched WORK and not only matched weights.

## The oracle side, and its chain of custody

`llama-completion` at `b10451` prints token PIECES and never token ids
(`tools/completion/completion.cpp:707-710`); `--verbose-prompt` prints the
PROMPT ids only (`:379-389`). A token-exact gate needs the generated ids, so
`job/oracle_tokens.cpp` is an **unavoidable harness adaptation**, per
`AGENTS.md` "Document only an unavoidable adaptation of the harness".

It is not a second oracle. It links the stock libllama built at the pin from a
byte-clean tree, calls the public `llama.h` API only, and mirrors
`completion.cpp`'s own choices: `llama_tokenize(add_special=true,
parse_special=true)` as `common_tokenize(ctx, prompt, true, true)` does at
`:279`, `llama_token_to_piece(..., special=false)` as `:707` does, and plain
argmax with first-maximum-wins, which is what `--temp 0 --top-k 1` resolves to
and is `llama_sampler_init_greedy`'s own tie rule. Building it left the pinned
tree's porcelain empty, re-asserted.

Two independent checks bind it to the stock binary:

1. **The stock control run reproduces #857 verbatim.** `llama-completion` on the
   recorded recipe printed, byte for byte, the same six capitals #857 recorded
   from a different build on 2026-08-22.
2. **`CHAIN_OF_CUSTODY=EXACT`.** The harness's detokenized prompt-plus-generation
   for prompt 0 is byte-identical to that stock stdout (228 vs 226 bytes,
   differing only in the trailing newlines the shell captured).

So the ids the gate compares against belong to a sequence the stock binary
demonstrably produced. The oracle also reported `ORACLE_ADD_BOS = 0`,
`ORACLE_EOS = 248046`, `ORACLE_N_VOCAB = 248320`.

## The gate

Six raw completion prompts, no chat template, greedy on both sides, 48 tokens
each, concurrency 1, `ignore_eos` so both sides emit exactly 48.

### Tokenizer: EXACT, 6 of 6

`examples/tokenize` reading the GGUF's own vocab produced the oracle's
`PROMPT_IDS` line for line, at lengths 6, 5, 6, 7, 11, 7. `vllm-cli` reported
the same six `prompt_tokens` counts through the C ABI independently, so three of
our paths agree with the oracle on tokenization. **The #1355 prompt-token
undercount does not appear on this path.**

### Generation: DIVERGES, 5 of 6

`TOKENIZER_DIVERGENCES=0/6`, `GENERATION_DIVERGENCES=5/6`, `TOKEN_GATE=FAIL`.

| Prompt | Verdict | First differing index | vllm.cpp | llama.cpp |
|---|---|---:|---:|---:|
| `The capital city of France is` | DIVERGE | 7 | 9338 | 9564 |
| `The three primary colors are` | DIVERGE | 34 | 198 | 3095 |
| `Water boils at a temperature of` | DIVERGE | 20 | 13 | 539 |
| `The Pythagorean theorem states that` | **TOKEN-EXACT 48/48** | — | — | — |
| `In 1969, humans first walked on` | DIVERGE | 14 | 4593 | 22486 |
| `A prime number is a natural number` | DIVERGE | 32 | 15 | 16 |

The full id sequences are in `out/<tag>/token_gate.txt`.

**Both frontends give the same answer**, so this is the engine and not the
harness: `vllm-cli`'s text for prompt 0 detokenizes exactly the ids
`vllm-bench --output-token-ids` recorded, and `vllm-cli` is a thin `include/vllm.h`
client while `vllm-bench` drives the V1 `AsyncLLM`.

**Neither side is producing nonsense.** Our continuations are fluent and
plausible; so are the oracle's. On prompt 0 ours repeats one capital while the
oracle enumerates six, and on prompt 4 the oracle repeats one sentence three
times while ours varies. That symmetry is the first sign that this is not a
structural defect.

## THE DIAGNOSIS: every divergence is a rank-2 loss under 0.18 logits

`job/oracle_margin.cpp` teacher-forces the oracle along **vllm.cpp's own** token
sequence and reports, at every step, where our token ranked in the oracle's
distribution and by how much it lost. It is a diagnosis and produces no gate
result.

Over all **288** steps (6 prompts x 48 tokens):

```text
our_rank=1  282
our_rank=2    6
```

**Not one step ranked 3 or worse.** The two engines order the vocabulary
identically everywhere except at six near-misses:

```text
MARGIN 0  7  9338 rank2 15.933463  top1 9564 15.991513  gap 0.058050  our=" France"    top1=" Germany"
MARGIN 1 34   198 rank2 19.653543  top1 3095 19.738977  gap 0.085434  our="\n"         top1=" When"
MARGIN 1 35  4350 rank2 18.953596  top1 5844 19.077843  gap 0.124247  our="When"       top1="Red"
MARGIN 2 20    13 rank2 22.465479  top1  539 22.643715  gap 0.178236  our="."          top1=" by"
MARGIN 4 14  4593 rank2 15.919413  top1 22486 16.034895 gap 0.115482  our=" heart"     top1=" satellite"
MARGIN 5 32    15 rank2 20.796000  top1   16 20.823185  gap 0.027185  our="0"          top1="1"
```

The absolute logits are 15.9 to 22.6, so the losing margins are **0.12% to
0.79%** of the logit magnitude. Prompt 3 was rank-1 on all 48 steps, which is why
it is token-exact.

**This is a precision difference in the quantized compute path, not a wiring or
structural defect.** A wrong graph, a mis-routed layer or a dequant fallback
would put our token far down the oracle's ranking, repeatedly, and would not
leave 282 of 288 steps at rank 1.

## What this is NOT, and what is not admissible

- **The near-tie band does not apply and was not reached for.** The spec admits
  it only where the oracle's own greedy decode is non-deterministic. This
  oracle's greedy decode reproduced #857's output byte for byte from a different
  build, so it is deterministic. A 0.058-logit loss is a small margin, not the
  exact fp32 tie #910 adjudicated. The gate FAILS.
- **No speed number is quoted.** Both engines were timed as a by-product and
  every figure below is recorded for completeness only. Correctness comes first,
  and this arm has none yet.
- **No memory axis is accepted either**, for the same reason.

### Recorded for completeness, quotable as nothing

| | vllm.cpp | llama.cpp `b10451` |
|---|---:|---:|
| decode | 0.42-0.45 tok/s (`vllm-bench`), 0.80-0.94 tok/s incl. load (`vllm-cli`) | 5.36 tok/s |
| prefill | 0.65 tok/s | 8.69 tok/s |
| peak RSS | 24.997 GiB (`vllm-bench`), 29.443 GiB (`vllm-cli`) | 30.917 GiB |

One repetition, no CPU clock pinned, no contention control, one shared box.

**The peak RSS is the one figure that answers a question the spec asked.**
`## Risks` wants a resident-bytes assertion so a Q4_K_M that silently
dequantizes to bf16 cannot pass a token gate. Ours is **lower** than the
oracle's on the same box and the same file, and both sit near 2x the 15.93 GiB
file because both repack quantized weights into a second buffer (`REPACK = 1` in
llama.cpp's own capability line). So there is no dequant-to-bf16 blow-up on our
side. That is an observation, not the assertion the spec owes, because the
assertion belongs beside a passing gate.

The decode gap is large enough to be worth naming and is NOT attributed here:
it was measured on an ungated arm, and an ungated arm's throughput ranks nothing.

## The next traceable hypothesis

No ceiling is declared. The next step is to find which stage's arithmetic
differs, and the margin data says where to look:

1. **Compare the final logits, not the tokens.** Our tree exposes generated
   token ids (`vllm-bench --output-token-ids`) but no logit vector on any
   production path, so the two distributions cannot be diffed today. A logits
   dump is the missing instrument and it is owed.
2. **Suspect the quantized dot product and the activation dtype first.** The
   file is Q4_K/Q5_K/Q6_K/Q8_0 and our CPU tier branches on `M`
   (`src/vt/cpu/cpu_quant_gemm.cpp:190`), taking the Arm i8mm `mmla` 2x2 tile
   only for even `M` and `N` and sending decode at `M=1` to the portable
   `nrc==1` path, while ggml uses its own repacked kernels. A different
   accumulation order or a different intermediate width produces exactly this
   signature: identical ordering, sub-1% logit offsets.
3. **Bisect by layer against a single forward.** One prompt, one position, our
   hidden state against llama.cpp's at each of the 64 layers, will localise
   whether the offset accumulates smoothly or appears at one block. The GDN
   (`ssm_*`) layers and the 16 full-attention layers are different code and
   should be separated.

None of that is W3's work and none of it was attempted here.
