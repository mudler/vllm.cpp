# `qwen4_exp` on `--device cuda`: the decode identifiers, before and after #2496

One artifact, one prompt, one device, three server runs. This file records what
was measured; the cause and the fix are argued in
[`.agents/specs/qwen4-exp-flash-next.md`](../../.agents/specs/qwen4-exp-flash-next.md)
under `## DECODEDIV`.

## What was run

| | |
|---|---|
| device | `thor:gpu0`, NVIDIA Thor, compute_cap 11.0, driver 595.78, nvcc 13.0.88, inside an `rc` lease |
| artifact | `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S, shard 1 `Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf`, sha256 `88a1420825a9304063e882ada29d438263617f51ac8923d438d927496693bafd`, staged on worker-local disk |
| entry point | `examples/vllm-server`, `POST /v1/completions`, `--block-size 16 --num-blocks 128 --max-model-len 256` |
| request | prompt `The capital of France is`, `max_tokens=8`, `temperature=0`, `logprobs=1` |
| env | `VT_CPU_QUANT_REPACK=0 VT_LOAD_STATS=1 VT_Q4EXP_STATE_FP=1`. **No `CUDA_LAUNCH_BLOCKING`** — this is the production configuration |
| before | tree `f10b3953b6bac7252a0fb3095a5c9de3907877ff` |
| after | tree `1ef7885ec28b4edde88c665a9c1ff00115b63e4a`, server sha256 `082451c96aa20ec8ee34f1ed429972882cf0f8768294914ae2ca6d0691686663` |

Both runs answered HTTP 200 with `completion_tokens=8`, 11 completed steps, and
no `illegal memory access` line.

## The tokens

| arm | token ids | text |
|---|---|---|
| `--device cpu`, both trees | `11751 13 15767 411 2029 11 1092 369` | ` Paris. Given this fact, what is` |
| `--device cuda`, BEFORE | `11751 271 271 271 271 271 0 0` | ` Paris\n\n\n\n\n\n\n\n\n\n!!` |
| `--device cuda`, AFTER | `11751 13 15767 411 1928 11 628 567` | ` Paris. Given this information, can we` |

**The CPU control was re-taken on each tree rather than inherited.** It is
byte-identical both times.

## The tensor that named the cause

`VT_Q4EXP_STATE_FP=1` prints one summary line per persistent state per step. The
PLE n-gram history is int64 TOKEN IDS, so its disagreement can never be a
rounding difference, and it is where the two arms first part:

| step | `--device cpu` | `--device cuda`, BEFORE | `--device cuda`, AFTER |
|---|---|---|---|
| 0 (prefill, T=5) | `[9338, 369]` | `[9338, 369]` | `[9338, 369]` |
| 1 | `[369, 11751]` | `[369, 0]` | `[369, 11751]` |
| 2 | `[11751, 13]` | `[0, 0]` | `[11751, 13]` |
| 3 | `[13, 15767]` | `[0, 0]` | `[13, 15767]` |
| 4 | `[15767, 411]` | `[0, 0]` | `[15767, 411]` |

Before the fix the FIFO rolled correctly and what it PUSHED was `0`: the forward
was handed token id 0 at every decode step, because it read the host `token_ids`
the asynchronous runner deliberately leaves stale for decode rows. After it, the
history carries the sampled ids on both arms.

## What is still open, and it is NOT #2496

The CUDA arm is fluent and agrees on five of eight tokens. It is not
token-exact, and the residual is present at PREFILL, before any decode state is
read:

```
step=0 hidden  cpu   n=12800 nonfinite=0 maxabs=192 sumabs=28054.1 v=0.353516,-0.171875,3.42188,0.486328
step=0 hidden  cuda  n=12800 nonfinite=0 maxabs=192 sumabs=27964.7 v=0.326172,-0.132812,3.42188,0.470703
```

About 0.3% on the aggregate, on the first forward, compounding until a near-tie
flips at token 4. That is [#2547](https://github.com/mudler/vllm.cpp/issues/2547)
and it needs a per-layer tap to attribute. **Token 0 agreed all along only
because ` Paris` after this prompt is not a near-tie**, which is why "the prefill
is right" had to be checked with the fingerprint rather than inferred from one
argmax.

Op-level arm-against-arm gates on the same binary: `test_qwen4_exp_cuda`
351/351, `test_qwen4_exp_cuda_reductions` 160/160.

## The gate, red then green, in both places

The gate runs on a CPU QUEUE, because the defect is not arithmetic and not a
device kernel: it is which array the hook reads. It was driven red-then-green
twice — once on a local CPU-only build and once on `thor:gpu0` inside the lease,
at `1fa9293f106de23a3bf3aff9e2010932a07dbe1e` — with the fix made inert by
`if (false && input.device_token_ids != nullptr)` and restored byte-for-byte.

| leg | local, CPU-only build | `thor:gpu0`, sm_110 CUDA build |
|---|---|---|
| green | rc 0, 64 assertions, 0 failed | rc 0, 64 assertions, 0 failed |
| mutation applied | 1 site | 1 site |
| mutation rebuild | rc 0 | rc 0, 1 object |
| **RED** | **rc 1**, `spliced != fresh`, `spliced == stale` byte for byte, history carries the stale id | **rc 1**, 64 assertions, **61 passed / 3 failed** |
| restored | `git diff` empty | sha256 == before: YES |
| green after restore | rc 0, suite 11 cases / 405 assertions | rc 0, 64 assertions |

**A mutation whose build fails reads as a passing test, and a case whose name
carries a comma reads as one too.** Both were paid for here. The first attempt at
this gate was named "… DEVICE identifiers, not the stale host vector"; `doctest`'s
`-tc` filter splits its argument on commas, so both legs returned
`assertions: 0 | 0 passed | 0 failed` at rc 0 and measured nothing. The rebuild
rc and the mutation's site count are therefore read rather than assumed, and the
assertion count is read beside them.

Whole suites at the same head on `thor:gpu0`, every rc read from the job's own
stdout:

| suite | rc | assertions |
|---|---|---|
| `test_qwen4_exp_layer_loop` | 0 | 523 / 523 |
| `test_qwen4_exp_cuda` | 0 | 351 / 351 |
| `test_qwen4_exp_cuda_reductions` | 0 | 160 / 160 |

`test_qwen4_exp_layer_loop` was rc **134** on the previous run of this wave, with
`365 | 365 passed | 0 failed` printed beside it — every assertion that RAN passed
and the binary still aborted, on a heap overflow in this wave's own test helper.
An assertion count is not an exit status.
