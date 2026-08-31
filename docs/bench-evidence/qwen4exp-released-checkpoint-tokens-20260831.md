# The RELEASED `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S artifact emits REAL TOKENS, 31 August 2026

W5s of [`MODEL-MM-QWEN4-EXP`](../../.agents/specs/qwen4-exp-flash-next.md),
[#2031](https://github.com/mudler/vllm.cpp/issues/2031). Issue OWED: `gh` on this
host returns `HTTP 403: Sorry. Your account was suspended` for API writes.

**The one-line result. The degenerate output is GONE.** On `origin/main`
`52f7ccbfc`, which carries W5r, the released checkpoint answers two different
prompts with two different, correct, prompt-dependent completions:

```text
prompt "The capital of France is"  (prompt_tokens=5) -> text " Paris. Given this fact, what is"
prompt "Water boils at"            (prompt_tokens=3) -> text " 100°C at sea level"
```

W5q, on `701606e51` (W5p but NOT W5r), got `"!!!!!!!!"` — token id 0 eight times —
from both prompts. **The cause was W5r's: `dense_attn::ResidentWeight` dropped the
load-time repack markers, so on an aarch64 i8mm host `kMatmulBTQuant` read
`block_q8_0x4` buffers as flat `q8_0`.** This run is the first to exercise that
fix on a host where `vt::cpu::QuantRepackActive()` is true.

**Nothing here is a benchmark, and nothing here is a token GATE.** Two prompts,
greedy, one repetition, no oracle comparison, on a shared box carrying four other
sessions' processes throughout. The completions are *plausible and correct-looking*;
no reference decoded the same prompts. No number below may be quoted as a speed.

## 1. What was run, and on what

`origin/main` at **`52f7ccbfc7d45673374895fbc6885eb1c2b6d3fd`**, unmodified except
for one read-only instrument (§4). `thor:gpu0`, pod `rc-worker-n8smh`, aarch64,
kernel `6.8.12-1021-tegra`, 14 cpus, Mem total 122 GiB.

**`thor` is the box the question required, and that is not incidental.** The
whole W5r chain is gated on `vt::cpu::QuantRepackActive()`, which is false off
aarch64 i8mm (`cpu_quant_repack_arm.cpp:275`). W5r was written and gated on x86,
where the chain is inert and reproduces nothing. This run asserted the precondition
on the box rather than assuming it:

```text
RESULT HOST ISA: i8mm PRESENT -- QuantRepackActive() is TRUE by default here, repack chain LIVE
RESULT MOUNT /workspace is cifs: YES
```

Built in the lease, CPU-only, `-j 4`: `cmake configure rc=0`, `ninja rc=0` in
**731 s**. No CUDA entered the build. **A build failure is not a test result**, so
every rc is read and the job refuses rather than proceeding.

**CONTENTION WAS PRESENT THROUGHOUT AND IS NOT ESTIMATED AWAY.** `ps` before the
first timing recorded another session's `vllm-server` (6361 s old, 4.4 % cpu), a
`vllm-cli`, a second `vllm-server` 51 h old, two `cc1plus` 56 h old, two `nsys`
and two `python3`. Loadavg ran **4.13 to 10.11** on 14 cores. Every wall time
below is an upper bound taken under that load.

## 2. The fixes were verified in the tree that was compiled, three times

Not assumed from a branch name. Read out of the source on the box, with the job
refusing (`exit 19`) if either count failed:

```text
RESULT FIXES IN THE COMPILED SOURCE: W5r marker lines=2 (expect 2), W5p IsBlockQuant=1 (expect >=1)
```

| fix | site | what it does |
|---|---|---|
| **W5r** `7a937db8a` | `dense_attn_block.h:235-236` | `t.repacked = w.repacked; t.elem_kn_repacked = w.elem_kn_repacked;` — the shared `ResidentWeight` stops dropping the load-time repack markers |
| **W5p** `cd49fd2d5` | `ops.cpp:161`, `cpu_qwen4_exp.cpp:165-169` | `vt::Qwen4ExpGatedResidual` forks on `IsBlockQuant` and routes a block-quantized mix weight through `MatmulBTQuant` |

The same two counts were checked in the worktree, again inside the shipped
tarball, and a third time against the extracted source before `cmake` ran.

## 3. The artifact, and that it is the pinned one

`/workspace/q4exp-bench/UD-IQ1_S/`, three shards,
10,946,624 + 49,990,818,368 + 22,544,696,352 = **72,546,461,344 bytes = 67.564 GiB**,
staged CIFS -> worker-local `/tmp` in **2762 s** (contended with the build, which
ran concurrently — so that figure is NOT a clean filesystem measurement and is not
offered as one).

Identity was asserted on the staged bytes inside the lease, because `checkpoints/`
is a sibling of `rc/` and invisible from a lease, and the same name on a different
mount is not the same bytes:

```text
RESULT ARTIFACT IDENTITY: shard1 sha256 VERIFIED = 88a1420825a9304063e882ada29d438263617f51ac8923d438d927496693bafd
```

Staged size was also required to equal 72,546,461,344 exactly (`exit 41` otherwise).

## 4. The instrument, and that it cannot be the thing that differs

The per-stage probe is **W7DIAG's, not this wave's** — reused verbatim from
`/workspace/q4exp-w7diag/diag.patch`, authored by the sibling wave on
[#1978](https://github.com/mudler/vllm.cpp/issues/1978). It reads bytes a kernel
has already written and writes nothing back. It is **compiled into every arm** and
inert unless `VT_Q4X_DIAG` is set, so all four arms below run a **byte-identical
binary** and the probe cannot be what separates them. It is not committed to this
branch.

## 5. The four arms

One build, one staged copy, four loads. Control and treatment interleaved
(A, B, Adiag, Bdiag) so a coherent B cannot be box drift.

| arm | env | load | server `VmRSS` HWM | system `used` peak |
|---|---|---|---|---|
| A | *(default: repack ON)* | 60 s | 77,526,020 kB = 73.93 GiB | 11,936,160 kB = 11.38 GiB |
| B | `VT_CPU_QUANT_REPACK=0` | 60 s | 76,750,336 kB = 73.19 GiB | 12,060,984 kB = 11.50 GiB |
| Adiag | `VT_Q4X_DIAG=1` | 45 s | 77,526,172 kB | 12,820,196 kB = 12.23 GiB |
| Bdiag | `VT_CPU_QUANT_REPACK=0 VT_Q4X_DIAG=1` | 45 s | 76,750,820 kB | 12,362,756 kB = 11.79 GiB |

**The two memory instruments are reported separately and labelled, because they
diverge by design.** The loader `mmap`s, so file-backed weight pages count in the
child's `VmRSS` but land in `buff/cache`, not in `used`. `used` is therefore the
one that discriminates a bf16 expansion (~95.4 GiB for the n-gram table alone),
and it stayed flat at **11-12 GiB** across all four arms against `buff/cache`
101-102 GiB. Keep-quant residency held; nothing expanded.

## 6. The responses, verbatim

**Arm A — default, repack ON, the arm that matters:**

```json
{"choices":[{"finish_reason":"length","index":0,"logprobs":null,"prompt_logprobs":null,"text":" Paris. Given this fact, what is"}],"created":1788155866,"id":"cmpl-0","model":"qwen4exp","object":"text_completion","usage":{"completion_tokens":8,"prompt_tokens":5,"total_tokens":13}}
```

```json
{"choices":[{"finish_reason":"length","index":0,"logprobs":null,"prompt_logprobs":null,"text":" 100°C at sea level"}],"created":1788155891,"id":"cmpl-1","model":"qwen4exp","object":"text_completion","usage":{"completion_tokens":8,"prompt_tokens":3,"total_tokens":11}}
```

The token ids, read off the `logprobs` request rather than inferred from the
decoded string — **eight distinct ids, none of them 0**:

```text
token_id:11751  token_id:13  token_id:15767  token_id:411
token_id:2029   token_id:11  token_id:1092   token_id:369
```

**Arm B — `VT_CPU_QUANT_REPACK=0`, the discriminator:** byte-identical text and
the identical eight ids. That is the correct outcome and is itself a result: the
repack is a performance transform, so with the marker fix in place the repacked
and portable paths **agree exactly**. Before W5r they did not, and that
disagreement was the defect.

## 7. Where the defect was, and that it is gone

The probe rows, identical in Adiag and Bdiag, beside the sibling wave's reading
on `701606e51` (**W5p but NOT W5r**, same box, same artifact):

| stage | pre-W5r (`w7diag`, job `b52d7ce8`) | **post-W5r (this run)** |
|---|---|---|
| `embed` | nan=0, l2=**0.473868** | nan=0, l2=**0.473868** |
| `stream.after_widen` | nan=0, l2=**0.947736** | nan=0, l2=**0.947736** |
| `stream.after_layer_0` | **nan=51200**, l2=0 | **nan=0**, min −0.211914 max 0.324219 l2=3.66967 |
| `hidden.after_mixer_collapse` | nan=12800 | nan=0 |
| `lm_head.input_hidden` | nan=2560, l2=0 | nan=0, min −32.75 max 161 l2=226.115 |
| `LOGITS` | nan=0, **zero=248320**, no maximum | nan=0, **zero=0**, min −9.89818 max 15.7873 l2=1338.92 |

`RESULT ARM Adiag PROBE first NaN stage:` is **empty** — no stage carries a NaN.

**Two things make this a mechanism and not a coincidence.** The `embed` and
`after_widen` rows are numerically *identical* across the two trees, so both runs
are provably the same workload up to the point of divergence — the pre-fix run is
a clean control rather than a different experiment. And the logits are now a
genuinely ranked distribution with a real margin:

```text
LOGITS v[0]=6.00011063  #0=[11751]15.7872601  #1=[6924]13.1099234  #2=[264]12.7006855
                        #3=[1259]12.0962963   #4=[279]11.9879637   #5=[12456]11.9694214
```

The argmax id **11751** is exactly the `" Paris"` token that came out of HTTP.
Before the fix the row was all zeros — no maximum — and `argmax` over it returns
index 0, which is `!` in this file's own vocabulary. That is the complete causal
chain from the dropped marker to the eight `!`.

## 8. What this does NOT establish

- **It is not a token gate.** No oracle decoded these prompts. llama.cpp is not a
  usable same-box control here: the sibling wave's llama.cpp arm exited **126**,
  and the recorded blocker is that it aborts in `build_delta_net_chunking` before
  loading a byte.
- **No speed number.** The box was contended throughout and every arm is n=1.
- **`logprobs` VALUES were not obtained, on either tree.** The request is accepted
  and the structure is built, but `token_logprobs` and `top_logprobs` came back
  all-`null`. That is **not** a serialized NaN: it is the
  `step == nullptr || step->empty() || step_token == nullptr` branch of
  `BuildCompletionLogProbs` (`serving_utils.cpp:130-135`), which emits the
  `"token_id:N"` fallback string alongside the nulls — exactly what both runs show.
  The engine produced no logprobs dict for those positions. So logprobs settled
  nothing here; §7's probe is what carries the answer. Owed below.
- **`VT_DEBUG_SAMPLED=1` produced no output** and this run cannot say why. The
  variable is real (`runner.cpp:3078-3086`) and its guard is
  `kDebugSampled && !toks.empty()`, stderr was captured, and the arms still logged
  zero `vt-debug sampled` lines. Treated as a **failed instrument contributing
  nothing**, not as evidence about the sampler. The token ids in §6 come from the
  `logprobs` array and the decoded text, which are independent of it.
- **Only the UD-IQ1_S arm ran.** The other six published quants remain unrun.
- **One sequence.** `num_reqs > 1` is still refused by name.

## 9. Reproduction

`rc` job `c3fc7aff-9b09-4d44-9561-d6324ea1a68f` on `thor:gpu0`, 05:10:27-06:03:39 UTC.
Job script and full 944-line log: `/workspace/q4exp-w5s/run.sh` and
`/workspace/q4exp-w5s-rc-worker-n8smh/out/`.

```sh
vllm-server --model <shard1>.gguf --device cpu --host 127.0.0.1 --port 8097 \
    --block-size 16 --num-blocks 128 --max-model-len 256 \
    --served-model-name qwen4exp --verbose
curl -H 'Content-Type: application/json' \
     -d '{"model":"qwen4exp","prompt":"The capital of France is","max_tokens":8,"temperature":0}' \
     http://127.0.0.1:8097/v1/completions
```

`--block-size 16` because the CPU attention backend accepts only multiples of 16.
