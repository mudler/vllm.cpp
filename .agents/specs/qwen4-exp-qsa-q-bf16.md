# The QSA block keeps `q` at the model dtype, as vLLM does

**Row:** `MODEL-MM-QWEN4-EXP` (wave QSABF16)
**Issue:** [#2488](https://github.com/mudler/vllm.cpp/issues/2488)
**State:** `ACTIVE`
**Base:** `origin/main` at `63889449c`

## Scope

`src/vllm/model_executor/models/qwen4_exp_qsa_block.cpp` splits the fused
`q_proj` output into an f32 `q` buffer and hands that to `vt::RmsNorm`. vLLM
keeps the same value at the model dtype, which for this architecture is bf16.
This spec narrows the query buffer to `hidden.dtype` and widens the shared
`vt::AttnGateSplit` seam by exactly the amount that requires.

**This is the cause of [#2477](https://github.com/mudler/vllm.cpp/issues/2477).**
[#2493](https://github.com/mudler/vllm.cpp/pull/2493) treated the symptom — it
taught the CUDA `RmsNorm` kernel to read a gamma whose dtype differs from the
activation — and said so in its own body and in
`.agents/specs/qwen4-exp-cuda-rmsnorm-weight-dtype.md`. That change stays. It is
correct on its own terms and it serves other callers. This change removes the
pairing that made it necessary here.

**In scope:** the query half. **Out of scope and recorded under `## Owed`:** the
output-gate half of #2488.

## Upstream anchors

vLLM implements this architecture. [#2502](https://github.com/mudler/vllm.cpp/pull/2502)
reconciled the row onto it, so vLLM is the primary oracle here.

**Every citation below is a FORWARD REFERENCE to an unpinned upstream.** The
pin in `.agents/upstream-sync.md` is `5559679229` (2026-07-26) and has no
`vllm/models/qwen4_exp/` at all — vLLM landed the architecture after the row was
pinned. The revision read for this spec is `origin/main` `cdefd9d499`
(2026-09-02), 1566 commits ahead of the pin. It is not a gateable oracle for this
row and is cited as a source of upstream *shape*, not of measured values.

| What | Where (`cdefd9d499`) |
|---|---|
| `Qwen4ExpQSAAttention(Qwen3NextAttention, ...)` — the QSA layer inherits the projection | `vllm/models/qwen4_exp/nvidia/qsa.py:168` |
| `self.attn_output_gate = True` for every Qwen4Exp full-attention checkpoint | `vllm/models/qwen4_exp/nvidia/qsa.py:230` |
| `self.q_norm = GemmaRMSNorm(self.head_dim, eps=...)` | `vllm/models/qwen4_exp/nvidia/qsa.py:254` |
| `q, gate = torch.chunk(q_gate, 2, dim=-1)` — a **view** of the qkv GEMM output; no dtype change | `vllm/model_executor/models/qwen3_next.py:430` |
| `q = self.q_norm(q.view(-1, num_heads, head_dim))` — `q` reaches the norm **unwidened** | `vllm/model_executor/models/qwen3_next.py:437` |
| `hidden_states.float()` … `.to(input_dtype)` — the promotion is INSIDE the norm | `vllm/models/qwen4_exp/nvidia/ple_layer.py:70,80` |

The last row is the whole argument. Upstream's `.float()` is a value promotion
inside one kernel's registers. It is not a materialised `[T, Hq, Dh]` allocation,
and the comment this change replaces read it as if it were.

## Design

**The seam.** `vt::AttnGateSplit(q, q_out, gate_out, qgate)` refused any
`q_out.dtype != kF32`, and `AttnGateSplitKernel` took `float* q_out` literally.
The op now accepts `q_out` at f32 **or** bf16 and templates the CUDA kernel on
that type. `gate_out` stays f32, because `vt::SigmoidGateBf16` — its only
consumer, on four backends — requires an f32 gate. Narrowing `gate_out` is the
`## Owed` half.

This is the extension the shared seam needed, not a parallel path. The op already
templated its INPUT on `Tin` for exactly the same reason (`VT_BF16_GEMM_OUT`
makes the `q_proj` GEMM emit bf16); the output was the half that stayed welded.

**The caller.** `q_f32` becomes `q_split` at `hidden.dtype`, and the norm reads
it. Two bf16 buffers replace one f32 and one bf16 buffer, which is upstream's own
shape: `torch.chunk` + `reshape` materialises the split at the model dtype and
`q_norm` returns a second tensor at the same dtype.

**Why the values cannot move.** `qgate` is allocated at `hidden.dtype` and the
`q_proj` GEMM stores into it, so the split's SOURCE is already bf16 on this
path. Widening each element to f32 and narrowing it back is the identity —
`F32ToBF16(BF16ToF32(x)) == x` for every bf16 `x`. The rounding count is
unchanged at one, at the `qgate` store, which is where upstream has its one.

## Risks

**Qwen3.5 shares the op.** `qwen3_5.cpp:5328` and `:5501` pass f32 `q_out`
buffers. They keep dispatching to the same `<float, Tin>` instantiation the
kernel had before, so the risk is a dispatch mistake, not a behaviour choice.
`## Tests` gates it directly rather than by argument.

**A token gate cannot see this.** `AGENTS.md` names the case: the tokens still
match while the path moves twice the bytes. Every gate below therefore observes a
dtype or a byte count, and the value gates exist only to prove nothing ELSE moved.

**The CUDA arm is unmeasured by CI.** No CI lane executes GPU tests. The CUDA
assertions in this change are `[SKIP]`ped documentation of intent unless a leased
device runs them, and `## Evidence` says which arms actually ran.

## Tests

1. **The production gate (`tests/vllm/models/test_qwen4_exp_layer_loop.cpp`).**
   A pass-through provider is installed for `OpId::kAttnGateSplit` on `kCPU` at a
   priority above `vt-native`; it records `q_out.dtype` and
   `q_out.Numel() * SizeOf(q_out.dtype)` and forwards to the native kernel
   captured before registration. The case then runs
   `vllm::ModelRegistry::Forward` over the GGUF fixture — whose layer 3 is
   `qwen_sparse_attention` — and asserts (a) the recorded call count is non-zero,
   (b) every recorded `q_out` is `kBF16`, and (c) its byte count is
   `T * kQHeads * kHeadDim * 2`.

   The count is the applied-ness property: deleting the QSA call site in
   `qwen4_exp_forward.cpp` takes it to zero, which is what makes this a
   reachability gate and not a class test.

2. **The seam's value identity (`tests/vt/test_ops_glue.cpp`).** Over a bf16
   `qgate`, the bf16 `q_out` arm is BIT-IDENTICAL to the f32 arm's values rounded
   to bf16, and `gate_out` is byte-identical between the two runs — the Qwen3.5
   operand, unchanged by the presence of the new one.

3. **The seam's refusals.** `gate_out` at bf16 and `q_out` at f16 each refuse by
   name.

4. **Qwen3.5 does not move.** `tests/vt/test_ops_attn_preamble.cpp` compares the
   unfused `AttnGateSplit + RmsNorm + RmsNorm + RopeNeox` sequence against the
   fused kernel byte-for-byte over f32 buffers, and is the right gate — but it
   opens `if (!HasCuda()) return;` and so gates nothing on a CPU-only build.
   `## Evidence` answers the question by counting Qwen3.5's executed calls
   instead, and reports what that measurement can and cannot support.

5. **The CPU behavioural control.** The released UD-IQ1_S artifact's CPU
   sequence `11751 13 15767 411 2029 11 1092 369` must not move.

   **RECONCILED 2026-09-03 by `KERNEL-GDN-CHUNKED-MIRROR`**
   ([#2612](https://github.com/mudler/vllm.cpp/issues/2612),
   [gdn-chunked-mirror.md](gdn-chunked-mirror.md)). That row moved the CPU GDN
   prefill default from the sequential recurrence to vLLM's chunked
   decomposition, so this control's PREMISE is gone: the ids are expected to
   move, and this row's gate does not own that change. The control still holds
   as written **with `VT_GDN_CHUNKED=0`**, which selects the arm it was measured
   on, and that is how to run it. The ids the new default emits have not been
   re-measured (it needs the 67.564 GiB artifact and a lease) and are owed by
   the mirror row, not by this one.

## Gates

```sh
cmake --build build -j 2 --target test_qwen4_exp_layer_loop test_ops_glue \
                            test_ops_attn_preamble test_qwen4_exp_qsa_block
ctest --test-dir build -R 'qwen4_exp_layer_loop|ops_glue|ops_attn_preamble|qwen4_exp_qsa_block' -V
scripts/agent-preflight.sh --staged
```

## Evidence

Measured on `mudler-ubuntu-box`, a CPU-only build (`cmake -G Ninja
-DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_BUILD_TESTS=ON`, no `nvcc` on this host),
at `origin/main` `63889449c` plus this branch. Every rc below is the literal exit
status of the command named.

### Red, then green

| # | What | Command | rc | Read |
|---|---|---|---|---|
| R1 | The seam refuses a bf16 `q_out` | `./build/tests/test_ops_glue -tc='attn_gate_split*'` at BASE | 1 | `vt: attn_gate_split: q_out/gate_out must be f32 at src/vt/ops.cpp:5092`; `assertions: 18 \| 16 passed \| 2 failed` |
| R2 | The production gate sees the f32 buffer | `./build/tests/test_qwen4_exp_layer_loop -tc='*MODEL dtype*'` at BASE | 1 | `CHECK( 256 == 128 )`, `calls seen through ModelRegistry::Forward: 1` |
| G1 | `test_ops_glue` | after | 0 | `15 passed`, `assertions: 192 \| 192 passed` |
| G2 | `test_qwen4_exp_qsa_block` | after | 0 | `13 passed`, `assertions: 5937 \| 5937 passed` |
| G3 | `test_qwen4_exp_layer_loop` | after | 0 | `10 passed`, `assertions: 352 \| 352 passed` |

R2 is the number this change exists for: the same tokens, `256` bytes where
upstream moves `128`.

### Mutations

| # | Mutation | Rebuild rc | Result |
|---|---|---|---|
| M1 | Delete the `RunQwen4ExpQsaBlockPaged` call in `qwen4_exp_forward.cpp` | 0 | `REQUIRE( 0 > 0 )` — `calls` goes 1 -> 0. The gate measures the production path, not a hand-built block. |
| M2 | (= R2) `DBuf q_split` back to `DType::kF32` | 0 | `CHECK( 256 == 128 )` |
| M3 | Drop the f32 arm from `AttnGateSplit`'s dtype check | 0 | The PRE-EXISTING `attn_gate_split: splits [q\|gate] per head` case — f32 in, f32 out, which is Qwen3.5's exact operand shape — reds. An existing gate covers the arm Qwen3.5 uses. |

Each mutation was applied to a tree that had just built green, each rebuild
returned 0 before the run, and the tree was restored to a clean `git status`
after each.

### Qwen3.5, measured rather than argued

`## Tests` item 4 named `test_ops_attn_preamble`, which does not run here. So the
question was answered by counting instead. A scratch `fprintf` counter was placed
in the CPU `AttnGateSplitKernel`, the affected binaries were rebuilt (rc 0), and
the counter was read. It was then removed and the tree restored to a clean
`git status`, and the rebuilt binary prints zero probe lines.

| Run | AttnGateSplit CPU calls | `q_out.dtype` | Suite |
|---|---|---|---|
| `test_qwen4_exp_layer_loop -tc='*MODEL dtype*'` | 1 | `2` = `kBF16` | pass |
| `test_qwen35_paged_forward`, `VT_FUSE_ATTN_PREAMBLE=1` (default) | 0 | — | 63/63 pass |
| `test_qwen35_paged_forward`, `VT_FUSE_ATTN_PREAMBLE=0` | **14** | `0` = `kF32`, all 14 | 63/63 pass |
| `test_qwen3_5_fa2_class`, either arm | 0 | — | 15/15 pass |

Three things follow, and only these three.

1. **Qwen3.5 still passes f32.** Fourteen executed calls, every one of them
   `kF32`, dispatching to the arm the kernel had before. Its goldens hold in that
   arm.
2. **The qwen4_exp buffer is bf16 at the production entry point**, read off the
   kernel rather than off the caller's own assertion.
3. **On its DEFAULT configuration Qwen3.5 does not reach this op on CPU at all.**
   `kAttnQkNormRopeGate` is registered on kCPU (`cpu_ops.cpp:4136`) and
   `FuseAttnPreambleOn` defaults true (`qwen3_5.cpp:1930-1934`), so the split is
   the `VT_FUSE_ATTN_PREAMBLE=0` rollback branch. Running the suite without that
   knob would have measured nothing and reported a pass — which is what the
   counter was installed to find out.

`test_qwen35_paged_engine` is UNRUN, not green: it exits 77 with
`*** GATE NOT RUN — SKIPPED (exit 77), this is NOT a pass ***` because the pinned
Qwen3.5-0.8B snapshot is not cached and its oracle was captured on gfx1100. That
is its pre-existing environment gate, not this change.

### What was NOT measured, and is not claimed

- **The CUDA arm did not compile.** This host has no `nvcc`, and the fleet's two
  x86 devices were held by other waves for the whole session (`thor:gpu0` by the
  #2509 verification this wave was told not to compete with). `cuda_glue.cu`'s
  templated dispatch is therefore reviewed and not built. The pull request's
  CUDA lane is the first thing that compiles it.
- **`test_ops_attn_preamble` is a SKIP wearing a PASS here.** It exits 0 with
  `assertions: 0` because both its cases open with `if (!HasCuda()) return;`.
  It is listed in `## Gates` because it is the right gate; on this build it
  gated nothing, and `## Tests` item 4 leans on M3 instead.
- **The released UD-IQ1_S control sequence was not re-run.** The artifact is
  71 GB across three shards on the NAS; this box had 61 GB available at a load
  average of 47, with four other waves building. Running it would have thrashed
  the box for everyone else and produced a number nobody could reproduce. The
  value argument is the bit-identity above, and the executable substitutes are
  G2 (5937 assertions against the oracle's own `Qwen4ExpTextAttention.forward`
  goldens) and G3 (352, including the forward goldens).
- **The rest of the suite.** `include/vt/ops.h` changed, so a full test build is
  1382 targets. At a load average of 45 on 20 cores with four other waves
  building, it advanced 4 targets in 5 minutes and was stopped rather than left
  to thrash the box. What ran instead is the set that touches the seam or the
  model: `test_op_provider` (497), `test_backend_cross_device` (3 — the rest of
  its cases are CUDA-gated), `test_qwen4_exp_forward` (429), `test_qwen4_exp_qsa`
  (7263), `test_qwen4_exp_qsa_device` (4697), `test_qwen4_exp_runner` (136),
  `test_qwen4_exp_ple_block` (112), `test_ops_fused_chain` (379),
  `test_fused_chain_additivity` (25), `test_ops_nvfp4_fp4` (919), all rc 0.
- **No throughput axis.** The change removes `T*Hq*Dh*2` bytes of traffic per
  QSA layer per step. That is arithmetic, not a measurement, and it is stated as
  arithmetic. Nothing here claims a speedup.

### One route that could have moved and does not

Narrowing `x` to bf16 makes `LaunchRmsNorm<__nv_bfloat16>` the CUDA
instantiation, which is the arm that can reach `TryLaunchRmsNormDecodeFast`.
It cannot here: that path returns false on `residual == nullptr`
(`cuda_ops.cu:395`) and again on `h < 1024` (`:396`), and the QSA `q_norm` call
passes no residual and `h = head_dim`. So the kernel is `RmsNormRowKernel` before
and after.

## Stop conditions

- Stop if narrowing `q_out` moves any Qwen3.5 byte. That would mean the dispatch
  is wrong, and no value argument rescues it.
- Stop if the CPU control sequence moves. The narrowing is an identity on this
  path; a moved token means the premise that `qgate` is bf16 is false somewhere.
- Stop and hand back if closing the gate half needs `vt::SigmoidGateBf16` widened
  on four backends. It does, which is why it is `## Owed` and not scope.

## Owed

- [#2488](https://github.com/mudler/vllm.cpp/issues/2488) stays open for its
  **output-gate half**: `gate` is still allocated f32. Closing it means widening
  `vt::SigmoidGateBf16`'s gate operand on kCPU, kCUDA, kVULKAN (a GLSL shader
  with dtype specialisation constants) and kTENSTORRENT, each of which states an
  f32 gate in its own refusal. That is a different unit of work with a different
  blast radius, and the value it buys here is zero rounding change — the same
  bandwidth argument and none of the precision one.
- The CUDA arm of every assertion in this change, until a CI lane executes GPU
  tests.

## Now

`ACTIVE`. The query buffer is narrowed; the gate buffer is `## Owed`.
