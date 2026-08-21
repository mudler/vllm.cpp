# Spec: ROCm Gemma-4 indexed MoE for T=2..63

- **Issue:** https://github.com/mudler/vllm.cpp/issues/838
- **Row slug:** `ROCM-GEMMA4-INDEXED-MAX-T` — child of `BACKEND-ROCM` (#41). Separate from #697.
- **Worktree / branch (this unit only):** `/home/don/llms/vllm.cpp-indexed-max-t` · `row/ROCM-GEMMA4-INDEXED-MAX-T`
- **Base / recipient:** `origin/main` `3ce5a1dc` `gemma4_moe.cpp:735` / `:1345`
- **Donor bytes:** `.agents/specs/rocm-gemma4-indexed-max-t-donor-*.log` slices hashed in `rocm-gemma4-indexed-max-t-donor.md` (dirty lab `/home/don/llms/vllm.cpp` HEAD `2bb4bd8a` **plus uncommitted**; HEAD is not a clean donor).
- **Implementer:** hermes-vllm. **Reviewer:** research. **Operator/smoke:** coord.
- **Git:** spec-only first (coord `25c9` / research `5071` / BLOCK `64cb`); impl after spec GREEN **on this same row branch**. Independent RED/GREEN from #837/#839. One PR per row. No shared `row/ROCM-GEMMA4-XDEV-MOE` landing history.
- **Supersedes for review:** `20332292` (BLOCK) and preview `c4fbe6e9` (not spec-GREEN).

## Now

`IMPLEMENTING` — d973/0f32 repair on this row. `7c416eb6` is not a review target.

**Not a confirmed fix.** Hypothesis (A) only: small-T **routing**. `9772` does not make this the T=2029 root. T=19 is **observed** on today's serial M1 route. Cause (serial M1 vs anything else) is **unconfirmed**. Do not call the serial path "racy" and do not say the T=19 hang "is this path."

## Upstream / source of the port

No vLLM Python equivalent. Source is the pinned donor slices. Product takes only the default-63 decode cap and the T<64 gate; **not** `PREFILL_INDEXED_NOSYNC` / `DEVICE_GROUP`.

| Tree | Indexed gate | T=2..63 path |
|---|---|---|
| `origin/main` `3ce5a1dc` | `gemma4_moe.cpp:735` `if (T == 1 && fp8_res && top_k <= 8 && top_k > 0)` | falls through to serial `:1345` `RunGemma4Fp8TopKOnExpertDevice` |
| hanging `vllm.cpp-bc64fa-r2` `1b1baf43` | `:735` same T==1 | same serial |
| donor slices | `:914-922` `VT_GEMMA4_DECODE_INDEXED_MAX_T` default 63; `:1096-1100` `indexed_ok_t` | per-token `ExpertGeGLUFp8TopKIndexed` / `RunGemma4Fp8TopKIndexedOnExpertDevice` |

Indexed **peer helper already exists on main**: `rocm_gemma4_experts.hip:543` `RunGemma4Fp8TopKIndexedOnExpertDevice`. Same-dev helper `vt::ExpertGeGLUFp8TopKIndexed` is already called at main `:775`. This row widens the **caller gate**, it does not invent a new kernel.

`kPrefillBatchMinT` stays 64 (`:980`). T≥64 remains prefill-batch (#839).

## Symptom this row owns

T=19 warmup / short prefill is **observed** to take the serial M1 peer path (`RunGemma4Fp8TopKOnExpertDevice`). Lab never enters that helper when `indexed_max=63`. Distinct from the T≥64 accumulation **class**. Cause unconfirmed.

## Scope

1. Parse once (cached `static const`):
   - `VT_GEMMA4_DECODE_INDEXED_MAX_T`: unset → **63**; `=1` → T=1 only; clamp `[1,63]`.
2. Replace the T==1 gate with:
   `T >= 1 && T <= indexed_max_t && T < kPrefillBatchMinT && fp8_res && top_k <= 8 && top_k > 0`.
3. T=1: keep hipGraph-stable TLS acc **and** TLS `rw_idx` (do not `pool-Release` either). Key `RwIdxTls` by `(compute_dev, T*top_k)`. T>1 keeps a per-call pooled `rw_idx_owned`.
4. T>1: owned `[T,H]` bf16 buffer, per-token existing indexed helpers (same-dev or peer).
5. Document the env in `docs/ENVIRONMENT.md` in the **implementation** commit.

## Out of scope

- **`ExpertGeGLUFp8TopKIndexedBatched`** and any packed T≥2 same-dev batch kernel. Lab has this at `:1154-1161`. Product already **REJECTED** that family. Do not port it.
- `VT_GEMMA4_PREFILL_DEVICE_GROUP`, `VT_GEMMA4_PREFILL_INDEXED_NOSYNC`, `kPrefillIndexedNoSyncMaxT`.
- Dual-slot `EscTls`/`AccFastTls` unless a host test proves T>1 + device hop needs them. Prefer the existing single T=1 TLS plus owned T>1 buffer.
- #837 GetBlas, #839 Launch/Finish, #697.

## Design

Honest boundary: **routing only**. The T=19 peer smoke uses `RunGemma4Fp8TopKIndexedOnExpertDevice` in a `for t` loop — the lab `else` branch, not the rejected batched same-dev arm.

`indexed_max_t` default 63 is a **product default change** for T=2..63 (warmups). It is not env-gated off. `=1` is the rollback to today's T==1-only gate.

T≥64 must still miss this gate so prefill-batch is unchanged.

## Risks

- Numerics: per-token indexed vs serial M1 may not be bit-identical. Token-exact Paris/arith is **necessary and not sufficient** (research `64cb` stop-ship 4).
- T>1 TLS acc sized `{T,H}` must not reuse the T=1 hipGraph buffer.
- Do not lower `kPrefillBatchMinT`.

## Tests

### Host predicate (required)

Extract a pure function or source+unit the env parse:

| T | env | expect |
|---|---|---|
| 1 | unset | indexed |
| 19 | unset (63) | indexed |
| 19 | `=1` | **not** indexed (serial) |
| 64 | unset | **not** indexed (prefill-batch) |
| 63 | unset | indexed |
| 0 / top_k>8 / !fp8_res | unset | not indexed |

RED: force the T==1 literal back → T=19 case fails.

### Direct tensor oracle (required on impl — this is the correctness gate)

Default-on numerical route change. Paris/arith can pass with hidden-state drift or indexing/ownership defects. Impl must ship a host-or-GPU oracle that compares **indexed output tensors** to **today's serial/reference math** for:

- T ∈ {2, 19, 63}
- same-dev arm **and** peer arm (`compute_dev != expert_dev`)

Oracle checks, all required:

1. All finite (no NaN/Inf).
2. Zero support: every exact-zero in the reference is exact-zero in the candidate (no 1e-6 floor that hides a miss).
3. Declared tolerance: `max_abs(cand-ref) <= max(abs_tol, rel_tol * max_abs(ref))` with **abs_tol and rel_tol written in the impl commit** (propose `abs_tol = 2^-7 * max_abs(ref)` class unless a tighter bound is proven; do not invent a looser floor).
4. Route witness: a test-visible flag/counter that the indexed helper ran (not serial M1, not packed-batched).
5. Owned-buffer lifetime: the `[T,H]` output buffer is still owned by the caller after return (no TLS alias, no free-before-read). Probe by writing a canary after return and re-reading the tensor.

RED mutations (must fail the oracle or the witness):

- wrong token stride (index `t` as `t*H` vs `t`);
- wrong `[T,H]` ownership (return a T=1 TLS pointer, or free before caller reads);
- accidental packed-batched dispatch (`ExpertGeGLUFp8TopKIndexedBatched` / any T≥2 packed kernel).

GPU (coord): T=19 generate succeeds; T=1 decode + Paris/arith unchanged. Generate is **not** a substitute for the tensor oracle.

## Gates

- Host predicate table GREEN without GPU.
- Tensor oracle GREEN for T=2,19,63 × {same-dev, peer} before any "A is GREEN" claim.
- Impl must not introduce `ExpertGeGLUFp8TopKIndexedBatched` or `PREFILL_INDEXED_NOSYNC`.
- Operator A/B (`5071`): this is **A**. Smoke **T=19 independently** first. p42k only after the smallest passing set.
- Default behavior outside Gemma-4 FP8 xdev `1<=T<64` eligibility is unchanged.
- `#697` files untouched.

## Stop conditions

- Research wants packed-batch after all — that is a new spec, not this one.
- Lab GPU smoke without coord.
- Landing this row on a shared branch with #837/#839.

## Evidence

Bus: `713f`, `a63e`, `25c9`, `5071`, `64cb`. Donor bytes hashed in `.agents/specs/rocm-gemma4-indexed-max-t-donor.md`.

## Open on gfx1201 hardware

Found while landing, and left open rather than papered over. Each needs the
RDNA4 pair this repository's maintainers do not have, so none of them can be
answered from a CPU host. The row does not reach `DONE` until they are.

1. **`rw_idx` capture-stability — SOURCE FIXED, GPU witness still owed.** T=1
   scaled router weights now live in `RwIdxTls` (`thread_local`, keyed
   `compute_dev` + `T*top_k`). Only T>1 `Release()`s a pooled `rw_idx_owned`.
   Host source invariants cover the shape. Owed on gfx1201: one decode
   hipGraph capture+replay witness that the baked `rw` pointer still matches
   the live TLS address (idle-beside `:8012`, never `:8010`).
2. **No measurement — protocol frozen to Researcher ca41; GPU after static GREEN.**
   Same `cff626f93`-derived binary (tree-identical to `9d4a16c5`), model, recipe,
   GPU placement, prompts, context, sampling, and exact active batch T:
   - A: `VT_GEMMA4_DECODE_INDEXED_MAX_T=63`
   - B: `=1` (T>1 host-gather). Separate processes (env once-cached).
   - T={2,8,63}; prove realized shape + selected arm in logs/counters.
   - Correctness first: deterministic greedy per-request token IDs A vs B
     must match before any timing is accepted.
   - Isolated `:8012`; never `:8010`; idle window; alternate AB/BA by T.
   - Exclude model load and first-request warmup. ≥3 warmups + ≥5 measured
     steady repeats/arm. Median **and range** for decode tok/s and per-step
     latency (not e2e load).
   - Record binary SHA, HEAD, ROCm/compiler, env, GPU mapping, prompt /
     context / output lengths, raw samples, thermal/clock, teardown.
   - Do not call a noisy single run a default-flip win.
3. **The per-expert scale costs one kernel launch per token.**
   `gemma4_moe.cpp:766-773` calls `vt::ApplyExpertScaleRw` inside the T-loop, up
   to 63 launches. `ApplyExpertScaleRwKernel`
   (`rocm_fp8_channel_gemv.hip:635`) is `<<<1, G>>>` and indexes `rw`/`ri`
   linearly with `ri` holding global expert ids, so one call with
   `G = T*top_k` is element-wise identical. `Gemma4IndexedOkT` bounds
   `top_k <= 8` and `T <= 63`, so `G <= 504` fits one block today — but
   collapsing the loop removes the structural reason `G` stays small, and
   `ApplyExpertScaleRwRocm` does not check `G` against the 1024-thread block
   limit or report a failed launch. The collapse therefore owes that guard.
4. Smaller, same hardware: `Gemma4IndexedHelperHits()` does a global atomic
   `fetch_add` per token on the decode hot path purely so a host test can
   observe it; `Gemma4IndexedScratchValidForT` cannot return false in the T>1
   branch that calls it; the new `RestoreComputeDev`
   (`rocm_gemma4_experts.hip:572`) adds a second `hipSetDevice` on a success
   path whose own comment says "no hipSetDevice between stream ops"; and
   `RetireGemma4Fp8TopKIndexedPeer` returns `ok = true` while skipping the
   expert-stream drain whenever `tls.edev != expert_dev`.
5. The six `Gemma4Indexed*` host helpers at `gemma4_indexed_gate.h:113-160` have
   no production caller and belong under `tests/`. Their "tensor oracle" is
   `s = (e+1) * rw[g]`, which models none of the FP8 GeGLU arithmetic, so it
   gates the T-loop's striding and nothing about the kernel. That is a real
   thing to gate; the header is the wrong place to keep it, and the name
   oversells it.
6. A comment string is load-bearing: the source-invariant case asserts
   `retire-before-acc_idx-dtor` appears in `gemma4_moe.cpp`. Reordering the
   statements it names while keeping the comment passes, and deleting the
   comment while keeping the order fails. The other invariants in that case
   assert on code.
