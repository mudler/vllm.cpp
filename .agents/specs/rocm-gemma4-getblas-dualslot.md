# Spec: ROCm GetBlas dual-slot TLS (Gemma-4 peer-MoE)

- **Issue:** https://github.com/mudler/vllm.cpp/issues/837
- **Row slug:** `ROCM-GEMMA4-GETBLAS-DUALSLOT` — child of `BACKEND-ROCM` (#41). Not a new KERNEL family. Separate from #697 / `KERNEL-ROCM-GEMMA4-BC64-FA-PREFILL`.
- **Worktree / branch (this unit only):** `/home/don/llms/vllm.cpp-getblas` · `row/ROCM-GEMMA4-GETBLAS-DUALSLOT`
- **Base / recipient:** `origin/main` `3ce5a1dc` `src/vt/rocm/rocm_matmul_hipblaslt.hip:72-99`
- **Donor bytes:** `.agents/evidence/rocm-gemma4-getblas/getblas-fn-67-99.txt` SHA256 `9df2b163bc817db0d9545570136666c8e07a0bb600a01e50288a8f78c4148c51` (dirty lab `/home/don/llms/vllm.cpp` HEAD `2bb4bd8a` **plus uncommitted**; HEAD is not a clean donor).
- **Implementer:** hermes-vllm (lab). **Reviewer:** research (spec then impl). **Operator/smoke:** coord-help-20260812.
- **Git:** spec-only first (coord `25c9` / research `5071` / BLOCK `64cb`); implementation is a later commit **on this same row branch** after spec GREEN. Independent RED/GREEN from #838/#839. One PR per row. No shared `row/ROCM-GEMMA4-XDEV-MOE` landing history.
- **Supersedes for review:** `20332292` (BLOCK) and preview `c4fbe6e9` (not spec-GREEN). Those SHAs lived on a combined branch and are not review targets.

## Now

`SPIKE` — spec for review. No product code in this commit.

**Not a confirmed fix.** `9772` is an accumulation failure class. This row is hypothesis (B) only. Route observed: T≥64 prefill-batch peer path calls `MatmulBT` → `GetBlas` on the expert queue. Cause (single-TLS destroy/create vs cache vs Launch/Finish) is **unconfirmed**. Do not label this a common root with #838 or #839.

## Upstream / source of the port

vLLM has no hipBLAS TLS. Source is the pinned donor slice above, cited as a **file:line + hash**, not a tree transplant.

| Tree | `GetBlas` | Shape |
|---|---|---|
| `origin/main` `3ce5a1dc` | `src/vt/rocm/rocm_matmul_hipblaslt.hip:72-99` | `static thread_local Tls tls`; on `tls.dev != device` → `hipblasDestroy` + `hipblasCreate` |
| hanging `vllm.cpp-bc64fa-r2` `1b1baf43` | `:72-99` | same single TLS |
| donor slice (hash above) | lab `:67-99` | `static thread_local Tls tls_slots[2]`; `Tls& tls = tls_slots[(device == 1) ? 1 : 0]` |

Callers (`MatmulBT` / `MatmulBTAlphaBeta` / Lt variants in the same file) stay unchanged. T≥64 prefill-batch peer path reaches `GetBlas` via `vt::MatmulBT` on the expert queue. Serial T=19 / `ExpertGeGLUFp8TopKM1` does **not**.

## Symptom this row owns

Coord `9772` / `25c9`: 274 matched `moe_prefill_peer_helper` BEGIN/END, then accumulation wedge. Hypothesis (B): single-TLS destroy/create on the hop is shared-resource churn for the **batch** path. This row does not claim to fix T=19 and does not claim T=2029 will generate from this change alone.

## Scope

Replace only the TLS storage in `GetBlas`:

```c
static thread_local Tls tls_slots[2];
Tls& tls = tls_slots[(device == 1) ? 1 : 0];
```

Keep the existing capture/`hipGetDevice`/`hipSetDevice`/`hipblasSetStream` body **inside the selected slot**. Device ids other than 0/1 share slot 0 (same as donor).

## Out of scope

- Launch/Finish / `PeerPipeTls` / `DequantCacheSlotFor` (#839).
- Indexed T<63 routing (#838).
- hipBLASLt product default, FP8×FP8 Lt, `VT_ROCM_HIPBLASLT`.
- `#697` / any edit of `rocm_paged_attn.hip`.
- Diagnostic `STAGE_SYNC` / `PREFILL_TRACE`.
- More than two slots; devices ≥2.


## Adjacent upstream (not this row)

- **#785** (joral, OPEN issue): host-dead `VT_ROCWMMA_OK` around SharedK launch. Hard same-hunk landing-order overlap with **#697**, not with GetBlas. Do not patch that guard here.
- **#523 / #509** (VikashLoomba, OPEN drafts): custom keep-quant grouped GEMM + `rocm_moe_chain.hip`. Path intersection with this row is `docs/FEATURES.md` + `docs/USAGE.md` only. Their expert GEMM does not call `GetBlas` / `ProductGetBlasHandle`.
- **#834** (unowned): router-lookahead prefetch. Adjacent cache policy only; not TLS lifetime.


## Design

1. Two process-lifetime per-thread handles. Hop 0→1 must not destroy GPU0's handle while GPU0 GEMMs may still be queued (donor comment at `:74-75`).
2. Create still happens lazily per slot on first use.
3. Stream bind remains per-slot (`tls.stream != stream` → `hipblasSetStream`).
4. Default ON inside `GetBlas` only. No new env. No behavior change outside hipBLAS handle lifetime on device 0/1 hops.
5. **Lifetime invariants:**
   - slot[i] handle is destroyed only when that slot is recreated for a **different** `tls.dev` than its index (should not happen if index is a function of `device`);
   - hopping 0→1 must leave slot[0].handle live (same pointer identity);
   - hopping 1→0 must leave slot[1].handle live;
   - `hipSetDevice` after a hop restores `device` before create/setStream;
   - capture path still skips setDevice.

## Risks

- Slot index `device==1` is a two-GPU lab assumption. A third visible device aliases slot 0. Named; do not invent a map.
- Does not by itself make T=2029 generate. Land before or with #839 as a **separate** immutable head.

## Tests

Text search for `tls_slots[2]` / `(device == 1)` is **not** sufficient (research `64cb` stop-ship 5). Tests must observe the new lifetime guarantees.

### Host load-bearing seam (required)

Extract or wrap:

1. `GetBlasSlotIndex(int device) -> 0|1` (`device==1 ? 1 : 0`).
2. A test-only lifecycle recorder (fake `hipblasCreate`/`Destroy`/`SetStream`/`SetDevice`, or a friend/hook compiled into `tests/vt/test_ops_getblas_dualslot.cpp`) that records, per slot: create count, destroy count, last handle identity, last bound stream, last `hipSetDevice` argument, whether the call was under capture.

Host cases (no GPU required if the seam is fakeable):

| Case | Expect |
|---|---|
| first use dev0 | slot0 create==1, slot1 create==0 |
| then hop 0→1 | slot0 destroy==0, slot1 create==1, slot0 handle identity unchanged |
| then hop 1→0 | slot1 destroy==0, slot0 create still 1 (no recreate), slot1 handle identity unchanged |
| then hop 1→0→1 | both handles survive; no extra destroy |
| stream change on slot0 | `SetStream` on slot0 only; slot1 stream untouched |
| capture path | no `hipSetDevice` |

RED mutations (must fail the table):

- swapped selector (`device==0` → slot 1);
- destroy-on-hop (old single-TLS `if (tls.dev != device) hipblasDestroy`);
- missing stream rebind (`tls.stream != stream` branch deleted);
- capture-path `setDevice` (setDevice runs even when `StreamIsCapturing`).

### Coord GPU probe (mandatory on impl, not this spec commit)

On dual visible devices: call `GetBlas(0,s0)`, `GetBlas(1,s1)`, `GetBlas(0,s0)`, `GetBlas(1,s1)` and assert handle pointer identities: 0→1→0 keeps the first GPU0 handle; 1→0→1 keeps the first GPU1 handle. Skip only when `HIP_VISIBLE_DEVICES` empty. This does **not** replace the host seam.

## Gates

- Host seam table GREEN without a GPU.
- T=1 decode + Paris + arith on the post-impl binary unchanged vs pre-change KEEP class.
- Operator A/B (`5071`): this is **B**. Run independently (or after A) on **T=2029**; do not bundle with C. p42k only after the smallest passing set.
- `#697` files untouched (`git diff` must not list `rocm_paged_attn.hip` or bc64 tests).
- Default path outside Gemma-4 FP8 xdev `GetBlas` hops is unchanged.

## Stop conditions

- Research BLOCK on this spec.
- Any attempt to transplant dirty-lab hipBLASLt / FP8 Lt / layer-split with this slot change.
- GPU smoke by lab without coord ownership.
- Landing this row on a shared branch with #838/#839.

## Evidence

Bus: `82b2`, `713f`, `9772`, `25c9`, `5071`, `64cb`. Donor bytes hashed in `.agents/evidence/rocm-gemma4-getblas/MANIFEST.md`.
