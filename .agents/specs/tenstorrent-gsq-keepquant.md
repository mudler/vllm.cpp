# Tenstorrent keep-quant arms for the GSQ-RCO Qwen3.8-27B GGUF census

Issue: `.agents/issues/BACKEND-TENSTORRENT/ISSUE-LOCAL-01M2YMVYBHPZ7VZW96BWKGBRG7.md`
(row `BACKEND-TENSTORRENT`).

Status: **DRAFT, 2026-09-13.** Spec-first: no implementation is in scope until
this file is committed and the row moves `READY`.

## Now

The two GSQ-RCO Qwen3.8-27B GGUF artifacts (the `IQ3_XXS`-mix and its Q4_K
companion) do not run end-to-end on the Tenstorrent keep-quant path today:
a third of the first file's tensors hit dtypes the TT admission set does not
name, and each such tensor either refuses by name or falls back to
expand-bf16 residency. This spec commits the widening that makes both files
run on-device, keep-quant, end-to-end.

## The admission chain, as verified on the tree (`a78fdba9e`)

The "TT admits only {Q4_K, Q5_K, Q6_K, Q8_0}" claim from the earlier research
pass describes **wave W3 of QUANT-GGUF-IQ-TENSTORRENT and is stale**. The
contradiction resolves against three real admission points:

1. **Device capability set** —
   `src/vllm/model_executor/model_loader/gguf_keep_quant.cpp:175-235`
   (`DeviceKeepQuantSupported`, `case kTENSTORRENT`). The set today is
   `{Q4_K, Q5_K, Q6_K, Q8_0}` (the W3 bit-exact decode set) **plus**
   `{IQ3_XXS, IQ2_XXS, IQ2_S, Q3_K}` — waves 1-3 of
   QUANT-GGUF-IQ-TENSTORRENT widened it — **plus** `{IQ3_S, IQ4_XS, IQ2_XS}`
   from waves 1-3 of this row (per-wave comments at :188-228) — **plus**
   `{Q2_K}` from wave 4 (per-wave comment at :230-236) — **plus** `{IQ1_S,
   IQ1_M}` from wave 5 (per-wave comments at :232-246), which closes the set.
   `tests/vllm/test_gguf_keep_quant.cpp` pins the set (:352-402); widening the
   predicate without widening the kernel reds it.
2. **Device dispatch** —
   `src/vt/tenstorrent/tenstorrent_keepquant.cpp:940-973`: `IQ3_XXS`,
   `IQ2_XXS`, `IQ2_S`, `Q3_K`, `IQ3_S`, `IQ4_XS`, `IQ2_XS`, `Q2_K`, `IQ1_S`,
   `IQ1_M` route to
   `MatmulBTQuantInt8DotKernel`
   **unconditionally (DEFAULT path, no env)**; `Q4_K/Q5_K/Q6_K/Q8_0` route
   to int8-dot only under `VT_TT_KEEPQUANT_INT8DOT` (opt-in since W4b,
   #3031), otherwise fall through to the W4a E=1 grouped arm
   (`MatmulBTQuantGroupedKernel`, registered set exactly
   `{Q4_K, Q5_K, Q6_K, Q8_0}`).
3. **On-core decodes** — `src/vt/tenstorrent/kernels/keepquant_kernel_code.h`:
   one ported `kq_vec_dot_*` per encoding
   (`iq3_xxs` :488, `iq3_s` :540, `iq4_xs` :617, `iq2_xxs` :664, `iq2_xs` :711,
   `iq2_s` :946, `q3_k` :999, `q2_k` :769, `iq1_s` :831, `iq1_m` :884, plus
   the four k-quants), selected by
   `enc_sel` 4..13 at `tenstorrent_keepquant.cpp:1857-1869`. This is
   why APEX-I-Nano runs
   IQ3_XXS/IQ2_S/IQ2_XXS/Q3_K on TT: they are admitted, dispatched, and
   decoded — the earlier "no IQ types admitted" pass read only the W3
   return list.

Env: `VT_TT_KEEPQUANT_INT8DOT` is **default off** for the four k-quant
encodings (the e2e anchor band failed on one non-tie flip; lever lands
op-level, default off, #3031) and **irrelevant** for the IQ set, which has no
grouped fall-through and therefore dispatches unconditionally.

## Census vs TT coverage (the GSQ-RCO IQ3_XXS artifact)

| Dtype | Count | TT today | Arm | Roles hit | Gap |
|---|---|---|---|---|---|
| F32 | 353 | yes (not block-quant) | — | norms | none |
| BF16 | 96 | yes | — | ssm params | none |
| IQ3_S | 97 | **no** | — | ffn, attn | **gap** |
| IQ3_XXS | 83 | yes | int8-dot (enc_sel 4, default) | ffn, attn | none |
| IQ2_S | 72 | yes | int8-dot (enc_sel 6, default) | ffn | none |
| IQ4_XS | 33 | **no** | — | ssm_out, attn | **gap** |
| IQ2_XS | 32 | **no** | — | ffn | **gap** |
| Q2_K | 28 | **no** (named owed, :184) | — | ffn, embd | **gap** |
| Q4_K | 19 | yes | int8-dot (env) / grouped (default) | embd | none |
| IQ1_M | 4 | yes | int8-dot (enc_sel 13, default) | ffn (tail) | none |
| IQ1_S | 4 | yes | int8-dot (enc_sel 12, default) | ffn (tail) | none |

Outstanding tensor counts: **198 of 622** block-quantized tensors
(IQ3_S 97 + IQ4_XS 33 + IQ2_XS 32 + Q2_K 28 + IQ1_M 4 + IQ1_S 4) are not
TT-resident-quantized. Q4_K (19) is already covered.

## Scope

**In.** Admit and arm on `kTENSTORRENT`, device-side (decode on-core, word
shadow residency, no expand-bf16 twin): `IQ3_S`, `IQ4_XS`, `IQ2_XS`, `Q2_K`,
`IQ1_S`, `IQ1_M`. Both GSQ-RCO Qwen3.8-27B files run keep-quant end-to-end,
token-verified, when this row is done.

**Out / non-goals.**
- MTP (multi-token prediction) variants of the checkpoint.
- Prism type 142 and any other fork-extended ggml type id (same disposition
  as NVFP4 type 40: a named, isolated extension).
- The `VT_TT_KEEPQUANT_INT8DOT` default-off lever for the four k-quants: it
  stays as W4b left it. This row widens the IQ/k-quant *census*, not the
  anchor-band disposition.
- Any performance claim: the decode f32-exact cost is recorded, the axis
  stays open (#1003).

## Upstream anchors

- **llama.cpp pin b10451** (`/tmp/llamacpp-b10451`): the ported decodes are
  the upstream `kq_vec_dot_iq3_s_q8_K`, `kq_vec_dot_iq4_xs_q8_K`,
  `kq_vec_dot_iq2_xs_q8_K`, `kq_vec_dot_q2_K_q8_K`,
  `kq_vec_dot_iq1_s_q8_K`, `kq_vec_dot_iq1_m_q8_K` (ggml-cpu
  `vec_dot_type = q8_K` for all six — the same q8_K activation pairing the
  existing eight encodings use). The `to_float` side of the same build is
  the golden-vector producer (`/tmp/lanegate-oracle/oracle-dump`).
- In-tree precedent for the pattern: waves 1-3 of
  QUANT-GGUF-IQ-TENSTORRENT (each wave = one `enc_sel`, one header decode,
  one predicate line, one sweep case, one golden header).
- vLLM's TT backend now exists ([`vllm-tt-plugin`](../oracles/vllm-tt-plugin.md), 2026-09-07, qwen35 registered) but serves HF weights only, so the llama.cpp k-quant/i-quant oracle registry row
  `llama-cpp` (pin b10451, gateable) is the secondary oracle for these
  decodes, and the b10451 greedy dump is the token denominator.

## Design

**Arm choice.** All six ride the **int8-dot kernel** (`MatmulBTQuantInt8DotKernel`):

- `IQ3_S`, `IQ4_XS`, `IQ2_XS`, `IQ1_S`, `IQ1_M` have no grouped decode and
  no fall-through — exactly the wave-1-3 situation; they dispatch
  unconditionally on the DEFAULT path.
- `Q2_K` is a **k-quant, not an i-quant**, but its vec_dot pairing is
  `q8_K` (same as Q3_K, which already rides int8-dot), so the same kernel
  shape serves it; the design assumption to validate first is that
  `kq_vec_dot_q2_K_q8_K`'s `scales` handling (the `q2_K` super-block scale
  layout) ports with the same 8-lane exactness. Risk §below carries the
  grouped-arm fallback if it does not.
- Dispatch order: assign `enc_sel` 8..13 (`IQ3_S`=8, `IQ4_XS`=9,
  `IQ2_XS`=10, `Q2_K`=11, `IQ1_S`=12, `IQ1_M`=13) — beyond 7 so the
  existing four are never renumbered (the ttnn program cache hashes the
  workload identity).

**Widening order** (tensor count × role impact):

1. `IQ3_S` (97, ffn+attn) — the single largest gap.
2. `IQ4_XS` (33, ssm_out+attn) — ssm_out residency decides whether the
   state path stays on-device.
3. `IQ2_XS` (32, ffn).
4. `Q2_K` (28, ffn+embd) — first k-quant-arm validation.
5. `IQ1_S` + `IQ1_M` (8, ffn tail) — sub-bit; land last, behind their own
   acceptance question.

Each step is independently landable and leaves the e2e leg one refusal
closer to green.

**Decode f32-exactness, P==1 requirement.** New arms inherit the existing
contract: ported `kq_vec_dot` verbatim, upstream 8-lane split preserved
(each `aux32[l] <= 2^24` keeps every per-lane f32 accumulate exact),
lane-sum/min-correction interleave preserved, compared red-first against
`vt::cpu::BlockVecDot`. Decode P==1 decode per output element (SFPU f32
dot) is the correctness floor the sweep pins; the whole-block sum is NOT
f32-exact and must not be re-derived (the :1414 warning).

**Capture safety.** Same rules as waves 1-3 / W4d:
- word shadows staged eagerly warm-first (`EnsureKeepQuantWords`); a
  capture-time miss refuses inside the same call — never stages during
  capture;
- no host-to-device writes inside the captured region; activation
  quantization stays on-core inside the launch so replay re-quantizes fresh
  activations (the #2812 class);
- shape variance reaches the kernel as runtime launch args; the eager warm
  step pays the one compile per page geometry;
- new `KeepQuantWordsPerBlock` word counts: `IQ3_S` = 129 B? — **verify
  from `ggml_get_type_traits` block sizes at implementation time**; stage
  zero-padded to the 128 B word grid exactly as the wave comments describe
  (98/66/82/110 B precedents).

**Refusal by name.** Anything deliberately deferred refuses with a message
naming the missing arm (`kQ2_K` precedent at gguf_keep_quant.cpp:184): the
predicate admits only what `keepquant_kernel_code.h` decodes, and the
refusal path names the dtype.

## Tests

1. **Red-first decode sweep** — extend the device-compiled sweep in
   `tests/vt/test_tenstorrent_backend.cpp` (the wave test cases at :5881,
   :6009, :6148 are the template): each new encoding gets a case that
   compares the on-core `kq_vec_dot_*` against `vt::cpu::BlockVecDot` across
   APEX decode shapes, bit-exact, on the DEVICE-COMPILED path.
2. **Golden vectors** — one header per dtype following
   `tests/vt/iq1m_golden_vectors.h` (produced by the pinned b10451
   `to_float`): `iq3s`, `iq4xs` (the `iq2xs_iq4xs_golden_vectors.h` header
   already carries reusable IQ4_XS decode vectors from the CUDA row),
   `iq2xs`, `q2_k`, `iq1s`, `iq1m`. Bit-exact per block.
3. **Admission test** — `tests/vllm/test_gguf_keep_quant.cpp`: widen the
   TT set pins in the same change as the kernels (the file reds otherwise,
   by design).
4. **Default-path dispatch test** — one case per new encoding asserting
   `MatmulBTQuantInt8DotKernel` serves it with `VT_TT_KEEPQUANT_INT8DOT`
   unset (the :5881 template).
5. **Production-reached gate** — the e2e leg: both GSQ files load and decode
   on TT keep-quant end-to-end through the loader, `ModelRegistry::Forward`,
   on the default configuration; tokens verified against the b10451 greedy
   oracle (`oracle-dump --greedy`, prompt tooling `/tmp/gsq_p0.i32` +
   `/tmp/genprompt.py`). IQ1_S/IQ1_M use the **near-tie disposition**
   (sub-bit encodings flip low-margin tokens; compare margin-ranked
   disagreements against the oracle's own greedy margins, as the IQ1_M
   golden header documents). This gate is what makes the work reached: a
   unit case that hand-builds the kernel measures the class, not the
   capability.

## Gates

1. Focused unit: decode sweep + golden vectors + admission set, red first.
2. Full backend suite: `tests/vt/test_tenstorrent_backend.cpp` green
   (device-leased run; the file needs the TT device).
3. E2E: both GSQ files token-verified vs the b10451 greedy oracle.
4. Perf: record the decode f32-exact cost per new arm (time-to-first-token
   and tok/s at the census shapes); the throughput axis stays **open**
   (#1003) — no ceiling declared, the ratio recorded, gap named.

## Risks

- **Q2_K is a k-quant**: scale layout differs from the i-quants; if the
  8-lane exactness argument fails for `kq_vec_dot_q2_K_q8_K`, fall back to
  a grouped-arm decode instead of int8-dot, and say so in the spec Outcome.
- **IQ1_S/IQ1_M sub-bit accuracy**: expect near-tie flips, not token-exact
  agreement. Disposition: near-tie vs the b10451 greedy margins; the IQ1
  arm may stay behind an opt-in like the `VT_TT_KEEPQUANT_INT8DOT`
  precedent if the default-path flip rate exceeds the anchor band — that
  decision is recorded in the Outcome, never inferred.
- **Word-count drift**: `KeepQuantWordsPerBlock` and the 128 B zero-pad
  table must be derived from `ggml_get_type_traits` block sizes, not
  copied from the wave comments (the comments' byte figures are per-32-
  block and easy to misread).
- **Predicate/kernel skew**: admitting before the kernel exists throws at
  first forward with the model resident — the exact failure
  `DeviceKeepQuantSupported` exists to prevent. Predicate and kernel land
  in the same commit, per wave.

## Stop conditions

- A `kq_vec_dot` port that cannot be made bit-exact against
  `vt::cpu::BlockVecDot` at the sweep shapes stops that encoding; the
  encoding stays refused-by-name and the rest of the order proceeds.
- The e2e leg blocked on device access stops the row with the unit gates
  recorded and the e2e named `Owed` — it never softens to a
  class-construction pass.
- The near-tie disposition for IQ1_S/IQ1_M needing a policy the repo does
  not yet have is a `NEEDS_DECISION`, not an inferred default.

## Owed

- The decode f32-exact cost measurement per new arm and the open #1003 axis.
- The Q2_K arm-choice confirmation (int8-dot vs grouped) — resolved in the
  Outcome, not silently.
- The MTP variants and prism type 142: refused by name, recorded here as
  deliberately deferred.
- If IQ1_S/IQ1_M land opt-in: the default-path e2e leg records which file
  tensor classes remain behind the lever, and the follow-up row id that
  closes it.

## Git integration

One pull request per the repository default; the spec lands first in the
same PR's commit history (spec commit precedes implementation commits).
