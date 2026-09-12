# Spec: qwen4_exp on ROCm — the nine operations with no arm

- Issue: `ISSUE-LOCAL-01M2A1DTCZQVAH7M193XT9PN2V`
- Row: `MODEL-MM-QWEN4-EXP` (`.agents/model-matrix.md`, `ACTIVE`). Owning
  backend row: `BACKEND-ROCM` ([#41](https://github.com/mudler/vllm.cpp/issues/41)).
- Claim: `CLAIM-MODEL-MM-QWEN4-EXP` (row-level, already held)
- Base: `51c248190` (`origin/main`, 2026-09-12). Rebased from `97cb6964b`
  on 2026-09-12; the three commits that arrived are #2959's Tenstorrent work
  (`.agents/backend-matrix.md`, `src/vt/tenstorrent/`, the TT spec) and touch
  nothing this row reads.

## Contract

| Field | Value |
|---|---|
| Scope | IN: a `kROCM` arm for each of the nine `vt::` operations the `qwen4_exp` forward calls and ROCm does not register — `kQwen4ExpPleConv`, `kQwen4ExpPleGate`, `kQwen4ExpGatedResidual`, `kQwen4ExpGatedResidualWriteBack`, `kQwen4ExpQsaCompress`, `kQwen4ExpQsaGatherAttention`, `kRmsNormGroup`, `kIndexSelect`, `kIndexCopy` — each with its cross-device gate against the CPU oracle, and the device-fit and refusal records that change with them. OUT: the MTP head (still unimplemented on every device); the vision tower (no ROCm-specific work, and the only published GGUF is text-only); multi-sequence decode (the engine clamps `--max-num-seqs` to 1 for this model on every device); any token-exactness claim; any throughput, latency or memory number. |
| Upstream chain | vLLM is the primary oracle and at the current pin `e126687a9a` it ships a FIRST-PARTY AMD backend for this architecture: `vllm/models/qwen4_exp/amd/`. Its divergence from `nvidia/` is about 840 lines confined to the Triton op layer, with `model.py`, `model_state.py` and `mtp.py` BYTE-IDENTICAL, and it carries `amd/ops/qsa.py` (6 `@triton.jit` kernels including `_compress_qsa_groups_kernel` and `_qsa_sparse_paged_gqa_splitk_kernel`), `amd/ops/hc.py`, `amd/ple_layer.py`, `amd/hyperconnection.py` and `amd/indexer_qsa.py`. `vllm/models/qwen4_exp/__init__.py` states the support surface in its own words: "Qwen4Exp currently supports CUDA and ROCm only". So the AMD arm is a MIRROR, not an extension, and not a transliteration of the CUDA arm. |
| Our baseline | Every one of the nine has BOTH a CPU and a CUDA implementation in this tree, which is what bounds the work: `cpu_qwen4_exp{,_ple,_qsa}.cpp` and `cuda_qwen4_exp{,_ple,_qsa}.cu`, `cuda_rms_norm_group.cu`, and `kIndexSelect`/`kIndexCopy` in `cpu_ops.cpp` and `cuda_gdn.cu`. About 1709 lines of CUDA across the four qwen4_exp files. `grep -rn 'Qwen4Exp' src/vt/rocm/` returns nothing. |
| Port map | Each CUDA kernel -> a `.hip` sibling under `src/vt/rocm/`, mirroring the vLLM file structure as AGENTS.md requires: `cuda_qwen4_exp.cu` -> `rocm_qwen4_exp.hip`, `cuda_qwen4_exp_ple.cu` -> `rocm_qwen4_exp_ple.hip`, `cuda_qwen4_exp_qsa.cu` -> `rocm_qwen4_exp_qsa.hip`, `cuda_rms_norm_group.cu` -> `rocm_rms_norm_group.hip`; `kIndexSelect`/`kIndexCopy` join an existing ROCm TU rather than earning a file, and W2 put them in `src/vt/rocm/rocm_gdn_state.hip` — the TU that already carries the other half of `cuda_gdn.cu`'s indexed data movement. They SELF-REGISTER there in W1's pattern rather than taking an entry in `rocm_ops.hip`, because that file is a surface every ROCm row writes and AGENTS.md "Records" calls such a file a lock. **Transcribe from the CPU bodies where the CUDA one uses a primitive RDNA lacks, and from the CUDA one otherwise** — measured, the four qwen4_exp CUDA files use only `__shfl_down_sync`, `atomicAdd` and bf16/fp16 types, with no `__dp4a` and no CUDA-only intrinsic, so most port directly. Read vLLM's `amd/` Triton kernels for the ALGORITHM where ours and theirs disagree, because that is the mirror source. |
| Tests to port | vLLM has no C++ test to port. The gate is INHERITED and must not be re-authored: `tests/vt/test_backend_cross_device.cpp` already holds any registered backend to NMSE <= 5e-4 against the CPU oracle, and the sibling cases REQUIRE-prove registration rather than skipping. Each op gains a case there in that shape. The existing `tests/vllm/models/test_qwen4_exp_*_device.cpp` suites are the per-op golden surface. |
| Gates | G1 per-op cross-device NMSE on `strix:gpu0`. G2 the model LOADS and the forward completes on ROCm with `VT_OP_PROVIDER_STATS=1` showing ZERO reference-tier hits — see D2, on this board a hit is impossible, so a non-zero count means the tier was somehow installed and the run is void. G3 first tokens. G4 token-exactness against an oracle — see `## Owed`, not this row. **No throughput, latency or memory number is admissible until G4.** |
| Dependencies | The IQ4_NL ROCm GEMM (`QUANT-GGUF-IQ4_NL`, PR #3149) for the `ffn_down_exps`, and #3097's ROCm gather for the n-gram table — BOTH have landed and are in this branch's base: the gather in #3097, the GEMM at `18e2a8c9a` on `origin/main`. Hardware: `strix:gpu0`, the only AMD fleet device, reachable ONLY through an `rc` lease. No CI lane has an AMD runner, so a green CI is not evidence for any arm here. |
| Work breakdown | `W0` this spec (LANDED) -> `W1` (LANDED) the two elementwise-shaped ops (`kQwen4ExpGatedResidual`, `kQwen4ExpGatedResidualWriteBack`) plus `kRmsNormGroup`, which are the cheapest and prove the file and registration shape -> `W2` (LANDED) `kIndexSelect`/`kIndexCopy`, the two GENERIC row gather/scatter helpers, which is why D3c records that D3's tie-break has nothing to arbitrate for them -> `W3` (LANDED) the PLE pair (`kQwen4ExpPleConv`, `kQwen4ExpPleGate`), where vLLM's AMD backend DOES define both behaviours and D3d records what the comparison found -> `W4` the QSA pair, the hardest, and the one where vLLM's `amd/ops/qsa.py` is the reference rather than the CUDA arm -> `W5` first load and forward on `strix:gpu0` -> `W6` first tokens. Each wave lands with its cross-device case; no wave lands unreached. |
| Risks/decisions | R1 a missing op HARD-REFUSES on this board rather than degrading (D2), so a partial port is not a slow model, it is the same refusal with a different name. R2 wave size: nine ops in one pull request would be unreviewable, and the waves above exist to keep each reviewable. R3 the QSA pair is a gather consumer and not a mask, so a fixture under 2048 tokens of context cannot distinguish a correct port from one attending pooled keys — that bound is stated in `.agents/specs/qwen4-exp-flash-next.md` and applies here. R4 `strix:gpu0` is a single shared device and every gate here needs it. R5 no AMD runner in CI. |

## D1. Why this row exists now rather than later

Two of the three things that stopped `qwen4_exp` reaching an AMD device are
already gone. [#3097](https://github.com/mudler/vllm.cpp/pull/3097) landed the
ROCm quantized gather, so `DeviceQuantGatherSupported(kROCM)` is true and the
loader no longer refuses the device by name at `qwen4_exp_weights.cpp:665`. The
IQ4_NL keep-quant GEMM gives the 48 `ffn_down_exps` a device arm. **The weights
are now loadable and multipliable on ROCm and the forward still cannot run**,
and these nine operations are the whole of the difference.

The artifact also fits, which it did not a day ago. `strix:gpu0` was re-carved
to 96 GiB on 2026-09-11 and plain `hipMalloc` reaches at least 76 GiB there,
measured by a bounded probe under an `rc` lease; the released `UD-IQ1_S` is
67.56 GiB.

## D2. A missing op on gfx1151 is a refusal, not a slow path

**This is the single most important fact in this spec and it is
counter-intuitive**, because Strix Halo is an APU and this tree does carry a
portable CPU reference tier. It does not apply here. The chain, each link read
in the tree rather than inferred:

- `ReferenceTierEligible` (`src/vt/op_provider.cpp:906`) gates on
  `DeviceMemoryIsHostAddressable()`, deliberately NOT on `UnifiedMemory()`.
  That narrowing cost two crashes, #844 and #1435.
- `RocmBackend::DeviceMemoryIsHostAddressable()` returns `unified_memory_`
  (`rocm_backend.hip:484`).
- `ResolveMemoryPolicy` (`include/vt/rocm/rocm_arch.h:192`) computes
  `unified_memory = managed_alloc || (pageable_memory_access && integrated)`.
- gfx1151 reports `pageableMemoryAccess = 0`, MEASURED on the device, and since
  the #2511 narrowing `managed_alloc = pageable_memory_access` under `kUnset`.

So `unified_memory_` is false, the tier is never installed, and `GetOp` refuses
by name. `rocm_backend.hip:313` says as much: "a discrete AMD board never
installs the tier in the first place" — post-#2511 this APU behaves like one for
this purpose.

**W1 MEASURED THIS RATHER THAN LEAVING IT ARGUED.** On `strix:gpu0`, with
`vt::RmsNormGroup`'s ROCm registration neutralised in a scratch copy and the
kernel itself left in place, a direct call on a ROCm queue throws
`vt: no kernel for op RmsNormGroup (id 142) on device rocm (type 5), and the
portable CPU reference tier is NOT eligible: ... this ROCm device reports
hipDeviceAttributePageableMemoryAccess = 0`. The same call on the unmutated
tree returns with `GetReferenceTierHits()` unchanged. So the chain above is not
a reading of four files, it is an observed refusal on the board, and the tier is
not merely unused there — it cannot be reached. Restored byte-for-byte
afterwards.

**The coupling is worth stating on its own, because nothing records it:** the
repair that stopped the gfx1151 GPU hang (#2511), by disabling managed
allocation, also removed the CPU reference tier on that board. Spec prose
elsewhere in `.agents/specs/` still asserts host-addressability is true on
gfx1151; that is pre-#2511 and stale.

Two consequences for how this row is planned. A partial port buys nothing
runnable, so the waves are ordered to reach a forward as early as the
dependencies allow rather than to maximise ops landed. And `VT_OP_PROVIDER_STATS`
showing zero reference-tier hits is not a target to work toward here; it is a
precondition that holds by construction, and a non-zero count means something
installed a tier that should not exist and the run is void.

## D3. vLLM's AMD backend is the mirror, not the CUDA arm

AGENTS.md requires mirroring vLLM wherever it defines behavior. For this
architecture vLLM defines an AMD behavior explicitly, so "port our CUDA kernel
to HIP" is the wrong default for any op where the two disagree. The AMD backend
is pure Triton over the standard vLLM linear methods — `amd/low_latency_gemm.py`
is a 376-byte no-op that keeps them — with no AITER and no MFMA dependency,
which also makes it plausible on RDNA where MFMA does not exist at all.

Where our CUDA arm and vLLM's AMD arm agree, port ours: it is already gated
against our CPU oracle and carries this tree's conventions. Where they disagree,
vLLM wins and the spec records the difference.

## D3a. What W1 read in vLLM's AMD backend, and the three differences

D3 says the mirror decides wherever it defines behaviour, so W1 read
`vllm/models/qwen4_exp/amd/ops/hc.py` at the pin `e126687a9a` before porting.
The mirror covers all three of this wave's ops:
`_grouped_gemma_rmsnorm_kernel` is `vt::RmsNormGroup`, `_hc_silu_kernel` and
`_hc_gate_mix_kernel` are stages 2 and 3 of `vt::Qwen4ExpGatedResidual`, and
`_hc_combine_kernel` is `vt::Qwen4ExpGatedResidualWriteBack` with its injection
scaling fused in.

**The algorithms agree.** The division by `hc_count` sits inside the SiLU and
not after it, the mix is a mean over the hc branches and not a sum, the gate
multiplies the NORMED stream and not the raw one, the injection is
`2 * sigmoid(logits / hc_count)`, and the grouped norm puts eps inside the
rsqrt over the mean square. Every one of those is what our CPU oracle and our
CUDA arm already compute, so D3's "where they agree, port ours" applies and the
ROCm arms are transcriptions of `cuda_qwen4_exp.cu` and `cuda_rms_norm_group.cu`.

Three differences exist and none of them is an algorithm difference. They are
recorded here rather than resolved silently:

1. **The Gemma affine's spelling.** vLLM writes `y = x*rrms; y += y*w` and says
   in its own comment that the form exists "to lower to an FMA". Ours is
   `(x*rrms) * (1 + w)`, which is what transformers writes at
   `modeling_qwen4_exp.py:177` and what the CPU arm computes. The two agree in
   exact arithmetic and are NOT bit-identical: once `|w| < 2^-24`, `fl(1 + w)`
   rounds `w` away entirely, so the difference is a real one and not only
   associativity. **Ours is kept, and it is also the mirror.** vLLM's own
   PyTorch reference for this norm writes
   `return (normalized * (1.0 + self.weight.float())).to(input_dtype)` at
   `vllm/models/qwen4_exp/common/hyperconnection.py:87`, in the `common/` file
   that both the `nvidia/` and `amd/` layers import. Our spelling is therefore
   what upstream DEFINES as the behaviour, and the Triton `y += y*w` is an FMA
   lowering of that same behaviour rather than a second one. The measurement
   agrees: the CPU arm is the only oracle that can fail this arm — G1 is NMSE
   against it — and adopting the lowering hint would move the ROCm arm away from
   that oracle AND away from upstream's own reference while matching nothing. A
   lowering hint is not a behaviour, and D3 is about behaviour.
2. **A shared `[GROUP_DIM]` norm affine.** vLLM's kernel admits one
   (`W_SHARED`) beside the full `[DIM]` layout. `vt::RmsNormGroup`'s contract
   requires the full-row weight (`src/vt/ops.cpp:1228`), so the shared form is
   not expressible at this op at all. It is OWED, not implemented: no
   `qwen4_exp` checkpoint this tree loads stores one, and widening an op
   contract to match a capability nothing exercises would be a second, untested
   arm.
3. **The injection fusion boundary.** vLLM computes
   `2*sigmoid(logits/hc)` inside `_hc_combine_kernel`, keeping it in f32
   registers; this tree computes it in the MIXER, stores it to the caller's
   `injection` tensor, and the write-back reads it back. Where that tensor is
   bf16 the two differ by one narrowing. The boundary is this tree's op
   contract, shared by the CPU oracle and the CUDA arm, and moving it for one
   device would make the ROCm arm answer a different op. Recorded as a known
   narrowing, owed to whichever wave revisits the op split.

## D3b. Two W1 guarantees were unmeasured, and what it took to measure them

A fresh review mutated the three arms and found TWO guarantees that the kernel
comments assert and the fixtures could not see. The kernels were correct. The
fixtures were wrong, and they are recorded here because a fixture that cannot
fail is indistinguishable from one that passes, and the next reader has no other
way to learn this axis was once blind.

**The mixer could not see DIVISION 1.** `rocm_qwen4_exp.hip:257-261` says the
division by `hc_count` sits INSIDE the SiLU and that the placement is
load-bearing because SiLU is not homogeneous. Deleting it measured NMSE
`0.000490056` against the `5e-4` bar and PASSED with 2% of margin. The cause was
the fixture's projection scale, not the tolerance: at `+/-0.2`, `down(normed)`
landed at `|a| ~ 0.5` inside SiLU's near-linear part, AND `up()` then left
`sigmoid(gate)` spanning only `0.488..0.512`, so the entire low-rank branch was
a near-constant `0.5` that could not move the output whatever it computed.
`mix_down` at `+/-2.0` and `mix_up` at `+/-0.5` put the pre-activation in the
knee and open the gate to `0.27..0.81`, still nowhere near saturation. The same
deletion now measures **`0.160672`** on both `use_combine` arms and FAILS — 321x
the bar, against 0.98x before.

**The MUTATION ITSELF was a false one the first time, exactly as W1's earlier
registration mutation was.** Writing `const float a = low[i];` drops the last
use of `hc_f`, and this tree builds HIP with `-Wall -Wextra -Werror`, so
`rocm_qwen4_exp.hip:262: error: unused parameter 'hc_f' [-Werror,-Wunused-parameter]`
fails the compile and the STALE binary then passes with the unmutated number.
The measurement above was taken with
`const float a = __fmul_rn(low[i], (hc_f > 0.0f) ? 1.0f : 2.0f);`, which deletes
the division, keeps the parameter live, and multiplies by an exact `1.0f`. The
binary was sha256-proven changed (`5b6117f8..` against the baseline
`69a716d6..`) and sha256-proven restored.

**The grouped norm could not see the eps PLACEMENT.** `rocm_rms_norm_group.hip:175-178`
says eps is inside the rsqrt and is added to the MEAN SQUARE. On `O(1)` data with
`eps = 1e-6` that is unmeasurable, and both ways of breaking it survived at NMSE
`2.39943e-12`. The fixture now rescales two rows, and RMS norm returns every row
to `O(1)` whatever its input scale, so neither row distorts what the others
contribute:

| Mutation | Before | After (`gemma=false` / `true`) |
|---|---|---|
| eps added to the ROOT, `1/(sqrt(ms) + eps)` | 2.39943e-12, PASS | **0.00910165 / 0.0106572**, FAIL |
| eps added OUTSIDE the reciprocal, `1/sqrt(ms) + eps` | 2.39943e-12, PASS | **0.250049 / 0.292251**, FAIL |

Row 0 is scaled by `1e-3`, which puts its mean square at the same order as eps
and makes the first mutation shift that row by ~32%. Row 1 is scaled by `1e6`,
which puts `sqrt(ms)` at `~1.2e6` and makes the second mutation more than double
that row. The unmutated arm stays BIT-EXACT against the CPU oracle at both
polarities with those rows present, so the widening cost no margin.

**One instrument defect was found and repaired alongside.** doctest renders a
`const char*` operand as `1`, so every W1
`MESSAGE(... << DeviceName(dt) << ...)` recorded its NMSE beside that text and
never named the device it measured; `CAPTURE` takes the same path. A
`DeviceTag()` helper returning `std::string` repairs the three W1 cases, and the
gate now prints `qwen4_exp mixer NMSE ROCM combine=true mixed = 6.98127e-16`.
The roughly forty older sites in the same file belong to other rows and are
owed.

**The MECHANISM this helper depends on was stated wrongly the first time, and a
later reader could have deleted the helper on that statement.** The comment
claimed doctest resolves a `const char*` through a `toString(bool)` overload,
and claimed the `std::string` operand "has its own overload at
`doctest.h:1158`". That declaration is real but sits inside
`#if DOCTEST_MSVC >= DOCTEST_COMPILER(19, 20, 0)` at `:1156-1159`, so it does
not exist on the clang/HIP toolchain that runs this gate; the sibling
`toString(const char*)` at `:1153` is behind
`DOCTEST_CONFIG_TREAT_CHAR_STAR_AS_STRING`, which a whole-tree grep finds
defined nowhere. **A comment that claims a verification and names the wrong
mechanism is worse than no comment**, because the helper it justifies reads as
removable.

Both halves were then RUN rather than read, with a standalone probe built from
this tree's own `third_party/doctest/doctest.h` 2.5.2: it prints `raw=1` and
`str=ROCM` for the same name. The mechanism is ostream insertion in both
directions. A pointer satisfies `types::is_pointer` at `:1114-1117`, so
`StringMaker` inherits `StringMakerBase<true>` and reaches `filldata<T*>::fill`
at `:1242`, which forwards to `filldata<const volatile void*>::fill` at
`:8359`; that does `*stream << in` on a `const volatile void*`, and
`std::ostream` carries an insertion for `const void*` and none for the
volatile-qualified one, so the operand converts to `bool` and prints `1`. A null
pointer escapes only because `:8360` branches to the literal `"nullptr"`. A
`std::string` satisfies `has_insertion_operator` at `:1034` and reaches the
GENERIC `filldata<T>::fill` at `:1193`, which uses the real `operator<<`. The
fix was correct throughout; only its justification was wrong, and no fixture
value or kernel moved when the comment was corrected.

## D3c. W2's two ops have NO vLLM counterpart, so D3 does not arbitrate them

D3 says that where our CUDA arm and vLLM's AMD arm disagree, vLLM wins. W1's
three ops each had an AMD arm to compare against and D3a records what the
comparison found. **W2's two do not, and a reader who assumed the same procedure
was followed would be reading a check that never happened**, so the difference is
recorded rather than left silent.

`vt::IndexSelect` and `vt::IndexCopy` are GENERIC row gather/scatter helpers and
not `qwen4_exp` kernels. At the pin `e126687a9a`, `vllm/models/qwen4_exp/`
spells both sides of this behaviour as `torch.index_select` and
`Tensor.index_copy_` and ships no Triton kernel for either, in `amd/ops/` or in
`nvidia/ops/`; the AMD backend's 840-line divergence is entirely in the HC, PLE
and QSA kernels. There is therefore no second definition of this behaviour to
reconcile and nothing for D3's tie-break to decide.

What that leaves is the ordinary port: the donor is this tree's own CUDA arm
`src/vt/cuda/cuda_gdn.cu:432-527`, the contract is `CheckIndexRowOp`
(`src/vt/ops.cpp:3288-3322`), and the CPU arm
(`src/vt/cpu/cpu_ops.cpp:3048-3085`) is the oracle and the only thing that can
fail this arm. Upstream still fixes the SEMANTICS through PyTorch's own
`index_select`/`index_copy_`, which both our arms already answer to; it just
does not fix an implementation.

**These two ops do no arithmetic**, which is what makes their gate stricter than
W1's rather than weaker: every output byte is a copy of an input byte, so there
is no rounding, no re-association and no libm between the arms, and the bar is
BYTE EQUALITY against the CPU oracle instead of an NMSE band. The NMSE is still
printed beside it, because it is the statistic this spec quotes mutation margins
in and a byte comparison cannot say how far a broken arm landed.

## D3d. W3's mirror: vLLM's AMD PLE layer is PYTORCH, not Triton

D3 makes vLLM's AMD backend the mirror wherever it defines behaviour, and D3c
records the opposite case, an op it does not define at all. **W3's pair is the
third shape: vLLM DOES define both behaviours, and it defines them in PyTorch.**
That is worth recording because the W3 dispatch asserted otherwise — it said
`amd/ple_layer.py` "carries 4 `@triton.jit` kernels for this component" — and a
reader who took that on trust would go looking for kernels that are not there.
MEASURED at the pin `e126687a9a`, with `git show <pin>:<path> | grep -c
'triton.jit'`: `amd/ops/hc.py` = 5, `amd/ops/qsa.py` = 6, **`amd/ple_layer.py`
= 0**. The four methods the claim most likely counted are
`_short_conv_dilated_{decode,prefill,spec}_batched` and
`_short_conv_dilated_dispatch`, which are plain `torch` bodies. The AMD backend's
divergence from `nvidia/` in this file is not a kernel rewrite at all: measured,
185 changed lines, chiefly the FP8 PLE-embedding quant method, the
`compute_ngram_ids` custom-op split, a dequantize hook and the n-gram workspace
slicing — **none of which touches either of W3's two ops**, whose bodies are
byte-identical between `amd/` and `nvidia/`.

**THE ALGORITHMS AGREE, so D3's "where they agree, port ours" applies and both
ROCm arms are transcriptions of `cuda_qwen4_exp_ple.cu`.** Read against
`amd/ple_layer.py`:

- The conv is `F.conv1d(history, conv_weights.unsqueeze(1), groups=C,
  dilation=self.short_conv_dilation)` followed by `F.silu`, over
  `history = cat(initial_state, x)` with `initial_state` the cached
  `conv_state_len = (conv_kernel_size - 1) * short_conv_dilation` columns, and
  the write-back is `next_state = history[..., -self.conv_state_len:]`. That is
  this tree's op tap for tap, including that the ring stores the RAW input and
  not the activation, and that a chunk shorter than the window keeps the tail of
  the old state ahead of it.
- The gate is
  `gate = torch.sigmoid(gate.sign() * gate.abs().clamp_min(1e-6).sqrt())` then
  `gated_value = gate * value.unsqueeze(-2)`, with
  `/ math.sqrt(self.hidden_size)` applied to the dot on the line above — which
  is exactly what `Qwen4ExpPleGateArgs::gate_divisor` holds and why it is held
  as the divisor rather than its reciprocal. Clamp before the root, sign after
  it, sigmoid outside, `value` broadcast across the hc axis.

Four differences exist and NONE of them is an algorithm difference:

1. **The null-slot remap.** vLLM's `_short_conv_dilated_decode_batched` remaps a
   `NULL_BLOCK_ID` state index to slot 0 for a safe gather, ZEROES that row's
   output, and preserves the slot's prior contents on write-back — the padded
   rows a FULL cudagraph decode carries. `vt::Qwen4ExpPleConv` has no null-slot
   concept: its wrapper requires every `conv_state_indices` entry to be in
   range, and a padded row is expressed as an EMPTY SEGMENT, which both arms
   early-out on as an identity. This is a caller-side convention, not
   arithmetic, and nothing in this tree emits `NULL_BLOCK_ID` for this op today.
   OWED to whichever wave gives ROCm a full-cudagraph padded decode.
2. **The four-tap accumulation width.** vLLM's AMD conv runs `F.conv1d` at the
   ACTIVATION dtype — it casts `conv_weights.to(dtype=inputs.dtype)` and
   `state.to(x_d.dtype)` first — so its taps accumulate at the model dtype. Our
   CPU oracle accumulates in **double**, to match the W2 host reference term for
   term at the model's 10240-channel width; the CUDA arm inherited that and this
   ROCm arm inherits it again. **Ours is kept**, for D3a §1's reason: the CPU arm
   is the only oracle that can fail this arm, G1 is NMSE against it, and adopting
   the narrower width would move the ROCm arm away from that oracle while
   matching a lowering choice rather than a behaviour. **No model-path BUFFER is
   widened by this** — `x`, `weight`, `out` and the ring are all read and written
   at the caller's dtype through the same tagged accessors — so this is not the
   "dtype that is too wide" AGENTS.md names; the double lives in four registers
   and dies there. gfx1151 runs fp64 at a reduced rate, and a same-width f32
   variant is a SPEED item for a later wave. No throughput number is admissible
   from this row in any case, so there is nothing to trade the width against yet.
3. **`has_initial_states`.** vLLM zeroes the gathered state for a request whose
   flag is false. `vt::Qwen4ExpPleConv` has no such operand, deliberately: a
   ZEROED cache row IS the first call's left zero pad, so the obligation is the
   caller's and the kernel needs no branch. Recorded because it is the most
   plausible thing for a reader to mistake for a missing feature. This and the
   null-slot remap are the same kind of difference — vLLM does inside the kernel
   what this tree's op contract puts outside it.
4. **The capture fallback.** `_short_conv_fallback` runs the conv for profiling
   and cudagraph capture WITHOUT updating the state. This tree has no such mode
   at the op; whether a call updates the ring is the caller's business. Recorded
   so a reader who finds the method does not look for its arm here.

## Tests

Red first, per op, each failing for the intended reason before the arm exists:

1. A cross-device case in `tests/vt/test_backend_cross_device.cpp` in the
   sibling shape — REQUIRE-proven registration, never `if (!OpAvailable)
   continue`, which that file states twice in comments at `:3939` and `:4048`
   and which a fresh review already corrected once on the IQ4_NL row.
2. The per-op golden suites under `tests/vllm/models/` extended to the ROCm
   device where they are device-parameterised.
3. For each op, the reachability mutation `.agents/reachability.md` requires:
   delete the `RegisterOp` line in a scratch copy and confirm the focused gate
   reds rather than silently falling back. On this board it must red by
   REFUSAL, which is also the executable proof of D2.

## Gates

- **G1 per-op.** Cross-device NMSE <= 5e-4 against the CPU oracle on
  `strix:gpu0`, every case REQUIRE-proving registration.
- **G2 load and forward.** `UD-IQ1_S` loads on `--device rocm` and the forward
  completes, with `VT_OP_PROVIDER_STATS=1` showing zero reference-tier hits.
- **G3 first tokens.** Real text, recorded as a load-and-decode result and NOT
  as a token gate.
- **G4 token-exactness.** Owed, not this row. See `## Owed`.
- **No throughput, latency or memory number is admissible from this row.**
  AGENTS.md admits no performance result from an arm whose token gate has not
  passed. The `gfx1151` token gate currently reads `TOKEN_GATE=FAIL` at 3 of 6
  on a simpler model. A baseline captured before G4 is recorded with the word
  INADMISSIBLE beside it, which is how the withdrawn 2.71x should have been
  recorded.

## Owed

- **W1, W2 AND W3 LAND UNREACHED, and this bullet is the record AGENTS.md
  "Nothing lands dead" requires.** The seven arms `kQwen4ExpGatedResidual`,
  `kQwen4ExpGatedResidualWriteBack`, `kRmsNormGroup` (W1), `kIndexSelect` and
  `kIndexCopy` (W2), `kQwen4ExpPleConv` and `kQwen4ExpPleGate` (W3) are
  registered on ROCm and are reached by NO production entry point on that
  device. The weights load since #3097, but `ModelRegistry::Forward` cannot
  complete a `qwen4_exp` step on `rocm`: it throws at the first of the TWO arms
  that are still missing (`kQwen4ExpQsaCompress`,
  `kQwen4ExpQsaGatherAttention`). Their only caller today is the cross-device
  suite. **The row that owns the wiring is `MODEL-MM-QWEN4-EXP`, this row,
  through wave W4**; the issue that tracks it is this row's own, named in the
  commit and pull request bodies rather than here, because a row-owned issue is
  not an owed reference. D2 is why the slice is staged rather than held back: on
  a board with no reference tier a partial port refuses by name, so there is no
  half-working forward to ship and no way to reach these seven from production
  until the second of the remaining arms lands.

  **W2's two arms are the one part of this slice that is NOT qwen4_exp-only**,
  and the scope of what that buys is deliberately NOT asserted here.
  `vt::IndexSelect` and `vt::IndexCopy` are generic row gather/scatter helpers
  and they have production callers outside this model —
  `spec_decode/dflash2/speculator.cpp:217-218`, `gpu/runner.cpp:4812,4951`,
  `models/muse_glimmer_vision.cpp:550` and `models/qwen3_dspark.cpp:185` as well
  as `qwen4_exp_forward.cpp:447` and `qwen4_exp_qsa_block.cpp:616,678`. Whether
  any of those reaches a ROCm device today is UNMEASURED by this row and no
  claim is made either way: this row gates the two arms against the CPU oracle
  and nothing else. What IS established is the `qwen4_exp` half — the forward
  still refuses on ROCm at the two missing arms above, so neither W2 nor W3 adds
  a reachable qwen4_exp capability. **W3's two arms, unlike W2's, are
  `qwen4_exp`-only**: `grep -rn 'Qwen4ExpPleConv\|Qwen4ExpPleGate' src/` reaches
  `models/qwen4_exp_ple_block.cpp` and nothing else, so there is no second consumer
  whose reachability would have to be measured separately.
- **The remaining `CAPTURE(DeviceName(dt))` sites in
  `tests/vt/test_backend_cross_device.cpp`**, tracked by
  `ISSUE-LOCAL-01M2A7P3C95W3PBAVT9SC6KKY5`. doctest stringifies a `const char*`
  operand through its `bool` overload and renders the device name as `1`, so a
  failing cross-device assertion cannot say which device failed. The three W1
  cases are repaired in place with a `DeviceTag()` helper; roughly forty older
  sites belong to other rows and are not swept here.
- **G4 and its oracle.** vLLM's own AMD `qwen4_exp` backend at the current pin
  is the natural denominator on gfx1151, and it is UNMEASURED: nobody has built
  or run it on this board. **The platform half of that question is already
  answered from evidence committed in this repository**, and the answer narrows
  what is still owed rather than closing it. Triton itself WORKS on gfx1151 at
  3.8.0 — `docs/bench-evidence/oracle-vllm-gfx1151-20260903/job-phase3.txt:69-70`
  and `job-phase2.txt:64` — conditional on `python3-dev`, because Triton's AMD
  driver compiles `hip_utils.c` at import time; without it the same probe reads
  `TRITON_JIT_ON_GFX1151 = FAIL ValueError`
  (`job-phase1b.txt:281`), and installing the package turned that FAIL into the
  PASS. Since vLLM's RDNA paths ARE Triton kernels, that precondition applies to
  any run of them. **What remains open is narrower:** whether vLLM's SPECIFIC
  `qwen4_exp` AMD Triton kernels compile and run on this board. That is now a
  build-and-run question, not a platform-viability one.
- **The oracle recipe is committed and must be followed rather than
  improvised.** A bare `rc` worker on `strix` carries no torch and no triton
  (probed: `ModuleNotFoundError: No module named 'torch'`), so any future oracle
  wave installs through `docs/bench-evidence/oracle-vllm-gfx1151-20260903/`
  (`phase1b.sh`, `phase2.sh`, `phase3.sh`, `phase4.sh`). That recipe also
  records that `HSA_OVERRIDE_GFX_VERSION` is never set, because it makes the
  runtime report a different device and no oracle measurement survives it.
- **The MTP head**, unimplemented on every device.
- **The vision tower**, which has no ROCm-specific work and no artifact.
- **The CPU comparison arm no longer fits `strix:gpu0`.** It peaked at 73.9 GiB
  `VmHWM` and the host side of that box is now 31 GiB after the re-carve. A
  CPU-versus-ROCm comparison must run on `thor` or `dgx`, or on-box against an
  oracle instead.

## Stop conditions

- An op cannot be expressed on RDNA without a primitive the target lacks. Stop
  and record which, rather than substituting a different algorithm silently.
- A cross-device case cannot be made to red by deleting its `RegisterOp`. The
  case is measuring nothing; stop and fix the case before landing the arm.
- `strix:gpu0` is unavailable or unhealthy. Arms stay `PENDING` on a named
  lease. Never convert an unrun gate into a pass.
- ~~The dependency PR #3149 does not land. W1 through W4 are unaffected, but
  G2 cannot run, because the `ffn_down_exps` have no device arm without it.~~
  RESOLVED 2026-09-12: it landed at `18e2a8c9a` and is in this branch's base.

## Now

`ACTIVE`, 2026-09-12. **W3 lands the PLE pair**, `kQwen4ExpPleConv` and
`kQwen4ExpPleGate`, in `src/vt/rocm/rocm_qwen4_exp_ple.hip`, each with a
cross-device case that REQUIRE-proves its ROCm registration. With W1's three and
W2's two that is SEVEN landed and TWO owed — the QSA pair (W4) — and the forward
still refuses on ROCm, because a partial port buys nothing runnable on a board
with no reference tier (D2). Next action is W4, the QSA pair, and it is the hard
one: `amd/ops/qsa.py` is its reference rather than the CUDA arm.

**vLLM's AMD backend DOES define both of these behaviours, and it defines them
in PYTORCH rather than Triton** — `amd/ple_layer.py` carries zero `@triton.jit`
kernels, against 5 in `amd/ops/hc.py` and 6 in `amd/ops/qsa.py`. D3d records the
comparison, the four non-algorithmic differences it found, and why the
double-width four-tap accumulator is kept rather than narrowed to vLLM's
activation dtype.

W3's measured numbers are recorded below when the gate runs; until then no
number in this section is a result.

**W2's OWN RECORD IS NOT DELETED, it is superseded here and kept below**, the
same way W2 kept W1's. Its counts are read against its own head and not against
this one: "five landed and four owed" was true at `52cecd733`.

`ACTIVE`, 2026-09-12. **W2 lands two more of the nine arms**, `kIndexSelect` and
`kIndexCopy`, in `src/vt/rocm/rocm_gdn_state.hip`, each with a cross-device case
that REQUIRE-proves its ROCm registration. With W1's three that is FIVE landed
and FOUR owed — the PLE pair (W3) and the QSA pair (W4) — and the forward still
refuses on ROCm, because a partial port buys nothing runnable on a board with no
reference tier (D2). Next action is W3, the PLE pair.

**These two ops have no vLLM counterpart at all**, so unlike W1 there was no AMD
arm to compare against and D3's tie-break had nothing to arbitrate. D3c records
that rather than leaving a reader to assume a check that never happened, and it
also records the consequence for the gate: with no arithmetic in either kernel
the bar is BYTE EQUALITY against the CPU oracle, not an NMSE band.

**The W2 numbers, measured on `strix:gpu0`**, `rc` job
`04c5e7ef-f6d3-4ea8-88cd-93e05e50ca79`, worker `rc-worker-lcjhd`, `gfx1151`,
ROCm 7.2.4 / HIP 7.2.53211, Release, `-DVLLM_CPP_HIP=ON
-DVLLM_CPP_HIP_ARCHITECTURES=gfx1151`, built in the lease from a clone of this
row's branch with `git rev-parse HEAD` asserted equal to the commit under test:

| Run | Head | `test_backend_cross_device` |
|---|---|---|
| RED | `d9b8b5b67` (test only) | `53 cases / 51 passed / 2 failed / 0 skipped`, `84135 assertions / 2 failed` |
| GREEN | `52cecd733` (the arms) | `53 cases / 53 passed / 0 failed / 0 skipped`, `84333 assertions / 0 failed`, `Status: SUCCESS!` |

**The red is the evidence, not a mishap.** Both failures were the two new cases
and both were the registration `REQUIRE`, not a number:
`REQUIRE( vt::OpRegistered(vt::OpId::kIndexSelect, DeviceType::kROCM) ) is NOT
correct!` and the same for `kIndexCopy`. **Two failed cases against two failed
assertions is the signature of a ROUTE gap**: nothing computed a wrong answer,
the work never started. The assertion count RISING by 198 across the pair is
what shows the device half executed rather than being skipped — in the red run
both cases aborted at the `REQUIRE` before their bodies ran — and `0 skipped` is
asserted on the green run, not assumed.

**Every one of the twelve measured NMSE values is `0`**, both ops x `{f32,
bf16}` x `{packed, padded, padded rank-3}`, and each also passes the byte
comparison it is graded on. That is the expected result and not a suspiciously
good one: these ops copy bytes and do no arithmetic, so an arm that addresses
correctly is bit-identical to the oracle by construction.

**FOUR NUMERICAL MUTATIONS, each red by three orders of magnitude against the
`5e-4` bar.** Every one was sha256-proven to have changed the test binary
(`c36081f3..` for the green arm) and every restore left
`git status --porcelain` empty:

| Mutation | What it breaks | NMSE, layout 0 / 1 / 2 | Ratio to the bar |
|---|---|---|---|
| `s1` | `index_select` uses the PACKED row as the base stride | 0 / **1.66989** / **1.56537** | up to 3340x |
| `s2` | `index_select` addresses by `row` instead of `idx[row]` | **1.37064** / **1.15318** / **1.43775** | up to 2876x |
| `c1` | `index_copy` uses the PACKED row as the destination stride | 0 / **0.753077** / **0.838575** | up to 1677x |
| `c2` | `index_copy` writes at the COMPACT offset, ignoring `idx` and the stride | **0.721975** / **0.924202** / **0.804197** | up to 1849x |

The `0`s in the stride rows are the point of the layout axis rather than a hole
in it: layout 0 is packed, so there the packed row IS the stride and the
mutation is a no-op. Layouts 1 and 2 pad the base side and both mutations red
there, and the case fails as a whole (8 failed assertions each). Had the fixture
carried only the packed layout, both stride claims would have been unmeasurable
— which is the W1 lesson applied before a review had to find it.

**THE MUTATIONS WERE WRITTEN TO STAY LIVE, because W1 was bitten twice by a
FALSE one.** This tree builds HIP with `-Wall -Wextra -Werror`, so deleting a
term usually orphans its parameter, fails the compile, and lets a STALE binary
pass with the unmutated number. Each mutation above therefore keeps every
parameter referenced through a branch that is never taken — for example `s1`
writes `((in_row_stride > 0) ? inner : in_row_stride)`, which removes the stride
from the arithmetic and keeps it live. All four compiled at `-Werror` with zero
warnings and all four changed the binary.

**THE REACHABILITY MUTATION REDS BY REFUSAL, which is D2 executed rather than
argued.** Deleting each `RegisterOp` in the scratch copy — with the kernel left
in place and referenced, so the deletion is not a false mutation — makes the call
throw:

```text
vt: no kernel for op IndexSelect (id 90) on device rocm (type 5), and the
portable CPU reference tier is NOT eligible: ... this ROCm device reports
hipDeviceAttributePageableMemoryAccess = 0 ...
```

and the same for `IndexCopy` (id 91). To reach the throw at all, the same
scratch edit relaxes the case's `REQUIRE` to a `CHECK` and drops its
`OpAvailable` guard; with the guard in place the case reds one line earlier, at
the registration assertion, which is the ordinary red and also correct.
Restored, the rebuilt binary is sha256-IDENTICAL to the green one, so nothing
about the landing arm depends on a mutation that was left behind.

**THE FULL SUITE WAS RUN AS A BASE-VERSUS-HEAD PAIR, because a bare failure
count is not a verdict.** `rc` job `7b593bf9-9448-4d1c-a234-ad17c5d8f61a` built
all 1429 targets and ran the whole 743-test `ctest` twice in one lease, at the
base `7b45fcfc7` and at this branch's `52cecd733`:

| Leg | Result | Failing set |
|---|---|---|
| base `7b45fcfc7` | `98% tests passed, 18 failed out of 743` | 18 |
| head `52cecd733` | `98% tests passed, 17 failed out of 743` | 17 |

**`comm` over the two sorted sets finds ZERO tests failing only at the head**,
so this branch introduces no regression. Seventeen failures are present in BOTH
legs and are pre-existing on `main` — among them `test_qwen4_exp_runner`,
`test_qwen4_exp_gguf_weights`, `test_cuda_quant_dot` (a CUDA suite on a build
with CUDA off), the seven `dflash2_*` suites and `test_ops_residual_rmsnorm`.
They are NOT diagnosed here and W2 makes no claim about them; they belong to
their own rows. The eighteenth, `test_ltx2_video`, failed in the BASE leg only
and passed at the head, which makes it a flake rather than a repair — nothing in
this branch touches LTX.

**One deviation from the declared gate, recorded rather than quietly
substituted.** The task asked for a full CPU `ctest`, and the local box has
16 GiB free on a 447 GiB disk with several sessions building, so a second full
build there was refused. What ran instead is the full suite on the HIP build in
the lease, which is a SUPERSET: every CPU test is built and executed, with a
ROCm backend additionally registered. A CPU-only `ctest` at this head is
therefore UNRUN, and the base-versus-head pair above is what stands in for it.

**W1's OWN NUMBERS ARE NOT DELETED, they are superseded here and kept in git.**
Its gate read `51 of 51 cases, 84127 assertions, 0 failed` at `adaa830ef` on the
base `51c248190`, with write-back `0`, `rmsnorm_group` `0` at both polarities,
mixer injection `0` and mixer `mixed` `6.98127e-16`. The full W1 record,
including the two fixtures a fresh review found blind and the margins that
repaired them, is in D3b above and in the commits `git log --grep
MODEL-MM-QWEN4-EXP` reaches. The 53-case count here is 51 plus W2's two, and the
84135-assertion red count is 84127 plus the eight preconditions the two new
cases assert before they reach the device.

No throughput, latency or memory number is admissible from this row, and W2 took
none.
