# EXL3 public documentation

## Now

Documentation correction for `QUANT-EXL3`, scoped to
`ISSUE-LOCAL-01M2CBGG169HDX0DWKHQWCBBGN`. The implementation row stays `ACTIVE`.
Base: `e030f1b90`. One pull request carries this spec and the documentation.

## Scope and inventory

| Surface | Current gap | Source and evidence | Change |
|---|---|---|---|
| README news | The older CUDA description omits mul1 and long-prefill dispatch | `src/vt/cuda/cuda_exl3.cu:2132`, `include/vllm/model_executor/models/dense_attn_block.h:315` | Replace the old EXL3 entry with a September capability update |
| `docs/FEATURES.md` EXL3 row | Attempt history obscures current support and contains a stale HumanEval obligation | `.agents/specs/quant-exl3-shared.md`, `.agents/specs/quant-exl3-perf.md`, `docs/benchmarks/qwen38-27b-exl3-gb10.md` | Concise support summary with evidence links |
| `docs/USAGE.md` EXL3 artifact entries | Current instructions mix with old attempts | Shared dense dispatch, EXL3 loader, and the quantization specs | Retain artifact pins, hashes, sizes, working paths, and refused arms. Remove obsolete chronology |

## Design and upstream anchors

This is an editorial correction, not a port. The existing `QUANT-EXL3` inventory
and committed implementation spec own the upstream surface and kernel gates.
The reference is exllamav3 at the revision in `.agents/oracles/exllamav3.md`.
`quant-exl3-shared.md` records the upstream dispatch and dependency chain.
Read the local loader, dispatch, CUDA instantiations, and CPU regression test
before editing their descriptions. Do not infer full model support from a kernel.

CUDA uses reconstruct plus cuBLASLt above 144 input rows when registered.
Other backends retain `Exl3Gemm`. Preserve the distinction between GEMM, GEMV,
and the DeepSeek-V4 fused MoE path. Do not generalize CUDA coverage to other
backends. Do not add new measured performance claims.

Preserve removed historical text in a dated file under `.agents/completed/`.
Keep the public feature row readable and link detailed constraints to the
existing specs. Change no benchmark values or benchmark dispositions.

## Tests and gates

No upstream tests to port: no executable behavior changes. No GPU, model
weights, external compute, or runtime benchmarks are required or authorized.
Use source inspection to verify every changed technical statement. Retain all
artifact filenames, sizes, revisions, and hashes byte-for-byte.

Focused CPU gate:

```sh
python3 scripts/check-readme-structure.py
python3 scripts/check-supported-models.py
python3 scripts/check-quickstart-recipes.py
python3 scripts/check-benchmark-index.py
python3 scripts/check-agent-record.py
python3 -m unittest discover -s tests/scripts -p 'test_check_readme_structure.py'
git diff --check
```

Run `scripts/agent-preflight.sh --quiet` and classify any failure against the
unchanged base. An independent reviewer checks the immutable implementation
commit and reruns the focused gate. Mutations apply only to executable test
claims. No new tests that merely copy the prose are needed for this edit.
The operator reruns the focused gate before the fork push.

## Work breakdown and authority

1. Operator commits this spec and the canonical local issue.
2. A fresh implementer edits only the three public documents above, the dated
   archive, and this spec's outcome. It records source anchors and gate output.
3. A fresh reviewer checks the immutable head without repairing findings.
4. Operator verifies and opens a pull request from the maintenance fork to
   `mudler/vllm.cpp:main`. Merge is not authorized by the user request.

## Risks and stop conditions

Do not promote a single completed benchmark round to an accepted headline.
HumanEval-style benchmark data is not a correctness gate. Preserve those limits.
Stop on an ambiguous source claim rather than guessing. Do not modify source,
tests, checkers, other rows, or unrelated documentation to repair baseline gates.

## Outcome: 13 September 2026

Implemented the three public-document corrections without changing model
support states or benchmark values. Removed prose is preserved verbatim in
[the dated archive](../completed/exl3-public-docs-20260913.md). The five edited
artifact rows retain their first five cells byte-for-byte, including filenames,
sizes, revisions, and hashes. The CUDA news replaces the older August entry.

Source verification at implementation base `0cf30baae`:

| Statement | Inspected source or existing evidence |
|---|---|
| Per-tensor width and codebook selection | `include/vllm/model_executor/models/dense_weight_loaders.h:700` derives bits from trellis shape; `:750` reads `mcg` and `mul1` presence |
| Shared dense method and long-prefill dispatch | `include/vllm/model_executor/layers/quantization/exl3.h:70` delegates to `dense_attn::Exl3MatmulD`; `include/vllm/model_executor/models/dense_attn_block.h:315` requires both M > 144 and a registered reconstruct op |
| CUDA GEMM coverage | `src/vt/cuda/cuda_exl3.cu:2132` admits exactly `(3,0)`, `(3,1)`, `(6,0)`, `(3,2)`, `(4,2)`, `(5,2)`, and `(6,2)`; reconstruct dispatch at `:2707` repeats those pairs |
| GEMV is a separate capability | `src/vt/cuda/cuda_exl3.cu:2284` admits `(3,1)`, `(3,2)`, and `(4,2)`; `:2314` declines uninstantiated arms; `tests/vt/test_exl3_gemv.cpp:161` covers artifact shapes and occupancy constraints |
| Non-CUDA long-prefill behavior | `tests/vt/test_exl3_matmul_dispatch.cpp:68` exercises M=145 through the shared seam on CPU, with no reconstruct registration. Inspected here, not compiled or executed |
| Draft uses packed target head | `src/vllm/model_executor/models/qwen3_dflash.cpp:105` selects the EXL3 head before the other formats |
| DeepSeek-V4 doubled widths no longer fail at load | `src/vllm/model_executor/models/deepseek_v4_weights.cpp:1145` derives compressor width from the ratio; `:1195` loads the indexer compressor at twice its head dimension |
| DeepSeek-V4 tokenizer caveat is obsolete | `tests/vllm/test_tokenizer_parity_deepseek_v3.cpp:136` compares encoding with HF goldens; `:144` checks round trips. `model-dsv4-exl3.md:53` records the real artifact's tokenizer parity evidence |
| DeepSeek-V4 MTP exclusion remains | `src/vllm/model_executor/models/deepseek_v4_weights.cpp:1306` explicitly skips and counts `mtp.*` tensors |
| Generation and performance limits | `docs/benchmarks/qwen38-27b-exl3-gb10.md:3` records generation; `:27` excludes correctness claims for sampled HumanEval legs. `quant-exl3-perf.md:45` records GEMV occupancy limits; `backend-rocm-exl3.md` owns gfx1151 generation evidence |

Focused verification uses Python from `/tmp/vllm-docs-tools/usr/bin`, with
`LD_LIBRARY_PATH=/tmp/vllm-docs-tools/usr/lib`:

| Command | Result |
|---|---|
| `python3 scripts/check-readme-structure.py` | PASS, exit 0 |
| `python3 scripts/check-supported-models.py` | PASS, exit 0; 44 registered architectures |
| `python3 scripts/check-quickstart-recipes.py` | PASS, exit 0 |
| `python3 scripts/check-benchmark-index.py` | PASS, exit 0 |
| `python3 scripts/check-agent-record.py` | PASS, exit 0 after correcting the new claim's missing lifecycle annotation; initial exit 1 |
| `python3 -m unittest discover -s tests/scripts -p 'test_check_readme_structure.py'` | PASS, exit 0; 19 tests |
| `git diff --check` | PASS, exit 0 |

A direct comparison with `git show 0cf30baae:docs/USAGE.md` confirms the five
artifact rows retain their first five cells. A second comparison confirms every
removed FEATURES and USAGE line occurs verbatim in the archive. Both pass.

No executable behavior or test guarantee changes, so red-first and negative
mutation checks are not applicable. No GPU or runtime benchmark was run. The
operator owns the unchanged-base comparison and full `agent-preflight.sh`
result; its broad run was still in progress at this implementation handoff.
Missing compiler, CMake, and readelf are reported environment limitations,
not passing gates. This helper does not modify unrelated code to repair them.

### Review corrections: 13 September 2026

The MTP bit width applies to its trellis modules, not every companion tensor.
`quant-exl3-mul1.md:246-248` records eight 4-bit MTP modules.
`dense_weight_loaders.h:722-723` requires F16 sign vectors, and `:765-767`
requires an I32 `mul1` marker. The usage description now names the trellis
modules explicitly.

The ROCm evidence includes a subsequent successful BF16 control at
`backend-rocm-exl3.md:81-104`. The usage limitation now names missing AMD clock
attribution and discrete-GPU validation. No benchmark values or artifact pins
changed. Both corrections address the existing documentation issue.

All seven focused gates passed again after these corrections, including all
19 README checker tests. The first five cells of both artifact rows remain
byte-identical to repair base `752b9628b`. No GPU work was run.
