ID: ISSUE-GH-2262
Title: **The llama.cpp arm's mutation sweep is not re-executable and its CUDA toolchain is asserted rather than pinned.** Two reproducibility debts narrowed rather than closed by the repair that brought `scripts/qwen4exp-llamacpp-build-cuda.sh` and `scripts/qwen4exp-llamacpp-decode-proof.sh` into the tree. (1) `docs/bench-evidence/qwen4exp-llamacpp-ladder-arm-20260829.md` records 11 mutations red with none unarmed, and the sweep DRIVER is not committed, so nine of them are a claim about a run that happened once on one machine; two are executable tests, `test_set_u_mutation_removing_one_default_goes_red` and `test_kv_mutation_restoring_the_fail_open_default_goes_red`. (2) `apt-get install -y cuda-toolkit-13-0` pins a CHANNEL, not a version, so a rerun gets whatever apt serves and, before this, nothing would have noticed; the build now carries `EXPECT_NVCC=13.0.88`, the version the evidence records, compares it against `nvcc --version` and exits 89 on a mismatch, which makes drift visible without making apt serve one version. Owed: a committed sweep driver (or the nine as tests), and a genuinely pinned toolchain — a versioned apt pin or a recorded container image
Row: MODEL-MM-QWEN4-EXP
State: UNKNOWN
Kind: record
GitHub: 2262
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:906`

### Frozen archive evidence

> | [#2262](https://github.com/mudler/vllm.cpp/issues/2262) | `MODEL-MM-QWEN4-EXP` | **The llama.cpp arm's mutation sweep is not re-executable and its CUDA toolchain is asserted rather than pinned.** Two reproducibility debts narrowed rather than closed by the repair that brought `scripts/qwen4exp-llamacpp-build-cuda.sh` and `scripts/qwen4exp-llamacpp-decode-proof.sh` into the tree. (1) `docs/bench-evidence/qwen4exp-llamacpp-ladder-arm-20260829.md` records 11 mutations red with none unarmed, and the sweep DRIVER is not committed, so nine of them are a claim about a run that happened once on one machine; two are executable tests, `test_set_u_mutation_removing_one_default_goes_red` and `test_kv_mutation_restoring_the_fail_open_default_goes_red`. (2) `apt-get install -y cuda-toolkit-13-0` pins a CHANNEL, not a version, so a rerun gets whatever apt serves and, before this, nothing would have noticed; the build now carries `EXPECT_NVCC=13.0.88`, the version the evidence records, compares it against `nvcc --version` and exits 89 on a mismatch, which makes drift visible without making apt serve one version. Owed: a committed sweep driver (or the nine as tests), and a genuinely pinned toolchain — a versioned apt pin or a recorded container image | record |

## Resolution

-
