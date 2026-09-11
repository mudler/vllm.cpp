ID: ISSUE-GH-3113
Title: fix(BACKEND-ROCM-QUANT-GATHER): reject unqualified model captures
Row: BACKEND-ROCM-QUANT-GATHER
State: OPEN
Kind: UNKNOWN
GitHub: 3113
Mirror: DIVERGED
Availability: FULL
Created: 2026-09-09
Updated: 2026-09-09
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BACKEND-ROCM-QUANT-GATHER`
>
> The bounded model comparator in MR #3097 accepts incomplete or unqualified primary records. Fresh review of commit `75d92614e84ff428eab49324f3b4fb2aba183ac3` reproduced the failures in `tools/rocm_quant_gather/compare_models.py`.
>
> A synthetic validator corpus uses the actual native and secondary captures. Changing one primary record to PENDING with a RuntimeError after generation still returns PASS. Wrong primary and plugin pins, a wrong runtime version, and an FP8 cache request also pass. The separate memory record accepts PENDING, a wrong primary pin, and different sampling or logical capacity. A nonzero cache view storage offset is also accepted. These synthetic mutations test the validator and are not oracle measurements.
>
> The driver records tokens before its second memory observation. Its exception path preserves those tokens. Comparing tokens alone therefore cannot establish successful execution. The operator separately checks actual process exits and sealed inputs; the comparator must enforce its own documented qualification contract.
>
> Require successful execution without an exception, the pinned primary and plugin identities, the resolved runtime identity, and the complete identical workload in both token and memory records. Validate the cache view offset and its physical bounds. Add focused rejection tests for each demonstrated invalid record and retain the valid actual captures as the positive control.
>
> Evidence: `/home/vikash/.cache/rdna3-gather-oracle-review/comparison-input-audit.json` records 31 cases, with 22 rejected mutations and nine invalid records accepted. The existing spec `.agents/specs/rocm-quant-gather.md` owns the correction. A fresh implementer repairs the finding, a different agent reviews the immutable result, and the operator reruns the complete comparison. No model, workload, pin, or performance requirement changes.
>

## Resolution

-
