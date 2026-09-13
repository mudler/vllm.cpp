ID: ISSUE-LOCAL-01M2E921GPVNYCJNC51CNJXP57
Title: vLLM DEFERS the HC combine to the next mix boundary and fuses it with that mix's RMSNorm; we read the 10240-wide residual twice
Row: MODEL-MM-QWEN4-EXP
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-13
Updated: 2026-09-13
Closed: -

## Problem

Verdict (b) at `.agents/specs/qwen4-exp-flash-next.md:177` and `:4630-4631`, carried as 'NO issue yet' until now. At the active parity pin `e126687a9a`, vLLM does NOT apply the hyper-connection combine where it is computed. It defers it to the next mix boundary and FUSES it with that mix's RMSNorm (`nvidia/hyperconnection.py:152-186`, `combine_and_mix`; kernel `nvidia/ops/hc.py:266-375`, `_hc_combine_norm_kernel`), materialising the combine early only where PLE adds into the stream (`nvidia/model.py:288-297`). The ARITHMETIC agrees with ours (verdict (c)); the SCHEDULE does not. Upstream reads the 10240-wide residual ONCE per boundary. This tree applies the combine immediately as its own op on the raw stream (`qwen4_exp_forward.cpp:466`, `:544`) and then reads the same residual again in the grouped norm, so it reads it TWICE. THIS IS A MEMORY-TRAFFIC LEVER AND IT IS LARGER THAN THE ONE ISSUE-LOCAL-01M2CJXQMV9R9JGRSKZMW4W21F CLOSES: that one fixes the grouped norm's launch shape and leaves the duplicated 10240-wide read in place. NOT MEASURED on this tree; the claim is upstream's structure plus the width, not an A/B.

## Resolution

-
