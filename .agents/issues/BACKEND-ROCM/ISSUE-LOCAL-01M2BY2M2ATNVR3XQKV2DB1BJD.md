ID: ISSUE-LOCAL-01M2BY2M2ATNVR3XQKV2DB1BJD
Title: gfx1151: greedy decode on the production server path hits an illegal GPU memory access on 2 of 5 identical runs, on a model that fits host RAM
Row: BACKEND-ROCM
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-12
Updated: 2026-09-12
Closed: -

## Problem

On gfx1151 (strix:gpu0, Radeon 8060S, ROCm 7.2.4) the production server path emits an ILLEGAL GPU MEMORY ACCESS on 2 of 5 identical greedy-decode requests, at origin/main 8ea148d4e, on a model that fits host RAM. Greedy is the arm #937 explicitly reports as working ("Greedy decoding (--temperature 0) works fine"), so this is a different shape from #937 and that claim is falsified on this board.

MEASURED 2026-09-12 under three rc leases on strix:gpu0, box idle and exclusively leased. Model Qwen3.8-27B-Q4_K_M.gguf, 17106775008 B (15.93 GiB), sha256 7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169, at /workspace/ckpt/qwen38-27b-q4km/. Architecture Qwen3_5ForConditionalGeneration. Built on the worker from a clean clone of 8ea148d4e with cmake -GNinja -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_HIP=ON -DVLLM_CPP_HIP_ARCHITECTURES=gfx1151, ninja -j 4 vllm-server; the artifact links /opt/rocm-7.2.4/lib/libamdhip64.so.7 and ninja reports "Linking HIP executable examples/vllm-server". Run through examples/vllm-server with --device auto --max-num-seqs 1 --no-enable-thinking --enable-force-include-usage, temperature 0. VT_ROCM_MANAGED_ALLOC was NOT set in any run.

THE DEVICE IS PROVEN, not assumed. With VT_OP_PROVIDER_STATS=1 every announced op reads device=5 (kROCM) selected=vt-native priority=0 registered=1: 17 ops in one run and 20 in another, and `grep -c 'vt reference-tier'` is 0 in both, so no op fell to the portable CPU tier.

FIVE RUNS, SAME BINARY, SAME BOARD, SAME FLAGS, temperature 0 throughout:
1. max_tokens=16, VT_OP_PROVIDER_STATS unset -> HTTP 200, 10 completion tokens, "Hello! How can I help you today?", elapsed_s=13.063.
2. max_tokens=64, VT_OP_PROVIDER_STATS=1 -> CRASH. stderr: "Memory access fault by GPU node-1 (Agent handle: 0x1815e0e0) on address 0x7f93bf15e000. Reason: Page not present or supervisor privilege." curl exit 52 (empty reply), process died and spent ~60 s in the kernel coredump path (sampled tid state=D wchan=dump_vma_snapshot then get_dump_page, syscall 234).
3. max_tokens=64, VT_OP_PROVIDER_STATS unset -> HTTP 200, 64 completion tokens, elapsed_s=114.738, gen_tok_s=0.557793, server alive afterwards.
4. max_tokens=16, VT_OP_PROVIDER_STATS=1 -> HTTP 500 InternalServerError: "EngineCore encountered an issue. See stack trace (above) for the root cause. [vt rocm: hipMemcpyAsync: an illegal memory access was encountered]", and the process then Aborted. 17 ops had announced on device=5 first.
5. max_tokens=64, VT_OP_PROVIDER_STATS=1 -> HTTP 200, 64 completion tokens, elapsed_s=58.0899, gen_tok_s=1.10174, 20 ops on device=5.

VT_OP_PROVIDER_STATS IS NOT THE CAUSE, and that was checked rather than assumed. Runs 4 and 5 both set it and disagree; runs 2 and 5 use the same max_tokens and disagree. The knob only fprintf()s to stderr once per (op, device). What the correlation does say is that the failure is TIMING-SENSITIVE: a perturbation as small as a once-per-op stderr write moves it. That is a race signature, not a deterministic bad address.

The two failures are two views of one event. "Memory access fault by GPU node-1 ... Page not present" is the HSA runtime reporting the fault out of band and killing the process; "hipMemcpyAsync: an illegal memory access was encountered" is the same poisoned stream surfacing at the next synchronizing HIP call, which vllm.cpp turns into a 500 rather than a crash. Which one a run shows depends on where the queue was when the fault landed.

NO PERFORMANCE READING IS ADMISSIBLE FROM THIS WORK, and the numbers above are recorded as liveness evidence only. The three passing runs read 0.558, 0.766 and 1.102 gen_tok_s, a 1.98x spread on one binary and one board, and the loads are CIFS-bound (12.176 GiB prefaulted at 69.8-123.9 MiB/s over three runs), so neither axis can gate anything.

WHY IT MATTERS. This is the production server path on its default configuration, at temperature 0, on a model that fits, with no reference-tier fallback and no refused op. A 40% failure rate on greedy decode means every gfx1151 gate on this board is running on an arm that can fail without the gate noticing, because 3 of 5 runs look perfect.

FOUND BY MODEL-MM-QWEN4-EXP W7 while answering a different question (does a model that fits host RAM forward on gfx1151, the NEXT HYPOTHESIS of ISSUE-LOCAL-01M2BRWKNZM8031JT1QE67WYHZ). It is NOT that stall: this fault is loud, fast and intermittent, and no thread was ever in svm_range_set_attr during any of the five runs, whereas the 67.56 GiB UD-IQ1_S rung sat there in uninterruptible sleep for the whole 805 s of its request in the same lease on the same binary. Filed rather than fixed in flow because the repair needs its own red-before reproduction of a race and would change kernel or stream-synchronization semantics, which the in-flow rule excludes.

Related: #937 (same board, same class of crash, but its trigger is temperature above 0 and its text asserts greedy is safe), ISSUE-LOCAL-01M2BRWKNZM8031JT1QE67WYHZ (the qwen4_exp SVM stall, a different failure on the same board).

## Resolution

-
