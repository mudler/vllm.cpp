# llama.cpp, CPU

| RPi5/A76 arm (R4-R6, `GATING`) | Result |
|---|---|
| Scope | 4-core A76, DotProd, no i8mm; 20-core binding arm does NOT transfer. buildx/QEMU-built, Pi-executed unthrottled; hashes pinned in the [campaign spec](../../.agents/specs/rpi5-cortex-a76-cpu-optimization.md) |
| Assembly vs compiler SDOT (one GCC 13.3 binary) | AAPCS64 leaf wall +3.66% M1/T1, +5.08% M128/T1, +3.69% M128/T4, with 9.74-10.24% fewer instructions; M1/T4 is the -2.43% residual ([assembly evidence](../bench-evidence/rpi5-a76-q8-dot-20260806.md)) |
| 64-token Qwen model gate | Byte-identical across x86, portable, SDOT and assembly arms; asm vs SDOT median TTFT -1.55%, TPOT neutral, E2E -0.13%; vs portable TTFT -33.40%, E2E -2.67%. Cortex-A76+DotProd selects assembly by default |
| Same-file llama.cpp floor (pp17/tg64) | **NOT MET on speed**: prefill 12.81 vs 27.77 tok/s (0.461x), decode 2.55 vs 3.91 (0.653x), E2E 26,018.39 vs 16,998.49 ms ([competitor evidence](../bench-evidence/rpi5-a76-llamacpp-20260806.md)); vs stock `b9892`, SUPERSEDED |
| Peak RSS | **2.841 vs 3.747 GiB, 24.2% less**; 3 clean unthrottled reps; same-text 64-token greedy output byte-identical after trailing-newline normalization |
| `PENDING` | Pi concurrency; BF16 GEMM / speed closure (the 2.17x prefill, 1.53x decode gap profiling is W6) |

Same GGUF file both arms, `dgx.casa` GB10 aarch64 (20 cores), idle, 3 reps,
llama.cpp built fresh on the same host from `237ad9b96`. That commit is our own
local-only fork, 65 performance commits deep, six of them on the CPU path, so
this denominator is **SUPERSEDED** and every llama.cpp number on this page is
owed a re-take against the new stock pin (#1003).

| Axis | vllm.cpp | llama.cpp | Ratio | Result |
|---|---:|---:|---:|---|
| Prefill | **223.8 tok/s** | 177.3 | **1.18x** | **PASS**, denominator SUPERSEDED |
| Decode | 24.7 tok/s | 25.4 | 0.97x | tie, denominator SUPERSEDED |
| Peak memory | 2.83 GiB | 2.80 GiB | 1.01x | **PARITY**, denominator SUPERSEDED |

Decode lands inside llama.cpp's own run-to-run spread, and the memory difference
is 30 MiB on a 2.8 GiB working set. Prefill is the only axis with a real gap and
it goes our way. Output tokens are **byte-identical** to llama.cpp's greedy
decode and to our own CPU reference path. Single-stream only: we have not
measured concurrent serving against llama.cpp's server. Seven favourable
llama.cpp verdicts can flip when the denominator is re-taken, five on this page,
one on the README, and prefill is not the most exposed. #1003 owes all seven. The
set is swept over every tracked file, not listed: the query is in the
[spec](../../.agents/specs/oracle-llamacpp-repin-stock.md).

| Flips first | Verdict | Exposure |
|---|---|---|
| 1 | Vulkan `BENCH-VK-LLAMA` decode 4.36 vs 4.35, `MET` | 0.23% margin inside a 0.69% 7-leg spread. Its own source calls it "a narrow pass, not a comfortable one", so any move can flip it. The README states it as "matches" |
| 2 | Muse Glimmer in128 prefill `1.023x` (also `STATUS.md`) | 2.3% margin inside our own arm's 4.5% leg spread, n=4. Its denominator is already stock `7044859`, 84 commits from `b10451`, so the noise floor is what exposes it |
| 3 | This table's peak memory `1.01x` PARITY and decode `0.97x` tie | ties by declaration, not wins. A denominator that moves at all in llama.cpp's favour turns both into recorded gaps |
| 4 | keep-f16's `1.01x` RSS and "prefill `1.18x` AHEAD", quoted in product code for `VT_GGUF_KEEP_F16` default ON | buys 1.05 GiB of peak RSS (3.885 to 2.832) for ~9% prefill (224 to 204 tok/s) and ~1.4% decode, tokens identical. **Decided 2026-08-17: the default stays ON on our own arms, so it needs no re-take.** Both ratios do |
| 5 | `KERNEL-GEMM-CPU-TILED` NEON vs stock ggml sgemm, "at parity, ahead on 4 of 6 shapes" | off this page, in the kernel matrix. Bands overlap, 216-242 vs 208-215 GFLOP/s, and one shape is already behind |
| 6 | This table's prefill `1.18x` PASS, and the same figure on the README | an 18% margin. Flipping it needs upstream's 624-commit window to beat our fork's CPU GDN and SSM_CONV work outright |
| 7 | Pi 5 peak RSS 2.841 vs 3.747 GiB, `0.758x` | a 24.2% margin, and its denominator was already stock `b9892`, so only the 559 commits of `b9892`-to-`b10451` drift apply |

**x86_64 arm, first measured 2026-08-11 (#433).** Both arms above are AArch64 and their levers are Arm-only. Peak RSS **1.0022x, a hairline OPEN GAP**; prefill/decode/E2E **`PENDING` a quiet host**; CIQ `G5` open ([evidence](../bench-evidence/cpu-x86-llamacpp-20260811.md)).
