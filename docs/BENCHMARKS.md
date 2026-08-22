# Benchmarks <!-- ENG-RELEASE-WINDOWS: state=ACTIVE publication=pending artifact=unpublished -->

## At a glance: W5/W6 green; validated release artifacts pending

| Reference | Workload | Headline | Tokens |
|---|---|---|---|
| **Weight load (`ENG-LOAD-DIRECT-UPLOAD`, #150)** | Qwen3.6-27B bf16 (50.098 GiB), GB10 Vulkan, same binary both arms, interleaved under one GPU lock | **Load phase 1.54x warm** (19.27 -> 12.48 s), **1.61x cold** (52.62 -> 32.75 s); bytes moved **100.196 -> 81.260 GiB**. Every ON leg beat every OFF leg | byte-identical; 6/6 token-exact |
| **Expert streaming vehicle (`ENG-EXPERT-STREAM`, [#912](https://github.com/mudler/vllm.cpp/issues/912))** | Qwen3.8-2.4T-A95B `UD-Q1_0` (370 GiB) on ONE GB10, 119 GiB, c1 | **Loads and generates**, streaming OFF: resident 62 GiB, decode 66.7 s/tok. Experts BORROW the mmap; ~6.7 GB/token, ~50x off NVMe. Streaming-ON decode **VOID** (#912 F1); re-measured LIVE by `ENG-EXPERT-STREAM-DEVICE` | correct answer; no oracle runs this |
| **Device expert slices (`ENG-EXPERT-STREAM-DEVICE`, [#1124](https://github.com/mudler/vllm.cpp/issues/1124), [#1299](https://github.com/mudler/vllm.cpp/issues/1299))** | Qwen3.8-2.4T-A95B `UD-Q1_0` 370 GiB on `cuda` vs `cpu`, one GB10, greedy 32 tokens, **4000 expert-stream slots** (8000 is 3.6-4.1x slower), [record](../.agents/benchmark-record.md) | CPU **11.05 s/token** steady, replacing #912 F1's VOID. CUDA **G0-LIVE PASS**: 32/32 steps, `exhausted` delta 0, peak RSS 97.75 GiB. **G0-CORRECT FAIL**, cause NOT identified (not the alias: GB10). **G0-SPEED VOID** | correct; no oracle |
| **BPE merge loop (`SPEC-BPE-QUADRATIC-MERGE`, [#1365](https://github.com/mudler/vllm.cpp/issues/1365))** | 65,536 bytes in ONE word, both tokenizer goldens; one harness source, both sides, 20-core host at load 1.98 | **Quadratic to linear.** Mistral **23,620.7 -> 7.797 ms**; Qwen3.6 **22,813.1 -> 10.563 ms**. Per 4x input: 16x before, 4x after. At matched load, PARITY with HF `tokenizers` (10.563 vs 10.546 ms), not faster | 320/320 ids match HF `tokenizers` 0.22.2; re-derived at the close with the eight tokenizer suites green |
| **Request-length guard (`SERVE-REQUEST-LENGTH-GUARD`, [#1541](https://github.com/mudler/vllm.cpp/issues/1541))** | **No number owed, and none taken:** a refusing byte bound before tokenization, with no throughput, latency or memory gate declared | Not a rate. `67823aee2` already took 64 KB in one word to tens of milliseconds, so this bounds a 100 MB body and a future regression, not a measured cost | n/a |
| **Structured state record (active)** | v1 scalar + relational + Git-history contracts | No benchmark. At `776c56f1`: 157 imports = 3,231,342 exact bytes; append preserved all 156 wrappers/rows. 95 tests: validator/core 44 (checker 20 + core 24), NOW 18, migration 22, cutover 11. New raw-row mutation guard. | n/a |
| **Binary release (ACTIVE; Windows pre-alpha pending)** | v0.0.2 shipped eight primary archive/checksum/provenance triplets + two indexes (26 assets) from source SHA `7020de93652ca920424a10ac5255b34810dd2f24`, run `31466516224` | Windows W14-W16 implemented. **PENDING:** native hosted gates, merged-SHA ten-tuple dry run, matching-hardware evidence, v0.0.3-pre.1 publication, 32-asset audit | W12 optional/non-primary |
| **Container images (ACTIVE; arm64 cuda verified on GB10 + Orin 2026-08-11)** | `ENG-RELEASE-CONTAINERS` ([spec](../.agents/specs/container-images.md)) | cpu amd64 783 MB; cuda arm64 **1.71 GB**. GB10 `sm_121a`: `/health`+`/version`+SIGTERM on `--gpus all`. Orin `sm_87` (Tegra): Qwen3-0.6B **generates**, GPU **GR3D 95-97%** | n/a |
| **Developer/row protocol** | Contribution entry point; `ENG-NOW-DERIVED` #374 @`dbd0d51c` | Entry-point gates retained. #374 W1-W5 DONE; benchmark/runtime/parity `VOID`; row specs now carry `## Now` | n/a |
| **NemotronH host re-expansion / decode token** (`A2-Q2b`, [#810](https://github.com/mudler/vllm.cpp/issues/810)) | Real Nemotron-3.5-Lightning-30B NVFP4 (rev `29f2d174`), T=1 decode step, counted at the `DenseBf16` seam via `examples/nemotron_h_gen` | Full host arm **3 078 512 640 elements (6.157 GB bf16)**, 369 calls. `lm_head` **352 321 536 (11.44%)**, **28.35%** of the post-A2-Q2a residue, **100%** once the mamba arm lands | Per-group table in [the record](../.agents/benchmark-record.md). Attribution only, no speed number. Device `lm_head` gate PENDING a `dgx:gpu0` window |
| **Record-anchor ratchet** (`ENG-RECORD-ANCHOR-RATCHET`, #632) | **No number owed:** a record checker. At `8daa67b39`, **832 of 867** in-scope citations (**96.0%**) were already parsed; no symbol test and no report ran. Rot **38** (32 stale, 6 broken), **32 in range** |
| **NemotronH paged forward** (`MODEL-NEMOTRON-H-ABI-A2P`, [#810](https://github.com/mudler/vllm.cpp/issues/810)) | **No speed number, by the unit's own rule** ([spec](../.agents/specs/nemotron-h-a2p-paged-forward.md) §5) | **A3 96/96 `STRICT PASS`, mode=decode, GB10**, on the tree landed as `0ea5d249f` ([#1221](https://github.com/mudler/vllm.cpp/pull/1221)); NO run against current `main`. Fix reverted: 4/24 | CPU gate 12/12. Load 264.4 s, peak host 43 405 MB. NOT a benchmark |
| **LoRA runtime W2** (`LORA-RUNTIME`, #278) | **No number owed:** correctness-only; a grid PENDS the W7 model gate |
| **ARCH audit: ABI is text-only** | 4 capabilities (H3 video, Laguna, Kimi-Linear, DeepSeek-V4) reachable only from `examples/`, none registry-backed. No gate asks whether a CONSUMER can reach a capability. Documentation only |
| **CUDA-graph break seam W1** (`ENG-CUDAGRAPH-BREAK`, [#1192](https://github.com/mudler/vllm.cpp/issues/1192)) | **No number owed, and none taken:** coverage and correctness row, no throughput gate declared | Capability, not a rate: mid-forward capture re-begin holds on a leased GPU; scoped forward matches eager, 500 logits, 0 differing. Probe committed, recipe and sha256 in the [record](../.agents/benchmark-record.md) |
| **DSR fix: server TU profiler guards (2026-08-09)** | **No number owed:** comments only. #189 moved the server body into the shared layer with its 5 `VT_BENCH_PROFILE_CONTROL` guards, taking DSR 32 -> 37; they are `DSR-ALLOW`'d per site, baseline unchanged at 32 |
| **DSR fix: async readback capability (2026-08-08)** | **No number owed**: behavior-neutral (CPU/CUDA async-ON, discrete non-CUDA async-OFF, unchanged); moves a `kCUDA` check onto `Backend`, unblocking red CI on #127/#154/#155 |
| **`ROAD-V1-MEM` M1+M2 (2026-08-08)** | KV auto-sizing CPU brick: `--kv-cache-memory` sizes the pool from a byte budget via the group-aware `KVBytesPerBlock` divisor (ABI v16, CPU-gated). M3 profile run dgx-gated |
| **Record/checker repair 2026-08-07–08** | Gates fixed. Public: `VT_GEMMA4_EXPERT_VRAM_MB` (positive-MiB LRU cap; unset/0 unlimited), `VT_SERVER_MAX_{PROMPT_CHARS,NEW_TOKENS}` (200000/4096; 0 disables); nine Gemma4/ROCm tuners internal. No runtime/perf change. |
| **Clock attribution (`BENCH-ASSERT-CLOCK-STATE`, #543)** | dgx.casa GB10, driver 580.159.03, no throttling either side | **A CLOCK-CONTROLLED series now exists:** flat **2184 MHz** (requested 2190), one `boot_id`, one leg n=861 / min 2158. Every OTHER figure predates clock assertion, where med 2470 vs 2190 repriced `marlin::Marlin` +9.65% | n/a, nothing withdrawn |
| **vLLM** | Qwen3.6-27B NVFP4 `unsloth` @`890bdef7`, GB10 | ahead 4.5% at c1, **tie** at c2 to c32, all **SUPERSEDED**: rollback oracle (#520) + unfused denominator (#414), both flattering us. No pin re-run yet | identical |
| **vLLM** | Qwen3.6-27B NVFP4 `nvidia` @`0893e160` (ModelOpt `modelopt_mixed`), GB10 | **BINDING at the pin, clocks pinned: 0.976x c1 / 0.946x c4 decode TPOT**, TTFT 0.944x c1. c16 VOID (#577). The 0.937x-0.956x canonical is **SUPERSEDED and optimistic**: rollback oracle (#520) + unfused denominator (#414) | near-tie |
| **vLLM** | Qwen3.6-35B-A3B NVFP4 `nvidia` @`491c2f1e`, GB10 | **BINDING at the pin, clocks pinned: 0.995x c1 / 0.946x c4 decode TPOT**; TTFT 0.920x c1, 0.849x c4; c16 NOT ESTABLISHED. The 0.918x-0.972x canonical is **SUPERSEDED and optimistic** (#520, #414) | near-tie |
| **vLLM** | DeepSeek-V2-Lite (MLA), GB10 | 0.86x to 0.95x throughput, TTFT wins at c4/c8 | identical |
| **vLLM** | Laguna-S-2.1 NVFP4 (118B/8B MoE), GB10 | **parity+, 1.03x** (44.46 vs 43.10 tok/s, byte-exact, default config; bf16 weights now device-resident) | near-tie |
| **llama.cpp** | Qwen3.5-2B GGUF, CPU aarch64 | 20-core Arm/i8mm: prefill **1.18x ahead**, decode tie, memory parity. RPi5/A76: vllm.cpp is **0.461x prefill / 0.653x decode+E2E**, but uses **24.2% less RSS** | byte-identical on both Arm lanes |
| **MLX-LM** | Qwen3-0.6B, Apple M4 | 97.6% warm total, prefill ahead | near-tie |
| **DwarfStar** | DeepSeek-V4-Flash GGUF, GB10 | **beats ds4, 1.144x** (18.69 vs 16.33 tok/s, byte-exact, default config) | n/a, GGUF peer |
| **vLLM** | Kimi-Linear-48B-A3B, GB10 | no binding number: the published checkpoint is tiktoken-only, so it cannot drive the warm-server harness | golden 122/128, near-tie profile |
| **Muse Glimmer 30B (#268)** | no vLLM denominator (pin lacks `muse_glimmer`); SECONDARY llama.cpp, same GGUF, idle GB10, after the [#391 fix](../.agents/specs/cpu-decode-barrier-and-attn-dispatch.md) | **vLLM axis is an OPEN GAP.** vs llama.cpp: in128 prefill **1.023x** (was 0.878x); in512 decode **0.194x** (3.41x), prefill 0.175x flat. Denominator stock `7044859`, SUPERSEDED (#1003) | coherent, NOT token-exact |
| **#1047 prefill-peer Finish barriers (item 3)** | Gemma-4-26B-A4B-it-fp8, dual R9700, isolated :8012, T=2029, PEER_ACT=1 | **Attribution only, not a ship claim.** Two Finish success-path barriers cost **2.55%** (1122.10 vs 1094.24 tok/s; 46.05 ms/req) vs retained RetirePinThenUnpin wait. [record](../.agents/benchmark-record.md) | ident-equal 4/4 |

Reading the ratios: throughput is ours/reference, latency is reference/ours, so
**1.0 or higher is a win** everywhere on this page. Which architecture each number
is measured on, and the per-architecture correctness gate behind it, is the
registry-bound list in [FEATURES.md](FEATURES.md).

## vLLM, online serving

The binding comparison. vLLM runs its **production graphed config**, never
`--enforce-eager`. Every `0.25.0` row below is **SUPERSEDED and OPTIMISTIC**, on
two counts that both flattered us: the harness enforced the `0.25.0` ROLLBACK
and *raised* on the pin until 2026-08-12
([#520](https://github.com/mudler/vllm.cpp/issues/520)), and the same runs never
passed `--language-model-only`, so the denominator ran vLLM's UNFUSED
QK-norm+RoPE+gate path
([#414](https://github.com/mudler/vllm.cpp/issues/414)). Nothing is withdrawn.
The first series free of both, at the pin, graphed, and at a pinned clock is in
[the benchmark record](../.agents/benchmark-record.md).

| Model | Quant | vLLM oracle | Axes passing | Disposition |
|---|---|---|---:|---|
| Qwen3.6-27B | NVFP4 (`unsloth` @`890bdef7`) | 0.25.0 ROLLBACK, SUPERSEDED | **115/124** | Effective parity-or-better, two-grid totality. Revision-PINNED (the gate no longer lets `readdir` choose): @`ccdaab7e` is the same repo name re-quantized to FP8 W8A8 throughout, not NVFP4 |
| Qwen3.6-27B | NVFP4 (`nvidia` @`0893e160`, ModelOpt `modelopt_mixed`) | 0.25.0 ROLLBACK, SUPERSEDED | 0/6 | BEHIND, uniformly 0.94x on decode, flat c1-c32, but **VOID as a ratio** (#520, #414). At the pin, clocks pinned: **0.976x c1 / 0.946x c4 TPOT**. Different model from the `unsloth` row (NVFP4 MLP + FP8 W8A8 tower) |
| Qwen3.6-35B-A3B | NVFP4 `modelopt_mixed` | 0.25.0 ROLLBACK, SUPERSEDED | 2/18 | 3-rep grid 2026-08-05 @`1ea26427`: 0.93-1.03x, VOID as ratios (#520, #414). At the pin, clocks pinned: **0.995x c1 / 0.946x c4 TPOT**. ★ probe found a prod async batch-1 greedy DEGENERATION bug the mirror fixes |
| DeepSeek-V2-Lite | bf16 MLA | 0.25.0 ROLLBACK, SUPERSEDED | 4/25 | Attributed miss, row stays `ACTIVE` |
| Qwen3.5-4B | bf16 direct-load | 0.26.0.dev0, UNFUSED and vision-tower-carrying denominator ([#1345](https://github.com/mudler/vllm.cpp/issues/1345)) | **1.0283x tput, `PENDING` and OPTIMISTIC** | OPTIMISTIC on all axes (#414). TTFT/TPOT/E2E 1.085/1.017/1.029x; VRAM **+118.7 MiB** OPEN, not like-for-like: the oracle built the tower the flag elides ([data](bench-evidence/qwen35-4b-sm120-main-20260807.md)) |
| Qwen3.8-27B | bf16 (@`1d4bf0f2`) | 0.26.0.dev0 at the pin, graphed; c4 at a pinned 2184 MHz, the 2026-08-19 re-measure SAMPLED only | **1 of 3 concurrency cells** | Token gate PASSES. c4 like-for-like: tput **0.963x**, ITL **1.008x**. c1 re-measured COMPLETE, pairing DISCARDED on clock spread; c8 vLLM denominator NOT MEASURABLE ([#915](https://github.com/mudler/vllm.cpp/issues/915)) |

### GDN prefill kernels by GPU

| GPU | Workload and basis | vllm.cpp | vLLM | Ratio | Status |
|---|---|---:|---:|---:|---|
| RTX 5070 Ti (`sm_120`) | Qwen3.5-4B BF16 c32: conv / post-conv | 234.605→**219.506**; 122.587 ms | 145.421; 108.035 ms | 1.509x; 1.135x | Opt-in, byte-exact: K4 **6.436%**, tile **1.859x** ([conv](../.agents/specs/sm120-qwen35-conv-channel-tile-2026-08-08.md), [tile](../.agents/specs/sm120-qwen35-postconv-token-tile-2026-08-08.md)) |
| GB10 (`sm_121a`) | Qwen3.6-27B NVFP4, historical normalized prefill | 0.43 us/token/layer | 0.18 us/token/layer | **2.39x slower** | Directional only: unequal token clusters, older pin ([ledger](../.agents/parity-ledger.md)) |
| GB10 (`sm_121a`) | Qwen3.6-35B NVFP4, later local kernel A/B | 321.148 us c1; 960.313 us c6 | - | `PENDING` | Register vs tiled improved 4.7%/7.3%; no paired vLLM denominator ([record](../.agents/specs/gdn-prefill-conv-reg-2026-07-18.md)) |
| Jetson Thor (`sm_110`), AGX Orin (`sm_87`) | No matched GDN workload | - | - | `PENDING` | Runtime correctness only; no causal-conv speed trace |

### Qwen3.6-27B by concurrency

Medians of three interleaved repetitions, 1,024 in / 128 out, cache off, closed
loop. Output is token-for-token identical to vLLM at every point. The `nvidia`
ModelOpt table that follows is a DIFFERENT checkpoint on a DIFFERENT axis and
must not be compared against this one: a 7-token prompt scored on OUTPUT tokens
per second, with method, attribution and the owed 1,024 in / 128 out axis in
[the benchmark record](../.agents/benchmark-record.md).

| Concurrency | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| **vllm.cpp** TOTAL tok/s | **86.05** | 159.68 | 292.34 | 508.77 | 801.76 | 1095.01 |
| vLLM TOTAL tok/s | 82.32 | 158.03 | 290.31 | 505.46 | 789.16 | 1076.25 |
| **Ratio** | **1.045x** | 1.011x | 1.007x | 1.007x | 1.016x | 1.017x |
| Axes passing | 20/20 | 20/20 | 18/20 | 15/20 | 19/20 | 18/20 |

TOTAL tokens per second (prompt plus output), not a decode rate: c1's 86.05 is
about 10 tok/s of decode plus about 940 of prefill. The `nvidia` table below
counts OUTPUT only, hence its c1 of 10.41.
We are nominally ahead at all six, but only c1 means anything. Our run-to-run
noise band is 0.5% and c2 through c32 land between 0.7% and 1.7%, so **treat
those five as ties**, not as wins. The nine axes that fail in both grids are one
tradeoff, not nine problems: our synchronous deterministic forward loses on
low-concurrency *median* decode and TTFT, and wins the corresponding *tail* and
the same metric at higher concurrency (c8 p99 ITL 0.86x, but 1.055x at c16 and
1.078x at c32).

### Qwen3.6-27B NVFP4 `nvidia` @`0893e160` by concurrency (ModelOpt)

| Concurrency | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| **vllm.cpp** tok/s (canonical 2026-08-11) | 10.756 | 19.232 | 32.365 | 50.520 | 69.040 | 84.064 |
| vLLM 0.25.0 tok/s (canonical 2026-08-11) | 11.250 | 20.153 | 34.281 | 53.666 | 73.114 | 89.706 |
| Ratio, SUPERSEDED and OPTIMISTIC (main @`348c265d`, rollback oracle #520, unfused denominator #414) | 0.9561x | 0.9543x | 0.9441x | 0.9414x | 0.9443x | 0.9371x |
| **BINDING at the pin, graphed, `--language-model-only`, clocks 2184 MHz: decode TPOT** | **0.976x** | - | **0.946x** | - | VOID #577 | - |
| Same series: decode tput / prefill TTFT (c4 TTFT is a median) | 0.973x / 0.944x | - | 0.958x / 0.982x | - | VOID #577 | - |
| Levers `VT_GDN_PACKED_DECODE_FP8_TOWER=1 VT_GDN_FP8_IN_BF16=1` (both DEFAULT OFF), same-binary A/B: decode / TTFT | 1.007x / 1.048x | - | 1.012x / 1.041x | - | 1.027x / - | - |
| Driver verdict | `{"gate_pass": false}`; first six-point grid here. Levers `lm_head`+qkvz ACTIVE by kernel signature | | | | | |
| OPEN, and what it retracts | vLLM reproduces to 1.0%, OURS moved +14.8% at c1; lottery REFUTED (6 loads, 1.0046x); build diff leads. "c1 did not move" withdrawn: c1 was noise-dominated ([#349](https://github.com/mudler/vllm.cpp/issues/349)) | | | | | |

| Noise band, measured BEFORE any delta | ±0.03% c1 back-to-back; 0.29-1.85% leg-to-leg with reload, drifting down on BOTH arms so it cancels in the ratio | | |
| Startup, cold to `/health` | 33.38 s vs vLLM 182.41 s = **5.46x faster** | | |
| Peak host RSS | 21.10 vs 13.09 GiB = **1.612x, BELOW FLOOR, open gap** | | |
| Peak GPU memory | PENDING: `nvidia-smi` returns N/A on this unified-memory part | | |
| Step attribution (nsys, node-level, both arms same tool) | ours 98.906 vs vLLM 81.577 ms/step, 99.2/99.3% GPU-busy; lm_head 8.6414 + fp8 tower 7.6068 + splitK 0.0532 + other 1.0279 = **17.3292 vs measured 17.3292** | | |
| Lever 1, `lm_head` | ships U8/NVFP4 (0.666 GiB), we read 2.368 GiB BF16: **+1.702 GiB/step**, 11.183 ms. Marlin efficiency is EQUAL (207.9 vs 210.0 GiB/s), only bytes differ | | |
| Lever 2, GDN fp8 in_proj | identical 6.7188 GiB/step both arms; ours 96 GEMMs at 165.9 GiB/s vs vLLM 48 merged qkvz at 204.3; `in_proj_qkv` at **129.3 vs 213.6 GiB/s** | | |
| Lever 2b, packed GDN decode (#365) | separate c1 in16/out256 harness: 0.977x -> 0.984x, INDICATIVE not binding (arms not interleaved); see the record | | | | | |
| OPEN: host-memory-state sensitivity | same binaries read c1 0.7604 pre-reboot vs 0.8289 post; vLLM barely moved. Protocols also differed, variables not separated | | |
| Spread, ours / vLLM | 1.000 / 1.006 | 1.009 / 1.069 | 1.001 / 1.001 | 1.005 / 1.003 |
| Method | medians of 3, warm servers, one `flock`, greedy, `ignore_eos` so both emit exactly 128 tokens, `--gpu-memory-utilization 0.55 --max-model-len 4096`, vLLM in its production graphed config | | | |
| Tokens | greedy continuation IDENTICAL between engines, captured from the same warm processes as these numbers | | | |
| Reading | still a real loss, not a tie, and still FLAT across the sweep, so batching and the scheduler are not the cause | | | |
| Roof | 20.42 GiB over about 273 GB/s: vLLM's 79 ms/token is about 95% of the bandwidth limit, ours 96 ms/token about 78% | | | |
| Peak host RSS | 21.0 GiB, down from 24.2 GiB, because the FP8 tower is no longer expanded to BF16 | | | |

#### NVFP4 `lm_head` kept packed (`PERF-27B-LMHEAD-FP4`, #213)

| Axis | Packed (`VT_LMHEAD_FP4=1`) | Dequant (`=0`) | Result |
|---|---:|---:|---|
| Peak host RSS | 19.36 GiB | 21.06 GiB | **-1.70 GiB**, but measured BEFORE `ENG-LOAD-DIRECT-UPLOAD` (#150) made `LoadCtNvfp4Raw` borrow mmap'd bytes, which moves the RSS accounting; re-measurement OWED |
| Peak host RSS, non-CUDA | | | **Arithmetic, not measured.** A backend with no fp4 GEMM keeps packed + one bf16 operand: Vulkan **-1.70 GiB** (it used to stage a host bf16 head *and* a device copy), plain CPU **+0.67 GiB**. See `docs/USAGE.md` |
| Greedy continuation | identical to the dequant leg, byte for byte | | SOLID |
| `test_qwen27_paged_engine` | 235/235 | 235/235 | unchanged |
| tok/s, leg A / leg B | 11.197 / 11.193 | 9.418 / 10.163 | **INDICATIVE ONLY** |
| Reading, throughput | packed faster in all four legs; packed legs agree to 0.04%, dequant legs disagree by 7.9% | | DIRECTION established, MAGNITUDE not |
| Owed | binding grid: 3 reps per leg, order-alternated, c1/c2/c4/c8, medians of per-rep medians, before any ratio is quoted | | PENDING |
| Method | same model and revision as the table above, same-binary A/B on GB10 | | |
| Provenance | the row's fresh reviewer, gate checkpoint `nvidia/Qwen3.6-27B-NVFP4`@`0893e1606ff3d5f97a441f405d5fc541a6bdf404` | | |

### Qwen3.6-35B-A3B by concurrency

| Concurrency | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| **BINDING at the pin, graphed, `--language-model-only`, clocks 2184 MHz: decode TPOT** | **0.995x** | - | **0.946x** | - | NOT ESTABLISHED, 6.8% spread | - |
| Same series: decode tput / prefill TTFT (c4 TTFT is a mean) | 0.987x / 0.920x | - | 0.944x / 0.849x | - | 0.930x / NOT ESTABLISHED, 15.0% spread | - |
| Ratio, SUPERSEDED and OPTIMISTIC @`348c265d` (rollback oracle #520, unfused denominator #414) | 0.9708x | 0.9293x | 0.9719x | 0.9183x | 0.9264x | 0.9377x |
| **vllm.cpp** tok/s @`348c265d` | 65.262 | 93.012 | 140.174 | 193.477 | 251.490 | 311.897 |
| vLLM tok/s @`348c265d` | 67.223 | 100.088 | 144.226 | 210.679 | 271.479 | 332.606 |
| Ratio, superseded @`a0fa12c7` | 0.979x | 0.937x | 0.949x | 0.935x | 0.939x | 0.969x |
| Mean TPOT @`a0fa12c7` | 0.978x | 0.945x | 0.943x | 0.938x | 0.930x | 0.967x |
| Mean TTFT @`a0fa12c7` | 0.972x | **0.872x** | 0.970x | 0.965x | 0.969x | 0.968x |
| Our CoV @`a0fa12c7` | 0.39% | 0.26% | 0.59% | 0.60% | 0.37% | 0.41% |
| vLLM CoV @`a0fa12c7` | 0.62% | 0.35% | 0.81% | 0.57% | 0.50% | 0.35% |

**There is no isolated c2/c8 weakness.** Binding grid at `a0fa12c7`
(2026-08-10), 3 reps, binding-eligible 12/12; the prior c2 0.87x / c8 0.92x
"weak cells" came from a different harness and were never comparable. The
deficit is a flat mid-band (c2 to c16 within 0.935x-0.949x, CoV under 0.81% on
both engines) and is entirely MARGINAL per-token work, since our FIXED per-step
cost already beats vLLM (9.02 ms against 9.43 ms). Two glue levers land against
it, both default ON: the fused shared-expert gate_up sink off the MoE-marlin
route (**+1.31% c8 / +1.38% c4**, `VT_MARLIN_DENSE_PAIR`) and the shared
`down_proj` emitted as bf16 rather than f32 (**+2.05% c8 / +0.79% c4**,
bit-identical, `VT_SHARED_DOWN_BF16`).

The SiLU lever is NEGATIVE ([spec](../.agents/specs/moe-silu-vectorize.md)):
the 9.2x per-launch gap that motivated it was a MEAN over a bimodal kernel
(min 1.34 us, max 979 us) and our decode-phase SiLU is already faster than
vLLM's, so the remaining band is unattributed. The
marlin block-size lever is REFUTED (block 8 at c8 is 1.16% SLOWER against a
+0.29% control), and a per-launch marlin gap is WITHDRAWN as cross-tool
uncertainty. Memory passes decisively: peak PSS **3.81x**, peak GPU **1.40x**.
Detail in `.agents/benchmark-record.md`.

**Device-resident sampled tokens on integrated (`VT_ASYNC_DEVICE_MIRROR`) A/B'd
2026-08-06, speed-NEUTRAL** (same-binary): c16 OFF median 2305.8 vs ON 2303.3
(0.999x), c32 2928.9 vs 2919.1 (0.997x), bands overlap. It is a drain MOVE (relocate
the drain past the host prep, not remove it), so it overlaps only the small host
prep; the drain still serializes GPU input staging, so c16 does not recover. The
drain REMOVAL, `VT_ASYNC_EXECUTOR` Option A (`row/SERVE-ASYNC-OPTION-A`, default-OFF,
input H2D staged out of capture, the faithful vLLM structure), is GREEN + RED-reproducing
but binding-A/B speed-NEUTRAL (c8/c16/c32 +0.0-0.1%), so the c16 gap is NOT the async
input path. Detail in `.agents/benchmark-record.md`.

**But the mirror FIXES a shipping correctness bug, so it is now DEFAULT ON
(ROW-SERVE-ASYNC-LLM, 2026-08-06).** Baseline async (AsyncLLM depth-2) batch-1 greedy
decode degenerated into nondeterministic token-0 garbage; the mirror is deterministic
and coherent. The missing gate now exists, `test_qwen36_async_serving` (depth-2
AsyncLLM, batch-1 + concurrency, token-exact vs the SACRED oracle): RED on `=0`, GREEN
on the default. c16 re-checked on the default: 2312.9/2303.9/2294.4 (median
**2303.9**), c32 2942.7 (no regression). Root cause + file:line in the benchmark
record. The same P0 hit classic dense `Qwen3ForCausalLM` (quant-independent), fixed by `ROW-SERVE-ASYNC-DENSE-MIRROR` (see the MXFP4 Qwen3-8B row). The intake-drain lever likewise measured NEUTRAL (2026-08-06, `VT_INTAKE_DRAIN` A/B 3+3 reps): admitting during the forward wait collapses intake -91% but shifts it into queued, arrival-to-scheduled invariant, so the recorded INTAKE term is an attribution boundary over a GPU-bound prefill wait, not reducible; lever reverted, byte-exact `VT_LOOP_TRACE` probe kept.

### Qwen3.8-27B (bf16) by concurrency

| Axis | c1 | c4 | c8 |
|---|---:|---:|---:|
| Requests completed, ours / vLLM | 5,5,5 / 6,6,6 of 6 | 24,24,24 / 24,24,24 of 24 | 36,37,36 / 48,48,48 of 48 |
| Output token throughput | NOT ESTABLISHED as a ratio; both arms' absolutes are re-measured below | **0.963x** | NOT ESTABLISHED as a ratio; ours is re-measured below and vLLM has no leg |
| Total token throughput | NOT ESTABLISHED | **0.918x** | NOT ESTABLISHED |
| Status of the two withheld cells | SUPERSEDED 2026-08-19 by the re-measure rows below: the cause was our SSE keepalive frame and our arm now completes every request ([#931](https://github.com/mudler/vllm.cpp/issues/931)) | (cell stands) | SUPERSEDED: our arm is complete, and the missing half is now vLLM's, for a different reason ([#915](https://github.com/mudler/vllm.cpp/issues/915)) |
| Median ITL, over completed only | 1.013x | **1.008x** | 1.021x |
| Median TPOT, over completed only | 1.014x | 0.980x | 0.925x |
| Median TTFT, over completed only | 0.733x | 0.881x | 1.268x |
| Median E2EL, over completed only | 1.003x | 0.974x | 0.983x |
| ours / vLLM output tok/s | 2.37 / 3.50 | 15.01 / 15.58 | 15.96 / 27.85 |
| ours / vLLM median TPOT ms | 220.6 / 223.6 | 239.0 / 234.3 | 261.1 / 241.4 |
| Cold start to first `/health` | **53 s vs 780 s = 14.7x**, medians of 3 (ours 53/53/53, vLLM 786/780/771), NOT like-for-like readiness (next two rows) | same binary | same binary |
| Why that ratio is not like-for-like | ours answers `/health` on process liveness only (`api_server.cpp:286-294`); no dummy run, no kernel warmup, decode CUDA graph captures lazily on first use | vLLM warms up and captures before it serves (`gpu_worker.py:697-708`, `api_server.py:780-785`) | 53 s is "weights loaded", 780 s is "warmed and graph-captured" |
| What our readiness signal defers | first request TTFT **91.613 s**, the same with the SSE keepalive on and off, so it is genuine first-inference cost, not the [#931](https://github.com/mudler/vllm.cpp/issues/931) defect | cell stands as measured; a like-for-like comparison has not been taken | forensics in `.agents/benchmark-record.md` |
| Host memory after warmup | **42.5 vs 110.1 GiB = 2.59x**, but vLLM's is set by `--gpu-memory-utilization 0.85` pre-reserving KV, so it is what the configured engine holds, not what the model needs | | |
| **RE-MEASURED 2026-08-19, absolutes only** ([#915](https://github.com/mudler/vllm.cpp/issues/915), [#979](https://github.com/mudler/vllm.cpp/issues/979)) | ours **4.4040 tok/s** (CV 0.039%), vLLM **4.2835 tok/s** (CV 0.033%) | (cell stands) | ours **22.6402 tok/s** (CV 0.205%); vLLM **NOT MEASURABLE**, below |
| Requests completed on the re-measure | ours 6,6,6 of 6; vLLM 6,6,6 of 6; `failed=0` every leg | (cell stands) | ours 48,48,48 of 48, `failed=0`; 162/162 over the whole series |
| Median TPOT / ITL on the re-measure | ours 218.11 / 216.56 ms; vLLM 228.36 / 226.86 ms | (cell stands) | ours 250.57 / 232.83 ms |
| **Why no c1 ratio is quoted** | `gpu_clock_state compare` returned `PAIRING_VERDICT=DISCARD` on all three pairings. The ratio is OWED, not withheld for being unflattering | (cell stands) | no vLLM leg exists to pair |
| What the clock gate actually refused | cross-arm rule PASSED: same boot, both arms 2489 MHz, median offset **0.0%**. The within-run rule FAILED on both arms | (cell stands) | ours 12.92-14.99% spread, same breach |
| Within-run SM-clock spread, 5% ceiling | ours 13.58 / 26.36 / 14.34%, vLLM 10.16 / 17.48 / 18.52%; `SwThermal` in every window, `HwSlowdown+HwThermal` in one | (cell stands) | **all six of our legs and all three vLLM legs breached it** |
| Clocks were SAMPLED, never pinned | `nvidia-smi -lgc` returns `LGC_RC=4`, permission denied, as root inside an `rc` lease. Every prior pinned figure used the retired `ssh`+`flock` path | (cell stands) | same lease, same refusal |
| **vLLM c8 denominator** | (c1 landed) | (cell stands) | **NOT MEASURABLE on this box at the recorded config**, which is the answer and not a gap. Not a claim that vLLM is defective |
| Evidence for that verdict | (c1 landed: vLLM's own c1 cold start was 426 s and all three legs ran) | (cell stands) | `/health` at 373 s, then the worker was lost during warmup; the KV reservation took **48,715 MB in one 4 s window** (58,453 -> 9,738); last value 6,261 MB; death inside one 2 s sample |
| Why no watchdog can guard it | (c1 landed) | (cell stands) | ~6-7 GB headroom at `--gpu-memory-utilization 0.85 --max-num-batched-tokens 8192`; 12,000 MB kills a healthy server and 5,000 MB is never reached in time. Every other fix is an engine knob |
| Settled: the HOST rebooted | (c1 landed) | (cell stands) | OBSERVED: `boot_id` moved `3fd9745a-...` -> `64c495a3-...` in a later leased job, so not a lost pod. DERIVED, only if it does not virtualize `/proc`: boot at or before 10:41:47.6Z, one-sided. Causation NOT traced |
| #915's own-arm debt, DISCHARGED | prior 2.37 -> 4.4040 tok/s while median TPOT moved 220.6 -> 218.11 | (cell stands) | prior 15.96 -> 22.6402 while TPOT moved 261.1 -> 250.57. Different boots, so read the SHAPE, not the percentage |
| What that shape proves | throughput moved a lot, per-token barely at all; qualitative, the boots differ: the throughput axis was dividing live tokens by a wall still holding dead requests ([#931](https://github.com/mudler/vllm.cpp/issues/931)) | (cell stands) | withholding 0.677x rather than publishing it was correct, and this is the evidence for that |
| A divergence found in the raw files | our `usage.prompt_tokens` reports 5,942 where vLLM reports 6,144 on identical prompts; 19 of 48 short at c8 | (cell stands) | `output_lens` is `[128]xN` on both arms, so TPOT and ITL stand. Total-token throughput is CORRUPTED, and output throughput is BIASED UP, next row |
| Size of that bias, which EXCEEDS our own stated precision | if the prompts were truly truncated the wall is short by ~0.22 s of 174.39 s, about **0.13%**, beside a published CV of 0.039% | (cell stands) | ~1.1-1.6 s of 271.0 s, about **0.4-0.6%**, beside a published CV of 0.205%. A lower bound: shorter context also cheapens decode |
| Are the two TTFT medians still comparable? | YES, by evidence and not by luck: our two short prompts are the two LOWEST TTFTs in all three reps, so both arms' medians fall on 1024-token requests | (cell stands) | no vLLM leg exists to compare against |
| Why only c4 counts | `output_throughput` divides tokens by a wall duration that still contains the dead request, so a cell where one arm dropped requests is withheld, not quoted | 3 paired reps, clocks 2184 MHz | token gate: 4/7 strict, 3 exact fp32 ties ([#915](https://github.com/mudler/vllm.cpp/issues/915)) |

### Qwen3.8-27B quantized arms, both gates PENDING and no number quoted

| Arm | Oracle, and why that one | Gate | Blocked on |
|---|---|---|---|
| `Qwen3.8-27B-Q4_K_M.gguf` (`unsloth` @`fe1e2a23`) | llama.cpp `b10451`, the ONLY comparator: vLLM has no in-tree GGUF reader at pin `555967922` and SGLang's aliases miss `qwen3_5` ([#979](https://github.com/mudler/vllm.cpp/issues/979)). Its oracle, never its mirror | **PENDING**, no number | [#857](https://github.com/mudler/vllm.cpp/issues/857): `.agents/oracles/llama-cpp.md` records the pin `gateable = no` and #857 owes that measurement |
| `unsloth/Qwen3.8-27B-NVFP4` @`7d6f8d4d` | vLLM `555967922`, the MIRROR and primary oracle, which runs this format | **PENDING**, no number | [#1185](https://github.com/mudler/vllm.cpp/issues/1185): the pinned oracle builds, installs and imports in an `rc` lease, but RUNNING a model there is untested |
| Why no number may be quoted yet | Correctness precedes speed and neither arm loads: the "NVFP4" file is `mixed-precision` with ZERO `*.input_scale` tensors, and the Q4_K_M file's block 64 is the MTP drafter (`nextn_predict_layers = 1`) | spec'd, not implemented | [#821](https://github.com/mudler/vllm.cpp/issues/821), [spec](../.agents/specs/qwen38-27b-quant-arms.md) |
| What a token gate alone would miss | Each arm owes a resident-bytes assertion beside its tokens: a Q4_K_M arm that silently dequantizes to bf16 passes every token gate and defeats the point of the arm | owed, not measured | [#821](https://github.com/mudler/vllm.cpp/issues/821) |

### DeepSeek-V2-Lite (MLA)

Medians of 3 reps, 1,024 in / 128 out. The vLLM arm runs `--moe-backend triton`,
which is its best **stable** graphed config on GB10: the auto-selected FlashInfer
CUTLASS MoE backend hard-rebooted the box five times. The substitution does not
flatter us, we lose against it.

| Concurrency | Output tok/s ours / vLLM | Ratio | Median TTFT | Median TPOT |
|---:|---|---:|---:|---:|
| 1 | 33.18 / 38.17 | 0.87x | 0.95x | 0.90x |
| 2 | 52.63 / 55.27 | 0.95x | 0.88x | **1.03x** |
| 4 | 70.36 / 81.51 | 0.86x | **1.04x** | 0.86x |
| 8 | 102.37 / 116.35 | 0.88x | **1.12x** | 0.86x |

Peak memory is the decisive win: **31.38 GiB against vLLM's 68.5 GiB**, with the
caveat that vLLM pre-reserves a fixed fraction up front while we allocate the KV
blocks the workload needs. Real difference in operating footprint, not evidence
of a lower per-token KV cost.

### Laguna-S-2.1 (NVFP4)

`poolside/Laguna-S-2.1-NVFP4`, 118B total / ~8B active MoE, 48 layers,
256 experts, ~67 GiB. Both arms NVFP4, single request, batch 1, GB10. The local
checkpoint directory is named `laguna-xs-nvfp4`, which is where the "Laguna-XS"
label in earlier revisions of this page came from; the measured model is S-2.1.

| Arm | Decode tok/s | Ratio |
|---|---:|---:|
| vLLM NVFP4, graphed | 43.10 | 1.00x |
| **vllm.cpp NVFP4**, resident decode + CUDA graph | **44.46** | **1.03x** |

The gap is CLOSED (2026-08-04, same-tool nsys graph-node tracing on both
engines). Root cause: the bf16 M=1 projection GEMVs (o_proj, qkv, router,
dense, lm_head) read their weights from GB10 UNIFIED/ATS host memory (a
`w.View()` device retag, no `cudaMalloc` staging), which the GPU reads slower
than true device memory. `VT_LAGUNA_RESIDENT_BF16W` (default-ON parity enabler)
stages every projection device-resident (one H2D copy at load); byte-exact
(ids bit-identical to the `=0` retag arm). Binding new-default decode is 44.46
tok/s (median-of-3), 1.03x vs vLLM's graphed 43.10.

Per-call the residency recovers o_proj 194 to 131 us (about 168 to 249 GB/s),
qkv 245 to 225 us, and lm_head 2410 to 1620 us/call. This supersedes the earlier
invocation / bf16-output-`cublasGemmEx` framing (measured a wash): the
invocation was never the cause, the weight's memory RESIDENCY was. Full
forensics in `.agents/benchmark-record.md`.

Memory: the roughly 2.6 GB of bf16 device copies sit in the 119 GB GB10 unified
pool; peak host RSS is unchanged at about 40.7 GiB (device/unified allocations
are not counted in `ru_maxrss`), so there is no memory regression.

## Memory

Qwen3.6-27B NVFP4, GB10, whole serving window.

| Axis | vllm.cpp | vLLM | Ratio | Result |
|---|---:|---:|---:|---|
| Peak PSS | 24.88 GiB | 28.18 GiB | 1.133x | **PASS** |
| Peak RSS | 24.88 GiB | 28.56 GiB | 1.148x | **PASS** |
| Peak GPU memory | 40,996 MiB | 70,531 MiB | 1.720x | **PASS** |
| Peak `MemAvailable` drop | 68.35 GiB | 80.66 GiB | 1.180x | **PASS** |
| Weight offload, resident device bytes (`ENG-WEIGHT-OFFLOAD` W6) | not measured | not measured | n/a | **BLOCKED**, not pending: unmeasurable on every host we own (GB10 shares one pool, so `cpu_offload_gb` frees nothing). Needs a discrete-GPU rig ([record](../.agents/benchmark-record.md)) |
| Disk residency via `--offload-config` (`ENG-RESIDENCY-CONFIG`, [#1110](https://github.com/mudler/vllm.cpp/issues/1110)) | not measured | n/a (no disk tier upstream) | n/a | **PENDING** a GB10 run. The row changes no kernel, dtype or allocation, so it claims no throughput axis; the 370 GiB reproduction through the JSON form is owed ([spec](../.agents/specs/weight-residency-config.md)) |
| Decode-graph executables and device bytes, `VT_CUDA_GRAPH_DEDUP` (`ENG-CUDAGRAPH-DEDUP`, [#1162](https://github.com/mudler/vllm.cpp/issues/1162)) | COARSE key 3/7 and 5/11 execs; 15.40 vs 29.24 MiB nominal at 7 buckets | n/a | 0.43x / 0.45x execs; bytes NOT ESTABLISHED | **NEGATIVE, decided.** The fold engages; the saving fails its null control -- 0.42% of process at 7 buckets, none at 11. Default stays OFF. No time figure, clocks unpinned ([record](../.agents/benchmark-record.md)) |

35B steady-serving PSS is 3.53 GiB against vLLM's 13.3 GiB after the routed-expert
host mirror is freed once the device Marlin resident is built.

## llama.cpp, CPU

| RPi5/A76 arm (R4-R6, `GATING`) | Result |
|---|---|
| Scope | 4-core A76, DotProd, no i8mm; 20-core binding arm does NOT transfer. buildx/QEMU-built, Pi-executed unthrottled; hashes pinned in the [campaign spec](../.agents/specs/rpi5-cortex-a76-cpu-optimization.md) |
| Assembly vs compiler SDOT (one GCC 13.3 binary) | AAPCS64 leaf wall +3.66% M1/T1, +5.08% M128/T1, +3.69% M128/T4, with 9.74-10.24% fewer instructions; M1/T4 is the -2.43% residual ([assembly evidence](bench-evidence/rpi5-a76-q8-dot-20260806.md)) |
| 64-token Qwen model gate | Byte-identical across x86, portable, SDOT and assembly arms; asm vs SDOT median TTFT -1.55%, TPOT neutral, E2E -0.13%; vs portable TTFT -33.40%, E2E -2.67%. Cortex-A76+DotProd selects assembly by default |
| Same-file llama.cpp floor (pp17/tg64) | **NOT MET on speed**: prefill 12.81 vs 27.77 tok/s (0.461x), decode 2.55 vs 3.91 (0.653x), E2E 26,018.39 vs 16,998.49 ms ([competitor evidence](bench-evidence/rpi5-a76-llamacpp-20260806.md)); vs stock `b9892`, SUPERSEDED |
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
[spec](../.agents/specs/oracle-llamacpp-repin-stock.md).

| Flips first | Verdict | Exposure |
|---|---|---|
| 1 | Vulkan `BENCH-VK-LLAMA` decode 4.36 vs 4.35, `MET` | 0.23% margin inside a 0.69% 7-leg spread. Its own source calls it "a narrow pass, not a comfortable one", so any move can flip it. The README states it as "matches" |
| 2 | Muse Glimmer in128 prefill `1.023x` (also `STATUS.md`) | 2.3% margin inside our own arm's 4.5% leg spread, n=4. Its denominator is already stock `7044859`, 84 commits from `b10451`, so the noise floor is what exposes it |
| 3 | This table's peak memory `1.01x` PARITY and decode `0.97x` tie | ties by declaration, not wins. A denominator that moves at all in llama.cpp's favour turns both into recorded gaps |
| 4 | keep-f16's `1.01x` RSS and "prefill `1.18x` AHEAD", quoted in product code for `VT_GGUF_KEEP_F16` default ON | buys 1.05 GiB of peak RSS (3.885 to 2.832) for ~9% prefill (224 to 204 tok/s) and ~1.4% decode, tokens identical. **Decided 2026-08-17: the default stays ON on our own arms, so it needs no re-take.** Both ratios do |
| 5 | `KERNEL-GEMM-CPU-TILED` NEON vs stock ggml sgemm, "at parity, ahead on 4 of 6 shapes" | off this page, in the kernel matrix. Bands overlap, 216-242 vs 208-215 GFLOP/s, and one shape is already behind |
| 6 | This table's prefill `1.18x` PASS, and the same figure on the README | an 18% margin. Flipping it needs upstream's 624-commit window to beat our fork's CPU GDN and SSM_CONV work outright |
| 7 | Pi 5 peak RSS 2.841 vs 3.747 GiB, `0.758x` | a 24.2% margin, and its denominator was already stock `b9892`, so only the 559 commits of `b9892`-to-`b10451` drift apply |

**x86_64 arm, first measured 2026-08-11 (#433).** Both arms above are AArch64 and their levers are Arm-only. Peak RSS **1.0022x, a hairline OPEN GAP**; prefill/decode/E2E **`PENDING` a quiet host**; CIQ `G5` open ([evidence](bench-evidence/cpu-x86-llamacpp-20260811.md)).

## MLX-LM, Apple M4

Qwen3-0.6B, warm, batch 1, 6 interleaved runs.

| Axis | vllm.cpp | MLX-LM | Ratio |
|---|---:|---:|---:|
| Prefill TTFT | **524.5 ms** | 532.6 ms | **1.015x** |
| Decode | 27.23 tok/s | 27.85 | 0.978x |
| Warm total | 24.37 tok/s | 24.96 | 0.976x |

The 2.4% is a real gap, not noise: our spread was 0.12% and MLX-LM's 0.34%. All
of it sits in decode, 0.81 ms per token. Indicative rather than binding: two
models, 18 of 75 ops native, and the 97.6% needs the optional MLX GEMM provider
shape-gated to prefill (95.9% on the default build).

## DwarfStar, GGUF

DeepSeek-V4-Flash cannot run on vLLM on a single GB10 at all: every
vLLM-loadable checkpoint is 156 GB or larger against a 119 GiB unified pool, so
the only quant that fits is extreme-low-bit GGUF, which vLLM cannot load here.
GGUF was forced by the hardware. A policy-correct vLLM comparison needs 2x GB10
Sparks with TP2 and is owed.

| Engine | Quant | Decode tok/s | Ratio |
|---|---|---:|---:|
| DwarfStar (`ds4`) | IQ2_XXS mixed | 16.33 | 1.00x |
| **vllm.cpp** (default) | same GGUF | **16.28** | **0.997x, parity** |
| **vllm.cpp** (`VT_V4_RESIDENT_W` default-ON) | same GGUF | **18.69** | **1.144x, byte-exact** |

The default arm is parity, measured same-session clean (2026-08-04, single-load
steady both arms); the earlier 15.87/96% and 17.13 figures are superseded.

Weight residency is the beat-path (2026-08-05, `VT_V4_RESIDENT_W`, default-ON). Env var allowlisted (env-doc gate green).
The dense Q8_0 MLA/shared-expert/lm_head projection tower is read from the GGUF
mmap over ATS/unified memory, which the GB10 GPU reads about 20% slower per-GEMV
than `cudaMalloc`'d device memory. Staging that ~6 GiB tower device-resident once
at load (same bytes, same kernel, same invocation) lifts decode 16.23 to 18.69
(median-of-3, drop_caches), generated ids byte-identical. It is the same lever
that took Laguna to vLLM parity+ (`VT_LAGUNA_RESIDENT_BF16W`).

PEAK RESIDENT is flat at 86.68 GiB in both arms: the staged copy is additive but
the clean mmap file pages evict under the unified pool, so net usage does not
grow. An nsys A/B (identical instance counts) confirms the mechanism is
residency-bound, not latency-bound: per-launch time drops about 20% on every
dense Q8_0 kernel (`QuantDotGemmQ8_0Kernel` 184 to 147 µs, `Q8_0GroupDiagKernel`
212 to 166 µs, `Q8_0PairKernel` 74 to 60 µs). This corrects the earlier
"per-launch GEMV parity / Q8_0 weight-stream floor" framing: our GEMV was
ATS-bound, not at ds4 parity.

Phase-2 staged the routed-expert slabs too (the ~70 GiB IQ2/Q2_K bulk,
`VT_V4_RESIDENT_EXPERTS`, first-touch `cudaMalloc` plus immediate
`madvise(MADV_DONTNEED)` per slab so the transient stays ~flat). It was **measured
NEGATIVE (2026-08-05) and is HELD default-OFF** as a characterized artifact.
Same-binary median-of-3, warm-cancelled steady, drop_caches: OFF (Phase-1) **19.43
tok/s** vs ON **18.76 tok/s** (0.966x, ~3.4% slower), generated ids byte-identical
(md5 equal across all 6 runs), PEAK RESIDENT flat at 86.6 GiB. The move itself
works: host RSS drops 86 to 14 GiB as the mmap pages are reclaimed.

The regression matches the roofline. Unlike the dense Q8_0 tower (63% of DRAM
peak, bandwidth-bound), the grouped-MoE `QuantDotGemmGrouped<IQ2_XXS>`/`<Q2_K>`
kernels run at only ~19-24% of peak (dequant/latency-bound), so weight residency,
a bandwidth lever, cannot help them. It also adds a large one-time graph-capture
cost, and pinning the 70 GiB as `cudaMalloc` (vs evictable mmap file cache) cuts
the unified-pool reclaimable headroom from ~103 to ~30 GiB avail. The lever stays
in the tree, default-OFF, for reproducibility; detail in the benchmark record.

## Speculative decoding

| Speculator | Model | Result | Status |
|---|---|---|---|
| MTP | Qwen3.6-27B NVFP4 | token-identical to vLLM MTP, **~4% faster at c1**; on-par at c2-c8 | `DONE` |
| DFlash | Qwen3.6-27B NVFP4 | **2.9x over spec-off** (10.16 → 29.32 tok/s), at/above vLLM DFlash-on (**1.003x**, non-overlapping bands) | `DONE` |
| n-gram | Qwen3.6-27B NVFP4 | draft-free (`SPEC-NGRAM`); 27B 5/5 STRICT our-ngram-ON == vLLM-ngram-ON, 180/180 drafts accepted (correctness only, no speed row yet) | `DONE` |
| DSpark | 27B NVFP4 dense k=15; 35B-A3B MoE k=8 | MoE 35B-A3B: **0.835x** paired on kairos-17dd (matched 89 tokens, warm oracle cache). Prior 0.957-0.989 came from a different machine with a cold oracle (#442) | `ACTIVE` |
| DSpark block floor | Qwen3.8-27B + `RadixArk/Qwen3.8-27B-DSpark` @ `85ef153b` | a `k` below the draft's block is refused instead of drafted; the run gate that exhibits the garbling is **owed** and needs a GPU lease (#1225) | `ACTIVE` |
| DSpark draft routing | Qwen3.8-27B + `RadixArk/Qwen3.8-27B-DSpark` @ `85ef153b` | **PENDING**, no number. The token-exact run gate needs the 2.53 GiB draft and GPU time, and neither authority is recorded; only the CPU classification gate has run (`.agents/specs/dspark-qwen3-routing.md` §6) | `ACTIVE` |
| DFlash2 | `Qwen3.8-27B` + `z-lab/Qwen3.8-27B-DFlash2`, k=7, GB10, oracle BEYOND-PIN on `TRITON_ATTN` (not vLLM's default here, #1456) | **Correctness gates READ**: 4/4 token-exact, 45/47 draft blocks identical, acceptance IDENTICAL per prompt (209 both). Acceptance is a COROLLARY of the draft result, not a second measurement. **No speed number** | Off the paged CUDA-graph fast path while the oracle GRAPHS its draft step; a GGUF drafter costs 3.584 GiB resident against 1.06 on disk; gate head one merge behind vLLM main; ours-tree RECONSTRUCTED (#1314, #1561, #1562) | `ACTIVE` |
| Drafter chain (`SPEC-DRAFTER-CHAIN`, [#1522](https://github.com/mudler/vllm.cpp/issues/1522)) | n/a, nothing runs a chain yet | **PENDING, no number admissible.** W1 landed the field and its refusals only. The premise is an ACCEPTANCE claim, invisible to a token gate, so W2/W4 must read first. G1 and G5 PASS on CPU | `ACTIVE` |
| Breadth (EAGLE1/3, suffix, ngram-gpu, dynamic-k, ...) | n/a | enumerated from vLLM source + `INVENTORIED` 2026-08-06 (`.agents/specs/spec-decode-inventory.md`), unmeasured | `INVENTORIED` |

## How we measure


Record dates are CI-guarded: structured state event timestamps and ordered
indexes are validated by `check-state-record`, so scoreboard stamps remain
traceable. The
review protocol behind these numbers is guarded the same way: the reviewer and
implementer sub-agent prompts are tracked artifacts checked by
`check-protocol-consistency` (orchestration harness step 5/5), and
`check-gate-commands` pins the record rows that name a gate command able to
FAIL. That pin is exact, not shrink-only: gaining a gate command reddens it too,
so the set is never re-pinned silently in either direction (#621). Since 2026-08-07,
a PR verified green merges in that same session (disposition rule).

**Hardware.** NVIDIA GB10 / DGX Spark (sm_121a) for CUDA, `dgx.casa` aarch64 for
CPU, Apple M4 for Metal. GB10's 119 GiB pool is unified, so host and device
memory compete; end-to-end wall-clock on a cold page cache is unusable there,
and steady-state per-step timing or `nsys` GPU-busy is the anchor. The
2026-08-06 #77-slip tree-revert changed no benchmark content or number.

**Oracle pin.** vLLM 0.26.0.dev0 (`55596792`) plus transformers 5.14.1, built from
source for sm_121a; the running oracle reports `0.23.1rc1.dev1511+g555967922` with
FlashInfer `0.6.15.post1`, and the binding series selects it by explicit path and
asserts that identity per leg. Speed figures labelled 0.25.0 ran the ROLLBACK the
harness enforced until 2026-08-12 and are SUPERSEDED, never binding (#520).
Correctness re-validated bit-identical across the advance, zero golden drift.
The llama.cpp oracle is stock `b10451` since 2026-08-16, `gateable = no` until
someone builds it (#857). Every llama.cpp figure here is SUPERSEDED and owed a
re-take (#1003), and a sweep finds it ran one of **three** revisions, none of
them the pin. The Reproduce table names all three.

**Protocol.** Greedy, closed loop, three interleaved repetitions per point, one
`flock` across the whole series, same-binary A/B for every lever, cold legs
discarded. Workload equivalence between arms is audited, not assumed: batch cap,
token budget, context, corpus bytes, KV and SSM dtypes, kernel family, and
graphed decode all match, and the audit is
[recorded](../.agents/specs/benchmark-equivalence-audit-2026-07-15.md). The
2026-08-04/08 governance checkpoints (record/CI substrate, anchor backfill,
operator/helper W0-W5, upstream/device inventory, onboarding probe, the
review-hardened `agent-start.py` entrypoint, and CI-bound review-until-PASS
policy) touched no engine code and moved no number: **NOT APPLICABLE**, nothing
to reproduce.

The PR #28 sanitizer repair is also NOT APPLICABLE to performance: both full
333-test CPU detector lanes pass after merging upstream `main`, while the
ASan+UBSan build footprint falls from 93 GiB to 5.7 GiB and TSan occupies
1.9 GiB. Reproduce with the sanitizer
CTest commands preserved in the structured state evidence. The 2026-08-06 live-state audit and the 2026-08-08 state-record migration plus range-gate and stale-reference repair are likewise NOT APPLICABLE: bookkeeping, record checkers, and prose. No engine code, kernel, or number on this page changed.

**Vocabulary.** *Token-exact* means our output ids equal the reference's, byte
for byte. *Near-tie* means the reference's own greedy decode is not deterministic
at this precision, so the gate is distributional: our output must fall inside the
set the reference produces across K runs. *Tie* means the difference is inside
the measured run-to-run noise band, which is 0.5% on GB10 and 0.12% to 0.34% on
M4. We never publish a partial, contended, or stale-denominator number as
binding, and when a denominator turns out to be wrong we correct every ratio
built on it rather than keeping the flattering one.

**CPU elementwise GEMM, wide x86 tiers (2026-08-07).** INDICATIVE ONLY, not binding: the x86 dev box is VOID for timing per `CLAIM-KERNEL-CPU-ELEM-GEMM-1`. The AVX-512 tier measures 1.56x to 2.83x over SSE2 on the elementwise micro-kernels, byte-identically. A binding number needs a qualified x86 host, which the project does not have.

**CPU elementwise GEMM, transpose-free `[K,N]` path (2026-08-07).** On dgx aarch64 the `[K,N]` path beats `[N,K]` by 1.16x to 1.30x, byte-identically. The x86 arm is INDICATIVE ONLY, not binding: that box is VOID for timing per `CLAIM-KERNEL-CPU-ELEM-GEMM-1`. `VT_CPU_MATMUL_STEAL` ships default OFF and is NOT measured; it must justify itself by measurement and may measure neutral.

## Open gaps

| Track | Status | Next gate |
|---|---|---|
| Surface coverage (`ARCH-ONE-SURFACE`) | **CORRECTNESS COMPLETE:** #139 restores DSR 32 (`kcuda=0`) via registry/name resolution; ABI-v14 selection unchanged; no speed claim | Selector 2/2·11 plus execution-bound CMake/File-API/CTest + CI/preflight + manifest-integrity guard 52/52; CPU platform/loader/C-ABI tests green; CUDA A/B remains residual |
| 35B prefill TTFT | **0.920x c1 / 0.849x c4** against a correctly FUSED denominator at the pin; the 0.93x-0.98x reading came from an UNFUSED oracle (#414) and is void | Re-attribute the residual against the fused denominator, then close |
| c16 and above, both models (#577) | **VOID, not a number.** 93/96 against the pin's 96/96, the three missing being the SLOWEST: our SSE keepalive was ON. Recurred on Qwen3.8-27B at 5/6 and 36/48 ([#931](https://github.com/mudler/vllm.cpp/issues/931)) | Keepalive now default OFF and rate harnesses refuse `failed != 0`; re-run c16/c32 on a binary carrying both |
| Clock-controlled pin grid (#520, #414, #543) | **First defensible series LANDED** at 1024/128, n=3, interleaved, one boot, flat 2184 MHz. Only c1, c4 and a partial c16 exist | c2, c8 and c32 at the pin under clock control, so the sweep is a sweep |
| 35B low-batch MoE decode | CLOSED at low batch (c1 0.975x, c4 wins); c16 0.93x. `VT_ASYNC_DEVICE_MIRROR` **default ON for correctness**. `VT_ASYNC_EXECUTOR` Option A (H2D out of capture) A/B'd speed-NEUTRAL | c16 lever is prefill glue (task #61), not the decode drain. `test_qwen36_async_serving` GREEN |
| CPU keep-quant MoE decode | **No number owed**: correctness-only P0. The grouped keep-quant GEMM read activations as f32 whatever their dtype, so CPU MoE decode emitted token-0 garbage from `b4f5610a` (2026-07-31) | Speed unmeasured and unclaimed; `test_ops_quant_dot` GREEN (150224 assertions) |
| Accepted-and-inert serve args (`SERVE-RECIPE-ARGS`, #606) | **No number owed**: argument parsing only, so nothing to time and no oracle leg. Correctness gate 4 cases / 58 asserts GREEN, RED-first, mutation-proven | None. A speed axis would be fabricated; closes on review plus the operator gate rerun |
| GDN core/z at the model dtype, MoE arms (`GDN-MOE-BF16-OUT`, [#1168](https://github.com/mudler/vllm.cpp/issues/1168)) | **No number, none claimed.** `GATING`, CPU tier only. A token gate cannot see this axis: f32 is the MORE precise deviation ([spec](../.agents/specs/gdn-moe-bf16-out.md)) | 35B correctness first, then 315/315, 235/235 dense inertness, the `VT_GDN_OUT_BF16` 0-versus-1 A/B per leaf, and `nsys` for the memory format. All `PENDING` a GPU host |
| NemotronH model row (`MODEL-TEXT-nemotron-h-...`, [#517](https://github.com/mudler/vllm.cpp/issues/517)) | **No number, none claimed.** The row's FALSE `KERNEL-SSM-MAMBA` block is corrected (#1074); it STAYS `INVENTORIED`, because the A3 96/96 belongs to `0ea5d249f`, not to current `main` | A re-run of the A3 gate against `main`. Only a pass measured there moves this row, or gives a grid a denominator |
| DeepSeek-V2-Lite MLA | Attributed miss, `ACTIVE` | Throughput at every concurrency |
| Qwen3.5 text-only arms (#490) | **No number; run gates OWED**, both `PARTIAL`. The loader half is CLOSED (#740, #864 `DONE`), so what blocks these is hardware, not a refusal | No fitting ckpt for either causal-LM arm: no denominator. `Qwen3.8-2.4T-A95B` is ~4.8 TB vs 128 GB; its load plan resolves, which is not a token |
| Darwin Qwen3.5 build repair (#1054, 2026-08-16) | **NOT APPLICABLE.** Removing a redundant namespace-scope lambda capture that Apple Clang rejects under `-Werror` changes no generated refusal text, no model math and no runtime path | None. The binding gate is the Apple Clang build |
| Qwen3.6-35B-A3B published BF16 (#740, #864) | **No number, and none was owed: the 2026-08-15 gate measured TOKENS.** Correctness MET vs the pinned oracle: 6/7 prompts STRICT 16/16, the 7th an exact tie (#910); SACRED 3/3 byte-identical | A throughput / latency / memory grid on this checkpoint. Nothing is measured, so nothing is claimed |
| MoE vision tower image + video (#891) | **NOT gated, no number.** The 333 `model.visual.*` tensors load and the tower computes, on sm_110 FALLBACK attention, not the shipped GB10 path. The token-exact mm gates never ran | Both modality gates on GB10 through the shipped fast path, then a per-modality speed grid |
| Dense image/video after the #891 merge (#908) | **UNVERIFIED, network-blocked.** Dense TEXT is 235/235 at `2f2bce926`, a true before/after (binary md5 `db889909d4…` vs `49ded1ece8…`, 500 TUs recompiled). The modality arms were not re-run | Re-run the dense image and video gates once the fixtures are reachable |
| Qwen3.5 upstream throughput levers (roadmap C10) | NOT MEASURED. vLLM's 2026-08-06 25K tok/s/GPU is a GB200/NVLink72 disaggregated cluster result, not comparable to one GB10, and is NOT adopted as our bar | Advance the parity pin past `555967922` so the referenced PRs exist, re-capture goldens at zero drift, then port the GDN prefill kernel |
| DeepSeek-V4-Flash | **Parity with ds4 (0.997x)** | Optional beat-path: f16 tensor-core DSA/router (near-tie class) |
| DeepSeek-V4-Flash vs vLLM | Infeasible on one Spark | 2x GB10 with TP2 over the NCCL seam |
| Tensor parallelism (task #287 spike, 2026-08-08) | **No number owed** (`benchmark_binding=false`): records-only scope at pin `555967922` ([spec](../.agents/specs/tensor-parallelism-spike.md)). TP-W1 landed (group table, 6/6); TP-W2..W7 gate on token-exactness, not speed | Perf gate is TP-W6: at or above vLLM TP=2 on every axis, same 2-GPU box (PENDING-HW) |
| Multimodal image, audio, video | Correctness gated, speed unmeasured | Per-modality speed grids |
| Voxtral enc attn (`VT_WHISPER_ENC_FA2`, #432) | FA-2 hd-64: 731.7 -> 133.0 ms (5.50x); enc fwd (not TTFT) vs pin TTFT 46.02: 15.90x -> 2.89x | **OPT-IN**: 3 near-tie diverg (band PASS) vs 0. Dev call |
| `/v1/videos` OpenAI + ONE-SURFACE ROW 2 | **No number owed:** ABI-v12 device selection is backend-dispatch plumbing; generation math and speed paths are unchanged | DSR 34→32; baseline/allowlist unchanged; 25/25 checker mutations; CPU fold 6/137, including one-queue/device-provenance mutations |
| Qwen3-dense decode CUDA-graph | Token-exact pass, ~4.3% e2e directional | Steady-state per-step tok/s |
| Kimi-Linear-48B-A3B (KDA+MLA+MoE) | **RUNNER FOLD LANDS (ROW 7, §21, #122): engine==CLI 128/128 byte-identical; vs golden 122/128 (near-tie profile); FA2 MLA default-ON; SACRED green.** Server 19.0 tok/s wall; CLI 18.93 reproduced | vLLM ~21 (#111 floor; in-session re-measure ABORTED by GB10 reboot at util 0.82, §21): **~0.90×**, >= vLLM NOT met; residual = KDA host islands + grouped MoE + decode graph |
| vLLM 0.26 re-benchmark | Pending | Re-run the binding grids on the advanced pin |
| MiniMax-H3 FP4 speed (W-FP4a) | **Measured GB10 (`row/H3-FP4-GPU-E2E`).** Marlin W4A16 byte-exact vs bf16; fp4 a memory win, 0.8x bf16/forward. Real-ckpt fp4-resident e2e RUNS (mp4/wav) | fp4 speed CLOSED. bf16-vs-quant A/B: ENCODER half MEASURED (§8.15), DiT half NOT (no bf16 render exists). Detail: benchmark-record + spec §8 |
| LTX-2.5 axes | Speed `PENDING` (vllm-omni#6066 has no native 2.5), binding oracle too. **SIZE: 704x448/25f and 448x256/25f both COMPLETE on GB10 (4231 s, 3085 s)**; one run each, contended box, no oracle, no ceiling (#1088) | NOT the VAE decode (#1041/#1009): 39-100% of the ~1731 s phase is the caption projection (#1208), x86 671.8→78.4 s; LoRA fusion 17.78→0.124 s (#1202), add-back 59% (#1254). **#1286 REFUTED (#1317)**; alloc bound 464 MiB |
| LTX-2.5 video VAE decode device arm | Speed **PENDING, and deliberately unmeasured**: the row that landed `vt::Conv3d` (#1007) was dispatched with the GPU lease withheld and made no ratio claim | Correctness only: `test_ops_conv3d` 4/4 byte-exact vs an independent reference, goldens unmoved at 44/44, non-CPU dispatch byte-identical on a fake backend. Owed: a CUDA run (#1452); the non-conv stages (#1451) |
| LTX-2.5 FULL 21.004B (`one_stage`, bf16 dev DiT) | **FIRST render on the full model: 768x448/25f on GB10 in 2990 s**, 25 frames + 1.01 s 48 kHz stereo, verifier PASS both arms. Binary `0a43a750` from `7b9e207b1`, run `20260819T150230Z` | REFUSED 1024x576, same lease: 162.0 s/fwd x 60 = 10803 s vs the rung's 7153 s LEASE BUDGET (#1375). STG/modality were OFF (#1092); that refusal is LIFTED, full fidelity is 4 fwd/step. Detail: benchmark-record |
| LTX-2.5 DiT self-attention kernel (`LTX25-DIT-ATTN-FLASH`, #1549) | Flash arm MEASURED on GB10: **7.680 s per DiT forward** at 768x448/49f (2352 tok), n=19, from the engine's own `last=` lines. Naive kernel 47.84 s (n=119). Ratio **~6.0x** (6.03-6.23x) | **A/B `PENDING`:** naive arm never ran (`rc` worker LOST); arms differ in binary, lease, prompt and sampler (~3.2%). No pixel gate (#1612). Cap-raise REVERTED, head_dim evidence WITHDRAWN (#1578). See benchmark-record |
| MiniMax-Music3 (`MiniMaxMusic3ForConditionalGeneration`) | **Every axis vs the reference stays `PENDING`.** A PARTIAL device arm now exists (#672): the 8.6B LM and the 2.4B fp32 DiT run on the accelerator, so the rows below are internal two-arm numbers and NOT parity ratios | Denominator: SGLang-Omni `748a0b43` in its production configuration (both CUDA graphs, compiled DIT and DAV, batched seeded sampling) |
| MiniMax-Music3 device arm, Jetson Thor sm_110 (#672) | `--device 1` vs `--device 0`, same request/seed, idle box: 2 AR frames **846.6 vs 835.1 s (1.014x SLOWER)**; 10 frames **1430.4 vs 1512.1 s (0.946x)**. Fit: **-11.65 s/frame, +34.8 s fixed** | A third duration (the fit has no residual), and moving the depth decoder + DiT + vocoder, which are 5 of 6 stages and still host scalar loops |
| MiniMax-Music3 DiT device arm, `thor:gpu0` sm_110 (#672) | Per DiT forward at the capture's geometry, same binary/weights/inputs, idle box: **204.955 s host vs 0.186-0.187 s device, 1094-1102x** (1201x fitted). Staged ONCE (0.61 s; loop intercept 0.063 s). Whole process 3.5-4.5x | e2e song pair NOT runnable (host DiT alone ~37.6 h at 30 steps). Depth decoder/condition mix (bf16-storage), vocoder (no `ConvTranspose1d`) still host. Detail: benchmark-record |
| MiniMax-Music3 depth stage, x86-64 20-core (#672) | Thor sm_110, real checkpoint, 3 pairs (`7b22b5b0`): `depth_forward` **348.27 -> 78.32 s, 4.45x** (1414 -> 808 calls); wall **446.33 -> 163.00 s, 2.74x**. E2E audio byte-identical, 6 runs. x86 synthetic: 3.50x | Wall 2.74x not 4.45x: `vocoder.decode_window` a fixed 53.6 s, 32.9%, unowned. `lm_decode_step` fell 2.65 s unexplained; Thor vs x86 a HYPOTHESIS. Arms proved by CALL COUNT, not sha256 (#1516). Detail: benchmark-record |
| MiniMax-Music3 vocoder DECODE WINDOW (#672, [#1334](https://github.com/mudler/vllm.cpp/issues/1334)) | Thor sm_110 `rc` lease, host arm, two trees, two binary sha256, 3 alternated rounds: **1.364x at 20 latents to 1.439x at 344**; second lease, 1 thread **2.157x** | Never the conv kernel alone (5.2-5.8x at `-O2`). **Do NOT multiply onto an e2e bucket**: e2e reads 54.091 s where this reads 97.4463 s at 344, unexplained (#1512). CUDA `memcmp` holds. Detail: benchmark-record |
| MiniMax-Music3 CPU host kernels, x86-64 20-core (#672) | KERNEL A/B at the vocoder's real geometry, min of 5 interleaved rounds: convolution chain **13.36 -> 1.25 s, 10.7x**; `Conv1d` 12.03x, `LinearNoBias` 10.88x. Output fingerprints IDENTICAL on both arms | e2e pair VOID (cold CIFS cache; a foreign `ctest` at load 76.6) and re-running. Stages 0/1 only ~2x: the pivot trades WEIGHT locality for accumulator locality. Detail: benchmark-record |
| MiniMax-Music3 END TO END on main, `thor:gpu0` sm_110 (#672, [#1512](https://github.com/mudler/vllm.cpp/issues/1512)) | Two clones `d0598a255` -> `a50c57d69`, one lease, staged ckpt, medians of 3 alternated rounds. 4 s/4 steps wall **166.04 -> 48.07 s, 3.45x**; `ar.depth_forward` **80.73 -> 4.27 s, 18.90x**. 20 s/30 steps **5.49x** | 4.53x with both loads excluded (the 20 s pair also crosses CIFS -> local). The DiT is now **62.24 %** at 370.556 s/120 calls, unchanged, owned by no row. `lm_decode_step` fell 1.49x unexplained. Detail: benchmark-record |
| MiniMax-Music3 DiT intra-forward split, `thor:gpu0` sm_110 (#672, [#1542](https://github.com/mudler/vllm.cpp/issues/1542)) | One binary, `VLLM_CPP_MUSIC3_DIT_SPANS` the only variable, idle box, staged ckpt, seq 690: **`vt::AttentionCross` is 43.9% of the forward for 4.0% of its flops, 0.204 vs the GEMMs' 3.98 TFLOP/s**; 27.6% of the run | Instrument only; nothing got faster. GEMMs sit at 71.5-79.0% of the fp32 peak, so the lever is a kernel ([#1555](https://github.com/mudler/vllm.cpp/issues/1555)), not precision. Detail: benchmark-record |
| MiniMax-Music3 DiT attention kernel, `thor:gpu0` sm_110 (#672, [#1555](https://github.com/mudler/vllm.cpp/issues/1555)) | ONE binary, arms by `VT_OP_PROVIDER_DISABLE`, engine-announced provider as the control, staged ckpt: at 20 s/30 steps **wall 595.496 -> 449.969 s, 1.3234x**, `denoise.dit_device` **1.6461x**, `dit.attn` **8.607x** | Same 1.6461x at 2 steps over 3 alternated pairs, spread 0.080%. All 6 mutations RED. `ncu` UNAVAILABLE here, so the mechanism is ABLATION: occupancy and bandwidth REFUTED. Detail: benchmark-record |
| MiniMax-Music3 vocoder CUDA arm, `thor:gpu0` (#672, [#1474](https://github.com/mudler/vllm.cpp/issues/1474)) | Same binary, `VLLM_CPP_VOCODER_DEVICE` the only variable, idle box, one lease. 344 latents host **15.078** vs CUDA **19.288 s = 0.782x**; ~689x4 **122.169 vs 150.060 = 0.814x**. f32 bought the device arm 2.78x | It LOSES: #1356 + #1474 moved the HOST kernel 3.59x against the device arm's 2.78x, so §13.10's f64 suspicion was right and is no longer the blocker. WAV byte-identical across arms, 12 runs. Detail: benchmark-record |
| MiniMax-H3 render coherence (`row/H3-RENDER-CLOSE` #77) | **CLOSED: a COHERENT scene on GB10.** #70/#74 white was wrong-PARTITION usage (t2va on the ref2va ckpt); t2va on the FL2VA GGUF renders a prompt-matched orange cat (adj-cos 0.95 vs 0.06, no patch-grid) | Verified first: t2va inputs byte-exact vs upstream; CUDA device==host at seq 1920. Follow-up `H3-TASK-PARTITION-GUARD`: the task/partition mismatch now RAISES 1:1 with `_resolve_task` (spec §8.6-8.7) |
| MiniMax-H3 image conditioning (`row/H3-CONDITIONED-E2E`, `row/H3-VISION-SCATTER`, `row/H3-REF2VA-ASSEMBLY`) | **fl2va COHERENT; ref2va assembly bug FIXED+gated.** vision→cond scatter gated; ref2va block-dim double-division fixed + RED-first gated (128 vs 512) + a permanent ref2va DiT-forward rung (§8.10) | grid RE-ATTRIBUTED: with the fix ref2va grids in fp4 AND bf16, and t2va with no refs on the ref2va NVFP4 also grids while FL2VA-GGUF renders, so it is the **NVFP4 checkpoint/loader**, NOT assembly/fp4 (§8.10) |
| MiniMax-H3 Thor render speed (sm_110, no FA2) | **34.6 s/step** at 864x480/124f/50 steps on Q4_K_M, **16.6x** off 574.5 (render ~28 min, was ~8 h). Landed: warp-per-query, chunked warp reduce-scatter (1.76x), bf16 `mma.sync` (9.82x) | Shared-memory K/V tiling (23% SLOWER) and register Q-blocking (-0.8%) both measured and REVERTED: memory traffic is not the bound (one head's K+V is 3.9 MB against 32 MB of L2) |
| MiniMax-H3 render duration (audio halving) | **CLOSED**: a 124-frame render silently muxed as **61**. The decoded audio ran half the video's duration and the muxer passes `-shortest`. Every structural check passed: shapes were self-consistent, just halved | Gated on the duration invariant (latent steps / 40 Hz equals the video duration), not on shape self-consistency, which a halved pipeline satisfies |
| MiniMax-H3 quantization floor | **Use Q4_K_M, not Q3_K_M.** H3's split-half RoPE produces channel-wise magnitude outliers 3-bit cannot hold; a controlled A/B (same prompt, seed, code) turned a murky lattice-covered silhouette into a photoreal close-up | Per-tensor mixed precision, if a smaller footprint is ever owed |
| MiniMax-H3 image conditioning (`row/H3-CONDITIONED-E2E`, `row/H3-VISION-SCATTER`, `row/H3-REF2VA-ASSEMBLY`, `row/H3-NVFP4-LOADER-DIFF`, `row/H3-NVFP4-STREAM-DIFF`) | **fl2va COHERENT; ref2va NVFP4 nibble loader bug FIXED (byte-verified); grid residual DIAGNOSED §8.12** | Activation diff + fingerprints (PR #95): NO load-path defect; all weights/islands/RoPE quant-noise-close to the coherent GGUF; grid = community-NVFP4 quant fidelity, not a loader fix. See benchmark-record |
| MiniMax-H3 encoder quantization (`H3-ENC-BF16-COND-DIFF`) | **Measured Thor (`d1085374`).** Q4_K_M vs bf16 encoder, same 233-token prompt, same forward: rel RMS **0.0340** (0.0685 excl. sink), per-token cosine mean **0.99745** / min 0.909, median rotation **3.5°** | NOT a scale change (best rescale 0.0340->0.0328). Same energy as a ONE-WORD prompt edit but DIFFUSE: 232/233 tokens rotate vs 172/233 untouched. Render A/B owed. Detail: benchmark-record |
| MXFP4 Qwen3-8B (W4A16 Marlin) | **`KERNEL-MARLIN-DENSE-EXEC` x3 (dense-ON default): c1 1.020, c2/c4/c8 0.962/0.966/0.969, GPU mem 2.63x less** (beats #51 1.005/0.925/0.939/0.953 EVERY axis); #44 3/3, 32B-NVFP4A16 6/6; -Werror test-guard fixes x2 | **VT_MARLIN_DENSE default-ON** (+951us). `FLASH-PTXAS` #82: cuModule A/B ties our+vLLM PTX across ptxas 13.0/13.2/driver-JIT (~144us); +10us is engine CONTEXT not codegen, no ptxas lever/flip (retires #75) |
| Vulkan load memory (`BACKEND-VULKAN-LOADMEM`) | **The load held the model TWICE.** 27B bf16, GB10, `VT_ADOPT_DEVICE_BYTES` A/B: **VmRSS 100.759 -> 53.413 GiB**, MemAvailable floor 13.85 -> 47.3 of 119.6. Device bytes identical. [Detail](../.agents/benchmark-record.md) | Load-phase peak (host build), and the page cache that tracks copied bytes 1:1 |
| Memory footprint vs declared workload (`ROAD-V1-MEM`, #83) | **Never measured, and not measurable today**: there is no auto-sizing to compare against, because the KV pool is a hand-typed `--num-blocks`, so "what the run actually needed" has no number | Once M1's `MemoryBudget` lands: predicted-vs-actual bytes per allocation class, then peak footprint ours-auto vs vLLM at its 0.9 default on the same model and config |
| Startup latency (cold to first `/health`) | **36.51 s vs vLLM 0.25.0's 221.51 s = 6.07x** (medians of 3, 27B-NVFP4, GB10). PROVISIONAL: 3 of 6 legs contended, repeat killed by a host reboot. [Detail](../.agents/benchmark-record.md) | Uncontended 3-rep re-run on a quiet box |
| Speculation depth (`ROAD-V1-D3-SPEC-K`, #81) | **PENDING.** 27B k=4 acceptance 0.875/0.750/0.618/0.507 vs control 0.000. One-window re-run 2026-08-17: void lifted, 6/6 clear. vLLM leg blocked, the oracle OOM-reboots this box ([spec](../.agents/specs/mtp-k-gt-1.md)) | the vLLM leg (our-ON == vLLM-ON) on the 27B and the whole 35B lane, then the c1/c>1 A/B at matched k + the prose vs code acceptance-vs-depth curve any depth policy needs |
| Vulkan vs llama.cpp Vulkan ([`BENCH-VK-LLAMA`](../benchmarks/demo/vulkan_27b_llamacpp.json)) | 25 NATIVE (+8 GDN). **27B prefill 21.5x, a SELF-ratio not a llama.cpp one**; decode **4.36 vs 4.35, MET**, denominator SUPERSEDED (7 clean legs). Smart barriers skip 19.8%/tok, GPU -1.09 ms; e2e 8/12, unresolved. OFF. | `VK-C` coopmat A/B on Thor (`VT_VULKAN_COOPMAT=0` A/Bs it): **11.1x-32.9x** vs our UNTILED scalar kernel, not a competent GEMM. `VK-E`: llama.cpp `-DGGML_VULKAN=ON` at `237ad9b96` on dgx, SUPERSEDED, same GGUF, 3 columns |
| ROCm (`BACKEND-GATE-ROCM-VLLM` / `-SGLANG`) | **PENDING: no binding throughput number.** The directional `d=128` decode row below is not one. Runtime-green on 5 gfx archs. Gemma-3 is 48/48 exact vs two vLLM-ROCm oracles; Qwen3.5-0.8B correctness remains open | Same model, quantization, request shape and cache policy vs pinned vLLM-ROCm on one idle AMD host. Add equivalent SGLang; close correctness first ([#41](https://github.com/mudler/vllm.cpp/issues/41)) |
| ROCm `d=128` decode arm (`BACKEND-ROCM`, [#382](https://github.com/mudler/vllm.cpp/issues/382)) | **DIRECTIONAL, not binding.** gfx1200, both sides in the pinned oracle container, 1024/128 c1, 8 prompts, 3 reps: TPOT 42.40 -> **11.67 ms** with `VT_ATTN_DECODE_D128=1`; vLLM `555967922` 6.68 ms, so **6.35x -> 1.75x** | Harnesses differ (oracle over HTTP, ours in-process) and no same-tool per-call trace exists, so [#488](https://github.com/mudler/vllm.cpp/issues/488) stays open. Owed: decode-windowed `rocprofv3` both sides |
| Gemma4 prefill-peer Finish-barrier cost (#1047 item 3) | **Attribution GREEN at T=2029, not a product ship number.** A (wait-only BEFORE) 1122.10 tok/s vs B (AFTER) 1094.24; +2.55% / 46.05 ms/req. Wait deletion is not authorized to land. | T=19 not run. Overlap/async retirement unmeasured. Detail: [benchmark-record](../.agents/benchmark-record.md) |
| Tenstorrent Blackhole (`BACKEND-TENSTORRENT`) | **NOT APPLICABLE (speed).** Correctness: OPT-125m STRICT 6/6 e2e on real hardware. Qwen3-0.6B has a device-specific golden and short 4-token warm smoke (~0.28 tok/s), not a completed speed run | Full 16x16 Qwen3 gate, then device-resident tensors + `ttnn::sdpa_decode` before any performance comparison. [Spec](../.agents/specs/tenstorrent-backend.md) |
| Mistral-7B-v0.3 on Tenstorrent (`BACKEND-TENSTORRENT-MISTRAL`) | **PENDING (speed).** First data point: 4.26 tok/s warm, batch 1, 32 tok, single run on a P150. Not a gate, not reproduced. No vLLM ratio exists or can (no TT backend). Correctness 16/16 | Reproduce idle with a same-binary A/B before quoting. [Record](../.agents/benchmark-record.md), [spec](../.agents/specs/tenstorrent-mistral.md) |
| Host-free decode graph (`BACKEND-TENSTORRENT-HOST-FREE-FORWARD`) | **PENDING (golden re-adjudication).** Operator gate 2026-08-20: the #1476 block-boundary degeneration is gone; captured vs eager carries one step-46 near-tie flip (0.25/0.125 nats). Env-gated. No vLLM ratio | [#1488](https://github.com/mudler/vllm.cpp/issues/1488) golden re-adjudication, then a same-binary A/B. [Record](../.agents/benchmark-record.md), [spec](../.agents/specs/tenstorrent-host-free-forward.md) |
| Prompt logprobs (`SAMPLE-PROMPT-LOGPROBS`, #223) | **NO number measured, claimed or owed.** Correctness-only, CPU. Upstream ships this path explicitly unoptimized (`gpu_model_runner.py:5622-5623`); a step where no request asks is unchanged | Floor if one is ever wanted: vLLM's own `prompt_logprobs=k`, same model and prompt |
| `logprobs_mode` (`SAMPLE-LOGPROB-TOKEN-IDS`, #238) | **NO number measured, claimed or owed.** Correctness-only, CPU. One [n, vocab] device->host copy per step when a processed mode is engaged, nothing when not | Nothing to close: observation modes, not a path vLLM optimizes either |
| `logprob_token_ids` scoring (#264) | **No number owed:** correctness-only, CPU-gated; inert unless set | Owed once the OpenAI field is wired |
| SGLang floor arms (`BACKEND-GATE-CUDA-SGLANG`) | **PARTIAL, not "never ran".** 27B-NVFP4 c8/c16, 3 reps, 0 errors, 2026-07-28. This cell read "Never ran" until 2026-08-16 ([#979](https://github.com/mudler/vllm.cpp/issues/979)) | c1/c2/c4, the in-series vLLM arm, 35B, the token-ID cross-check, paired traces, and every Qwen3.8-27B point. **None reachable: the image path is forbidden** ([#1265](https://github.com/mudler/vllm.cpp/issues/1265)) |
| llama.cpp on a CURRENT CUDA card (`BACKEND-GATE-CUDA-LLAMACPP`) | **No run.** Row filed 2026-08-16 ([#979](https://github.com/mudler/vllm.cpp/issues/979)); this arm had no owner before | Our Q4_K_M arm ([#821](https://github.com/mudler/vllm.cpp/issues/821)), then a same-file same-box quant-matched run |
| Qwen3.8-27B three-way at each engine's best (`BENCH-QWEN38-27B-SOTA`) | **No run**, filed 2026-08-21 ([#1574](https://github.com/mudler/vllm.cpp/issues/1574)). The competitors' own numbers on their own checkpoint are quoted in the spec as THEIR claim | fp8 KV on CUDA gates every cell (`KV-FP8` W2-W4 read `later`), then `QUANT-QWEN38-27B-NVFP4-ARM` W5. DFlash2's mechanism landed 2026-08-20. [Spec](../.agents/specs/bench-qwen38-27b-sota.md) |
| Embeddings on the ONE surface (ROW 6, `LlamaModel` + `vllm_embed` + `/v1/embeddings`) | **NO number measured, claimed or owed.** Correctness-gated only, CPU: the 2026-08-08 fold (engine path == direct registry path, f64 LAST+normalize reference on the committed fixture) is plumbing, no speed claim | A REAL embedding checkpoint (e5-mistral class) + a same-box `vllm.LLM(task="embed")` oracle; only then does an embed-throughput bar exist |
| Parakeet/FastConformer ASR (P1-P4 + ONE-SURFACE fold ROW 1) | **NO number measured, claimed or owed.** Correctness-gated only, CPU f32; the 2026-08-07 surface fold (`vllm_transcribe`, `/v1/audio/transcriptions`) is transcript-byte-identical plumbing, no speed claim. | Floor is `parakeet.cpp`, same clip and box; needs a CUDA provider and a pretrained checkpoint |
| Async serving correctness (#323) | **FIXED**: the decode graph replayed stale HOST token ids, degenerating concurrent requests past slot 0 (classic-dense, graph on = default). Graph declines while the mirror is live; async 7/7, SACRED 184/184 | Graph to read ids at REPLAY |
| Ampere consumer (`sm_86`, RTX 3090 class) | **No number owed; no such board here.** 2026-08-06 build-verify: 7/7 FA2 TUs 0-warn, real `sm_86` SASS. [Detail](../.agents/benchmark-record.md) | External RTX 3090 report. Floor is llama.cpp on that card (GGUF, not our Blackwell-only NVFP4 grid) |
| Pre-Ampere breadth (Turing `sm_75` / Volta `sm_70` / Pascal) | **No number owed; nothing runs on these arches.** 2026-08-06 `sm_75`: 20/20 TUs PASS (0 err/warn), WMMA bodies + all 3 selectors arch-gated; GB10 SASS byte-identical. [Detail](../.agents/benchmark-record.md) | Full-library LINK at `sm_75` + `cuobjdump` SASS, then a build-supported row. The fp16 `fattn` port is speed-only now; its floor when a card exists is llama.cpp on that card |

## Reproduce

| Benchmark | Entry point |
|---|---|
| vLLM online grid | `.agents/specs/competitive-benchmarks.md`, evidence under `dgx:~/work/vllm.cpp-online-gate/evidence/` |
| Clock-controlled pin series | `$HOME/gpu.lock` FIRST, then `sudo -n nvidia-smi -lgc 2190` under an always-fires `-rgc` trap; oracle by EXPLICIT PATH, identity asserted per leg; a `gpu_clock_state.py` window per leg |
| CPU vs llama.cpp | Same GGUF both arms, 3 reps under one `flock $HOME/gpu.lock`; `VT_GGUF_KEEP_F16=0` reproduces the pre-L7 baseline |
| Laguna NVFP4 decode | `flock $HOME/gpu.lock ./build-cuda/examples/laguna-gen --model ~/laguna-xs-nvfp4 --gpu` (that directory holds the S-2.1 checkpoint); `drop_caches` first, create the CUDA context before loading weights |
| DeepSeek-V4-Flash decode | `deepseek-v4-gen --gpu --kv-cache` on `ds4flash.gguf`, captured under tmux |
| Metal vs MLX-LM | Paired A/B harness, interleaved runs, cold legs discarded |
| Vulkan vs llama.cpp Vulkan | Same GGUF both arms: ours `-DVLLM_CPP_VULKAN=ON`, llama.cpp `-DGGML_VULKAN=ON` at `237ad9b96`, SUPERSEDED, via `llama-bench`; clean legs only, one `flock $HOME/gpu.lock`. GEMV sweep: `benchmarks/vulkan_gemv_ab.cpp` |
| Which llama.cpp a figure ran | Three revisions on this page, all SUPERSEDED (#1003): fork `237ad9b96` (GB10 CPU, Vulkan, x86, kernel matrix), stock `b9892` (Pi 5), stock `7044859` (Muse Glimmer, #391). Pin is stock `b10451`, unbuilt (#857) |
| Revisions repo-wide | **Five**, not three, enumerated in the [spec](../.agents/specs/oracle-llamacpp-repin-stock.md). Absent here: stock `030ebb5` (NON-BINDING) and a Poolside fork BRANCH with no commit recorded, behind Laguna's `27.8 tok/s` |

Build flags, environment variables, and the full gate list are in
[BUILD.md](BUILD.md) and [ENVIRONMENT.md](ENVIRONMENT.md).
