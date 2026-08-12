# ROCm M4 — the pinned vLLM-ROCm oracle on gfx1100 and the ROCm near-tie gate lane

## Goal

Give `BACKEND-ROCM` (issue #41) the milestone that only this box can host: the M4
correctness gate — our ROCm paged engine held against a **pinned vLLM-ROCm oracle
running on the same gfx1100 hardware**, not against the dgx (CUDA) capture.

## The oracle

- **Pin:** upstream vLLM commit `5559679229bc961848b121ccdeaa8fa5d79bec98`
  (runtime identity `0.23.1rc1.dev1511+g555967922`, verified against the
  parity-pin block in `.agents/upstream-sync.md`).
- **Base image:** `rocm/vllm-dev:base` (HIP 7.2.5 userland, host ROCm 7.14
  driver). `PYTORCH_ROCM_ARCH=gfx1100` covers the 7900 XTX.
- **Build:** `python3 setup.py bdist_wheel` inside the container after
  `apt-get install binutils mold` (the base image ships no linker) and the
  pyproject build-system requirements. Wheel:
  `vllm-0.23.1rc1.dev1511+g555967922.rocm723-cp312-cp312-linux_x86_64.whl`
  (2,696 files; `_C`/`_rocm_C`/`_moe_C` ABI extensions present).
- **Committed image:** `vllm-rocm-oracle:555967922-gfx1100`
  (`/home/vikash/oracle/Dockerfile.oracle`), wheel installed with deps.
- **Determinism:** K=10 per-prompt greedy capture on the 16 gate prompts is
  **deterministic in every cell** (0 multi-member (prompt,pos) cells) — a
  well-posed strict gate on this board, matching the dgx finding for batch=1
  `enforce_eager=True`.

## The gate lane

`tests/parity/test_qwen3_paged_engine.cpp` already has device-aware goldens for
Metal and Tenstorrent (`our_ids_<dev>.npy` + `neartie_gap_mnats_<dev>.npy` under
the shared `qwen3_greedy_0_6b/` dir, with the dgx CUDA pair as base). This spec
adds `kROCM` to that lane, unchanged logic:

- hard anchor REQUIRE (our deterministic tokens vs the committed ROCm anchor),
- near-tie band ≤ 500 milli-nats (vLLM teacher-forced on OUR exact prefix),
- backend proof (all 8 Qwen3-dense ops `selections>0 ∧ declines==0` on kROCM,
  with the fused-RoPE alternative counted as in the existing lane).

New committed goldens (captured on gfx1100, 4x RX 7900 XTX, ROCm 7.14,
`enforce_eager=True`, batch=1):

| file | content |
|---|---|
| `our_ids_rocm.npy` | our engine's greedy tokens (16×16) |
| `neartie_gap_mnats_rocm.npy` | oracle teacher-forced gaps in milli-nats (16×16) |
| `greedy_ids_rocm.npy` | the ROCm oracle's own greedy (evidence; the base `greedy_ids.npy` stays the dgx capture) |
| `greedy_dist_rocm.npy` | K=10 run set (evidence) |

Flow (the three commands, all GPU-locked):
1. oracle capture: `scripts/qwen3-oracle-capture.py --runs 10 --per-prompt` in the
   committed container against `/models` (Qwen3-0.6B).
2. bootstrap dump: `VT_DUMP_IDS=1 ./build-hip/tests/test_qwen3_paged_engine`
   → `our_ids_rocm.i32`.
3. teacher-forced gaps: `scripts/qwen3-neartie-gap.py --golden-dir ...` in the
   container → `our_ids_rocm.npy` + `neartie_gap_mnats_rocm.npy`.

## Result (Qwen3-0.6B, gfx1100)

**16/16 prompts PASS** — STRICT token-exact 11/16 vs the base greedy, 5/16 via
the near-tie band, **max gap 0.125 nats** (prompt[3] tok=1), **0
forward-divergent**; backend proof: kPagedAttention selections 7,168, 0 declines;
125/125 assertions. The 28 token-divergent positions vs the oracle's own greedy
are all within the band; the known p0 France/Italy flip is a literal 0.0000-nat
tie in the oracle's own logits.

## Boundaries

- No source/kernel change; the only edited file is the parity gate test.
- The dgx base goldens are byte-untouched; the ROCm additions are purely additive.
- Qwen3-4B (the strict deterministic dense) is not on this box (disk); the 0.6B
  near-tie-robust gate is the M4 evidence for the lane.
- The oracle image and `/home/vikash/oracle/` scratch are machine-local, not
  committed; this spec is the reproduction recipe.
