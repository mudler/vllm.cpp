# Parity harness (test-time oracle)

Goldens under `tests/parity/goldens/` are produced by UPSTREAM vLLM code —
never re-implemented math. Oracle venv (dgx.casa): `~/venvs/vllm-oracle`
(`python3 -m venv ~/venvs/vllm-oracle && pip install vllm`).

Regenerate:
    ssh dgx.casa 'cd ~/work/vllm.cpp && ~/venvs/vllm-oracle/bin/python \
      tools/parity/dump_ops.py --out tests/parity/goldens'
then commit the changed goldens. Manifests record the oracle version + pin.
Tolerances: f32 tight (1e-5); bf16 cases 8e-3 (we compute in f32, upstream
rounds to dtype mid-pipeline); RoPE pos>=32k cases 2e-2 (upstream f32 cache
drift — see .agents/state.md 2026-07-03 note).
