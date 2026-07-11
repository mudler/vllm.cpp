# Parity harness (test-time oracle)

Goldens under `tests/parity/goldens/` are produced by UPSTREAM vLLM code —
never re-implemented math. Oracle venv (dgx.casa): `~/venvs/vllm-oracle`
(`python3 -m venv ~/venvs/vllm-oracle && pip install vllm==0.24.0`).

Regenerate:
    ssh dgx.casa 'cd ~/work/vllm.cpp && ~/venvs/vllm-oracle/bin/python \
      tools/parity/dump_ops.py --out tests/parity/goldens'
then commit the changed goldens. Manifests record the oracle version + pin.

The W5/W6 YaRN/MRoPE/Llama 3 fixtures have their own pinned-source dumper:

    python3 tools/parity/dump_long_context.py \
      --out tests/parity/goldens --only yarn llama3

It first tries the installed oracle. If that import is unavailable, it verifies
the full `/home/mudler/_git/vllm` commit and executes the exact pinned
`common.py`, `base.py`, `yarn_scaling_rope.py`, `mrope.py`, and
`llama3_rope.py` class code with
only unrelated runtime-registration scaffolding stubbed. The manifest records
which loader produced each fixture; this fallback is not formula reimplementation.
Tolerances: f32 tight (1e-5); bf16 cases 8e-3 (we compute in f32, upstream
rounds to dtype mid-pipeline); RoPE pos>=32k cases 2e-2 (upstream f32 cache
drift — see .agents/state.md 2026-07-03 note).

## Oracle version vs upstream pin

The oracle venv runs pip vLLM **0.24.0**, which is NEWER than the porting pin
(`e24d1b24`, see .agents/upstream-sync.md). rmsnorm, silu_and_mul and rope are
drift-verified vs the pin (byte-identical); matmul/embedding are pure-torch
oracles. Before adding any new dumper (MoE, attention, quant), diff that op's
`forward_native` between the installed 0.24.0 package and the pinned checkout
at /home/mudler/_git/vllm — if semantics differ, dump from the pinned source
instead and record how in the manifest args. Manifests always record the
oracle version, so a silent regen with a different version shows up in git.

Model/layer-level goldens (M0.7+) MUST be dumped from the pinned checkout
(`PYTHONPATH=/home/mudler/_git/vllm` at the pin), not pip vLLM — per-op drift
audits don't scale to whole-model forwards. torch version: manifests record it
(2.11.0); keep the venv's torch matching when regenerating.
