# Parity harness (test-time oracle)

Goldens under `tests/parity/goldens/` are produced by UPSTREAM vLLM code —
never re-implemented math. Oracle venv (dgx.casa): `~/venvs/vllm-oracle`
(`python3 -m venv ~/venvs/vllm-oracle && pip install vllm==0.24.0`).

Regenerate:
    ssh dgx.casa 'cd ~/work/vllm.cpp && ~/venvs/vllm-oracle/bin/python \
      tools/parity/dump_ops.py --out tests/parity/goldens'
then commit the changed goldens. Manifests record the oracle version + pin.
Tolerances: f32 tight (1e-5); bf16 cases 8e-3 (we compute in f32, upstream
rounds to dtype mid-pipeline); RoPE pos>=32k cases 2e-2 (upstream f32 cache
drift — see .agents/state.md 2026-07-03 note).

## Oracle version vs upstream pin

The oracle venv runs pip vLLM **0.24.0**, which is NEWER than the porting pin
(`e24d1b24`, see .agents/upstream-sync.md). rmsnorm was verified drift-free.
Before adding any new dumper (rope, MoE, attention, quant), diff that op's
`forward_native` between the installed 0.24.0 package and the pinned checkout
at /home/mudler/_git/vllm — if semantics differ, dump from the pinned source
instead and record how in the manifest args. Manifests always record the
oracle version, so a silent regen with a different version shows up in git.
