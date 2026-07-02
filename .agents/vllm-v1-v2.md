# vLLM V1/V2 terminology

There is no "V2 engine". vLLM V0 was removed in 2025; everything lives under
`vllm/v1/`. "V2" refers only to the **Model Runner V2** (`vllm/v1/worker/gpu/`
package, "MRV2" in commit messages), an in-progress rewrite of the model
runner *inside* the V1 engine, gated by `VLLM_USE_V2_MODEL_RUNNER` /
`VllmConfig.use_v2_model_runner` (default: on for an allowlist of
architectures, required for DSpark/diffusion).

**We port MRV2**, not the legacy `gpu_model_runner.py` — upstream development
is converging on it.
