# Bounded oracle configuration amendment

Issue: [#3093](https://github.com/mudler/vllm.cpp/issues/3093).
Contract: [row spec](../../../../.agents/specs/rocm-quant-gather.md#bounded-oracle-configuration-and-allocation-amendment).

The operator ran these exact primary commands on 9 September 2026 UTC.
The helper prepared the scratch drivers and inspected their results.
Each attempt uses the same 944,192-byte Q4_0 GGUF and 974-byte HF config.
The operator receipts seal those inputs, the driver, plugin extension, and image.
The complete input audit remains in each original task evidence directory;
the receipt records its digest and the 2,500 files checked before and after.
These paths record this host's evidence and are not environment defaults.

| Attempt | Configuration change | Measured result |
|---|---|---|
| v3 | Pinned `model_class_overrides` selects the registered text class | Renderer refusal resolved; model load fails on rotary width 16 versus section sum 32 |
| v4 | Top-level `partial_rotary_factor=1.0` preserves rotary width 64 | Model load, normal compilation, and graph capture succeed; four physical blocks leave 48 usable cells and fail admission for length 64 |
| v5 | Sixteen physical blocks preserve logical length 64 | Exit 0; first prompt and repeat emit `[47, 19, 4, 20]`; resolved BF16 model and 256 physical cells |

The configuration corrections use public arguments in the pinned primary.
No primary source, plugin source, generated GGUF, or HF config changed.
Default compilation, graph capture, and kernel selection remain enabled.
The v5 driver reports the requested cache contract and pinned null-block
semantics separately from the observed `CacheConfig`.
Neither field proves the actual allocation bytes or cache tensor layout.
Identical-tool memory measurements and the full model token matrix remain pending.

The CPU RoPE probe imports the pinned configuration class and the executing
image's Transformers normalization helper. It demonstrates that normalization
copies the default top-level factor of 0.25 over the nested factor of 1.0.
Setting the top-level factor to 1.0 preserves the original rotary width of 64.
The probe ran without GPU devices and with networking disabled.

`manifest.json` gives file sizes, SHA256 values, and original evidence paths.
The retained logs, JSON reports, launch commands, receipts, and driver diffs are
unchanged copies. `../oracle-amendment-raw-captures.tar.gz` preserves the three
logs with their original terminal carriage returns and the three driver diffs.
The diffs document scratch adaptations; their presence here
does not replace the committed harness implementation or fresh review.

Both pristine secondary bounded runs still abort on their first decode.
The allocation overlay in the amendment is a separate, unqualified proposal.
It changes no primary result and does not advance an oracle pin or global
gateability record. Its red/green allocation witness, recurrent control, full
token comparison, and independent mutation review remain required.
