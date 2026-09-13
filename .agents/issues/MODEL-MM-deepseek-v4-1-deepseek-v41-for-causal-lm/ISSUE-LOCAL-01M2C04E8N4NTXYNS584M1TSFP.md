ID: ISSUE-LOCAL-01M2C04E8N4NTXYNS584M1TSFP
Title: DeepSeek-V4.1-Flash W1: resolve and register `deepseek_v41`, parse and VALIDATE the nested config, and refuse every unimplemented arm BY NAME. The architecture is UNRECOGNIZED today and refused only as unknown (the opposite sense of the `refuses BY NAME` this wave delivers: its name is ABSENT from every table rather than present in the message) -- `kGgufArchArms` carries `deepseek4` only (`src/vllm/entrypoints/model_loader.cpp:1243`) and no HF class resolves `DeepseekV41ForCausalLM` -- so nothing can open the artifact to be wrong about it. W1 needs NO pin advance and NO device: the tree already carries two ahead-of-pin registrations at `PARTIAL` (`Qwen3_5ForCausalLM` and `Qwen3_5MoeForCausalLM`, #490, 'REGISTERED, NOT RUN-GATED') and one registration of an architecture vLLM implements at no revision (`glm5_next`), because registration claims no gate against the oracle. Scope mirrors the GLM-5.3-Flash W1 shape (`47a2b35a5`): a new TU registering the ONE architecture string with zero edits to shared registries, a config parser that descends the nested `text_config`/`vision_config` where V4's was flat, the real published `config.json` @ `dba1be0a40` checked in byte-for-byte as a fixture, and a scaffold test proving the architecture resolves, the config descends and validates, and every unimplemented arm refuses with a message naming the missing part. It does NOT claim a load and does NOT claim a forward
Row: MODEL-MM-deepseek-v4-1-deepseek-v41-for-causal-lm
State: CLOSED
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-12
Updated: 2026-09-13
Closed: 2026-09-13

## Problem

The architecture cannot be opened at all, so no later wave can be gated.

## Resolution

LANDED 2026-09-13 on branch row/MODEL-MM-dsv41-w1-impl. `DeepseekV41ForCausalLM` RESOLVES through `ModelRegistry` (one `REGISTER_VLLM_MODEL` in the new `src/vllm/model_executor/models/deepseek_v4_1_registry.cpp:199`, zero edits to any shared registration array), and the REAL published `config.json` @ revision `dba1be0a40aa45a94ad051997016db3960a90277` (sha256 `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879`, 3311 bytes, checked in byte-for-byte at `tests/vllm/models/fixtures/deepseek_v4_1/config.json`) descends and validates through `ParseDeepseekV41Params`. Every unimplemented arm -- the safetensors loader, the forward, the KV-cache spec and the `deepseek41` GGUF container -- refuses with a message naming this architecture, the missing part and the wave that owes it, and each of the four has its own message assertion. EVIDENCE: `tests/vllm/models/test_deepseek_v4_1_scaffold.cpp`, 33 cases / 295 assertions green. REACHABILITY: deleting the `REGISTER_VLLM_MODEL` line in a scratch copy reds 16 of the cases; mutating the forward to `return {};` or to `VT_CHECK(false, "boom")` reds the forward case. It does NOT load and does NOT forward -- that is the wave's boundary, not a gap. W2-W10 remain owed and are scoped in `.agents/specs/deepseek-v4-1-flash.md` `## Work breakdown`; the row is `PARTIAL` and still gate-blocked on the pin and on hardware.
