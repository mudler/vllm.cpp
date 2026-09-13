ID: ISSUE-LOCAL-01M2C04E8N4NTXYNS584M1TSFP
Title: DeepSeek-V4.1-Flash W1: resolve and register `deepseek_v41`, parse and VALIDATE the nested config, and refuse every unimplemented arm BY NAME. The architecture is refused by name today -- `kGgufArchArms` carries `deepseek4` only (`src/vllm/entrypoints/model_loader.cpp:1243`) and no HF class resolves `DeepseekV41ForCausalLM` -- so nothing can open the artifact to be wrong about it. W1 needs NO pin advance and NO device: the tree already carries two ahead-of-pin registrations at `PARTIAL` (`Qwen3_5ForCausalLM` and `Qwen3_5MoeForCausalLM`, #490, 'REGISTERED, NOT RUN-GATED') and one registration of an architecture vLLM implements at no revision (`glm5_next`), because registration claims no gate against the oracle. Scope mirrors the GLM-5.3-Flash W1 shape (`47a2b35a5`): a new TU registering the ONE architecture string with zero edits to shared registries, a config parser that descends the nested `text_config`/`vision_config` where V4's was flat, the real published `config.json` @ `dba1be0a40` checked in byte-for-byte as a fixture, and a scaffold test proving the architecture resolves, the config descends and validates, and every unimplemented arm refuses with a message naming the missing part. It does NOT claim a load and does NOT claim a forward
Row: MODEL-MM-deepseek-v4-1-deepseek-v41-for-causal-lm
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-12
Updated: 2026-09-12
Closed: -

## Problem

The architecture cannot be opened at all, so no later wave can be gated.

## Resolution

-
