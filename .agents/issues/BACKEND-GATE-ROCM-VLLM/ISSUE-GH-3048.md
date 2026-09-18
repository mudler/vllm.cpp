ID: ISSUE-GH-3048
Title: fix(BACKEND-GATE-ROCM-VLLM): avoid callable serialization in Strix metadata RPC
Row: BACKEND-GATE-ROCM-VLLM
State: CLOSED
Kind: UNKNOWN
GitHub: 3048
Mirror: SYNCED
Availability: FULL
Created: 2026-09-07
Updated: 2026-09-12
Closed: 2026-09-12

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BACKEND-GATE-ROCM-VLLM`
>
> Owner: Strix #3043 campaign operator. Spec: .agents/specs/rocm-strix-vllm-headpin.md.
>
> Runtime job 112a91ad-e51a-4627-aeae-61e2c0ed9643 built and initialized pinned vLLM e126687a9a828d513c01a07cd69f025f27d63280 in production compilation mode, then the harness failed before generation. LLM.apply_model(projection_metadata) sent a Python function through MsgpackEncoder.enc_hook. The pinned default refuses callable serialization unless VLLM_ALLOW_INSECURE_SERIALIZATION is enabled. Do not enable that flag or weaken the build-state/provenance checks.
>
> Evidence: /workspace/strix-vllm-3043.GkMABu/run-9624441-01/logs/generation.log; generation exited1 after412.604 seconds. The prior fake LLM.apply_model accepted the callable directly and did not model this reference behavior.
>
> Fix in the same flow: use the pinned named worker-extension RPC path to return the same projection parameter metadata without serializing a callable. Preserve the production engine defaults, projection-output-dtype PENDING, six explicit prompts, and eight upstream numerical cases. Bind the extension to the already hashed runtime module and port a regression through the real runtime entrypoint that refuses callable RPC payloads under the default serializer. Fresh implementation, mutation review, and operator gates are required. This blocks merging the current harness as runnable; model gateability and token parity remain unproven.

## Resolution

Fixed by PR #3052: the Strix oracle harness now strips tuning/injection env from the test subprocess, avoiding callable serialization in the metadata RPC.
