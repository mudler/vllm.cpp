# vLLM-Omni — the diffusion and TTS half of the primary reference

A separate repository from `vllm-project/vllm`, on its own release cadence, and
the home of the architectures vLLM proper never registers: the MiniMax-H3 joint
video+audio DiT, LTX-2.5, and the TTS family (`MossTTSDelayModel`,
`MossTTSRealtime`, `Qwen3TTSForConditionalGeneration`, IndexTTS-2.5). It is
still *vLLM* for mirroring purposes — its structure is the one we mirror — but
it is not covered by the vLLM parity pin, which names a commit in the other
repository.

**It has no pin, and that is tracked debt, not an omission.** #633 owes it.
Until it is pinned, every omni-only lane is compared against source read at
whatever revision happened to be checked out, which is why H3 W3+ and LTX-2.5
carry no oracle-run gate.

```oracle-pin
id = vllm-omni
role = secondary
upstream = https://github.com/vllm-project/vllm-omni
scope = diffusion, TTS and the omni-only architectures absent from the pinned vLLM registry
pin = UNPINNED
pin_label = none
pinned_on = 2026-08-13
gateable = no
evidence = #633
```
