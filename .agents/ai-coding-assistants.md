# AI Coding Assistants — contribution policy

This project accepts AI-assisted contributions under the same discipline the
Linux kernel project uses for coding assistants:
<https://docs.kernel.org/process/coding-assistants.html>. AI is a tool; the
**human submitter owns the change** and is fully responsible for it.

## Attribution & sign-off (commit trailers)

- **No `Signed-off-by` from an AI.** The Developer Certificate of Origin (DCO)
  is a legal attestation that only a human can make. Only the human submitter
  signs off.
- **No `Co-Authored-By: <AI>` trailers.** An AI is not a co-author. The human
  contributor owns and takes credit/responsibility for the change.
- **Use an `Assisted-by:` trailer** to disclose AI involvement:

  ```
  Assisted-by: AGENT_NAME:MODEL_VERSION [TOOL1] [TOOL2]
  ```

  Examples:

  ```
  Assisted-by: Claude Code:claude-opus-4-8 [ClaudeCode]
  Assisted-by: Cursor:gpt-5 [Cursor]
  ```

  Put the model/version you actually used and the tool(s) in brackets.

## Human responsibility

The human submitter **must review, test, and understand every line** of
AI-generated code before submitting — exactly as if they had written it by
hand. "The AI wrote it" is never an excuse for a bug, a license violation, or
an unreviewed change. If you cannot explain and defend a line, do not submit it.

## The `FOLLOWING_AGENTS_PROTOCOL` tag

Every commit MUST carry the trailer `FOLLOWING_AGENTS_PROTOCOL`. This is a
deliberate speed-bump: it asserts the contributor has **read
[AGENTS.md](../AGENTS.md)** and follows this project's protocol (mirror vLLM,
ground every check in vLLM source, compare vs the vLLM oracle on the identical
workload, keep the parity ledger updated, etc.). CI (`commit-protocol-tag` in
`.github/workflows/ci.yml`) rejects any commit lacking it, so a contributor who
skips AGENTS.md is caught at the gate.

## Example commit

```
perf(attn): vectorized K/V staging in the prefill kernel

Address-once + 128-bit int4 direct bf16 copies (no bf16->f32->bf16
round-trip), mirroring flash_attn's vectorized gmem->smem loads.
Bit-identical output; 16/16 token-for-token vs the vLLM oracle.

FOLLOWING_AGENTS_PROTOCOL
Assisted-by: Claude Code:claude-opus-4-8 [ClaudeCode]
```

(The human submitter adds their own `Signed-off-by:` when the project's DCO
flow requires it; the AI never does.)
