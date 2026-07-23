# SPIKE/IMPL: SentencePiece BPE + Metaspace pre-tokenizer (`LOAD-SENTENCEPIECE`)

*Engine-matrix row `LOAD-SENTENCEPIECE`. Roadmap `ROAD-V1-C2`/`ROAD-V1-C8` (tokenizer). Base `adca14c`. vLLM oracle `~/venvs/vllm-oracle` = vLLM 0.25.0 (HF `tokenizers` 0.22.2, `transformers` 5.13.1). Unblocks `MODEL-TEXT-mistral-mistral-for-causal-lm` (the Mistral paged-engine SACRED gate).*

---

## Scope

Add the **SentencePiece (Metaspace + byte-fallback) BPE** tokenizer family — the gate to Mistral, Gemma, and every SentencePiece-derived `tokenizer.json`. Before this, `Tokenizer::FromHfJson` was byte-level-BPE-only and threw `unsupported pre_tokenizer component "Metaspace"`; Mistral's forward was gated tokenizer-free. In scope: the `Metaspace` pre-tokenizer (space→▁ U+2581, `prepend_scheme` first/always/never, `split` flag), byte-fallback BPE (out-of-vocab char → `<0xNN>` byte tokens), the HF decoder chain (Replace ▁→space, ByteFallback, Fuse, Strip), and byte-exact encode/decode vs vLLM 0.25.0. Out of scope (fail loudly, documented): `Metaspace split=true` (no golden checkpoint in scope — Mistral/Gemma use `split=false`); a normalizer other than the already-accepted set.

## Upstream chain (the oracle)

HF `tokenizers` (the Rust crate that produced `tokenizer.json`; vLLM tokenizes through it via `transformers`). Mirrored `file:line` (tokenizers 0.22):
- `pre_tokenizers/metaspace.rs::Metaspace::pre_tokenize` — replace `' '`→replacement; prepend the replacement to the first split (scheme `First`) / every split (`Always`) **iff the split does not already start with the replacement**; `split=false` keeps the whole normalized string as one pretoken.
- `models/bpe/model.rs::BPE::tokenize`/`merge_word` — per character: vocab hit → symbol; else `byte_fallback` → the char's UTF-8 bytes as `<0xNN>` tokens (all-or-nothing); else unk (`fuse_unk` collapses consecutive). Merges run over the constructed symbols.
- `decoders/{replace,byte_fallback,fuse,strip}.rs` — Sequence decoder: Replace(▁→" ") → ByteFallback (run of `<0xNN>` → raw bytes; invalid UTF-8 run → one U+FFFD per byte) → Fuse → Strip(content=" ", start=1).
- `processors/template.rs` (already handled) — `TemplateProcessing` "single" `[<s>, A]` prepends BOS=1 (the shared `ExtractBosEos`/`EncodeWithSpecialTokens` seam).

Mistral-7B-v0.3 `tokenizer.json`: `pre_tokenizer` Metaspace(replacement ▁, prepend_scheme "first", split false); model BPE `byte_fallback=true`, `fuse_unk=true`, `unk_token=<unk>`, `ignore_merges=false`, vocab 32768 (incl. all 256 `<0xNN>` ids 771..1026), 58980 legacy-string merges, 771 special added tokens; `normalizer` null; post_processor TemplateProcessing BOS=1.

## Our baseline

The existing byte-level tokenizer (`LOAD-HF-BPE`): `Tokenizer::FromHfJson`/`FromGguf`, `Pretokenize` (Qwen/Llama-3/GPT-2/DeepSeek `SplitPattern`s), `BpeSplit` (GPT-2 bytes_to_unicode + merge-ranked BPE), `EncodeWithSpecialTokens` (TemplateProcessing BOS/EOS — the Llama `Sequence`-descend fix already handles Mistral's BOS=1), and the `SlowIncrementalDetokenizer` (ByteLevel reversal). The SP family REUSES: the added-token longest-match pre-pass in `Encode`, the merge loop (factored to `BpeMerge`), the `TemplateProcessing`/`ExtractBosEos` seam, `FinalizeTables`, and the whole incremental-detokenizer machinery (`ConvertIdsToTokens`, `AnalyzeLossy`/hold-back, stop strings). Only the per-segment pretokenize+BPE input (metaspace transform + byte-fallback, raw UTF-8 not byte-mapped) and the per-window decode chain are new, dispatched by `family_`.

## Port map

(See §Scope for the file:line port map.) New files: `tests/vllm/test_tokenizer_parity_mistral.cpp`, `tests/parity/goldens/tokenizer_mistral/{tokenizer.json,encodings.json}`, `tools/parity/dump_tokenizer_mistral.py`, and the Mistral paged goldens `tests/parity/goldens/mistral_greedy_7b/`. Shared additive touches: `tests/CMakeLists.txt` (one test row). Modified: `include/vllm/tokenizer/{bpe,tokenizer}.h`, `src/vllm/tokenizer/{bpe,tokenizer}.cpp`, `src/vllm/v1/engine/detokenizer.cpp` (SP dispatch). ZERO edit to any engine/KV/model/scheduler/kernel file.

## Dependencies

Builds on delivered seams: the byte-level tokenizer (`LOAD-HF-BPE`), the `TemplateProcessing` BOS/EOS post-processor seam (OPT + Llama bring-ups), and the incremental detokenizer. No new third-party dependency (the byte-fallback token format `<0xNN>` and the Metaspace ▁ are self-contained). The Mistral paged gate depends additionally on `MODEL-TEXT-mistral-mistral-for-causal-lm` (W0-W3, already landed) and the vLLM 0.25.0 oracle for goldens. No multi-GPU. Checkpoint `mistralai/Mistral-7B-v0.3` (dgx HF cache).

## Work breakdown

- **W0 detect + dispatch** — `DetectMetaspace`, `Family` enum, family dispatch in `FromHfJson`/`Encode`/`Decode`. Gate: CPU build; byte-level suites byte-identical.
- **W1 encode** — `EncodePlainSp` (metaspace transform + prepend guard + byte-fallback initial symbols + `BpeMerge` + fuse_unk). Gate: `Encode`/`EncodeWithSpecialTokens` == HF goldens.
- **W2 decode** — `SpDecodeTokens` (Replace/ByteFallback/Fuse/Strip) shared by `Decode` + the incremental detokenizer. Gate: `Decode` + incremental == HF decode.
- **W3 gate** — commit the Mistral `tokenizer.json` + generated goldens; `test_tokenizer_parity_mistral` byte-exact; SACRED cross-check vs vLLM `AutoTokenizer`; **run the Mistral paged-engine SACRED gate** (was SKIP) + the regression suite; CUDA `-Werror`, memcheck, DSR, record checkers.

## Structured contract — Work breakdown status

W0 detect/dispatch, W1 encode, W2 decode, W3 gate — CLOSED 2026-07-23. `test_tokenizer_parity_mistral` 6/6·421 byte-exact vs HF `tokenizers` 0.22.2; SACRED cross-check 0/45; byte-level suites byte-identical; CUDA `-Werror` 0-warn; ASan+UBSan clean; **Mistral `test_mistral_paged_engine` 16/16** (unblocked). Row `ACTIVE`.

## Risks/decisions

- `include/vllm/tokenizer/bpe.h` + `src/vllm/tokenizer/bpe.cpp` — factor the merge loop out of `BpeSplit` into `BpeMerge` (byte-level path byte-identical) so the SP path can pre-seed byte-fallback symbols before merging.
- `include/vllm/tokenizer/tokenizer.h` — `Family {kByteLevel,kSentencePiece}`, `IsSentencePiece()`, Metaspace params (replacement, `PrependScheme`, split), `byte_fallback_`/`fuse_unk_`/`unk_id_`, `EncodePlainSp`, `SpDecodeTokens`.
- `src/vllm/tokenizer/tokenizer.cpp` — `DetectMetaspace` dispatch (bare Metaspace node → SP family; everything else → `DetectPattern` byte-level, which still fails loudly on Metaspace so the families never overlap); `EncodePlainSp` (metaspace transform + prepend-with-starts-with-guard + byte-fallback initial symbols + `BpeMerge` + fuse_unk id mapping); `SpDecodeTokens` (the decoder chain, strict-UTF-8 ByteFallback); `Encode` passes `at_input_start=(pos==0)` so Metaspace `prepend_scheme="first"` fires only for the offset-0 segment (a segment after a special token is NOT "first").
- `src/vllm/v1/engine/detokenizer.cpp` — the incremental detokenizer's `ConvertTokensToString` dispatches to `SpDecodeTokens` for the SP family (byte-level ByteLevel reversal untouched).
- Tests + goldens (below).

## Tests to port / evidence

`tests/vllm/test_tokenizer_parity_mistral.cpp` (+ `tests/parity/goldens/tokenizer_mistral/{tokenizer.json,encodings.json}`, generator `tools/parity/dump_tokenizer_mistral.py`): the committed Mistral `tokenizer.json` + a 45-entry corpus stressing Metaspace edges (leading/trailing/repeated spaces, literal ▁, empty), byte-fallback (emoji, ℤ, `\t`/`\n`/NUL, CJK/Hangul), and special-token interaction ([INST]/[/INST], `</s>`, a mid-string special where "first" prepend must NOT re-fire), plus the 16 paged-engine battery prompts. Per entry: `Encode==ids`, `EncodeWithSpecialTokens==ids_special` (BOS=1), `Decode==decode`, incremental detok id-by-id reproduces `decode`. **MEASURED byte-exact vs HF `tokenizers` 0.22.2 (= vLLM 0.25.0's backend): 6/6, 421 assertions.** SACRED cross-check: vLLM's own `AutoTokenizer` (transformers 5.13.1) matches all 45 entries' ids (±BOS) with 0 mismatches.

Regression net (byte-level path byte-identical): `test_bpe` 852, `test_detokenizer` 221, `test_tokenizer_parity` 1175, `test_tokenizer_parity_deepseek` 2461 — all unchanged.

## Gates

CPU/CUDA `-Werror` 0-warn; the SP encode/decode/incremental tests byte-exact vs vLLM goldens; every existing tokenizer suite byte-identical. **Mistral `test_mistral_paged_engine` now RUNS** (was SKIP) — the paged-engine SACRED gate result closes the Mistral row's tokenizer block. Regressions each STANDALONE UNCHANGED: 27B 235/235 · 35B 315/315 · Qwen3-Coder 6/6 · Qwen3-dense 16/16 · OPT 6/6 · DeepSeek-V2 8/8 · Llama 16/16 (all byte-level — hold by byte-identity). memcheck 0 on the tokenizer path; DSR 32 (tokenizer adds ZERO device references).

## Risks/decisions

- **Prepend rule** (empirically locked + source-mirrored): after space→▁, prepend one ▁ iff scheme active for this segment AND result non-empty AND it does not already start with ▁; scheme "first" ⇒ only the offset-0 segment. Verified against `" hello"`→`▁hello` (no double), `"  hello"`→`▁▁hello`, `"▁already"` (no double), `"[INST]hi"`→`hi` (no prepend after a special).
- **byte-fallback before merge**: unknown char → `<0xNN>` symbols pre-merge (HF constructs the Word first); for Mistral all 256 bytes are in vocab so `unk` is never reached (kept correct + loud if ever hit).
- **split=true unsupported**: fail loudly rather than tokenize subtly wrong without a golden (project "no silent wrong tokenization" policy).
- **Decode U+FFFD**: the SP ByteFallback path emits U+FFFD for invalid byte runs (mirrors HF), which the incremental hold-back (`endswith("�")`) already handles — more faithful than the byte-level raw-bytes deviation.
