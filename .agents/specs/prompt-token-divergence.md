# SPEC-PROMPT-TOKEN-DIVERGENCE — the served prompt-token count, and what it is not

Issue: [#1355](https://github.com/mudler/vllm.cpp/issues/1355)
Row: `SPEC-PROMPT-TOKEN-DIVERGENCE`
State: `READY` (diagnosis committed; the defect is not yet located)

## Now

The benchmark campaign of 19 August 2026 measured our server at **5,942** prompt
tokens where the pinned vLLM oracle measured **6,144** for a corpus both arms
took from byte-identical invocations of one `vllm bench serve` client. This spec
records what that divergence **is not**, with reproductions, and names the one
probe that can still locate it. It changes no product code.

## Scope

In scope: attributing the count. Out of scope: the speed verdict itself, which
this spec only corrects arithmetically (`## Consequence`).

## Upstream anchors (pin `5559679229`, `/home/mudler/_git/vllm`)

- `vllm/benchmarks/datasets/datasets.py:557` `RandomDataset` — a prompt is a
  contiguous slice of vocabulary ids, DECODED to text and then re-encoded.
- `vllm/benchmarks/datasets/datasets.py:495-549` `gen_prompt_decode_to_target_len`
  — the decode/re-encode loop that pads or truncates until
  `len(tokenizer.encode(prompt, add_special_tokens=False)) == target`, retrying
  ten times and reporting the residual as `token_mismatch`.
- `vllm/benchmarks/datasets/utils.py:41` `get_sampling_params`.
- `vllm/benchmarks/serve.py:74-144` `_align_prompts_to_server_tokenizer` — posts
  the FIRST prompt to the server's `/tokenize`, and on a disagreement re-tokenizes
  every prompt through the server and truncates it to `prompt_len` ids.
- `vllm/benchmarks/lib/endpoint_request_func.py:245-247` — `output.prompt_len`
  starts at the dataset's value and is OVERWRITTEN by `usage.prompt_tokens` from a
  usage-only SSE frame.
- `vllm/benchmarks/serve.py:606,734,1222` — `total_input`, hence
  `total_token_throughput`, and `input_lens`.

**vLLM reporting exactly 1024 is not the surprising half.** The dataset
calibrates each prompt string until the client's own tokenizer encodes it to
exactly `--random-input-len` tokens, so a server whose tokenizer agrees with the
client's reports 1024 by construction. Regenerated locally at the pin: 48/48
prompts, `token_mismatch == 0`.

## What was measured

Corpus regenerated from `--seed 0`, `--random-input-len 1024`,
`--random-range-ratio 0`, tokenizer `/mnt/nas_share/rc/ckpt/qwen3.8-27b-hf`
(`tokenizer.json` sha256 `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`,
byte-identical to the `ckpt-sha256.txt` manifest the campaign wrote from the
STAGED copy the server read).

1. **The corpus reproduces exactly.** The regenerated c8 prompts come short at
   the same 19 of 48 indices as the campaign's
   `bench-20260819T035148Z/c8-r1.json` — 0, 1, 9, 10, 11, 13, 14, 15, 16, 19,
   21, 22, 25, 28, 29, 30, 33, 45, 47. Prompts 0-5 of the c8 leg are
   byte-identical to the c1 leg because `numpy` short-circuits a zero-width
   `integers()` draw and consumes no state, so both legs share one offset
   stream. A 48-bit index agreement is not a coincidence: these are the
   campaign's prompts.
2. **Our tokenizer is byte-exact.** `vllm::tok::Tokenizer::FromHfJson` +
   `Encode` at this revision returns the SAME IDS, not merely the same count, as
   HF `tokenizers` for all 48 prompts. The tokenizer sources are unchanged
   between the campaign's build `1dac4f9a7` and `origin/main`
   (`git diff 1dac4f9a7 origin/main -- src/vllm/tokenizer include/vllm/tokenizer`
   is empty), so the comparison is against the code that ran.
3. **The transport is not it.** The same result over a real socket through
   `third_party/httplib` + `nlohmann::json::parse` + `EncodeWithSpecialTokens`,
   with the body sent three ways: `ensure_ascii` escapes (what `aiohttp` sends),
   raw UTF-8, and `Transfer-Encoding: chunked`. 48/48 at 1024.
4. **Not a build-flag or memory defect.** `-O0`, `-O3`, `-O3 -funsigned-char`
   (the aarch64 `char` signedness, which is the obvious portability suspect) and
   `-fsanitize=address,undefined` all return 1024 on all 48 with no diagnostic.
5. **Not a pre-tokenizer or BPE-option mistake.** Every wrong `SplitPattern`
   (`kQwen2Classic`, `kLlama3`, `kTekken`, `kGpt4o`, `kGpt2`) and every
   misclassification of `\p{M}` produces MORE tokens (1117-1306), never fewer;
   `ignore_merges=true` produces 1022. **Nothing that can go wrong inside the
   tokenizer produces a count below the reference**, because a coarser split
   cannot remove tokens. Fewer tokens means less text.
6. **Not the client's alignment pass.** `_fix_one` either leaves a prompt alone
   or truncates it to exactly `prompt_len` SERVER ids; simulated against our
   tokenizer under every wrong pattern it always predicts 1024, never 915.
7. **Not Unicode normalisation.** The checkpoint declares `normalizer: NFC` and
   `tokenizer.cpp:388-391` records the deviation that we accept it and do not
   apply it. Every one of the 48 prompts is ALREADY NFC, so the deviation is
   inert here. (It is still owed: a client that sends non-NFC text gets a
   different tokenization from HF. Not this bug.)

The divergence correlates perfectly with combining marks: all 19 short prompts
carry 74-150 characters in `Mn`/`Mc`, all 29 exact prompts carry 0-2. Equivalently
they are the prompts drawn from the multilingual region of the vocabulary
(bytes-per-character 1.4-2.2), against pure-ASCII (1.0) and pure-CJK (3.0)
neighbours that are exact. No transformation tried — deleting or replacing any
category, page or codepoint range, prefix truncation, NFC/NFD/NFKC — lands on the
served numbers.

## Diagnosis

**This is not a harness artifact and not a corpus property, and it is not a
defect in `vllm::tok::Tokenizer`.** All three were candidate explanations and all
three are falsified above. What remains is that the server binary
(`bin/vllm-server`, sha256 `ab0b9a1e6144…`, aarch64, built from `1dac4f9a7`)
reported a count that the identical source, the identical `tokenizer.json` and
the identical prompt bytes do not produce on x86-64.

Because `usage.prompt_tokens` is `res.prompt_token_ids.size()`
(`serving_completion.cpp:78`) and those ids are what the model consumed
(`input_processor.cpp:259-260`, `output_processor.cpp:174`), a wrong count is not
a reporting error. **The model saw a different prompt.** If the mechanism is
confirmed, it reaches every token-exact gate driven over HTTP with non-ASCII
prompts, not only benchmarking.

## The one probe left

Everything reproducible without hardware has been reproduced. The remaining
question needs the box, and needs no GPU beyond a server that is already up:

1. On `dgx:gpu0` inside a lease, against a running `vllm-server` on this
   checkpoint, POST prompt 0 of the regenerated corpus to `/tokenize`
   (`add_special_tokens: false`) and to `/v1/completions`. `/tokenize` and the
   completions `usage` share one string and two DIFFERENT tokenizer objects (the
   `ApiServer` member and the engine's `InputProcessor` member). If they
   disagree, the defect is between them and not in the tokenizer. If they agree
   at 915, build `examples/tokenize` natively on aarch64 and run it over the same
   prompt: that separates an ARM code-generation difference from a server-path
   difference in one step.
2. Whichever way it falls, the prompt that reproduces it is the smallest failing
   test, and it enters through the HTTP entry point rather than by constructing a
   tokenizer by hand.

## Owed

- **The aarch64 lane runs no tokenizer gate.** `.github/workflows/ci.yml:1086`
  builds exactly four targets on `ubuntu-24.04-arm` — `test_cpu_isa_arm`,
  `test_ops_matmul_elem`, `test_ops_quant_dot`, `test_ops_quant_repack`. The
  tokenizer parity goldens never execute on ARM, so an ARM-only tokenizer defect
  is unguarded by construction. The golden corpus itself is not the gap: it
  carries 28 combining marks and passes on x86.
- The NFC deviation at `src/vllm/tokenizer/tokenizer.cpp:388-391` is accepted
  rather than applied, and no gate feeds it non-NFC text.

## Consequence for the campaign's numbers

`total_token_throughput = (total_input + total_output) / duration`
(`serve.py:734`) takes `total_input` from the server's own report, so the two
arms divide different numerators and the metric is **not comparable at all**:

| leg | reported | corpus-matched (input = 1024 x N) | error |
|---|---|---|---|
| c1, mean of 3 | 38.4819 | 39.6404 | 3.01 % |
| c8, mean of 3 | 195.9628 | 203.6222 | 3.91 % |

The sign of the c1 verdict flips: as reported we are 0.16 % SLOWER than the
oracle (38.4819 against 38.5449); corpus-matched we are 2.84 % FASTER — the same
2.84 % the `output_throughput` ratio already showed (4.4045 against 4.2828).

`output_throughput` is biased upward because the shortfall is real work our
server did not do: 3.29 % fewer prompt tokens at c1, 4.23 % at c8. Weighted by
the prefill share of wall time the bias is **0.10-0.15 % at c1** and **about
0.6 % at c8** — measured run-to-run CV over the three reps is **0.048 %** (c1)
and **0.251 %** (c8), so the bias exceeds the noise on both legs and is not
absorbed by it. (The campaign's published 0.039 % / 0.205 % CVs are of the same
order; recomputed here from `c1-r{1,2,3}` and `c8-r{1,2,3}` they are 0.048 % and
0.251 %.)

One correction to the campaign's record: the absence of a
`WARNING: tokenizer mismatch` line is NOT evidence that the alignment pass found
the tokenizers in agreement. `bench.sh:236` pipes the client through `tail -40`,
and both that warning and `WARNING: /tokenize unavailable` are printed before the
result block, so both are cut. The saved logs cannot say which happened.

## Reproduction

No GPU, no lease, no download. `python3` with `numpy` + `transformers`, and a
C++20 compiler.

1. Mirror `RandomDataset` at the pin (seed 0, `input_len` 1024, `range_ratio` 0,
   `prefix_len` 0) against the checkpoint's tokenizer; dump the prompts.
2. Compile `src/vllm/tokenizer/{tokenizer,bpe,pretokenizer,unicode_data}.cpp`
   with a stub for `vllm::GgufFile::FindKv` (needed only to link `FromGguf`) and
   encode each prompt.
3. Compare ids against HF and counts against
   `bench-20260819T035148Z/c8-r1.json`'s `input_lens`.

## Stop conditions

Stop and report `NEEDS_DECISION` if the probe shows `/tokenize` and
`/v1/completions` agreeing at 1024 on the box, because that would mean the
campaign's `usage` frames did not come from the tokenizer at all and the
attribution above is wrong.
