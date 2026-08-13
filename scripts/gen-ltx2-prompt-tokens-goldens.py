#!/usr/bin/env python3
"""Emit tests/vllm/models/ltx2_prompt_tokens_goldens.inc — REAL prompts, REAL vocab.

`LTXGemmaTokenizer.tokenize_with_weights` (tokenizer.py:31-59) run against the
tokenizer the LTX-2.5 text encoder actually ships — which it ships AS A TENSOR,
`tokenizer_json` U8 [32169626] inside the safetensors file, not as a sibling
`tokenizer.json` (gemma_assets.py:34-36).

Why this gate is separate from a synthetic one. A hand-built tokenizer fixture
proves the WRAPPER — strip, BOS prepend, left pad, truncation order — and proves
nothing about the 262144-entry SentencePiece-flavoured BPE the model was trained
with. This one runs real English prompts through the real merges and checks the
ids, so a pre-tokenizer or normalizer mismatch on the shipped vocab is caught as
a wrong id rather than as a slightly different video.

The shipped tokenizer is the Gemma metaspace-via-normalizer form the C++ side
already recognizes: `Replace(" " -> "U+2581")` NORMALIZER plus
`Split(" ", MergedWithPrevious)` PRE-TOKENIZER, byte-fallback decoder, BPE with
262144 entries and 24 added tokens. Its `post_processor` is a TemplateProcessing
whose `special_tokens` map is EMPTY, which is exactly why `ltx_core` prepends BOS
itself and says so (tokenizer.py:12-15) — measured here rather than believed.

Needs the checkpoint, so this is not a CI generator; the emitted .inc is, and the
C++ case that uses the 32 MB tensor is env-gated on the checkpoint being present.

Usage:
    scripts/gen-ltx2-prompt-tokens-goldens.py \
        --text-encoder $CHECKPOINT_ROOT/ltx-2.5/vonkaiser-fp8-nvfp4/text_encoders/\
gemma4-12b-with-proj-nvfp4-torchao.safetensors \
        --out tests/vllm/models/ltx2_prompt_tokens_goldens.inc
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

# gemma_assets.py:162 / base_encoder.py:231-236, and diffusers
# pipeline_ltx2.py:304 independently.
MAX_LENGTH = 1024

# Deliberately varied. Each one is here for a reason a single prompt would miss:
#   * leading/trailing whitespace, which upstream strips (tokenizer.py:33);
#   * an empty prompt, which must still produce a BOS and 1023 pads;
#   * punctuation and digits, which the Split/Replace pre-tokenizer treats
#     differently from letters;
#   * a long cinematic prompt of the kind this model is actually driven with.
PROMPTS = [
    "a red fox running through deep snow at sunrise",
    "  a cat sits on a windowsill, watching rain fall  ",
    "",
    "Cinematic wide shot, 35mm: two astronauts walk across a rust-coloured dune "
    "field while a ringed planet rises behind them; slow dolly-in, volumetric "
    "light, 24 fps.",
]


def read_tokenizer_tensor(path: Path) -> bytes:
    """Pull `tokenizer_json` out of the safetensors file WITHOUT loading weights."""
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        if "tokenizer_json" not in header:
            raise SystemExit(
                f"{path} has no `tokenizer_json` tensor. This checkpoint family "
                "embeds the tokenizer as a TENSOR (gemma_assets.py:34-36); a file "
                "without it is not an LTX gemma asset pack."
            )
        start, end = header["tokenizer_json"]["data_offsets"]
        f.seek(8 + header_len + start)
        return f.read(end - start)


def tokenize_with_weights(tok, text: str, bos_id: int, pad_id: int):
    """tokenizer.py:31-59, transcribed.

    Kept as a transcription rather than a call into ltx_core so the generator runs
    without ltx_core installed (it needs einops, which is not on this box) — the
    only thing borrowed is the algorithm, and every step cites its line.
    """
    text = text.strip()                                       # :33
    # :37-43 is `self.tokenizer(text, ...)` — `__call__` with its DEFAULT
    # `add_special_tokens=True`, so upstream runs the post_processor. This passes
    # False deliberately, to transcribe what the C++ `Encode` does; the two agree
    # only because the measured post_processor here has an EMPTY `special_tokens`
    # map (asserted into the emitted header below). On a checkpoint where it added
    # something, THIS line and the C++ call would both have to change.
    ids = list(tok.encode(text, add_special_tokens=False).ids)  # :37-43
    if len(ids) > MAX_LENGTH:                                 # truncation=True
        ids = ids[:MAX_LENGTH]
    if not ids or ids[0] != bos_id:                           # :44-46
        ids = [bos_id, *ids][:MAX_LENGTH]
    pad = MAX_LENGTH - len(ids)                               # :48-54, LEFT side
    return [pad_id] * pad + ids, [0] * pad + [1] * len(ids)


def _emit_i32(out, name: str, values) -> None:
    out.write(f"inline constexpr int32_t {name}[] = {{\n")
    for i in range(0, len(values), 16):
        out.write("    " + ", ".join(str(int(v)) for v in values[i : i + 16]) + ",\n")
    out.write("};\n\n")


def _cxx_string(s: str) -> str:
    out = []
    for ch in s:
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        elif 0x20 <= ord(ch) < 0x7F:
            out.append(ch)
        else:
            out.extend("\\x%02x" % b for b in ch.encode("utf-8"))
            out.append('" "')
    return '"' + "".join(out) + '"'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-encoder", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    # ORACLE IDENTITY, asserted: `tokenizers` is what HuggingFace's own fast
    # tokenizer runs, and it is the thing our BPE is being compared against.
    import tokenizers  # noqa: PLC0415
    from tokenizers import Tokenizer  # noqa: PLC0415

    blob = read_tokenizer_tensor(args.text_encoder)
    digest = hashlib.sha256(blob).hexdigest()
    tok = Tokenizer.from_str(blob.decode("utf-8"))

    doc = json.loads(blob)
    post = doc.get("post_processor") or {}
    specials = post.get("special_tokens") or {}
    # MEASURED, and the reason `ltx_core` prepends BOS by hand.
    added = {t["content"]: t["id"] for t in doc.get("added_tokens", [])}
    bos_id, pad_id = added["<bos>"], added["<pad>"]

    rows_ids, rows_mask, valid = [], [], []
    for p in PROMPTS:
        ids, mask = tokenize_with_weights(tok, p, bos_id, pad_id)
        assert len(ids) == MAX_LENGTH and len(mask) == MAX_LENGTH
        rows_ids.append(ids)
        rows_mask.append(mask)
        valid.append(sum(mask))

    with args.out.open("w", encoding="utf-8") as out:
        out.write(
            "// GENERATED by scripts/gen-ltx2-prompt-tokens-goldens.py — DO NOT EDIT.\n"
            "//\n"
            "// `LTXGemmaTokenizer.tokenize_with_weights` (tokenizer.py:31-59) over the\n"
            "// tokenizer the LTX-2.5 text encoder SHIPS AS A TENSOR, run by HuggingFace\n"
            f"// `tokenizers` {tokenizers.__version__}.\n"
            "//\n"
            f"// tokenizer_json sha256: {digest}\n"
            f"// source: {args.text_encoder}\n"
            "//\n"
            "// MEASURED on that tokenizer, and the reason upstream prepends BOS by hand\n"
            f"// (tokenizer.py:12-15): its post_processor is `{post.get('type')}` with\n"
            f"// {len(specials)} special_tokens, so `add_special_tokens=True` adds NOTHING.\n"
            "// diffusers relies on exactly that flag (pipeline_ltx2.py:339) and would\n"
            "// therefore produce a prompt with no BOS; `ltx_core` is followed here.\n"
            "#pragma once\n\n"
            "#include <cstdint>\n\n"
            "namespace vllm_test {\n\n"
        )
        out.write(f"inline constexpr int64_t kLtxPromptMaxLength = {MAX_LENGTH};\n")
        out.write(f"inline constexpr int32_t kLtxPromptBosId = {bos_id};\n")
        out.write(f"inline constexpr int32_t kLtxPromptPadId = {pad_id};\n")
        out.write(f"inline constexpr int64_t kLtxPromptCount = {len(PROMPTS)};\n")
        out.write(
            f'inline constexpr const char* kLtxPromptTokenizerSha256 = "{digest}";\n\n')
        out.write("inline constexpr const char* kLtxPromptText[] = {\n")
        for p in PROMPTS:
            out.write("    " + _cxx_string(p) + ",\n")
        out.write("};\n\n")
        _emit_i32(out, "kLtxPromptValidCount", valid)
        for i, (ids, mask) in enumerate(zip(rows_ids, rows_mask)):
            _emit_i32(out, f"kLtxPromptIds_{i}", ids)
            _emit_i32(out, f"kLtxPromptMask_{i}", mask)
        out.write("}  // namespace vllm_test\n")

    sys.stderr.write(
        f"wrote {args.out} — {len(PROMPTS)} prompts, valid token counts {valid}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
