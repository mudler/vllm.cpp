#!/usr/bin/env python3
"""Golden generator for Laya sequence construction (Phase 3, MODEL-LAYA).

Creates a minimal byte-level BPE tokenizer (all 256 byte-level chars + 4
special tokens, no merges), runs build_sequence from laya/common.py:48-85 with
known inputs, and dumps expected token IDs and marker positions as C++ arrays
in tests/vllm/models/laya_sequence_goldens.inc.

Both this generator and the C++ test (test_laya_sequence.cpp) build the SAME
tokenizer.json, load it, and run sequence construction. The goldens verify
parity with the Python reference.

The tokenizer has no merges, so every character becomes one token. This isolates
the sequence-construction logic (budget management, marker tracking, truncation)
from BPE merge correctness, which is tested in test_bpe.cpp.

Run: python3 scripts/gen-laya-sequence-goldens.py
"""
import json
import os
import sys
import tempfile


def bytes_to_unicode():
    """GPT-2 bytes_to_unicode bijection (mirrors vllm::tok::ByteToUnicode)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))


# Special token IDs (must match the C++ test fixture).
CLS_ID = 256
SEP_ID = 257
PAD_ID = 258
MASK_ID = 259
MASK_STR = "<mask>"

SPLIT_REGEX = (
    r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|"""
    r"""[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
)


def build_tokenizer_json():
    """Build a minimal byte-level BPE tokenizer.json matching the C++ fixture."""
    b2u = bytes_to_unicode()
    vocab = {}
    for byte_val in range(256):
        vocab[b2u[byte_val]] = byte_val
    vocab["[CLS]"] = CLS_ID
    vocab["[SEP]"] = SEP_ID
    vocab["[PAD]"] = PAD_ID
    vocab[MASK_STR] = MASK_ID

    added_tokens = [
        {"id": CLS_ID, "content": "[CLS]", "single_word": False,
         "lstrip": False, "rstrip": False, "normalized": False, "special": True},
        {"id": SEP_ID, "content": "[SEP]", "single_word": False,
         "lstrip": False, "rstrip": False, "normalized": False, "special": True},
        {"id": PAD_ID, "content": "[PAD]", "single_word": False,
         "lstrip": False, "rstrip": False, "normalized": False, "special": True},
        {"id": MASK_ID, "content": MASK_STR, "single_word": False,
         "lstrip": False, "rstrip": False, "normalized": False, "special": True},
    ]

    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": None,
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": SPLIT_REGEX},
                 "behavior": "Isolated", "invert": False},
                {"type": "ByteLevel", "add_prefix_space": False,
                 "trim_offsets": False, "use_regex": False},
            ],
        },
        "post_processor": {"type": "ByteLevel", "add_prefix_space": False,
                            "trim_offsets": False, "use_regex": False},
        "decoder": {"type": "ByteLevel", "add_prefix_space": False,
                     "trim_offsets": False, "use_regex": False},
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": vocab,
            "merges": [],
        },
    }


def build_sequence(tok, state, q, max_len=512, head_max_len=192,
                   option_order=None, truncate_left=False):
    """Mirror of build_sequence from laya/common.py:48-85.

    Uses tok.encode(text, add_special_tokens=False).ids instead of the HF
    __call__ API, which is equivalent (the tokenizers Rust core does the work
    in both cases).
    """
    mask_tok = MASK_STR
    opts = q["opts"]
    order = option_order if option_order is not None else list(range(len(opts)))
    ins = str(q["ins"]).replace(mask_tok, " ")
    head_ids = tok.encode(
        "%s question: %s" % (q["t"], ins), add_special_tokens=False
    ).ids

    opt_ids = []
    for i in order:
        opt_ids.append(
            [MASK_ID]
            + tok.encode(
                " " + opts[i].replace(mask_tok, " "),
                add_special_tokens=False,
            ).ids[:48]
        )

    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    if opt_budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)

    head_ids = head_ids[: max(8, opt_budget)]
    ids = [CLS_ID] + head_ids + [SEP_ID]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(SEP_ID)

    room = max(0, max_len - len(ids) - 1)
    st = tok.encode(
        state.replace(mask_tok, " "), add_special_tokens=False
    ).ids
    st = st[-room:] if truncate_left else st[:room]
    ids = ids + st + [SEP_ID]

    return ids[:max_len], [m for m in markers if m < max_len]


def fmt_array(name, values):
    if not values:
        return f"inline constexpr int {name}[] = {{}}"
    return "inline constexpr int %s[] = {%s}" % (
        name, ", ".join(str(v) for v in values))


def main():
    from tokenizers import Tokenizer

    tj = build_tokenizer_json()
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(tj, f, ensure_ascii=False)
        tok_path = f.name
    tok = Tokenizer.from_file(tok_path)
    os.unlink(tok_path)

    test_cases = [
        {
            "name": "choice_basic",
            "q": {"t": "choice", "ins": "which color",
                  "opts": ["red", "blue", "green"]},
            "state": "the car is fast",
            "max_len": 512,
            "head_max_len": 192,
        },
        {
            "name": "score_basic",
            "q": {"t": "score", "ins": "rate quality",
                  "opts": ["level 0: bad", "level 1: ok", "level 2: good"]},
            "state": "product review",
            "max_len": 512,
            "head_max_len": 192,
        },
        {
            "name": "noul_basic",
            "q": {"t": "noul", "ins": "is it true",
                  "opts": ["false: no", "true: yes"]},
            "state": "the statement",
            "max_len": 512,
            "head_max_len": 192,
        },
        {
            "name": "truncation",
            "q": {"t": "choice", "ins": "which color is best",
                  "opts": ["red", "blue"]},
            "state": "the car is fast and red",
            "max_len": 25,
            "head_max_len": 18,
        },
        {
            "name": "truncate_left",
            "q": {"t": "noul", "ins": "is it true",
                  "opts": ["false: no", "true: yes"]},
            "state": "the car is fast and red",
            "max_len": 25,
            "head_max_len": 18,
            "truncate_left": True,
        },
    ]

    out = []
    out.append("#pragma once")
    out.append("// Generated by scripts/gen-laya-sequence-goldens.py.")
    out.append("// DO NOT EDIT — regenerate with: "
               "python3 scripts/gen-laya-sequence-goldens.py")
    out.append("")
    out.append("// Tokenizer: 256 byte-level chars (id=byte), no merges,")
    out.append("// [CLS]=256 [SEP]=257 [PAD]=258 <mask>=259.")
    out.append("// Every character becomes one token (no BPE merges), so the")
    out.append("// goldens isolate sequence-construction logic from BPE.")
    out.append("")
    out.append("namespace laya_sequence_goldens {")
    out.append("")
    out.append("inline constexpr int kClsId = 256;")
    out.append("inline constexpr int kSepId = 257;")
    out.append("inline constexpr int kPadId = 258;")
    out.append("inline constexpr int kMaskId = 259;")
    out.append("")

    for tc in test_cases:
        tl = tc.get("truncate_left", False)
        ids, markers = build_sequence(
            tok, tc["state"], tc["q"],
            tc["max_len"], tc["head_max_len"],
            truncate_left=tl,
        )
        name = tc["name"]
        out.append(f"// {name}: type={tc['q']['t']!r} "
                   f"max_len={tc['max_len']} "
                   f"head_max_len={tc['head_max_len']}"
                   f"{' truncate_left' if tl else ''}")
        out.append(f"inline constexpr int k{name}_max_len = {tc['max_len']};")
        out.append(f"inline constexpr int k{name}_head_max_len = "
                   f"{tc['head_max_len']};")
        out.append(f"inline constexpr int k{name}_n_ids = {len(ids)};")
        out.append(fmt_array(f"k{name}_ids", ids) + ";")
        out.append(f"inline constexpr int k{name}_n_markers = "
                   f"{len(markers)};")
        out.append(fmt_array(f"k{name}_markers", markers) + ";")
        out.append("")

    out.append("}  // namespace laya_sequence_goldens")
    out.append("")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(
        script_dir, "..", "tests", "vllm", "models",
        "laya_sequence_goldens.inc",
    )
    with open(out_path, "w") as f:
        f.write("\n".join(out))
    print(f"Wrote {out_path} ({len(test_cases)} test cases)")

    # Summary for verification
    for tc in test_cases:
        tl = tc.get("truncate_left", False)
        ids, markers = build_sequence(
            tok, tc["state"], tc["q"],
            tc["max_len"], tc["head_max_len"],
            truncate_left=tl,
        )
        print(f"  {tc['name']}: {len(ids)} ids, {len(markers)} markers")


if __name__ == "__main__":
    main()
