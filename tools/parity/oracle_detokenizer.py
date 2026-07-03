#!/usr/bin/env python3
"""Detokenizer parity oracle: runs the REAL upstream slow incremental
detokenizer (vllm/v1/engine/detokenizer.py + vllm/tokenizers/
detokenizer_utils.py at the porting pin) against a mock byte-level BPE
tokenizer identical to the fixture in tests/vllm/test_detokenizer.cpp, and
asserts the exact values that C++ test hardcodes.

Needs no venv (stubs replace the heavy imports; the algorithm under test is
executed verbatim from the pinned checkout):

    python3 tools/parity/oracle_detokenizer.py [/path/to/pinned/vllm]

Every expected value in tests/vllm/test_detokenizer.cpp was produced by this
script. Known recorded deviation (see include/vllm/v1/engine/detokenizer.h):
the C++ port emits raw bytes where Python's lossy str shows U+FFFD, so the
"invalid flush" case reads '\\xf0\\x9f\\x8d!' in C++ and '\\ufffd!' here —
same character boundaries, same held-back windows.
"""

import sys
import types

VLLM = sys.argv[1] if len(sys.argv) > 1 else "/home/mudler/_git/vllm"


def mod(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


class _Version:
    def __init__(self, s):
        self.t = tuple(int(x) for x in s.split("."))

    def __ge__(self, other):
        return self.t >= other.t


def load_upstream():
    """Exec the pinned detokenizer modules with their imports stubbed."""
    tokenizers = mod("tokenizers", __version__="0.0.0", Tokenizer=object)
    tokenizers.decoders = mod("tokenizers.decoders", DecodeStream=object)
    mod("packaging")
    mod("packaging.version", parse=_Version)
    mod("transformers", PreTrainedTokenizerFast=type("PTF", (), {}))
    mod("vllm")
    mod("vllm.logger", init_logger=lambda n: types.SimpleNamespace(
        exception=lambda *a: None, warning=lambda *a: None))
    mod("vllm.tokenizers", TokenizerLike=object)
    du = mod("vllm.tokenizers.detokenizer_utils")
    with open(f"{VLLM}/vllm/tokenizers/detokenizer_utils.py") as f:
        exec(compile(f.read(), "detokenizer_utils.py", "exec"), du.__dict__)
    mod("vllm.utils",
        length_from_prompt_token_ids_or_embeds=lambda ids, emb: len(ids or []))
    mod("vllm.v1")
    mod("vllm.v1.engine", EngineCoreRequest=object)
    dt = mod("vllm.v1.engine.detokenizer")
    with open(f"{VLLM}/vllm/v1/engine/detokenizer.py") as f:
        exec(compile(f.read(), "detokenizer.py", "exec"), dt.__dict__)
    return du, dt


def bytes_to_unicode():
    bs = (list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) +
          list(range(0xAE, 0x100)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(0x100 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


B2U = bytes_to_unicode()
U2B = {v: k for k, v in B2U.items()}


def mapped(bs):
    return "".join(B2U[b] for b in bs)


# Mirror of the C++ test fixture (tests/vllm/test_detokenizer.cpp).
VOCAB = {0: "h", 1: "e", 2: "l", 3: "o", 4: "w", 5: "r", 6: "d",
         7: mapped(b" "), 8: "1", 9: "2", 10: "ll", 11: "he", 12: "llo",
         13: "hello", 14: mapped(b" w"), 15: "or", 16: "orld",
         17: mapped(b" world"), 18: "ld",
         19: "<|end|>", 20: "<tool>", 21: "<|end|>of",
         22: mapped(b"\xF0\x9F"), 23: mapped(b"\x8C\x8D"),
         24: mapped(b"\xC3"), 25: mapped(b"\xA9"),
         26: mapped(b"\xF0"), 27: mapped(b"\x9F"),
         28: mapped(b"\x8C"), 29: mapped(b"\x8D"),
         30: mapped(b"\x8D!")}
SPECIAL = {19, 21}


class MockTokenizer:
    """Byte-level BPE TokenizerLike surface used by the slow detokenizer."""

    is_fast = True

    def __len__(self):
        return 31

    def get_added_vocab(self):
        return {"<|end|>": 19, "<tool>": 20, "<|end|>of": 21}

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        return [VOCAB.get(i) for i in ids
                if not (skip_special_tokens and i in SPECIAL)]

    def convert_tokens_to_string(self, tokens):
        bs = bytearray()
        for t in tokens:
            for ch in t:
                if ch in U2B:
                    bs.append(U2B[ch])
                else:
                    bs.extend(ch.encode("utf-8"))
        return bs.decode("utf-8", errors="replace")


def main():
    du, dt = load_upstream()

    class Req:
        def __init__(self, prompt, **params):
            self.prompt_token_ids = prompt
            self.prompt_embeds = None
            defaults = dict(skip_special_tokens=True,
                            spaces_between_special_tokens=True, stop=None,
                            include_stop_str_in_output=False, min_tokens=0)
            defaults.update(params)
            self.sampling_params = types.SimpleNamespace(**defaults)

    def run(prompt, gen, stop_terminated_last=False, **params):
        det = dt.SlowIncrementalDetokenizer(MockTokenizer(),
                                            Req(prompt, **params))
        deltas, stops = [], []
        for i, t in enumerate(gen):
            last = i == len(gen) - 1
            stops.append(det.update([t], stop_terminated_last and last))
            deltas.append(det.get_next_output_text(last, delta=True))
        return deltas, stops, det.output_text, det.output_token_ids

    checks = [
        ("emoji 2tok", run([], [22, 23]), (["", "🌍"], "🌍")),
        ("2byte 2tok", run([], [24, 25]), (["", "é"], "é")),
        ("emoji 4tok", run([], [26, 27, 28, 29]), (["", "", "", "🌍"], "🌍")),
        ("unfinished tail", run([], [13, 22]), (["hello", ""], "hello")),
        ("invalid flush", run([], [22, 30]), (["", "�!"], "�!")),
        ("prompt straddle", run([13, 22], [23, 17]),
         (["", " world"], " world")),
        ("oov", run([], [31]), ([""], "")),
        ("negative id", run([], [-5, 13]), (["", "hello"], "hello")),
        ("specials skipped", run([], [13, 19, 19, 20, 17]),
         (["hello", "", "", "<tool>", " world"], "hello<tool> world")),
        ("specials included",
         run([], [13, 19, 19, 20, 17], skip_special_tokens=False),
         (["hello", "<|end|>", "<|end|>", "<tool>", " world"],
          "hello<|end|><|end|><tool> world")),
    ]
    for name, (deltas, _stops, text, _ids), (want_deltas, want_text) in checks:
        assert deltas == want_deltas, (name, deltas)
        assert text == want_text, (name, text)
        print(f"ok {name}: deltas={deltas!r} text={text!r}")

    # Stop-string bookkeeping (detokenizer.py owns check_stop_strings).
    d, s, text, _ = run([], [13, 17], stop=["wor"])
    assert s == [None, "wor"] and text == "hello "
    d, s, text, _ = run([], [13, 17], stop=["wor"],
                        include_stop_str_in_output=True)
    assert s == [None, "wor"] and text == "hello wor"
    d, s, text, ids = run([], [13, 17], stop_terminated_last=True)
    assert text == "hello" and ids == [13, 17]
    d, s, text, _ = run([], [13, 17], stop=["hello"], min_tokens=1)
    assert s == [None, None] and text == "hello world"
    d, s, text, _ = run([], [13], stop=["hello"])
    assert s == ["hello"] and text == ""
    print("ok stop strings / min_tokens / stop_terminated")

    css = dt.check_stop_strings
    assert css("hello world", 0, ["wor", "ld"], False) is None
    assert css("hello world", 6, ["wor", "ld"], False) == ("wor", 6)
    assert css("hello world", 6, ["wor", "ld"], True) == ("wor", 9)
    assert css("hello world", 2, ["ld"], True) == ("ld", -1)
    assert css("hello world", 11, ["ld", "hell"], False)[0] == "ld"
    assert css("abcabc", 1, ["ab"], False) is None
    assert css("abcabc", 1, ["abc"], False) == ("abc", 3)
    print("ok check_stop_strings")

    tokens, prefix, read = du.convert_prompt_ids_to_tokens(
        MockTokenizer(), [13] * 10)
    assert (len(tokens), prefix, read) == (7, 2, 7)
    assert du.convert_prompt_ids_to_tokens(
        MockTokenizer(), [13, 19, 17], True) == (["hello", "Ġworld"], 0, 2)
    assert du.detokenize_incrementally(
        MockTokenizer(), [13, 17], None, 0, 0, False, True) == (
            ["hello", "Ġworld"], " world", 1, 2)
    print("ok detokenizer_utils")
    print("ALL OK")


if __name__ == "__main__":
    main()
