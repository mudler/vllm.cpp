#!/usr/bin/env python3
"""Write `glm-dsa.attention.indexer.types` into a published `glm-dsa` GGUF.

W7 of `.agents/specs/glm-dsa-latest-deepseek.md` §3.7, discharging **O17**
([#2214](https://github.com/mudler/vllm.cpp/issues/2214)).

**Why this file exists.** `unsloth/GLM-5.3-GGUF` writes no
`glm-dsa.attention.indexer.types` key, and it writes no `index_topk_freq` /
`index_skip_topk_offset` either. Its shard 1 KV block was read tensor by tensor
and key by key at revision `346b3591c7f28d1a23716f97a065ecf12ec14771`: 64 keys,
none of them the schedule. The file also declares `indexer.*` weights on **all
79 blocks** while the source checkpoint ships them on 22, so the conversion
broadcast the shared layers' tensors and the file's own tensor list cannot be
read as the schedule either. Spec D3 records both facts.

So the published artifact states its indexer schedule NOWHERE, and
`ParseGlmMoeDsaParams` refuses it by name rather than inventing one:

    the indexer schedule is stated nowhere: there is no `indexer_types` list
    and no `index_topk_freq` / `index_skip_topk_offset` to derive one from

**That refusal is correct and this script does not weaken it.** llama.cpp
survives the same file by falling back to a hardcoded 78-entry table
(`b10451:src/models/glm-dsa.cpp:6-27`, `GLM_5_2_DEFAULT_INDEXER_TYPES`), and
spec D3 rejects copying it: "a hardcoded 78-entry constant that happens to be
right is the shape that silently becomes wrong on GLM-5.4". The repair belongs
in the FILE, not in the loader. This script writes the key llama.cpp's own
converter would have written — `b10451:conversion/glm.py:337-339` emits
`indexer.types` whenever the source `config.json` carries `indexer_types`, which
`zai-org/GLM-5.3`'s does — so the output is a file that describes itself, and
the loader keeps refusing any file that does not.

**The value is never this script's to invent.** `--from-config` is required and
names the model author's own `config.json`. The schedule is transcribed from its
`indexer_types` list; nothing is derived, defaulted or guessed, and a config
that carries no such list is refused. The provenance therefore travels with the
artifact rather than with the port.

**What it rewrites.** Only the metadata shard, which for this artifact is shard
1 of 6: 9,428,677 bytes, `tensor_count = 0`, and its header ends exactly at
end-of-file, so it carries no tensor payload and no alignment padding to
preserve. The five shards that hold tensors are untouched and are not even
opened. The output is a new file; the input is never modified in place.

**Label the output as DERIVED.** The result is not `unsloth/GLM-5.3-GGUF`'s
shard 1 and must never be quoted as if it were. It has its own sha256, which
this script prints, and the row that uses it records both hashes.

Usage:

    scripts/glm-dsa-write-indexer-types.py \\
        --shard  /path/GLM-5.3-UD-IQ1_S-00001-of-00006.gguf \\
        --from-config /path/GLM-5.3/config.json \\
        --out    /path/GLM-5.3-UD-IQ1_S-00001-of-00006.indexed.gguf

`--dry-run` reads and validates everything and writes nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys

# GGUF metadata value type ids, `b10451:gguf-py/gguf/constants.py::GGUFValueType`.
GGUF_UINT8, GGUF_INT8 = 0, 1
GGUF_UINT16, GGUF_INT16 = 2, 3
GGUF_UINT32, GGUF_INT32 = 4, 5
GGUF_FLOAT32 = 6
GGUF_BOOL = 7
GGUF_STRING = 8
GGUF_ARRAY = 9
GGUF_UINT64, GGUF_INT64 = 10, 11
GGUF_FLOAT64 = 12

# type id -> (struct format, byte width). Strings and arrays are handled apart.
_SCALAR = {
    GGUF_UINT8: ("<B", 1),
    GGUF_INT8: ("<b", 1),
    GGUF_UINT16: ("<H", 2),
    GGUF_INT16: ("<h", 2),
    GGUF_UINT32: ("<I", 4),
    GGUF_INT32: ("<i", 4),
    GGUF_FLOAT32: ("<f", 4),
    GGUF_BOOL: ("<B", 1),
    GGUF_UINT64: ("<Q", 8),
    GGUF_INT64: ("<q", 8),
    GGUF_FLOAT64: ("<d", 8),
}

ARCH = "glm-dsa"
SCHEDULE_KEY = "glm-dsa.attention.indexer.types"


class GgufFormatError(RuntimeError):
    pass


class _Reader:
    """A byte-exact GGUF header reader. It records the raw span of every key so
    an untouched key is re-emitted verbatim rather than re-encoded."""

    def __init__(self, buf: bytes) -> None:
        self.buf = buf
        self.pos = 0

    def take(self, n: int) -> bytes:
        if self.pos + n > len(self.buf):
            raise GgufFormatError(
                f"header ends after {len(self.buf)} bytes while reading {n} "
                f"more at offset {self.pos}; this is not a whole GGUF header"
            )
        out = self.buf[self.pos : self.pos + n]
        self.pos += n
        return out

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def string(self) -> str:
        return self.take(self.u64()).decode("utf-8")

    def value(self, type_id: int):
        if type_id == GGUF_STRING:
            return self.string()
        if type_id == GGUF_ARRAY:
            elem_type = self.u32()
            count = self.u64()
            return [self.value(elem_type) for _ in range(count)]
        if type_id not in _SCALAR:
            raise GgufFormatError(f"unknown GGUF value type id {type_id}")
        fmt, width = _SCALAR[type_id]
        raw = struct.unpack(fmt, self.take(width))[0]
        return bool(raw) if type_id == GGUF_BOOL else raw


def _enc_string(s: str) -> bytes:
    raw = s.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _enc_bool_array(values: list[bool]) -> bytes:
    # llama.cpp writes this key as an array of bool
    # (`b10451:gguf-py/gguf/gguf_writer.py:806-808`), and our reader accepts an
    # array whose elements read as integers (`glm_moe_dsa.cpp::OptIndexerTypes`).
    out = struct.pack("<I", GGUF_ARRAY)
    out += struct.pack("<I", GGUF_BOOL)
    out += struct.pack("<Q", len(values))
    for v in values:
        out += struct.pack("<B", 1 if v else 0)
    return out


def read_header(path: str):
    """Return (version, kv_pairs, tensor_count, header_end, file_size).

    `kv_pairs` is a list of `(key, type_id, value, raw_bytes)` in file order.
    """
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        # 64 MiB is far beyond any real header; this artifact's is 9.4 MB
        # because it carries a 20 MB-class tokenizer.
        buf = fh.read(min(size, 64 * 1024 * 1024))
    r = _Reader(buf)
    if r.take(4) != b"GGUF":
        raise GgufFormatError(f"{path}: not a GGUF file (bad magic)")
    version = r.u32()
    if version != 3:
        raise GgufFormatError(
            f"{path}: GGUF version {version}; this script was written and "
            f"gated against version 3 only"
        )
    tensor_count = r.u64()
    kv_count = r.u64()

    kv_pairs = []
    for _ in range(kv_count):
        start = r.pos
        key = r.string()
        type_id = r.u32()
        value = r.value(type_id)
        kv_pairs.append((key, type_id, value, buf[start : r.pos]))

    header_end_kv = r.pos
    return version, kv_pairs, tensor_count, header_end_kv, size


def _kv(kv_pairs, key):
    for k, _t, v, _raw in kv_pairs:
        if k == key:
            return v
    return None


def load_schedule(config_path: str) -> list[str]:
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    types = cfg.get("indexer_types")
    if types is None:
        raise SystemExit(
            f"{config_path} carries no `indexer_types` list. This script "
            f"transcribes the schedule the model author published and derives "
            f"nothing, so a config that does not state it cannot be a source. "
            f"llama.cpp's converter takes the same position "
            f"(b10451:conversion/glm.py:337-339 writes the key only when the "
            f"source config carries the list)."
        )
    if not isinstance(types, list) or not all(isinstance(t, str) for t in types):
        raise SystemExit(
            f"{config_path}: `indexer_types` is {type(types).__name__}, and "
            f"the schedule is a list of strings"
        )
    bad = sorted({t for t in types if t not in ("full", "shared")})
    if bad:
        raise SystemExit(
            f"{config_path}: `indexer_types` carries {bad}, and upstream's "
            f"schedule has exactly two states — `full` (the layer runs its own "
            f"lightning indexer) and `shared` (it reuses the preceding full "
            f"layer's selection, deepseek_v2.py:1097-1101)"
        )
    return types


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Write the per-layer indexer schedule into a `glm-dsa` GGUF "
            "metadata shard, transcribed from the model author's config.json."
        )
    )
    ap.add_argument("--shard", required=True,
                    help="the GGUF shard carrying the KV block (shard 1 of a split)")
    ap.add_argument("--from-config", required=True, dest="from_config",
                    help="the model author's config.json, the ONLY source of the schedule")
    ap.add_argument("--out", help="output path (required unless --dry-run)")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate and report; write nothing")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing --out")
    args = ap.parse_args(argv)

    if not args.dry_run and not args.out:
        ap.error("--out is required unless --dry-run is given")

    version, kv_pairs, tensor_count, header_end, size = read_header(args.shard)

    arch = _kv(kv_pairs, "general.architecture")
    if arch != ARCH:
        raise SystemExit(
            f"{args.shard}: general.architecture is {arch!r}, not {ARCH!r}. "
            f"This script knows one architecture and refuses to guess at "
            f"another's key spellings."
        )

    if tensor_count != 0:
        raise SystemExit(
            f"{args.shard}: tensor_count is {tensor_count}, so this shard "
            f"carries tensor payload. Growing the header would move every "
            f"tensor's data without moving its recorded offset, which "
            f"silently corrupts the file. Point --shard at the metadata-only "
            f"shard (shard 1 of this split has tensor_count 0)."
        )
    if header_end != size:
        raise SystemExit(
            f"{args.shard}: the KV block ends at {header_end} but the file is "
            f"{size} bytes, so {size - header_end} trailing bytes exist that "
            f"this script does not understand. Refusing rather than dropping "
            f"them."
        )

    if _kv(kv_pairs, SCHEDULE_KEY) is not None:
        raise SystemExit(
            f"{args.shard} already carries `{SCHEDULE_KEY}`. This file states "
            f"its own schedule and needs no repair; refusing rather than "
            f"overwriting a value the converter wrote."
        )

    block_count = _kv(kv_pairs, "glm-dsa.block_count")
    nextn = _kv(kv_pairs, "glm-dsa.nextn_predict_layers") or 0
    if block_count is None:
        raise SystemExit(f"{args.shard}: no `glm-dsa.block_count` key")
    backbone = int(block_count) - int(nextn)

    schedule = load_schedule(args.from_config)
    if len(schedule) != backbone:
        raise SystemExit(
            f"{args.from_config} states {len(schedule)} layers and "
            f"{args.shard} describes a {backbone}-layer backbone "
            f"(block_count {block_count} - nextn_predict_layers {nextn}). "
            f"These are two different models."
        )

    full = [i for i, t in enumerate(schedule) if t == "full"]
    print(f"shard        : {args.shard}")
    print(f"  version    : {version}, kv={len(kv_pairs)}, tensors={tensor_count}, bytes={size}")
    print(f"  block_count: {block_count} ({backbone} backbone + {nextn} MTP)")
    print(f"config       : {args.from_config}")
    print(f"  schedule   : {len(schedule)} entries, {len(full)} full")
    print(f"  full layers: {full}")
    print(f"  pattern    : {''.join('1' if t == 'full' else '0' for t in schedule)}")

    # Re-emit every existing key VERBATIM from its recorded raw span, then
    # append the one new key. Nothing else in the file changes by a byte.
    body = b"".join(raw for _k, _t, _v, raw in kv_pairs)
    body += _enc_string(SCHEDULE_KEY)
    body += _enc_bool_array([t == "full" for t in schedule])

    out_bytes = (
        b"GGUF"
        + struct.pack("<I", version)
        + struct.pack("<Q", tensor_count)
        + struct.pack("<Q", len(kv_pairs) + 1)
        + body
    )

    digest = hashlib.sha256(out_bytes).hexdigest()
    print(f"output       : {len(out_bytes)} bytes (+{len(out_bytes) - size}), "
          f"kv={len(kv_pairs) + 1}")
    print(f"  sha256     : {digest}")
    print("  NOTE: this is a DERIVED artifact. It is not "
          "unsloth/GLM-5.3-GGUF's shard 1 and must not be quoted as if it "
          "were. Record this hash beside the published one.")

    if args.dry_run:
        print("dry run: nothing written")
        return 0

    if os.path.exists(args.out) and not args.force:
        raise SystemExit(f"{args.out} exists; pass --force to overwrite")
    tmp = args.out + ".partial"
    with open(tmp, "wb") as fh:
        fh.write(out_bytes)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, args.out)
    print(f"wrote {args.out}")

    # Read the result back through the same parser, so the file this script
    # claims to have written is the file it actually wrote.
    _v2, kv2, tc2, end2, size2 = read_header(args.out)
    back = _kv(kv2, SCHEDULE_KEY)
    if back is None or len(back) != backbone:
        raise SystemExit(
            f"{args.out}: read-back found {SCHEDULE_KEY} = {back!r}, which is "
            f"not the {backbone}-entry schedule just written"
        )
    if [bool(b) for b in back] != [t == "full" for t in schedule]:
        raise SystemExit(f"{args.out}: read-back schedule differs from the source")
    if tc2 != tensor_count or end2 != size2:
        raise SystemExit(f"{args.out}: read-back header is malformed")
    print(f"read-back OK : {len(kv2)} keys, {SCHEDULE_KEY} has "
          f"{sum(1 for b in back if b)} full of {len(back)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
