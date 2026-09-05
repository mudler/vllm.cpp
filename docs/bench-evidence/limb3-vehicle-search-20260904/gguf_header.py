#!/usr/bin/env python3
# #2864 -- read a GGUF's OWN header and print the two facts a limb-3 vehicle
# search turns on: the `general.architecture` string our loader dispatches on,
# and the histogram of ggml tensor type ids the file actually stores.
#
# The histogram is the point. #2510 is usually restated as "our reader cannot
# load the UD quant family", which reads like a property of a NAME. It is a
# property of the BYTES: `UD-Q4_K_M` refuses over 4 tensors of ggml type 21,
# and a differently-quantized file under the same `UD` name would not. Two
# candidates in this scan are excluded by type ids their names do not disclose
# -- the Nemotron `UD-Q4_K_XL` stores NO k-quant tensor at all -- so the
# predicate is evaluated against the artifact rather than against its label.
#
# Reads only the header prefix, so it costs the same on a 17 GB file and a
# 108 GB one, and it never touches tensor data.
import json
import struct
import sys

# ggml_type ids, ggml/include/ggml.h. Only the ids this fleet's artifacts carry
# need to be named; anything else prints as UNK<id> and is still counted.
GGML = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
    20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
    26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0",
    35: "TQ2_0", 39: "MXFP4",
}


class Reader:
    def __init__(self, fh):
        self.fh = fh

    def raw(self, n):
        b = self.fh.read(n)
        if len(b) != n:
            raise EOFError("short read")
        return b

    def u32(self):
        return struct.unpack("<I", self.raw(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.raw(8))[0]

    def string(self):
        return self.raw(self.u64()).decode("utf-8", "replace")

    def value(self, t):
        if t == 0:
            return struct.unpack("<B", self.raw(1))[0]
        if t == 1:
            return struct.unpack("<b", self.raw(1))[0]
        if t == 2:
            return struct.unpack("<H", self.raw(2))[0]
        if t == 3:
            return struct.unpack("<h", self.raw(2))[0]
        if t == 4:
            return self.u32()
        if t == 5:
            return struct.unpack("<i", self.raw(4))[0]
        if t == 6:
            return struct.unpack("<f", self.raw(4))[0]
        if t == 7:
            return struct.unpack("<?", self.raw(1))[0]
        if t == 8:
            return self.string()
        if t == 9:
            et = self.u32()
            return [self.value(et) for _ in range(self.u64())]
        if t == 10:
            return self.u64()
        if t == 11:
            return struct.unpack("<q", self.raw(8))[0]
        if t == 12:
            return struct.unpack("<d", self.raw(8))[0]
        raise ValueError("unknown gguf kv type %d" % t)


# The KV keys a vehicle decision needs. `expert_count` is what separates a dense
# candidate from an MoE one, and limb 3 asks for a DENSE model.
WANT_SUFFIX = (
    ".block_count", ".embedding_length", ".feed_forward_length",
    ".expert_count", ".expert_used_count",
)


def main(path):
    with open(path, "rb") as fh:
        r = Reader(fh)
        magic = r.raw(4)
        if magic != b"GGUF":
            raise SystemExit("not a GGUF: %r" % magic)
        version = r.u32()
        n_tensors = r.u64()
        n_kv = r.u64()
        kv = {}
        for _ in range(n_kv):
            k = r.string()
            kv[k] = r.value(r.u32())
        hist = {}
        for _ in range(n_tensors):
            r.string()
            dims = r.u32()
            for _ in range(dims):
                r.u64()
            tid = r.u32()
            r.u64()
            name = GGML.get(tid, "UNK%d" % tid)
            hist[name] = hist.get(name, 0) + 1
    print("PATH               %s" % path)
    print("GGUF_VERSION       %d" % version)
    print("N_TENSORS          %d" % n_tensors)
    print("ARCHITECTURE       %s" % kv.get("general.architecture"))
    print("SIZE_LABEL         %s" % kv.get("general.size_label"))
    for k in sorted(kv):
        if k.endswith(WANT_SUFFIX):
            print("SHAPE_KV           %s = %s" % (k, kv[k]))
    print("TYPE_HISTOGRAM     %s" % json.dumps(hist, sort_keys=True))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
        print()
