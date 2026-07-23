# Fixture generator for the LMCache MODE-1 (lm://) wire codec.
#
# Produces byte-exact reference vectors for the C++ codec gate.  The framing is
# generated with the *real* Python stdlib `struct` module (which IS LMCache's
# codec — protocol.py calls struct.pack directly); the blake3 hashes are
# generated with the *real* `blake3` PyPI package (the exact package LMCache's
# token_hasher.py imports); the KV_2LTD layout with the *real* numpy.
#
# Sources mirrored (LMCache @ 8570aad):
#   lmcache/v1/protocol.py:214-321   ClientMetaMessage / ServerMetaMessage
#   lmcache/utils.py:348-374,449-457 dtype maps + CacheEngineKey.to_string
#   lmcache/v1/multiprocess/token_hasher.py:37-49,183-230  blake3 rolling hash
#   lmcache/v1/memory_management.py:79-101  MemoryFormat.KV_2LTD
import json
import struct
import numpy as np
import blake3

# ---- constants copied verbatim from lmcache/v1/protocol.py ---------------
# ClientCommand(IntEnum) auto() from 1 (protocol.py:28-33)
PUT, GET, EXIST, LIST, HEALTH = 1, 2, 3, 4, 5
# ServerReturnCode (protocol.py:36-39)
SUCCESS, FAIL = 200, 400
MAX_KEY_LENGTH = 150  # protocol.py:23

# MemoryFormat(Enum) auto() with UNDEFINED=0 (memory_management.py:79-114)
FMT = {"UNDEFINED": 0, "KV_2LTD": 1, "KV_T2D": 2, "KV_2TD": 3,
       "BINARY": 4, "BINARY_BUFFER": 5, "KV_MLA_FMT": 6, "EC_TD": 7, "HS_TD": 8}

# DTYPE_TO_INT (protocol.py:41-53).  NOTE: torch.half IS torch.float16 (same
# singleton), so the literal `{torch.half:1, torch.float16:2, ...}` collapses to
# fp16 -> 2 (the value 1 is shadowed/unreachable).  Likewise float/float32 -> 4,
# double/float64 -> 5.  We encode the *effective* post-collision map (torch is
# too heavy to install on the dev box; the collapse is deterministic Python dict
# semantics, documented here so the C++ side mirrors the same effective values).
DTYPE_TO_INT = {None: 0, "float16": 2, "bfloat16": 3, "float32": 4,
                "float64": 5, "uint8": 6, "float8_e4m3fn": 7, "float8_e5m2": 8}
# TORCH_DTYPE_TO_STR_DTYPE (utils.py:348-372)
DTYPE_TO_STR = {"float16": "half", "bfloat16": "bfloat16", "float32": "float",
                "float64": "double", "uint8": "uint8", "int8": "int8",
                "float8_e4m3fn": "fp8_e4m3fn", "float8_e5m2": "fp8_e5m2"}
LOCATION_TO_INT = {None: 0, "LocalCPUBackend": 1, "LocalDiskBackend": 2}


def pad4(shape):  # protocol.py:103-128
    assert len(shape) <= 4
    return list(shape) + [0] * (4 - len(shape))


def client_meta(command, key_str, length, fmt, dtype, shape, location):
    # protocol.py:228-251
    assert len(key_str) <= MAX_KEY_LENGTH
    p = pad4(shape)
    return struct.pack(
        f"iiiiiiiii{MAX_KEY_LENGTH}s",
        command, length, FMT[fmt], DTYPE_TO_INT[dtype], LOCATION_TO_INT[location],
        p[0], p[1], p[2], p[3], key_str.encode().ljust(MAX_KEY_LENGTH))


def server_meta(code, length, fmt, dtype, shape, location):
    # protocol.py:288-302  (NOTE: location LAST, unlike the client message)
    p = pad4(shape)
    return struct.pack(
        "iiiiiiiii",
        code, length, FMT[fmt], DTYPE_TO_INT[dtype],
        p[0], p[1], p[2], p[3], LOCATION_TO_INT[location])


def cache_key_to_string(model, world, worker, chunk_hash_int, dtype):
    # utils.py:449-453 + chunk_hash_hex (utils.py:557-561)
    chunk_hash_hex = f"{chunk_hash_int:x}"
    return f"{model}@{world}@{worker}@{chunk_hash_hex}@{DTYPE_TO_STR[dtype]}"


# ---- blake3 rolling token hash (token_hasher.py:37-49,183-230) -----------
def blake3_hash(prefix_hash, tokens):
    h = blake3.blake3()
    if isinstance(prefix_hash, bytes):
        h.update(prefix_hash)
    elif isinstance(prefix_hash, int):
        h.update(prefix_hash.to_bytes(8, byteorder="big", signed=True))
    else:
        h.update(bytes(prefix_hash))
    h.update(struct.pack(f">{len(tokens)}I", *tokens))
    return h.digest()  # 32 bytes


NONE_HASH = blake3_hash(0, (0,))  # token_hasher.py:179


def hash_tokens(tokens, prefix_hash=None):
    if prefix_hash is None:
        prefix_hash = NONE_HASH
    return blake3_hash(prefix_hash, tuple(tokens))


def compute_chunk_hashes(token_ids, chunk_size):
    # token_hasher.py:192-230
    hashes = []
    prefix = NONE_HASH
    n = len(token_ids)
    num_complete = n - n % chunk_size
    for i in range(0, num_complete, chunk_size):
        prefix = hash_tokens(token_ids[i:i + chunk_size], prefix)
        hashes.append(prefix.hex())
    return hashes


def normalize_hash_to_int(digest):  # token_database.py:34-56
    return int.from_bytes(digest[:8], "big")


def hx(b):
    return b.hex()


out = {"lmcache_commit": "8570aad", "blake3_c_tag": "1.5.5", "blake3_c_commit": "81f772a"}

# ---- header fixtures ------------------------------------------------------
client_cases = [
    dict(name="put_bf16_kv2ltd", command=PUT, key="gpt2@1@0@deadbeef@bfloat16",
         length=4096, fmt="KV_2LTD", dtype="bfloat16", shape=[2, 24, 256, 128],
         location=None),
    dict(name="get_bf16", command=GET, key="gpt2@1@0@deadbeef@bfloat16",
         length=0, fmt="KV_2LTD", dtype="bfloat16", shape=[2, 24, 256, 128],
         location=None),
    dict(name="exist_fp16_localcpu", command=EXIST, key="m@2@1@ff00@half",
         length=0, fmt="KV_2LTD", dtype="float16", shape=[2, 12, 64, 64],
         location="LocalCPUBackend"),
    dict(name="put_fp8_binary", command=PUT, key="x@1@0@1@fp8_e4m3fn",
         length=17, fmt="BINARY", dtype="float8_e4m3fn", shape=[17],
         location="LocalDiskBackend"),
    dict(name="list_none_dtype", command=LIST, key="q@8@7@abc123@float",
         length=0, fmt="UNDEFINED", dtype=None, shape=[0], location=None),
    dict(name="health", command=HEALTH, key="h@1@0@0@double",
         length=0, fmt="UNDEFINED", dtype="float64", shape=[1], location=None),
]
out["client_meta"] = []
for c in client_cases:
    b = client_meta(c["command"], c["key"], c["length"], c["fmt"], c["dtype"],
                    c["shape"], c["location"])
    assert len(b) == 4 * 9 + MAX_KEY_LENGTH == 186
    out["client_meta"].append({**c, "bytes_hex": hx(b), "nbytes": len(b)})

server_cases = [
    dict(name="success_bf16", code=SUCCESS, length=4096, fmt="KV_2LTD",
         dtype="bfloat16", shape=[2, 24, 256, 128], location=None),
    dict(name="fail", code=FAIL, length=0, fmt="UNDEFINED", dtype=None,
         shape=[0], location=None),
    dict(name="success_localcpu", code=SUCCESS, length=64, fmt="KV_2LTD",
         dtype="float16", shape=[2, 12, 64, 64], location="LocalCPUBackend"),
]
out["server_meta"] = []
for c in server_cases:
    b = server_meta(c["code"], c["length"], c["fmt"], c["dtype"], c["shape"],
                    c["location"])
    assert len(b) == 36
    out["server_meta"].append({**c, "bytes_hex": hx(b), "nbytes": len(b)})

# ---- CacheEngineKey.to_string fixtures -----------------------------------
key_cases = [
    dict(model="gpt2", world=1, worker=0, chunk_hash=0xdeadbeef, dtype="bfloat16"),
    dict(model="meta-llama/Llama-3.1-8B", world=2, worker=1,
         chunk_hash=0x0123456789abcdef, dtype="float16"),
    dict(model="Qwen/Qwen3-27B", world=8, worker=7, chunk_hash=0, dtype="float32"),
    dict(model="m", world=1, worker=0, chunk_hash=0xff, dtype="float8_e4m3fn"),
]
out["cache_key"] = []
for c in key_cases:
    s = cache_key_to_string(c["model"], c["world"], c["worker"],
                            c["chunk_hash"], c["dtype"])
    out["cache_key"].append({**c, "chunk_hash_hex": f"{c['chunk_hash']:x}",
                             "string": s})

# ---- blake3 fixtures ------------------------------------------------------
out["blake3"] = {"none_hash_hex": hx(NONE_HASH)}
# single-chunk hashes with none prefix
single = []
for toks in [[0], [1], [1, 2, 3], [65536, 4294967295, 0], list(range(10))]:
    single.append({"tokens": toks, "prefix": "none",
                   "digest_hex": hx(hash_tokens(toks))})
out["blake3"]["single"] = single
# rolling: explicit two-step (prefix = first digest)
step1 = hash_tokens([1, 2, 3, 4])
step2 = blake3_hash(step1, (5, 6, 7, 8))
out["blake3"]["rolling_pair"] = {
    "chunk0": [1, 2, 3, 4], "chunk1": [5, 6, 7, 8],
    "digest0_hex": hx(step1), "digest1_hex": hx(step2)}
# compute_chunk_hashes over sequences at several chunk sizes
chunk_cases = []
for cs, seq in [
    (4, list(range(12))),          # 3 full chunks
    (4, list(range(10))),          # 2 full + partial discarded
    (256, list(range(512))),       # 2 chunks of the real default size
    (256, list(range(600))),       # 2 full + 88 partial discarded
]:
    chunk_cases.append({"chunk_size": cs, "tokens": seq,
                        "chunk_hashes_hex": compute_chunk_hashes(seq, cs)})
out["blake3"]["chunk_hashes"] = chunk_cases
# fold-to-int (token_database normalize) for a couple of digests
fold = []
for toks in [[0], [1, 2, 3]]:
    d = hash_tokens(toks)
    fold.append({"tokens": toks, "digest_hex": hx(d),
                 "folded_uint64": normalize_hash_to_int(d),
                 "folded_hex": f"{normalize_hash_to_int(d):x}"})
out["blake3"]["fold_to_int"] = fold

# ---- KV_2LTD layout fixtures ---------------------------------------------
# KV_2LTD = [2, num_layers, num_tokens, hidden_dim] contiguous (row-major).
# Provide (a) the flat little-endian bytes of a known [2,L,T,D] array, and
# (b) the per-(kv,layer) source planes so the C++ repack can be checked both
# ways.  Use float32 so bytes are unambiguous.
kv_cases = []
for L, T, D in [(2, 3, 4), (24, 5, 8)]:
    arr = np.arange(2 * L * T * D, dtype=np.float32).reshape(2, L, T, D)
    kv_cases.append({
        "num_layers": L, "num_tokens": T, "hidden_dim": D, "dtype": "float32",
        "shape": [2, L, T, D],
        "packed_le_hex": arr.tobytes(order="C").hex(),
        # element (kv,l,t,d) value == flat index; lets C++ verify stride order
        "strides_elems": [L * T * D, T * D, D, 1],
        "nbytes": arr.nbytes,
    })
out["kv_2ltd"] = kv_cases

with open("lmcache_fixtures.json", "w") as f:
    json.dump(out, f, indent=1)
print("wrote lmcache_fixtures.json")
print("none_hash", out["blake3"]["none_hash_hex"])
print("client_meta[0] nbytes", out["client_meta"][0]["nbytes"])
print("server_meta[0] nbytes", out["server_meta"][0]["nbytes"])
print("key[0]", out["cache_key"][0]["string"])
