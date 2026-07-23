// Ported from: lmcache/v1/protocol.py:23-321 @ LMCache 8570aad.
//
// LMCache MODE-1 (lm:// remote store) wire framing: the fixed-`struct` binary
// header messages exchanged with lmcache.v1.server.LMCacheServer over plain
// TCP.  This header owns:
//   * ClientCommand / ServerReturnCode enums (protocol.py:28-39)
//   * the wire dtype-integer map DTYPE_TO_INT (protocol.py:41-65)
//   * the Location map LOCATION_TO_INT (protocol.py:68-78)
//   * ClientMetaMessage (186 bytes) / ServerMetaMessage (36 bytes)
//     (protocol.py:214-321)
//
// ENDIANNESS (upstream R3): protocol.py uses `struct.pack("iiiiiiiii150s", ...)`
// with NO byte-order prefix, i.e. NATIVE little-endian 4-byte ints on our
// x86-64 / aarch64 targets.  We encode little-endian EXPLICITLY here so the
// codec is well-defined regardless of host endianness; client and server must
// share byte order (true for every realistic lm:// deployment).
#ifndef VLLM_V1_KV_OFFLOAD_LMCACHE_REMOTE_PROTOCOL_H_
#define VLLM_V1_KV_OFFLOAD_LMCACHE_REMOTE_PROTOCOL_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "vllm/v1/kv_offload/lmcache/memory_format.h"

namespace vllm::v1::kv_offload::lmcache {

// protocol.py:23.
inline constexpr int kMaxKeyLength = 150;

// protocol.py:28-33 — ClientCommand(IntEnum), auto() starting at 1.
enum class ClientCommand : int32_t {
  kPut = 1,
  kGet = 2,
  kExist = 3,
  kList = 4,
  kHealth = 5,
};

// protocol.py:36-39 — ServerReturnCode(IntEnum).
enum class ServerReturnCode : int32_t {
  kSuccess = 200,
  kFail = 400,
};

// protocol.py:68-78 — LOCATION_TO_INT.
enum class Location : int32_t {
  kNone = 0,
  kLocalCPUBackend = 1,
  kLocalDiskBackend = 2,
};

// The wire/key dtype set.  A single enum drives BOTH maps below.
//
// NOTE on DTYPE_TO_INT (protocol.py:41-53): the upstream dict literal is
// `{torch.half: 1, torch.float16: 2, ...}` but `torch.half IS torch.float16`
// (the same singleton), so the entry for 1 is immediately shadowed and the
// EFFECTIVE map has float16 -> 2 (the integer 1 is never emitted).  Likewise
// float/float32 -> 4 and double/float64 -> 5 collapse to one entry each.  We
// mirror the effective post-collision map.  (torch is not installed on the
// build boxes; the collapse is deterministic Python dict semantics.)
enum class Dtype : int32_t {
  kNone = 0,
  kFloat16 = 2,       // torch.half / torch.float16 ; key str "half"
  kBFloat16 = 3,      // key str "bfloat16"
  kFloat32 = 4,       // torch.float / torch.float32 ; key str "float"
  kFloat64 = 5,       // torch.double / torch.float64 ; key str "double"
  kUint8 = 6,         // key str "uint8"
  kFloat8E4M3FN = 7,  // key str "fp8_e4m3fn"
  kFloat8E5M2 = 8,    // key str "fp8_e5m2"
};

// DTYPE_TO_INT (protocol.py:41-53) — the wire header integer for a dtype.
int32_t DtypeToInt(Dtype dtype);
// INT_TO_DTYPE (protocol.py:55-65).
Dtype IntToDtype(int32_t v);
// TORCH_DTYPE_TO_STR_DTYPE (utils.py:348-372) — the string used in the key.
std::string DtypeToStr(Dtype dtype);
// STR_DTYPE_TO_TORCH_DTYPE (utils.py:374).
Dtype StrToDtype(std::string_view s);

// protocol.py:214-267 — request header from client to server.
//
// `shape` is the LOGICAL shape (1..4 dims, unpadded, as a torch.Size); it is
// padded to 4-D with trailing zeros on the wire (protocol.py:103-128,236).
struct ClientMetaMessage {
  ClientCommand command = ClientCommand::kGet;
  std::string key;  // CacheEngineKey.to_string() result, <= kMaxKeyLength
  int32_t length = 0;
  MemoryFormat fmt = MemoryFormat::kUndefined;
  Dtype dtype = Dtype::kNone;
  std::vector<int32_t> shape;
  Location location = Location::kNone;

  // struct.pack("iiiiiiiii150s", command, length, fmt, dtype, location,
  //             shape0..3, key.ljust(150))  -> 186 bytes (protocol.py:238-251).
  // The key is right-padded to 150 bytes with SPACE (0x20), matching Python's
  // bytes.ljust default fill (protocol.py:249).
  std::string Serialize() const;
  static ClientMetaMessage Deserialize(std::string_view bytes);
  static constexpr std::size_t PackLength() {
    return 4 * 9 + kMaxKeyLength;  // 186
  }
};

// protocol.py:275-321 — reply header from server to client.  NOTE the field
// ORDER differs from the client message: `location` is LAST here, whereas the
// client message places it 5th (protocol.py:288-302 vs :238-251).
struct ServerMetaMessage {
  ServerReturnCode code = ServerReturnCode::kSuccess;
  int32_t length = 0;
  MemoryFormat fmt = MemoryFormat::kUndefined;
  Dtype dtype = Dtype::kNone;
  std::vector<int32_t> shape;
  Location location = Location::kNone;

  // struct.pack("iiiiiiiii", code, length, fmt, dtype, shape0..3, location)
  //  -> 36 bytes (protocol.py:288-302).
  std::string Serialize() const;
  static ServerMetaMessage Deserialize(std::string_view bytes);
  static constexpr std::size_t PackLength() { return 4 * 9; }  // 36
};

// protocol.py:103-160 — shape padding helpers, exposed for testing / reuse.
std::vector<int32_t> PadShapeTo4d(const std::vector<int32_t>& shape);
std::vector<int32_t> StripShapePadding(const std::array<int32_t, 4>& dims,
                                       MemoryFormat fmt);

}  // namespace vllm::v1::kv_offload::lmcache

#endif  // VLLM_V1_KV_OFFLOAD_LMCACHE_REMOTE_PROTOCOL_H_
