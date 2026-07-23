// Ported from: lmcache/v1/protocol.py:23-321 + lmcache/utils.py:348-374
//              @ LMCache 8570aad.
#include "vllm/v1/kv_offload/lmcache/remote_protocol.h"

#include <stdexcept>

namespace vllm::v1::kv_offload::lmcache {
namespace {

// Encode a signed 32-bit int as 4 little-endian bytes (explicit LE, see the
// ENDIANNESS note in the header).
void PutI32LE(std::string* out, int32_t v) {
  const auto u = static_cast<uint32_t>(v);
  out->push_back(static_cast<char>(u & 0xFF));
  out->push_back(static_cast<char>((u >> 8) & 0xFF));
  out->push_back(static_cast<char>((u >> 16) & 0xFF));
  out->push_back(static_cast<char>((u >> 24) & 0xFF));
}

int32_t GetI32LE(std::string_view s, std::size_t off) {
  const auto b0 = static_cast<uint32_t>(static_cast<uint8_t>(s[off + 0]));
  const auto b1 = static_cast<uint32_t>(static_cast<uint8_t>(s[off + 1]));
  const auto b2 = static_cast<uint32_t>(static_cast<uint8_t>(s[off + 2]));
  const auto b3 = static_cast<uint32_t>(static_cast<uint8_t>(s[off + 3]));
  return static_cast<int32_t>(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
}

}  // namespace

int32_t DtypeToInt(Dtype dtype) {
  // protocol.py:41-53.  The enum value IS the wire int by construction.
  return static_cast<int32_t>(dtype);
}

Dtype IntToDtype(int32_t v) {
  // protocol.py:55-65.
  switch (v) {
    case 0:
      return Dtype::kNone;
    case 1:  // INT_TO_DTYPE[1] == torch.half; its DTYPE_TO_INT is 2, but the
             // reverse map does list 1 -> half, so accept it on decode.
    case 2:
      return Dtype::kFloat16;
    case 3:
      return Dtype::kBFloat16;
    case 4:
      return Dtype::kFloat32;
    case 5:
      return Dtype::kFloat64;
    case 6:
      return Dtype::kUint8;
    case 7:
      return Dtype::kFloat8E4M3FN;
    case 8:
      return Dtype::kFloat8E5M2;
    default:
      throw std::invalid_argument("IntToDtype: unknown dtype int");
  }
}

std::string DtypeToStr(Dtype dtype) {
  // utils.py:348-372 (TORCH_DTYPE_TO_STR_DTYPE).
  switch (dtype) {
    case Dtype::kFloat16:
      return "half";
    case Dtype::kBFloat16:
      return "bfloat16";
    case Dtype::kFloat32:
      return "float";
    case Dtype::kFloat64:
      return "double";
    case Dtype::kUint8:
      return "uint8";
    case Dtype::kFloat8E4M3FN:
      return "fp8_e4m3fn";
    case Dtype::kFloat8E5M2:
      return "fp8_e5m2";
    case Dtype::kNone:
      throw std::invalid_argument("DtypeToStr: None has no key string");
  }
  throw std::invalid_argument("DtypeToStr: unknown dtype");
}

Dtype StrToDtype(std::string_view s) {
  // utils.py:374 (STR_DTYPE_TO_TORCH_DTYPE).
  if (s == "half") return Dtype::kFloat16;
  if (s == "bfloat16") return Dtype::kBFloat16;
  if (s == "float") return Dtype::kFloat32;
  if (s == "double") return Dtype::kFloat64;
  if (s == "uint8") return Dtype::kUint8;
  if (s == "fp8_e4m3fn") return Dtype::kFloat8E4M3FN;
  if (s == "fp8_e5m2") return Dtype::kFloat8E5M2;
  throw std::invalid_argument("StrToDtype: unknown dtype string");
}

std::vector<int32_t> PadShapeTo4d(const std::vector<int32_t>& shape) {
  // protocol.py:103-128.
  if (shape.size() > 4) {
    throw std::invalid_argument("PadShapeTo4d: shape dimension must be <= 4");
  }
  std::vector<int32_t> out = shape;
  out.resize(4, 0);
  return out;
}

std::vector<int32_t> StripShapePadding(const std::array<int32_t, 4>& dims,
                                       MemoryFormat fmt) {
  // protocol.py:131-160.
  if (fmt == MemoryFormat::kBinary || fmt == MemoryFormat::kBinaryBuffer) {
    return {dims[0], dims[1], dims[2], dims[3]};
  }
  int end = 4;
  while (end > 1 && dims[static_cast<std::size_t>(end - 1)] == 0) {
    --end;
  }
  return std::vector<int32_t>(dims.begin(), dims.begin() + end);
}

std::string ClientMetaMessage::Serialize() const {
  if (key.size() > static_cast<std::size_t>(kMaxKeyLength)) {
    throw std::invalid_argument("ClientMetaMessage: key exceeds 150 bytes");
  }
  const std::vector<int32_t> padded = PadShapeTo4d(shape);
  std::string out;
  out.reserve(PackLength());
  PutI32LE(&out, static_cast<int32_t>(command));
  PutI32LE(&out, length);
  PutI32LE(&out, static_cast<int32_t>(fmt));
  PutI32LE(&out, DtypeToInt(dtype));
  PutI32LE(&out, static_cast<int32_t>(location));
  PutI32LE(&out, padded[0]);
  PutI32LE(&out, padded[1]);
  PutI32LE(&out, padded[2]);
  PutI32LE(&out, padded[3]);
  // key.encode().ljust(150) — right-pad with SPACE (0x20) (protocol.py:249).
  out.append(key);
  out.append(static_cast<std::size_t>(kMaxKeyLength) - key.size(), ' ');
  return out;
}

ClientMetaMessage ClientMetaMessage::Deserialize(std::string_view bytes) {
  if (bytes.size() != PackLength()) {
    throw std::invalid_argument("ClientMetaMessage::Deserialize: bad length");
  }
  ClientMetaMessage m;
  m.command = static_cast<ClientCommand>(GetI32LE(bytes, 0));
  m.length = GetI32LE(bytes, 4);
  m.fmt = static_cast<MemoryFormat>(GetI32LE(bytes, 8));
  m.dtype = IntToDtype(GetI32LE(bytes, 12));
  m.location = static_cast<Location>(GetI32LE(bytes, 16));
  const std::array<int32_t, 4> dims = {GetI32LE(bytes, 20), GetI32LE(bytes, 24),
                                       GetI32LE(bytes, 28),
                                       GetI32LE(bytes, 32)};
  m.shape = StripShapePadding(dims, m.fmt);
  // key.decode().strip() — strip trailing (and leading) whitespace
  // (protocol.py:261).
  std::string_view key_field = bytes.substr(36, kMaxKeyLength);
  std::size_t begin = key_field.find_first_not_of(" \t\n\r\f\v");
  std::size_t end = key_field.find_last_not_of(" \t\n\r\f\v");
  if (begin == std::string_view::npos) {
    m.key.clear();
  } else {
    m.key = std::string(key_field.substr(begin, end - begin + 1));
  }
  return m;
}

std::string ServerMetaMessage::Serialize() const {
  const std::vector<int32_t> padded = PadShapeTo4d(shape);
  std::string out;
  out.reserve(PackLength());
  PutI32LE(&out, static_cast<int32_t>(code));
  PutI32LE(&out, length);
  PutI32LE(&out, static_cast<int32_t>(fmt));
  PutI32LE(&out, DtypeToInt(dtype));
  PutI32LE(&out, padded[0]);
  PutI32LE(&out, padded[1]);
  PutI32LE(&out, padded[2]);
  PutI32LE(&out, padded[3]);
  PutI32LE(&out, static_cast<int32_t>(location));  // location LAST
  return out;
}

ServerMetaMessage ServerMetaMessage::Deserialize(std::string_view bytes) {
  if (bytes.size() != PackLength()) {
    throw std::invalid_argument("ServerMetaMessage::Deserialize: bad length");
  }
  ServerMetaMessage m;
  m.code = static_cast<ServerReturnCode>(GetI32LE(bytes, 0));
  m.length = GetI32LE(bytes, 4);
  m.fmt = static_cast<MemoryFormat>(GetI32LE(bytes, 8));
  m.dtype = IntToDtype(GetI32LE(bytes, 12));
  const std::array<int32_t, 4> dims = {GetI32LE(bytes, 16), GetI32LE(bytes, 20),
                                       GetI32LE(bytes, 24),
                                       GetI32LE(bytes, 28)};
  m.shape = StripShapePadding(dims, m.fmt);
  m.location = static_cast<Location>(GetI32LE(bytes, 32));
  return m;
}

}  // namespace vllm::v1::kv_offload::lmcache
