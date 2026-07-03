#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/model_loader/gguf_reader.h"

namespace {

// Little-endian scalar encoders (GGUF is little-endian on the wire).
std::string U32Le(uint32_t v) {
  std::string s(4, '\0');
  for (int i = 0; i < 4; ++i) s[i] = static_cast<char>((v >> (8 * i)) & 0xff);
  return s;
}

std::string U64Le(uint64_t v) {
  std::string s(8, '\0');
  for (int i = 0; i < 8; ++i) s[i] = static_cast<char>((v >> (8 * i)) & 0xff);
  return s;
}

// GGUF string: u64 LE length + raw bytes (no NUL).
std::string GStr(const std::string& s) { return U64Le(s.size()) + s; }

// Writes raw bytes to a unique file under the system temp dir; removed in the
// destructor so test runs don't accumulate files.
class TempFile {
 public:
  explicit TempFile(const std::string& bytes) {
    static int counter = 0;
    path_ = (std::filesystem::temp_directory_path() /
             ("vllm_gguf_test_" + std::to_string(counter++) + ".gguf"))
                .string();
    std::ofstream out(path_, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }
  ~TempFile() { std::remove(path_.c_str()); }
  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

// GGUF header for `tensor_count` tensors and `kv_count` kvs.
std::string Header(uint32_t version, uint64_t tensor_count, uint64_t kv_count) {
  return "GGUF" + U32Le(version) + U64Le(tensor_count) + U64Le(kv_count);
}

// Pads to the alignment boundary the way writers do before the data section.
void PadTo(std::string& bytes, size_t alignment) {
  while (bytes.size() % alignment != 0) bytes.push_back('\0');
}

// One Q8_0 block: f16 scale + 32 int8 quants = 34 bytes.
std::string Q8Block(uint16_t scale_f16_bits) {
  std::string b = U32Le(scale_f16_bits).substr(0, 2);
  for (int i = 0; i < 32; ++i) b.push_back(static_cast<char>(i));
  return b;
}

// A full, valid GGUF file: 4 kvs (string, u32 alignment, string array, u64)
// and 2 tensors — "t_f32" F32 ggml-dims [4,2] at offset 0 and "t_q8" Q8_0
// [32] at offset 32 — with the data section aligned to 32.
std::string BuildValid(uint32_t version = 3) {
  std::string f = Header(version, /*tensors=*/2, /*kvs=*/4);
  f += GStr("general.architecture") + U32Le(8) + GStr("llama");
  f += GStr("general.alignment") + U32Le(4) + U32Le(32);
  f += GStr("tokenizer.ggml.tokens") + U32Le(9) +  // array kv
       U32Le(8) + U64Le(3) + GStr("a") + GStr("bb") + GStr("ccc");
  f += GStr("some.count") + U32Le(10) + U64Le(UINT64_C(1234567890123));
  // Tensor infos: name, u32 n_dims, u64 dims (ggml order), u32 type, u64 off.
  f += GStr("t_f32") + U32Le(2) + U64Le(4) + U64Le(2) + U32Le(0) + U64Le(0);
  f += GStr("t_q8") + U32Le(1) + U64Le(32) + U32Le(8) + U64Le(32);
  PadTo(f, 32);
  const float vals[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  f.append(reinterpret_cast<const char*>(vals), 32);
  f += Q8Block(0x3c00);  // f16 1.0
  return f;
}

}  // namespace

TEST_CASE("gguf: valid v3 file parses kvs and tensors") {
  TempFile f(BuildValid());
  vllm::GgufFile g = vllm::GgufFile::Open(f.path());

  // Scalar and string kvs.
  const vllm::GgufValue* arch = g.FindKv("general.architecture");
  REQUIRE(arch != nullptr);
  CHECK(arch->TypeId() == vllm::kGgufString);
  CHECK(std::get<std::string>(arch->v) == "llama");
  const vllm::GgufValue* align = g.FindKv("general.alignment");
  REQUIRE(align != nullptr);
  CHECK(std::get<uint32_t>(align->v) == 32);
  const vllm::GgufValue* count = g.FindKv("some.count");
  REQUIRE(count != nullptr);
  CHECK(std::get<uint64_t>(count->v) == UINT64_C(1234567890123));

  // String array kv (tokenizer vocab shape, needed by M0.5).
  const vllm::GgufValue* toks = g.FindKv("tokenizer.ggml.tokens");
  REQUIRE(toks != nullptr);
  REQUIRE(toks->TypeId() == vllm::kGgufArray);
  const vllm::GgufArray& arr = std::get<vllm::GgufArray>(toks->v);
  CHECK(arr.elem_type == vllm::kGgufString);
  REQUIRE(arr.elems.size() == 3);
  CHECK(std::get<std::string>(arr.elems[0].v) == "a");
  CHECK(std::get<std::string>(arr.elems[2].v) == "ccc");

  // Absent kv -> nullptr, not a throw.
  CHECK(g.FindKv("no.such.key") == nullptr);

  // Tensors, in file order; ggml dims [4,2] reversed to torch shape [2,4].
  REQUIRE(g.Tensors().size() == 2);
  const vllm::GgufTensorInfo& t0 = g.Tensors()[0];
  CHECK(t0.name == "t_f32");
  CHECK(t0.shape == std::vector<int64_t>({2, 4}));
  CHECK(t0.ggml_type == 0);
  REQUIRE(t0.nbytes == 32);
  float vals[8];
  std::memcpy(vals, t0.data, 32);
  CHECK(vals[0] == 0.0f);
  CHECK(vals[7] == 7.0f);

  const vllm::GgufTensorInfo& t1 = g.Get("t_q8");
  CHECK(t1.shape == std::vector<int64_t>({32}));
  CHECK(t1.ggml_type == 8);
  REQUIRE(t1.nbytes == 34);
  CHECK(t1.data[0] == 0x00);  // f16 1.0 scale, low byte
  CHECK(t1.data[1] == 0x3c);
  CHECK(t1.data[2 + 5] == 5);  // 6th quant
}

TEST_CASE("gguf: v2 file parses identically (same LE layout as v3)") {
  TempFile f(BuildValid(/*version=*/2));
  vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  CHECK(g.Tensors().size() == 2);
  CHECK(g.Get("t_f32").nbytes == 32);
}

TEST_CASE("gguf: general.alignment kv moves the data section start") {
  // One F32 tensor [2] at offset 0 with alignment 64: the header ends well
  // before 64, so honoring the kv is observable in the data the span sees.
  std::string f = Header(3, 1, 1);
  f += GStr("general.alignment") + U32Le(4) + U32Le(64);
  f += GStr("t") + U32Le(1) + U64Le(2) + U32Le(0) + U64Le(0);
  // The kv must be observable: default alignment 32 would place the data
  // section at a different offset than the declared 64.
  auto align_up = [](size_t v, size_t a) { return (v + a - 1) / a * a; };
  REQUIRE(align_up(f.size(), 32) != align_up(f.size(), 64));
  PadTo(f, 64);
  const float vals[2] = {5.0f, 6.0f};
  f.append(reinterpret_cast<const char*>(vals), 8);
  TempFile tf(f);
  vllm::GgufFile g = vllm::GgufFile::Open(tf.path());
  float got[2];
  std::memcpy(got, g.Get("t").data, 8);
  CHECK(got[0] == 5.0f);
  CHECK(got[1] == 6.0f);
}

TEST_CASE("gguf: move semantics keep the mapping alive") {
  TempFile f(BuildValid());
  vllm::GgufFile a = vllm::GgufFile::Open(f.path());
  vllm::GgufFile b = std::move(a);
  CHECK(b.Get("t_f32").nbytes == 32);
  vllm::GgufFile c = vllm::GgufFile::Open(f.path());
  c = std::move(b);
  CHECK(c.Get("t_q8").nbytes == 34);
  float first;
  std::memcpy(&first, c.Get("t_f32").data, 4);
  CHECK(first == 0.0f);
}  // moved-from a and b destroyed here; must not double-munmap

TEST_CASE("gguf: Get on absent tensor throws with name") {
  TempFile f(BuildValid());
  vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  CHECK_THROWS_WITH_AS(g.Get("nope"), doctest::Contains("nope"),
                       std::runtime_error);
}

TEST_CASE("gguf: missing file throws with path") {
  CHECK_THROWS_WITH_AS(vllm::GgufFile::Open("/nonexistent/no.gguf"),
                       doctest::Contains("/nonexistent/no.gguf"),
                       std::runtime_error);
}

TEST_CASE("gguf: bad magic throws") {
  std::string f = BuildValid();
  f.replace(0, 4, "GGML");
  TempFile tf(f);
  CHECK_THROWS_WITH_AS(vllm::GgufFile::Open(tf.path()),
                       doctest::Contains("magic"), std::runtime_error);
}

TEST_CASE("gguf: unsupported version throws") {
  std::string f = BuildValid();
  f.replace(4, 4, U32Le(1));
  TempFile tf(f);
  CHECK_THROWS_WITH_AS(vllm::GgufFile::Open(tf.path()),
                       doctest::Contains("version"), std::runtime_error);
}

TEST_CASE("gguf: big-endian file is rejected with a clear message") {
  std::string f = BuildValid();
  f.replace(4, 4, std::string("\x00\x00\x00\x03", 4));  // byte-swapped v3
  TempFile tf(f);
  CHECK_THROWS_WITH_AS(vllm::GgufFile::Open(tf.path()),
                       doctest::Contains("big-endian"), std::runtime_error);
}

TEST_CASE("gguf: truncated file throws") {
  std::string f = BuildValid();
  TempFile half(f.substr(0, f.size() / 2));
  CHECK_THROWS_AS(vllm::GgufFile::Open(half.path()), std::runtime_error);
  TempFile tiny(std::string("GGUF"));
  CHECK_THROWS_AS(vllm::GgufFile::Open(tiny.path()), std::runtime_error);
  TempFile empty("");
  CHECK_THROWS_AS(vllm::GgufFile::Open(empty.path()), std::runtime_error);
}

TEST_CASE("gguf: huge kv count throws before any allocation") {
  TempFile f(Header(3, 0, UINT64_C(1) << 40));
  CHECK_THROWS_WITH_AS(vllm::GgufFile::Open(f.path()),
                       doctest::Contains("kv count"), std::runtime_error);
}

TEST_CASE("gguf: huge tensor count throws before any allocation") {
  TempFile f(Header(3, UINT64_C(1) << 40, 0));
  CHECK_THROWS_WITH_AS(vllm::GgufFile::Open(f.path()),
                       doctest::Contains("tensor count"), std::runtime_error);
}

TEST_CASE("gguf: string length beyond file size throws") {
  // kv key claims 2^40 bytes.
  std::string f = Header(3, 0, 1) + U64Le(UINT64_C(1) << 40) + "x";
  TempFile tf(f);
  CHECK_THROWS_AS(vllm::GgufFile::Open(tf.path()), std::runtime_error);
}

TEST_CASE("gguf: array count beyond file size throws") {
  std::string f = Header(3, 0, 1) + GStr("k") + U32Le(9) +  // array kv
                  U32Le(7) + U64Le(UINT64_C(1) << 40);       // bool x 2^40
  TempFile tf(f);
  CHECK_THROWS_AS(vllm::GgufFile::Open(tf.path()), std::runtime_error);
}

TEST_CASE("gguf: unknown kv value type throws") {
  std::string f = Header(3, 0, 1) + GStr("k") + U32Le(99) + U64Le(0);
  TempFile tf(f);
  CHECK_THROWS_AS(vllm::GgufFile::Open(tf.path()), std::runtime_error);
}

TEST_CASE("gguf: tensor offset beyond data section throws") {
  std::string f = Header(3, 1, 0);
  f += GStr("t") + U32Le(1) + U64Le(2) + U32Le(0) + U64Le(1024);  // off 1024
  PadTo(f, 32);
  f += std::string(8, '\0');  // section is only 8 bytes
  TempFile tf(f);
  CHECK_THROWS_WITH_AS(vllm::GgufFile::Open(tf.path()),
                       doctest::Contains("t"), std::runtime_error);
}

TEST_CASE("gguf: tensor nbytes beyond data section throws") {
  std::string f = Header(3, 1, 0);
  f += GStr("t") + U32Le(1) + U64Le(64) + U32Le(0) + U64Le(0);  // needs 256B
  PadTo(f, 32);
  f += std::string(8, '\0');
  TempFile tf(f);
  CHECK_THROWS_AS(vllm::GgufFile::Open(tf.path()), std::runtime_error);
}

TEST_CASE("gguf: numel not divisible by quant block throws") {
  std::string f = Header(3, 1, 0);
  f += GStr("t") + U32Le(1) + U64Le(33) + U32Le(8) + U64Le(0);  // Q8_0 [33]
  PadTo(f, 32);
  f += std::string(64, '\0');
  TempFile tf(f);
  CHECK_THROWS_WITH_AS(vllm::GgufFile::Open(tf.path()),
                       doctest::Contains("divisible"), std::runtime_error);
}

TEST_CASE("gguf: unknown ggml type id in a tensor throws") {
  std::string f = Header(3, 1, 0);
  f += GStr("t") + U32Le(1) + U64Le(2) + U32Le(999) + U64Le(0);
  PadTo(f, 32);
  f += std::string(8, '\0');
  TempFile tf(f);
  CHECK_THROWS_WITH_AS(vllm::GgufFile::Open(tf.path()),
                       doctest::Contains("999"), std::runtime_error);
}

TEST_CASE("gguf: duplicate tensor names throw") {
  std::string f = Header(3, 2, 0);
  f += GStr("t") + U32Le(1) + U64Le(2) + U32Le(0) + U64Le(0);
  f += GStr("t") + U32Le(1) + U64Le(2) + U32Le(0) + U64Le(32);
  PadTo(f, 32);
  f += std::string(64, '\0');
  TempFile tf(f);
  CHECK_THROWS_WITH_AS(vllm::GgufFile::Open(tf.path()),
                       doctest::Contains("duplicate"), std::runtime_error);
}

TEST_CASE("gguf: dims product overflow throws, not wraps") {
  std::string f = Header(3, 1, 0);
  f += GStr("t") + U32Le(3) + U64Le(UINT64_C(1) << 40) +
       U64Le(UINT64_C(1) << 40) + U64Le(UINT64_C(1) << 40) +  // 2^120 elems
       U32Le(0) + U64Le(0);
  PadTo(f, 32);
  f += std::string(8, '\0');
  TempFile tf(f);
  CHECK_THROWS_AS(vllm::GgufFile::Open(tf.path()), std::runtime_error);
}

TEST_CASE("gguf: misaligned tensor offset throws") {
  std::string f = Header(3, 1, 0);
  f += GStr("t") + U32Le(1) + U64Le(2) + U32Le(0) + U64Le(7);  // 7 % 32 != 0
  PadTo(f, 32);
  f += std::string(64, '\0');
  TempFile tf(f);
  CHECK_THROWS_AS(vllm::GgufFile::Open(tf.path()), std::runtime_error);
}

TEST_CASE("gguf: non-power-of-two general.alignment throws") {
  std::string f = Header(3, 0, 1);
  f += GStr("general.alignment") + U32Le(4) + U32Le(24);
  TempFile tf(f);
  CHECK_THROWS_WITH_AS(vllm::GgufFile::Open(tf.path()),
                       doctest::Contains("alignment"), std::runtime_error);
}

TEST_CASE("gguf: zero-tensor file with kvs only is fine") {
  std::string f = Header(3, 0, 1) + GStr("k") + U32Le(5) + U32Le(0xfffffff6);
  TempFile tf(f);
  vllm::GgufFile g = vllm::GgufFile::Open(tf.path());
  CHECK(g.Tensors().empty());
  CHECK(std::get<int32_t>(g.FindKv("k")->v) == -10);
}

TEST_CASE("ggml traits: standard table values") {
  CHECK(vllm::GgmlTraits(0).block_elems == 1);   // F32
  CHECK(vllm::GgmlTraits(0).block_bytes == 4);
  CHECK(vllm::GgmlTraits(0).name == std::string("F32"));
  CHECK(vllm::GgmlTraits(1).block_bytes == 2);   // F16
  CHECK(vllm::GgmlTraits(2).block_elems == 32);  // Q4_0
  CHECK(vllm::GgmlTraits(2).block_bytes == 18);
  CHECK(vllm::GgmlTraits(8).block_bytes == 34);  // Q8_0
  CHECK(vllm::GgmlTraits(10).block_bytes == 84);   // Q2_K
  CHECK(vllm::GgmlTraits(11).block_bytes == 110);  // Q3_K
  CHECK(vllm::GgmlTraits(12).block_elems == 256);  // Q4_K
  CHECK(vllm::GgmlTraits(12).block_bytes == 144);
  CHECK(vllm::GgmlTraits(13).block_bytes == 176);  // Q5_K
  CHECK(vllm::GgmlTraits(14).block_bytes == 210);  // Q6_K
  CHECK(vllm::GgmlTraits(24).block_bytes == 1);    // I8
  CHECK(vllm::GgmlTraits(25).block_bytes == 2);    // I16
  CHECK(vllm::GgmlTraits(26).block_bytes == 4);    // I32
  CHECK(vllm::GgmlTraits(27).block_bytes == 8);    // I64
  CHECK(vllm::GgmlTraits(28).block_bytes == 8);    // F64
  CHECK(vllm::GgmlTraits(30).block_bytes == 2);    // BF16
  CHECK(vllm::GgmlTraits(30).name == std::string("BF16"));
}

TEST_CASE("ggml traits: unknown type id throws") {
  CHECK_THROWS_WITH_AS(vllm::GgmlTraits(999), doctest::Contains("999"),
                       std::runtime_error);
  // Real ggml ids we have not tabulated (e.g. 7 = Q5_1) must also throw
  // rather than return garbage size math.
  CHECK_THROWS_AS(vllm::GgmlTraits(7), std::runtime_error);
}
