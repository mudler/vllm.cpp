#include <doctest/doctest.h>

#include <unistd.h>

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"

namespace {

// Little-endian u64, as the safetensors header-length prefix requires.
std::string U64Le(uint64_t v) {
  std::string s(8, '\0');
  for (int i = 0; i < 8; ++i) s[i] = static_cast<char>((v >> (8 * i)) & 0xff);
  return s;
}

// Assembles a full .safetensors byte stream: u64 LE header length + JSON
// header + raw data section.
std::string MakeSafetensors(const std::string& header, const std::string& data) {
  return U64Le(header.size()) + header + data;
}

// Writes raw bytes to a unique file under the system temp dir; removed in the
// destructor so test runs don't accumulate files.
class TempFile {
 public:
  explicit TempFile(const std::string& bytes) {
    static int counter = 0;
    path_ = (std::filesystem::temp_directory_path() /
             ("vllm_safetensors_test_" + std::to_string(counter++) +
              ".safetensors"))
                .string();
    std::ofstream out(path_, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }
  ~TempFile() { std::remove(path_.c_str()); }
  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

// 24-byte data section: tensor "a" = F32 [2,2] at [0,16), tensor "b" =
// BF16 [4] at [16,24).
std::string ValidData() {
  std::string data(24, '\0');
  const float a_vals[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  std::memcpy(data.data(), a_vals, 16);
  const uint16_t b_vals[4] = {0x3f80, 0x4000, 0x4040, 0x4080};  // bf16 1,2,3,4
  std::memcpy(data.data() + 16, b_vals, 8);
  return data;
}

constexpr const char* kValidHeader =
    R"({"__metadata__":{"format":"pt","producer":"vllm.cpp-test"},)"
    R"("a":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},)"
    R"("b":{"dtype":"BF16","shape":[4],"data_offsets":[16,24]}})";

}  // namespace

TEST_CASE("safetensors: valid two-tensor file with metadata") {
  TempFile f(MakeSafetensors(kValidHeader, ValidData()));
  vllm::SafetensorsFile st = vllm::SafetensorsFile::Open(f.path());

  // Names are reported in header-appearance order, __metadata__ excluded.
  REQUIRE(st.Names().size() == 2);
  CHECK(st.Names()[0] == "a");
  CHECK(st.Names()[1] == "b");

  const vllm::StTensor& a = st.Get("a");
  CHECK(a.dtype == "F32");
  REQUIRE(a.shape == std::vector<int64_t>({2, 2}));
  REQUIRE(a.nbytes == 16);
  float a_vals[4];
  std::memcpy(a_vals, a.data, 16);
  CHECK(a_vals[0] == 1.0f);
  CHECK(a_vals[3] == 4.0f);

  const vllm::StTensor& b = st.Get("b");
  CHECK(b.dtype == "BF16");
  REQUIRE(b.shape == std::vector<int64_t>({4}));
  REQUIRE(b.nbytes == 8);
  uint16_t b_vals[4];
  std::memcpy(b_vals, b.data, 8);
  CHECK(b_vals[0] == 0x3f80);
  CHECK(b_vals[3] == 0x4080);

  REQUIRE(st.Metadata().size() == 2);
  CHECK(st.Metadata().at("format") == "pt");
  CHECK(st.Metadata().at("producer") == "vllm.cpp-test");
}

TEST_CASE("safetensors: Get on absent tensor throws with name") {
  TempFile f(MakeSafetensors(kValidHeader, ValidData()));
  vllm::SafetensorsFile st = vllm::SafetensorsFile::Open(f.path());
  CHECK_THROWS_WITH_AS(st.Get("nope"), doctest::Contains("nope"),
                       std::runtime_error);
}

TEST_CASE("safetensors: move semantics keep the mapping alive") {
  TempFile f(MakeSafetensors(kValidHeader, ValidData()));
  vllm::SafetensorsFile a = vllm::SafetensorsFile::Open(f.path());
  vllm::SafetensorsFile b = std::move(a);
  CHECK(b.Get("a").nbytes == 16);
  vllm::SafetensorsFile c = vllm::SafetensorsFile::Open(f.path());
  c = std::move(b);
  CHECK(c.Get("b").nbytes == 8);
  float first;
  std::memcpy(&first, c.Get("a").data, 4);
  CHECK(first == 1.0f);
}  // moved-from a and b destroyed here; must not double-munmap

TEST_CASE("safetensors: missing file throws with path") {
  CHECK_THROWS_WITH_AS(
      vllm::SafetensorsFile::Open("/nonexistent/no.safetensors"),
      doctest::Contains("/nonexistent/no.safetensors"), std::runtime_error);
}

TEST_CASE("safetensors: empty file throws with path") {
  TempFile f("");
  CHECK_THROWS_WITH_AS(vllm::SafetensorsFile::Open(f.path()),
                       doctest::Contains(f.path()), std::runtime_error);
}

TEST_CASE("safetensors: file shorter than the 8-byte prefix throws") {
  TempFile f(std::string("\x08\x00\x00", 3));
  CHECK_THROWS_AS(vllm::SafetensorsFile::Open(f.path()), std::runtime_error);
}

TEST_CASE("safetensors: header_len beyond file size throws") {
  // Prefix claims a 100-byte header but only 10 bytes follow.
  TempFile f(U64Le(100) + std::string(10, '{'));
  CHECK_THROWS_AS(vllm::SafetensorsFile::Open(f.path()), std::runtime_error);
}

TEST_CASE("safetensors: huge header_len throws instead of wrapping") {
  TempFile f(U64Le(UINT64_C(1) << 63) + std::string("{}"));
  CHECK_THROWS_AS(vllm::SafetensorsFile::Open(f.path()), std::runtime_error);
}

TEST_CASE("safetensors: malformed JSON header throws") {
  TempFile f(MakeSafetensors("{not json", ""));
  CHECK_THROWS_AS(vllm::SafetensorsFile::Open(f.path()), std::runtime_error);
}

TEST_CASE("safetensors: non-object header throws") {
  TempFile f(MakeSafetensors("[1,2,3]", ""));
  CHECK_THROWS_AS(vllm::SafetensorsFile::Open(f.path()), std::runtime_error);
}

TEST_CASE("safetensors: non-object tensor entry throws") {
  TempFile f(MakeSafetensors(R"({"a":42})", ""));
  CHECK_THROWS_AS(vllm::SafetensorsFile::Open(f.path()), std::runtime_error);
}

TEST_CASE("safetensors: data_offsets end beyond data section throws") {
  const char* header =
      R"({"a":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}})";
  TempFile f(MakeSafetensors(header, std::string(8, '\0')));  // section = 8
  CHECK_THROWS_AS(vllm::SafetensorsFile::Open(f.path()), std::runtime_error);
}

TEST_CASE("safetensors: data_offsets begin > end throws") {
  const char* header =
      R"({"a":{"dtype":"F32","shape":[0],"data_offsets":[8,4]}})";
  TempFile f(MakeSafetensors(header, std::string(16, '\0')));
  CHECK_THROWS_AS(vllm::SafetensorsFile::Open(f.path()), std::runtime_error);
}

TEST_CASE("safetensors: overlapping data_offsets throw") {
  const char* header =
      R"({"a":{"dtype":"F32","shape":[4],"data_offsets":[0,16]},)"
      R"("b":{"dtype":"F32","shape":[4],"data_offsets":[8,24]}})";
  TempFile f(MakeSafetensors(header, std::string(24, '\0')));
  CHECK_THROWS_AS(vllm::SafetensorsFile::Open(f.path()), std::runtime_error);
}

TEST_CASE("safetensors: reverse-offset key order is accepted") {
  // JSON key order need not match data-offset order: the official reader
  // sorts entries by offset before validating, so "a" at [8,16) listed
  // before "b" at [0,8) is a valid file and must Open.
  const char* header =
      R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[8,16]},)"
      R"("b":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})";
  std::string data(16, '\0');
  const float vals[4] = {1.0f, 2.0f, 3.0f, 4.0f};  // b = {1,2}, a = {3,4}
  std::memcpy(data.data(), vals, 16);
  TempFile f(MakeSafetensors(header, data));
  vllm::SafetensorsFile st = vllm::SafetensorsFile::Open(f.path());
  REQUIRE(st.Names() == std::vector<std::string>({"a", "b"}));
  float a0, b0;
  std::memcpy(&a0, st.Get("a").data, 4);
  std::memcpy(&b0, st.Get("b").data, 4);
  CHECK(a0 == 3.0f);
  CHECK(b0 == 1.0f);
}

TEST_CASE("safetensors: overlapping data_offsets in reverse order throw") {
  // Overlap must be caught regardless of key order: sorted spans are
  // [0,16) then [8,24).
  const char* header =
      R"({"a":{"dtype":"F32","shape":[4],"data_offsets":[8,24]},)"
      R"("b":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}})";
  TempFile f(MakeSafetensors(header, std::string(24, '\0')));
  CHECK_THROWS_WITH_AS(vllm::SafetensorsFile::Open(f.path()),
                       doctest::Contains("overlap"), std::runtime_error);
}

TEST_CASE("safetensors: gap between tensors is tolerated") {
  // [0,8) then a hole, then [16,24). The official spec wants full coverage;
  // we deliberately accept gaps (see reader comment) as long as ranges stay
  // ascending, non-overlapping, and inside the data section.
  const char* header =
      R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},)"
      R"("b":{"dtype":"F32","shape":[2],"data_offsets":[16,24]}})";
  TempFile f(MakeSafetensors(header, std::string(24, '\0')));
  vllm::SafetensorsFile st = vllm::SafetensorsFile::Open(f.path());
  CHECK(st.Get("b").nbytes == 8);
}

TEST_CASE("safetensors: duplicate tensor names throw") {
  const char* header =
      R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},)"
      R"("a":{"dtype":"F32","shape":[2],"data_offsets":[8,16]}})";
  TempFile f(MakeSafetensors(header, std::string(16, '\0')));
  CHECK_THROWS_WITH_AS(vllm::SafetensorsFile::Open(f.path()),
                       doctest::Contains("duplicate"), std::runtime_error);
}

TEST_CASE("safetensors: shape/dtype byte-count mismatch throws") {
  // F32 [2] needs 8 bytes, offsets only cover 4.
  const char* header =
      R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}})";
  TempFile f(MakeSafetensors(header, std::string(8, '\0')));
  CHECK_THROWS_AS(vllm::SafetensorsFile::Open(f.path()), std::runtime_error);
}

TEST_CASE("safetensors: numel overflow (2^40 x 2^40) throws, not wraps") {
  // 2^80 elements * 4 bytes would wrap size_t; must throw loudly.
  const char* header =
      R"({"a":{"dtype":"F32","shape":[1099511627776,1099511627776],)"
      R"("data_offsets":[0,4]}})";
  TempFile f(MakeSafetensors(header, std::string(8, '\0')));
  CHECK_THROWS_AS(vllm::SafetensorsFile::Open(f.path()), std::runtime_error);
}

TEST_CASE("safetensors: negative shape dim throws") {
  const char* header =
      R"({"a":{"dtype":"F32","shape":[-2],"data_offsets":[0,8]}})";
  TempFile f(MakeSafetensors(header, std::string(8, '\0')));
  CHECK_THROWS_AS(vllm::SafetensorsFile::Open(f.path()), std::runtime_error);
}

TEST_CASE("safetensors: zero-element tensor is fine") {
  const char* header =
      R"({"a":{"dtype":"F32","shape":[0,4],"data_offsets":[0,0]}})";
  TempFile f(MakeSafetensors(header, ""));
  vllm::SafetensorsFile st = vllm::SafetensorsFile::Open(f.path());
  CHECK(st.Get("a").nbytes == 0);
}

TEST_CASE("safetensors: unknown dtype keeps raw span, skips size cross-check") {
  const char* header =
      R"({"a":{"dtype":"F6_E3M2","shape":[5],"data_offsets":[0,4]}})";
  TempFile f(MakeSafetensors(header, std::string(4, '\xab')));
  vllm::SafetensorsFile st = vllm::SafetensorsFile::Open(f.path());
  const vllm::StTensor& a = st.Get("a");
  CHECK(a.dtype == "F6_E3M2");
  CHECK(a.nbytes == 4);
  CHECK(a.data[0] == 0xab);
}

TEST_CASE("safetensors: unknown dtype is still bounds-checked") {
  const char* header =
      R"({"a":{"dtype":"F6_E3M2","shape":[5],"data_offsets":[0,64]}})";
  TempFile f(MakeSafetensors(header, std::string(4, '\0')));
  CHECK_THROWS_AS(vllm::SafetensorsFile::Open(f.path()), std::runtime_error);
}

TEST_CASE("safetensors: known dtype sizes validate") {
  // One tensor per size class, laid out back-to-back.
  const char* header =
      R"({"t8":{"dtype":"I64","shape":[1],"data_offsets":[0,8]},)"
      R"("t4":{"dtype":"U32","shape":[1],"data_offsets":[8,12]},)"
      R"("t2":{"dtype":"F16","shape":[1],"data_offsets":[12,14]},)"
      R"("t1":{"dtype":"F8_E4M3","shape":[1],"data_offsets":[14,15]},)"
      R"("tb":{"dtype":"BOOL","shape":[1],"data_offsets":[15,16]}})";
  TempFile f(MakeSafetensors(header, std::string(16, '\0')));
  vllm::SafetensorsFile st = vllm::SafetensorsFile::Open(f.path());
  CHECK(st.Names().size() == 5);
  CHECK(st.Get("t1").nbytes == 1);
}

TEST_CASE("safetensors: non-string __metadata__ value throws") {
  const char* header = R"({"__metadata__":{"n":42}})";
  TempFile f(MakeSafetensors(header, ""));
  CHECK_THROWS_AS(vllm::SafetensorsFile::Open(f.path()), std::runtime_error);
}

TEST_CASE("safetensors: missing dtype/shape/data_offsets throw") {
  const char* no_dtype = R"({"a":{"shape":[1],"data_offsets":[0,4]}})";
  const char* no_shape = R"({"a":{"dtype":"F32","data_offsets":[0,4]}})";
  const char* no_offsets = R"({"a":{"dtype":"F32","shape":[1]}})";
  const char* bad_offsets =
      R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4,8]}})";
  for (const char* h : {no_dtype, no_shape, no_offsets, bad_offsets}) {
    TempFile f(MakeSafetensors(h, std::string(4, '\0')));
    CHECK_THROWS_AS(vllm::SafetensorsFile::Open(f.path()), std::runtime_error);
  }
}

TEST_CASE("safetensors index: parses weight_map") {
  TempFile f(
      R"({"metadata":{"total_size":24},)"
      R"("weight_map":{"a":"model-00001-of-00002.safetensors",)"
      R"("b":"model-00002-of-00002.safetensors"}})");
  auto index = vllm::LoadSafetensorsIndex(f.path());
  REQUIRE(index.size() == 2);
  CHECK(index.at("a") == "model-00001-of-00002.safetensors");
  CHECK(index.at("b") == "model-00002-of-00002.safetensors");
}

TEST_CASE("safetensors index: missing file / missing weight_map throw") {
  CHECK_THROWS_WITH_AS(vllm::LoadSafetensorsIndex("/nonexistent/idx.json"),
                       doctest::Contains("/nonexistent/idx.json"),
                       std::runtime_error);
  TempFile no_map(R"({"metadata":{}})");
  CHECK_THROWS_AS(vllm::LoadSafetensorsIndex(no_map.path()),
                  std::runtime_error);
  TempFile bad_value(R"({"weight_map":{"a":1}})");
  CHECK_THROWS_AS(vllm::LoadSafetensorsIndex(bad_value.path()),
                  std::runtime_error);
}

TEST_CASE("safetensors index: shard names with path components throw") {
  // The index is untrusted; shard names must be plain filenames, never
  // paths that could traverse outside the model directory.
  TempFile slash(R"({"weight_map":{"a":"sub/model.safetensors"}})");
  CHECK_THROWS_WITH_AS(vllm::LoadSafetensorsIndex(slash.path()),
                       doctest::Contains("plain filename"),
                       std::runtime_error);
  TempFile dotdot(R"({"weight_map":{"a":"..secret.safetensors"}})");
  CHECK_THROWS_AS(vllm::LoadSafetensorsIndex(dotdot.path()),
                  std::runtime_error);
  TempFile traverse(R"({"weight_map":{"a":"../../etc/passwd"}})");
  CHECK_THROWS_AS(vllm::LoadSafetensorsIndex(traverse.path()),
                  std::runtime_error);
}

// ---- Windowed source-page release (LOAD-SAFETENSORS memory checkpoint) ----
// Exercises the real reader (MAP_PRIVATE mmap) + the progressive-release path
// the 27B/35B loaders call after each tensor copy. See
// .agents/specs/safetensors-windowed-load.md.
//
// Residency is measured as the source MAPPING's resident set (per-VMA smaps
// Rss), NOT mincore: mincore reports page-CACHE residency for a file mapping, so
// after MADV_DONTNEED it still reads "resident" even though the pages have left
// the process's mapping. The process RSS (smaps Rss = what VmHWM tracks) is what
// this checkpoint reduces, so that is what we assert.
namespace {

size_t HostPageSize() {
  const long p = ::sysconf(_SC_PAGESIZE);
  return p > 0 ? static_cast<size_t>(p) : 4096;
}

// Resident set (KiB) of the /proc/self/smaps VMA that contains `addr` — the
// portion of that mapping present in this process's page tables.
size_t MappingRssKb(const void* addr) {
  const auto target = reinterpret_cast<uintptr_t>(addr);
  std::ifstream smaps("/proc/self/smaps");
  std::string line;
  bool in_vma = false;
  while (std::getline(smaps, line)) {
    // A VMA header line begins with "<startHex>-<endHex> perms ...". A field
    // line ("Rss:", "Size:", "Anonymous:", ...) never has that '-' after the
    // leading token, so the '-' guard distinguishes them robustly.
    if (!line.empty() && std::isxdigit(static_cast<unsigned char>(line[0]))) {
      char* p = nullptr;
      const auto start = static_cast<uintptr_t>(std::strtoull(line.c_str(), &p, 16));
      if (p != nullptr && *p == '-') {
        const auto end = static_cast<uintptr_t>(std::strtoull(p + 1, &p, 16));
        in_vma = (target >= start && target < end);
        continue;
      }
    }
    if (in_vma && line.rfind("Rss:", 0) == 0) {
      return static_cast<size_t>(std::strtoull(line.c_str() + 4, nullptr, 10));
    }
  }
  return 0;
}

// A valid U8 safetensors byte stream: one tensor per size, tensor i filled with
// byte value (i+1) so copies are byte-checkable. U8 keeps the header trivial.
std::string MakeVarFile(const std::vector<size_t>& sizes,
                        std::vector<std::pair<size_t, size_t>>* spans_out) {
  std::string header = "{";
  std::string data;
  for (size_t i = 0; i < sizes.size(); ++i) {
    const size_t begin = data.size();
    data.append(sizes[i], static_cast<char>(i + 1));
    const size_t end = data.size();
    if (i != 0) header += ",";
    header += "\"t" + std::to_string(i) + "\":{\"dtype\":\"U8\",\"shape\":[" +
              std::to_string(sizes[i]) + "],\"data_offsets\":[" +
              std::to_string(begin) + "," + std::to_string(end) + "]}";
    if (spans_out != nullptr) spans_out->emplace_back(begin, end);
  }
  header += "}";
  return MakeSafetensors(header, data);
}

}  // namespace

TEST_CASE("safetensors: windowed release drops consumed source mapping RSS") {
  const size_t page = HostPageSize();
  const size_t each = page * 512;  // multi-MB per tensor -> clear RSS signal
  const size_t total_kb = 3 * each / 1024;
  std::vector<std::pair<size_t, size_t>> spans;
  TempFile f(MakeVarFile({each, each, each}, &spans));
  vllm::SafetensorsFile st = vllm::SafetensorsFile::Open(f.path());
  REQUIRE(st.Names().size() == 3);

  // Copy every tensor (faults its source pages into the mapping RSS).
  for (const std::string& name : st.Names()) {
    const vllm::StTensor& t = st.Get(name);
    std::vector<uint8_t> owned(t.nbytes);
    std::memcpy(owned.data(), t.data, t.nbytes);
  }
  const size_t rss_after_copy = MappingRssKb(st.Get("t0").data);
  CHECK(rss_after_copy >= total_kb * 3 / 4);  // most of the source is resident

  // Release each consumed range: the mapping RSS collapses to a small residue
  // (header pages + <=1 edge page per tensor). RED until the real madvise lands.
  for (const std::string& name : st.Names()) {
    const vllm::StTensor& t = st.Get(name);
    vllm::ReleaseSourcePages(t.data, t.nbytes);
  }
  const size_t rss_after_release = MappingRssKb(st.Get("t0").data);
  CHECK(rss_after_release <= total_kb / 20);  // dropped below 5% of the source
  CHECK(rss_after_copy - rss_after_release >= total_kb / 2);
}

TEST_CASE("safetensors: windowed release preserves copied bytes") {
  const size_t page = HostPageSize();
  std::vector<std::pair<size_t, size_t>> spans;
  const std::string bytes = MakeVarFile({page * 4, page * 4, page * 4}, &spans);
  // Copy every tensor with release ON, then OFF; both reproduce the file bytes.
  for (const bool release : {true, false}) {
    vllm::detail::SetLoadWindowedReleaseOverrideForTesting(release);
    TempFile f(bytes);
    vllm::SafetensorsFile st = vllm::SafetensorsFile::Open(f.path());
    for (size_t i = 0; i < st.Names().size(); ++i) {
      const vllm::StTensor& t = st.Get(st.Names()[i]);
      std::vector<uint8_t> owned(t.nbytes);
      std::memcpy(owned.data(), t.data, t.nbytes);
      vllm::MaybeReleaseSourcePages(t.data, t.nbytes);  // gated
      bool all_ok = !owned.empty();
      for (uint8_t b : owned) all_ok = all_ok && (b == static_cast<uint8_t>(i + 1));
      CHECK(all_ok);
    }
  }
  vllm::detail::SetLoadWindowedReleaseOverrideForTesting(std::nullopt);
}

TEST_CASE("safetensors: VT_LOAD_WINDOWED_RELEASE gate semantics") {
  const size_t page = HostPageSize();
  const size_t each = page * 512;
  const size_t each_kb = each / 1024;
  std::vector<std::pair<size_t, size_t>> spans;
  TempFile f(MakeVarFile({each}, &spans));

  // Forced ON: MaybeReleaseSourcePages drops the source mapping RSS.
  {
    vllm::detail::SetLoadWindowedReleaseOverrideForTesting(true);
    CHECK(vllm::LoadWindowedReleaseEnabled());
    vllm::SafetensorsFile st = vllm::SafetensorsFile::Open(f.path());
    const vllm::StTensor& t = st.Get("t0");
    std::vector<uint8_t> owned(t.nbytes);
    std::memcpy(owned.data(), t.data, t.nbytes);
    REQUIRE(MappingRssKb(t.data) >= each_kb * 3 / 4);
    vllm::MaybeReleaseSourcePages(t.data, t.nbytes);
    CHECK(MappingRssKb(t.data) <= each_kb / 20);
  }
  // Forced OFF: inert; the source mapping stays resident (double-residency).
  {
    vllm::detail::SetLoadWindowedReleaseOverrideForTesting(false);
    CHECK_FALSE(vllm::LoadWindowedReleaseEnabled());
    vllm::SafetensorsFile st = vllm::SafetensorsFile::Open(f.path());
    const vllm::StTensor& t = st.Get("t0");
    std::vector<uint8_t> owned(t.nbytes);
    std::memcpy(owned.data(), t.data, t.nbytes);
    vllm::MaybeReleaseSourcePages(t.data, t.nbytes);
    CHECK(MappingRssKb(t.data) >= each_kb * 3 / 4);
  }
  vllm::detail::SetLoadWindowedReleaseOverrideForTesting(std::nullopt);
}

TEST_CASE("safetensors: interior-page release never drops a neighbor's bytes") {
  const size_t page = HostPageSize();
  // Tensor a ends 100 bytes into a page; tensor b begins in that same page.
  // Releasing a's interior must keep the shared edge page, so b's leading bytes
  // stay intact (and a clean re-fault would restore them anyway).
  const size_t a_size = page * 512 + 100;
  const size_t b_size = page * 512;
  const size_t b_kb = b_size / 1024;
  std::vector<std::pair<size_t, size_t>> spans;
  TempFile f(MakeVarFile({a_size, b_size}, &spans));
  vllm::SafetensorsFile st = vllm::SafetensorsFile::Open(f.path());
  const vllm::StTensor& a = st.Get("t0");
  const vllm::StTensor& b = st.Get("t1");
  std::vector<uint8_t> owned_a(a.nbytes), owned_b(b.nbytes);
  std::memcpy(owned_a.data(), a.data, a.nbytes);
  std::memcpy(owned_b.data(), b.data, b.nbytes);
  const size_t rss_both = MappingRssKb(a.data);

  vllm::ReleaseSourcePages(a.data, a.nbytes);

  // b is fully intact (its own pages, plus the retained shared edge page).
  bool b_ok = b.nbytes > 0;
  for (int64_t i = 0; i < static_cast<int64_t>(b.nbytes); ++i)
    b_ok = b_ok && (b.data[i] == static_cast<uint8_t>(2));
  CHECK(b_ok);
  // a's interior was dropped (mechanism active, RED until the real madvise
  // lands) yet b's whole resident set is retained.
  const size_t rss_after = MappingRssKb(a.data);
  CHECK(rss_after >= b_kb * 3 / 4);          // b retained
  CHECK(rss_both - rss_after >= b_kb / 2);   // a's interior dropped
}
