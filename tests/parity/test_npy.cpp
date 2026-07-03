// vllm.cpp original (parity harness); no upstream mirror.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "npy.h"

namespace {

std::string TempPath(const std::string& name) {
  return (std::filesystem::temp_directory_path() / name).string();
}

// Writes a well-formed NPY file. version 1 uses a 2-byte header length,
// version 2 a 4-byte one (little-endian, as on all supported hosts).
std::string WriteTemp(const std::string& name, const std::string& header_dict,
                      const void* payload, size_t payload_bytes,
                      unsigned char version = 1) {
  const std::string magic = "\x93NUMPY";
  std::string hdr = header_dict;
  const size_t preamble = (version == 1) ? 10 : 12;
  const size_t total = preamble + hdr.size() + 1;
  const size_t pad = (64 - (total % 64)) % 64;
  hdr += std::string(pad, ' ') + "\n";
  const std::string path = TempPath(name);
  std::ofstream f(path, std::ios::binary);
  f << magic << static_cast<char>(version) << '\x00';
  if (version == 1) {
    const uint16_t hlen = static_cast<uint16_t>(hdr.size());
    f.write(reinterpret_cast<const char*>(&hlen), 2);
  } else {
    const uint32_t hlen = static_cast<uint32_t>(hdr.size());
    f.write(reinterpret_cast<const char*>(&hlen), 4);
  }
  f << hdr;
  f.write(static_cast<const char*>(payload),
          static_cast<std::streamsize>(payload_bytes));
  return path;
}

}  // namespace

TEST_CASE("loads f4 2x2") {
  float vals[4] = {1.5f, -2.0f, 3.25f, 0.0f};
  auto p = WriteTemp("t1.npy",
                     "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 2), }",
                     vals, sizeof(vals));
  auto a = parity::LoadNpy(p);
  CHECK(a.dtype == "<f4");
  REQUIRE(a.shape == std::vector<int64_t>{2, 2});
  REQUIRE(a.data.size() == sizeof(vals));
  CHECK(reinterpret_cast<float*>(a.data.data())[2] == 3.25f);
  std::remove(p.c_str());
}

TEST_CASE("loads i8 1-d") {
  int64_t v[3] = {-1, 0, 1};
  auto p = WriteTemp("t2.npy",
                     "{'descr': '<i8', 'fortran_order': False, 'shape': (3,), }",
                     v, sizeof(v));
  auto a = parity::LoadNpy(p);
  CHECK(a.dtype == "<i8");
  CHECK(a.shape == std::vector<int64_t>{3});
  REQUIRE(a.data.size() == sizeof(v));
  CHECK(reinterpret_cast<int64_t*>(a.data.data())[0] == -1);
  std::remove(p.c_str());
}

TEST_CASE("loads u2 via v2 header") {
  uint16_t v[2] = {7, 65535};
  auto p = WriteTemp("t4.npy",
                     "{'descr': '<u2', 'fortran_order': False, 'shape': (2,), }",
                     v, sizeof(v), /*version=*/2);
  auto a = parity::LoadNpy(p);
  CHECK(a.dtype == "<u2");
  CHECK(a.shape == std::vector<int64_t>{2});
  REQUIRE(a.data.size() == sizeof(v));
  CHECK(reinterpret_cast<uint16_t*>(a.data.data())[1] == 65535);
  std::remove(p.c_str());
}

TEST_CASE("rejects bad magic and fortran order") {
  const std::string bad = TempPath("bad.npy");
  std::ofstream(bad, std::ios::binary) << "notnumpy";
  CHECK_THROWS_AS(parity::LoadNpy(bad), std::runtime_error);
  std::remove(bad.c_str());

  float x = 0;
  auto p = WriteTemp("t3.npy",
                     "{'descr': '<f4', 'fortran_order': True, 'shape': (1,), }",
                     &x, 4);
  CHECK_THROWS_AS(parity::LoadNpy(p), std::runtime_error);
  std::remove(p.c_str());
}

TEST_CASE("rejects missing file, big-endian, truncated payload") {
  CHECK_THROWS_AS(parity::LoadNpy(TempPath("does_not_exist.npy")),
                  std::runtime_error);

  float x = 0;
  auto pbe = WriteTemp("t5.npy",
                       "{'descr': '>f4', 'fortran_order': False, 'shape': (1,), }",
                       &x, 4);
  CHECK_THROWS_AS(parity::LoadNpy(pbe), std::runtime_error);
  std::remove(pbe.c_str());

  float two[2] = {1.0f, 2.0f};
  auto ptr = WriteTemp("t6.npy",
                       "{'descr': '<f4', 'fortran_order': False, 'shape': (4,), }",
                       two, sizeof(two));  // header claims 4 elems, only 2 written
  CHECK_THROWS_AS(parity::LoadNpy(ptr), std::runtime_error);
  std::remove(ptr.c_str());
}

TEST_CASE("loads real rmsnorm golden if present") {
  const std::string p =
      std::string(VLLM_CPP_GOLDENS_DIR) + "/rmsnorm_f32_8x128/x.npy";
  if (!std::filesystem::exists(p)) {
    MESSAGE("golden not present, skipping: " << p);
    return;
  }
  auto a = parity::LoadNpy(p);
  CHECK(a.dtype == "<f4");
  REQUIRE(a.shape == std::vector<int64_t>{8, 128});
  CHECK(a.data.size() == 8u * 128u * 4u);
}
