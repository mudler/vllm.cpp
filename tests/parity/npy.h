// vllm.cpp original (parity harness); no upstream mirror.
#pragma once

#include <cctype>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// Minimal NPY (.npy) reader for the parity harness.
// Supports format v1.0/v2.0, C-order, little-endian (or endian-agnostic '|')
// dtypes only. Throws std::runtime_error on anything malformed or unsupported.

namespace parity {

struct NpyArray {
  std::vector<char> data;
  std::vector<int64_t> shape;
  std::string dtype;  // numpy descr, e.g. "<f4", "<u2", "<i8"
};

namespace detail {

// Returns the remainder of the header dict after "'key':".
inline std::string ExtractField(const std::string& hdr, const std::string& key) {
  const auto k = hdr.find("'" + key + "'");
  if (k == std::string::npos) throw std::runtime_error("npy: missing key " + key);
  const auto colon = hdr.find(':', k);
  if (colon == std::string::npos)
    throw std::runtime_error("npy: malformed header near key " + key);
  return hdr.substr(colon + 1);
}

inline bool IsDigit(char c) {
  return std::isdigit(static_cast<unsigned char>(c)) != 0;
}

}  // namespace detail

inline NpyArray LoadNpy(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("npy: cannot open " + path);

  char magic[6];
  f.read(magic, 6);
  if (!f || std::string(magic, 6) != "\x93NUMPY")
    throw std::runtime_error("npy: bad magic in " + path);

  unsigned char ver[2];
  f.read(reinterpret_cast<char*>(ver), 2);
  if (!f) throw std::runtime_error("npy: truncated header in " + path);
  uint32_t hlen = 0;
  if (ver[0] == 1) {
    unsigned char b[2];
    f.read(reinterpret_cast<char*>(b), 2);
    hlen = static_cast<uint32_t>(b[0]) | (static_cast<uint32_t>(b[1]) << 8);
  } else if (ver[0] == 2) {
    unsigned char b[4];
    f.read(reinterpret_cast<char*>(b), 4);
    hlen = static_cast<uint32_t>(b[0]) | (static_cast<uint32_t>(b[1]) << 8) |
           (static_cast<uint32_t>(b[2]) << 16) | (static_cast<uint32_t>(b[3]) << 24);
  } else {
    throw std::runtime_error("npy: unsupported version in " + path);
  }
  std::string hdr(hlen, '\0');
  f.read(hdr.data(), hlen);
  if (!f) throw std::runtime_error("npy: truncated header in " + path);

  // descr
  const auto d = detail::ExtractField(hdr, "descr");
  const auto q1 = d.find('\'');
  const auto q2 = (q1 == std::string::npos) ? std::string::npos : d.find('\'', q1 + 1);
  if (q2 == std::string::npos)
    throw std::runtime_error("npy: malformed descr in " + path);
  NpyArray arr;
  arr.dtype = d.substr(q1 + 1, q2 - q1 - 1);
  if (arr.dtype.size() < 3)
    throw std::runtime_error("npy: unsupported descr '" + arr.dtype + "'");
  if (arr.dtype[0] != '<' && arr.dtype[0] != '|')
    throw std::runtime_error("npy: non-little-endian descr '" + arr.dtype + "'");

  // fortran_order: the value is the first token after the colon.
  const auto fo = detail::ExtractField(hdr, "fortran_order");
  const auto fb = fo.find_first_not_of(" \t");
  if (fb != std::string::npos && fo.compare(fb, 5, "False") == 0) {
    // C-order: supported.
  } else if (fb != std::string::npos && fo.compare(fb, 4, "True") == 0) {
    throw std::runtime_error("npy: fortran order unsupported in " + path);
  } else {
    throw std::runtime_error("npy: malformed fortran_order in " + path);
  }

  // shape
  const auto s = detail::ExtractField(hdr, "shape");
  const auto lp = s.find('(');
  const auto rp = (lp == std::string::npos) ? std::string::npos : s.find(')', lp);
  if (rp == std::string::npos)
    throw std::runtime_error("npy: malformed shape in " + path);
  const std::string dims = s.substr(lp + 1, rp - lp - 1);
  // Strict tokenizer: comma-separated [0-9]+ tokens; spaces around tokens are
  // allowed, and a trailing comma is valid ("(3,)"). Anything else (minus
  // signs, letters, empty inner tokens) is malformed.
  std::vector<std::string> tokens;
  size_t start = 0;
  while (true) {
    const size_t comma = dims.find(',', start);
    tokens.push_back(comma == std::string::npos
                         ? dims.substr(start)
                         : dims.substr(start, comma - start));
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  int64_t total = 1;
  for (size_t i = 0; i < tokens.size(); ++i) {
    std::string t = tokens[i];
    const auto b = t.find_first_not_of(" \t");
    const auto e = t.find_last_not_of(" \t");
    t = (b == std::string::npos) ? std::string() : t.substr(b, e - b + 1);
    if (t.empty()) {
      // Valid only as a trailing comma ("(3,)") or an empty tuple ("()").
      if (i + 1 == tokens.size()) break;
      throw std::runtime_error("npy: malformed shape in " + path);
    }
    for (char c : t) {
      if (!detail::IsDigit(c))
        throw std::runtime_error("npy: malformed shape in " + path);
    }
    int64_t v = 0;
    try {
      v = std::stoll(t);
    } catch (const std::exception&) {
      throw std::runtime_error("npy: bad dim in " + path);
    }
    if (v != 0 && total > INT64_MAX / v)
      throw std::runtime_error("npy: shape overflow in " + path);
    total *= v;
    arr.shape.push_back(v);
  }

  // element size: digits after the endian char and type char, e.g. "<f4" -> 4.
  const std::string esize_str = arr.dtype.substr(2);
  size_t esize = 0;
  for (char c : esize_str) {
    if (!detail::IsDigit(c))
      throw std::runtime_error("npy: unsupported descr '" + arr.dtype + "'");
    esize = esize * 10 + static_cast<size_t>(c - '0');
  }
  if (esize == 0)
    throw std::runtime_error("npy: unsupported descr '" + arr.dtype + "'");

  if (static_cast<size_t>(total) > SIZE_MAX / esize)
    throw std::runtime_error("npy: size overflow in " + path);
  arr.data.resize(static_cast<size_t>(total) * esize);
  f.read(arr.data.data(), static_cast<std::streamsize>(arr.data.size()));
  if (!f) throw std::runtime_error("npy: truncated payload in " + path);
  return arr;
}

}  // namespace parity
