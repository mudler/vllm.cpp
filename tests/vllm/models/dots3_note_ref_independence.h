// The run-time REFERENCE-INDEPENDENCE instrument, shared by the dots3-note
// gates that carry an independent double-precision reference.
//
// It is a header rather than a copy in each test because it is ONE instrument:
// the property it measures (every qualified name inside a reference namespace
// resolves through `std::`) and the pp-number rule that makes the measurement
// trustworthy have to be the same code in every file that claims them. See the
// note below for the history.
#ifndef VLLM_CPP_TESTS_DOTS3_NOTE_REF_INDEPENDENCE_H_
#define VLLM_CPP_TESTS_DOTS3_NOTE_REF_INDEPENDENCE_H_

#include <doctest/doctest.h>

#include <cstddef>
#include <fstream>
#include <iterator>
#include <set>
#include <string>

// ═══════════════════════════════════════════════════════════════════════════
// THE ENUMERATION INSTRUMENT.
//
// Reads a TEST SOURCE FILE at a path the target hands it as a compile
// definition (the same arrangement `MODELOPT_MIXED_FIXTURE_DIR` uses), strips
// comments and string/char literals, takes the span of one reference namespace,
// and counts every `scope::name`. Comments must be stripped or the enumeration
// LIST beside a reference would count itself and the instrument would agree
// with any list it was given — which is the exact failure this replaces.
//
// EVERY FREE FUNCTION HERE IS `inline`, and that is a requirement rather than a
// style. The header is included by two test translation units
// (`test_dots3_note_vision.cpp` and `test_dots3_note_audio.cpp`); a non-`inline`
// definition in a header gives each TU an external-linkage symbol with the same
// name, which is an ODR violation the moment those two objects are linked into
// one binary. They are separate executables today, so the hazard is latent --
// which is exactly the state in which it gets discovered by a link error in
// somebody else's change. Found by the fresh review of PR #2947 (F7).
//
// IT LIVES IN A HEADER SINCE W9d (#2881) and it did not before. W7b, W7c-2 and
// W8a each EXTENDED it to another namespace, for the stated reason that
// reference code the instrument does not read is reference code whose
// independence nothing measures. W9d adds a reference in a DIFFERENT FILE, and
// the only two ways to read it were to copy the instrument or to give it a
// path. A copy would be a second instrument with a second pp-number rule to
// keep in step, which is the failure this file's own prose names one paragraph
// up. So the source path became an argument and `QualifiedNamesIn` became
// `QualifiedNamesInFile`; the scan, the stripper and the pp-number gate are
// byte-for-byte the ones W7c-2 landed, and the audio suite's counts are
// unchanged, which is the regression check on that claim.
// ═══════════════════════════════════════════════════════════════════════════
namespace dots3_ref_independence {

struct RefNames {
  int distinct = 0;
  int occurrences = 0;
  std::set<std::string> scopes;
  std::set<std::string> names;
};

inline std::string Join(const std::set<std::string>& s) {
  std::string out;
  for (const std::string& v : s) {
    if (!out.empty()) out += ",";
    out += v;
  }
  return out;
}

inline bool IsIdentChar(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
         (c >= '0' && c <= '9') || c == '_';
}

// Comments and literals out, everything else through unchanged.
inline std::string StripCommentsAndLiterals(const std::string& code) {
  std::string out;
  const size_t n = code.size();
  for (size_t i = 0; i < n;) {
    const char c = code[i];
    if (c == '/' && i + 1 < n && code[i + 1] == '/') {
      const size_t j = code.find('\n', i);
      i = (j == std::string::npos) ? n : j;
    } else if (c == '/' && i + 1 < n && code[i + 1] == '*') {
      const size_t j = code.find("*/", i + 2);
      i = (j == std::string::npos) ? n : j + 2;
    } else if (c >= '0' && c <= '9' && (i == 0 || !IsIdentChar(code[i - 1]))) {
      // A pp-number, taken WHOLE. C++14 lets a numeric literal carry `'` digit
      // separators (`16'000`), and treating that `'` as a char-literal
      // delimiter makes the scan below run to the NEXT `'` and drop everything
      // in between. That is a hole in THIS instrument and not a cosmetic one:
      // two separators bracketing a `vt::` call would hide the call from the
      // enumeration, and the independence property would read GREEN while being
      // false. The token must START at a digit that does not continue an
      // identifier, so `u8'a'` is still a char literal and is still stripped.
      size_t j = i;
      while (j < n) {
        if (IsIdentChar(code[j]) || code[j] == '.') {
          ++j;
        } else if (code[j] == '\'' && j + 1 < n && IsIdentChar(code[j + 1])) {
          j += 2;
        } else if ((code[j] == '+' || code[j] == '-') && j > i &&
                   (code[j - 1] == 'e' || code[j - 1] == 'E' ||
                    code[j - 1] == 'p' || code[j - 1] == 'P')) {
          ++j;
        } else {
          break;
        }
      }
      out.append(code, i, j - i);
      i = j;
    } else if (c == '"' || c == '\'') {
      size_t j = i + 1;
      while (j < n && code[j] != c) j += (code[j] == '\\') ? 2 : 1;
      i = j + 1;
    } else {
      out += c;
      ++i;
    }
  }
  return out;
}

inline RefNames QualifiedNamesInFile(const char* source_path, const std::string& ns) {
  std::ifstream in(source_path, std::ios::binary);
  REQUIRE_MESSAGE(in.good(),
                  "the enumeration instrument could not open the source at "
                      << source_path);
  const std::string src((std::istreambuf_iterator<char>(in)),
                        std::istreambuf_iterator<char>());
  const std::string open = "namespace " + ns + " {";
  const std::string close = "}  // namespace " + ns;
  const size_t a = src.find(open);
  const size_t b = src.find(close);
  REQUIRE(a != std::string::npos);
  REQUIRE(b != std::string::npos);
  REQUIRE(b > a);
  const std::string body = StripCommentsAndLiterals(src.substr(a, b - a));

  RefNames r;
  for (size_t i = 0; i + 1 < body.size(); ++i) {
    if (body[i] != ':' || body[i + 1] != ':') continue;
    // the scope to the left
    size_t s = i;
    while (s > 0 && IsIdentChar(body[s - 1])) --s;
    if (s == i) continue;
    // the name to the right
    size_t e = i + 2;
    size_t t = e;
    while (t < body.size() && IsIdentChar(body[t])) ++t;
    if (t == e) continue;
    const std::string scope = body.substr(s, i - s);
    const std::string name = body.substr(e, t - e);
    r.scopes.insert(scope);
    r.names.insert(scope + "::" + name);
    ++r.occurrences;
  }
  r.distinct = static_cast<int>(r.names.size());
  return r;
}

}  // namespace dots3_ref_independence

#endif  // VLLM_CPP_TESTS_DOTS3_NOTE_REF_INDEPENDENCE_H_
