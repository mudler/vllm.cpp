// BPE encode-cost harness (#1365) — the artifact behind every timing figure in
// `.agents/specs/bpe-quadratic-merge.md`.
//
// WHY THIS FILE EXISTS. Three earlier harnesses for that spec lived in prose
// only. Nothing in the tree let a second reader re-derive one figure, and three
// successive rounds of certified constants each failed to reproduce for the
// next reviewer. AGENTS.md §Gates requires the exact build and run recipe to be
// recorded; a recipe nobody can execute is not one. So the harness is committed
// and the numbers are not.
//
// WHAT IT MEASURES. Wall time of `vllm::tok::Tokenizer::Encode` on one
// synthetic input, at the byte sizes given on the command line, through a
// `tokenizer.json` given on the command line. Encode is the whole path a
// request pays: added-token pre-pass, pretokenize, BPE merge, vocabulary
// lookup. It is deliberately NOT `BpeMerge` alone, because the request pays the
// whole thing and a component figure cannot be compared with HF's `encode`.
//
// WHAT ITS OUTPUT IS. A SESSION READING, never a bound. Every row carries the
// 1/5/15-minute load average sampled around it, because that is the term that
// moved these figures by 54% between two runs of one binary on one input. Do
// not copy a number out of this program into a spec, a matrix row, or a test
// bound without the load beside it, and do not treat any number it prints as
// reproducible on another box, another load, or another day. Take min-of-k and
// say what k was.
//
// DELIBERATELY NOT A REGISTERED TEST and deliberately NOT A GATE. A growth
// assertion over these figures was considered for this row and REJECTED — see
// `.agents/specs/bpe-quadratic-merge.md` §Tests to port item 3 — because the
// two halves of a ratio are independently preemptible on a shared runner and
// the red and green distributions overlap there. This program exists so that a
// human or an agent can re-derive a figure deliberately, on a host whose load
// they have looked at, not so that CI can decide anything from one.
//
// BUILD. The recipe is in a block comment because a `//` line may not end in a
// backslash under `-Werror=comment`.
/*
    g++ -O2 -std=c++20 -I include -I src -isystem third_party \
        tools/bench/bpe_encode_cost.cpp \
        src/vllm/tokenizer/bpe.cpp \
        src/vllm/tokenizer/tokenizer.cpp \
        src/vllm/tokenizer/pretokenizer.cpp \
        src/vllm/tokenizer/unicode_data.cpp \
        src/vllm/model_executor/model_loader/gguf_reader.cpp \
        src/vllm/model_executor/model_loader/read_only_file_mapping.cpp \
        -o /tmp/bpe_encode_cost
*/
//
// RUN.
/*
    # the SentencePiece arm, whose pre_tokenizer declares "split": false, so the
    # whole prompt is one word. English prose is the input because that is what
    # is slow there.
    /tmp/bpe_encode_cost tests/parity/goldens/tokenizer_mistral/tokenizer.json \
        --case english --repeats 5 --sizes 1000,8000

    # the byte-level arm. A single-character run is one pretoken, which is the
    # shape that reaches the same regime through the split regex.
    /tmp/bpe_encode_cost tests/parity/goldens/tokenizer_qwen36/tokenizer.json \
        --case a --case space --case newline --repeats 5 --sizes 1000,8000
*/
//
// SIZES ARE NOT DEFAULTED UPWARD ON PURPOSE. On the O(n^2) code one 65,536-byte
// English leg costs tens of seconds of one core, so a size list is something
// the caller states, having decided to spend that.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include "vllm/tokenizer/tokenizer.h"

namespace {

// The 1/5/15-minute load average, as one string, read fresh. Linux only; on a
// host without /proc/loadavg the harness says so rather than printing a figure
// with no load beside it, because an unlabelled figure is the defect this file
// exists to stop.
std::string LoadAverage() {
  std::ifstream in("/proc/loadavg");
  if (!in) return "UNKNOWN";
  double one = 0, five = 0, fifteen = 0;
  if (!(in >> one >> five >> fifteen)) return "UNKNOWN";
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.2f/%.2f/%.2f", one, five, fifteen);
  return buf;
}

// Repeats `unit` until the result is at least `bytes` long, then truncates on a
// UTF-8 boundary. Truncating mid-codepoint would feed the tokenizer invalid
// UTF-8 and measure the error path.
std::string Repeat(std::string_view unit, size_t bytes) {
  std::string out;
  out.reserve(bytes + unit.size());
  while (out.size() < bytes) out.append(unit);
  size_t end = bytes;
  while (end > 0 && (static_cast<unsigned char>(out[end]) & 0xC0) == 0x80) --end;
  out.resize(end);
  return out;
}

struct Case {
  const char* name;
  const char* unit;
};

// Each case is one input SHAPE. The first is ordinary English prose, which is
// what a user who pastes a document sends; the rest are single-class runs,
// which is what the split regex turns into one pretoken.
constexpr Case kCases[] = {
    {"english", "The quick brown fox jumps over the lazy dog. "},
    {"a", "a"},
    {"space", " "},
    {"newline", "\n"},
    {"tilde", "~"},
    {"cjk", "\xe7\x9a\x84"},
};

const Case* FindCase(std::string_view name) {
  for (const Case& c : kCases) {
    if (name == c.name) return &c;
  }
  return nullptr;
}

std::vector<size_t> ParseSizes(const char* spec) {
  std::vector<size_t> sizes;
  const char* p = spec;
  while (*p != '\0') {
    char* end = nullptr;
    const unsigned long long v = std::strtoull(p, &end, 10);
    if (end == p || v == 0) {
      std::fprintf(stderr, "bad --sizes near \"%s\"\n", p);
      std::exit(2);
    }
    sizes.push_back(static_cast<size_t>(v));
    p = end;
    if (*p == ',') ++p;
  }
  return sizes;
}

void Usage() {
  std::fprintf(stderr,
               "usage: bpe_encode_cost <tokenizer.json> [--case NAME]... "
               "[--sizes N,N,...] [--repeats K]\n"
               "cases:");
  for (const Case& c : kCases) std::fprintf(stderr, " %s", c.name);
  std::fprintf(stderr, "\n");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    Usage();
    return 2;
  }
  const std::string tokenizer_path = argv[1];
  std::vector<const Case*> cases;
  std::vector<size_t> sizes;
  int repeats = 3;
  for (int i = 2; i < argc; ++i) {
    const std::string_view arg = argv[i];
    const bool has_value = i + 1 < argc;
    if (arg == "--case" && has_value) {
      const Case* c = FindCase(argv[++i]);
      if (c == nullptr) {
        Usage();
        return 2;
      }
      cases.push_back(c);
    } else if (arg == "--sizes" && has_value) {
      sizes = ParseSizes(argv[++i]);
    } else if (arg == "--repeats" && has_value) {
      repeats = std::atoi(argv[++i]);
      if (repeats < 1) {
        Usage();
        return 2;
      }
    } else {
      Usage();
      return 2;
    }
  }
  if (cases.empty()) cases.push_back(&kCases[0]);
  if (sizes.empty()) sizes = {1000, 8000};

  const std::string load_at_load = LoadAverage();
  const auto load_start = std::chrono::steady_clock::now();
  const vllm::tok::Tokenizer tokenizer =
      vllm::tok::Tokenizer::FromHfJson(tokenizer_path);
  const double load_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - load_start)
          .count();

  // The banner is part of the output on purpose: a pasted table without it is
  // a number with no provenance, which is exactly how the figures this file
  // replaces got certified.
  std::printf("# bpe_encode_cost -- SESSION READING, NOT A BOUND\n");
  std::printf("# tokenizer: %s\n", tokenizer_path.c_str());
  std::printf("# load at tokenizer load (1/5/15): %s, load took %.2f ms\n",
              load_at_load.c_str(), load_ms);
  std::printf("# repeats k=%d, reporting MIN over k; the min is the least\n",
              repeats);
  std::printf("# contended estimate available, not an idle-host figure\n");
  std::printf("case\tbytes\tids\tk\tmin_ms\tmax_ms\tload_before\tload_after\n");

  for (const Case* c : cases) {
    for (const size_t bytes : sizes) {
      const std::string text = Repeat(c->unit, bytes);
      const std::string before = LoadAverage();
      double min_ms = 0;
      double max_ms = 0;
      size_t ids = 0;
      for (int r = 0; r < repeats; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        const std::vector<int32_t> out = tokenizer.Encode(text);
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0)
                              .count();
        ids = out.size();
        if (r == 0 || ms < min_ms) min_ms = ms;
        if (r == 0 || ms > max_ms) max_ms = ms;
      }
      const std::string after = LoadAverage();
      std::printf("%s\t%zu\t%zu\t%d\t%.3f\t%.3f\t%s\t%s\n", c->name,
                  text.size(), ids, repeats, min_ms, max_ms, before.c_str(),
                  after.c_str());
      std::fflush(stdout);
    }
  }
  // The spread between min and max within one k is printed above for the same
  // reason the load is: when it is large, the box moved under the measurement
  // and nothing on that row is quotable.
  return 0;
}
