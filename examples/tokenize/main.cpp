// tokenize: encodes a corpus file with the vllm.cpp tokenizer, one output
// line per corpus entry: space-separated token ids, or "EMPTY" for an entry
// that encodes to zero ids (avoids the empty-line ambiguity in diffs).
//
//   tokenize <tokenizer.json | model.gguf> <corpus.txt>
//
// The corpus line format ('#' comments skipped, empty line = empty string,
// \n \r \t \\ escapes) is documented at the top of
// tests/parity/goldens/tokenizer_qwen36/corpus.txt; the reference reader is
// tools/parity/dump_tokenizer.py::read_corpus. Truth side for the GGUF e2e
// check: tools/parity/verify_tokenizer_gguf.py (same output format).
#include <exception>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/tokenizer/tokenizer.h"

namespace {

bool EndsWith(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Mirrors dump_tokenizer.py::unescape (fail loud on unknown escapes).
std::string Unescape(const std::string& line) {
  std::string out;
  out.reserve(line.size());
  for (size_t i = 0; i < line.size(); ++i) {
    if (line[i] != '\\') {
      out += line[i];
      continue;
    }
    if (i + 1 >= line.size()) {
      throw std::runtime_error("bad escape at end of corpus line: " + line);
    }
    switch (line[++i]) {
      case 'n': out += '\n'; break;
      case 'r': out += '\r'; break;
      case 't': out += '\t'; break;
      case '\\': out += '\\'; break;
      default:
        throw std::runtime_error("bad escape in corpus line: " + line);
    }
  }
  return out;
}

// Mirrors dump_tokenizer.py::read_corpus ('#' lines skipped, empty line =
// empty string; the file's trailing newline does not create an entry).
std::vector<std::string> ReadCorpus(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open corpus file: " + path);
  std::stringstream buf;
  buf << in.rdbuf();
  const std::string raw = buf.str();
  if (raw.empty() || raw.back() != '\n') {
    throw std::runtime_error(path + " must end with a newline");
  }
  std::vector<std::string> entries;
  size_t start = 0;
  while (start < raw.size()) {
    const size_t nl = raw.find('\n', start);
    const std::string line = raw.substr(start, nl - start);
    start = nl + 1;
    if (!line.empty() && line[0] == '#') continue;
    entries.push_back(Unescape(line));
  }
  return entries;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0]
              << " <tokenizer.json | model.gguf> <corpus.txt>\n";
    return 2;
  }
  try {
    const std::string model_path = argv[1];
    vllm::tok::Tokenizer tokenizer =
        EndsWith(model_path, ".gguf")
            ? vllm::tok::Tokenizer::FromGguf(vllm::GgufFile::Open(model_path))
            : vllm::tok::Tokenizer::FromHfJson(model_path);
    for (const std::string& text : ReadCorpus(argv[2])) {
      const std::vector<int32_t> ids = tokenizer.Encode(text);
      if (ids.empty()) {
        std::cout << "EMPTY\n";
        continue;
      }
      for (size_t i = 0; i < ids.size(); ++i) {
        if (i > 0) std::cout << ' ';
        std::cout << ids[i];
      }
      std::cout << '\n';
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
