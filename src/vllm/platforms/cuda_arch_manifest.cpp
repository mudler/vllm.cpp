// The compiled-architecture matcher (issue #1357, umbrella #1332 M2). The header
// carries the reasoning; this file is the rule.
#include "vllm/platforms/cuda_arch_manifest.h"

#include <cctype>
#include <cstddef>

namespace vllm::platforms {

namespace {

// One CMake CUDA_ARCHITECTURES token -> a CompiledArch, or `false` when it does
// not parse. The generated form is `<major><minor>[a|f]`, the same spelling
// vt_cuda_archs_denormalize emits: "80", "121a", "100f". A token that does not
// match is REJECTED rather than repaired, because a repaired token is a claim
// about compiled code that nobody made.
bool ParseToken(const std::string& token, CompiledArch* out) {
  std::size_t begin = 0;
  std::size_t end = token.size();
  while (begin < end && std::isspace(static_cast<unsigned char>(token[begin]))) ++begin;
  while (end > begin && std::isspace(static_cast<unsigned char>(token[end - 1]))) --end;
  if (begin >= end) return false;

  char suffix = '\0';
  const char last = token[end - 1];
  if (last == 'a' || last == 'f') {
    suffix = last;
    --end;
  }
  // Need at least a major and a minor digit. The minor is ALWAYS the final
  // digit, which is what vt_cuda_archs_normalize assumes when it splits
  // `121` into `12.1`.
  if (end - begin < 2) return false;
  for (std::size_t i = begin; i < end; ++i) {
    if (!std::isdigit(static_cast<unsigned char>(token[i]))) return false;
  }
  int major = 0;
  for (std::size_t i = begin; i + 1 < end; ++i) {
    major = major * 10 + (token[i] - '0');
  }
  out->major = major;
  out->minor = token[end - 1] - '0';
  out->suffix = suffix;
  return true;
}

}  // namespace

std::vector<CompiledArch> ParseCompiledArchs(const std::string& manifest) {
  std::vector<CompiledArch> archs;
  std::string token;
  // Accept both separators: the generator renders a comma-separated literal, and
  // a raw CMake list reaching it unconverted is semicolon-separated. Taking both
  // means a generator change cannot silently empty the manifest — and an empty
  // manifest reads as "not compiled", which would disable FA2 everywhere rather
  // than fail loudly.
  for (std::size_t i = 0; i <= manifest.size(); ++i) {
    if (i == manifest.size() || manifest[i] == ',' || manifest[i] == ';') {
      CompiledArch arch;
      if (ParseToken(token, &arch)) archs.push_back(arch);
      token.clear();
      continue;
    }
    token.push_back(manifest[i]);
  }
  return archs;
}

bool ArchIsCompiled(const std::vector<CompiledArch>& compiled, int device_major,
                    int device_minor) {
  for (const CompiledArch& arch : compiled) {
    if (arch.major != device_major) continue;
    // An arch-SPECIFIC target (sm_121a) is emitted for exactly its own arch, so
    // it neither inherits forwards nor satisfies a base request.
    if (arch.suffix != '\0') {
      if (arch.minor == device_minor) return true;
      continue;
    }
    // A base target's SASS runs on the same major at an equal or later minor.
    if (arch.minor <= device_minor) return true;
  }
  return false;
}

bool ArchIsCompiled(const std::string& manifest, int device_major,
                    int device_minor) {
  return ArchIsCompiled(ParseCompiledArchs(manifest), device_major, device_minor);
}

}  // namespace vllm::platforms
