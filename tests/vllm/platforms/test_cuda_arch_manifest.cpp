// The compiled-architecture manifest matcher (issue #1357, umbrella #1332 M2).
//
// CPU-runnable by construction: the matcher is a pure function over an INJECTED
// manifest string and an INJECTED capability, which is the same trick the
// upstream attention-selector test uses when it monkeypatches
// `torch.cuda.get_device_capability`. No GPU, no CUDA build, no lease.
//
// What this file cannot measure, stated so a green run is not over-read: it
// proves the matcher, not the launch. Nothing here executes a kernel on a device
// whose architecture is absent from the manifest, and no CUDA build runs on this
// host. See `## Owed` in .agents/specs/cuda-compiled-arch-manifest.md.
#include <doctest/doctest.h>

#include <string>
#include <vector>

#include "vllm/platforms/cuda_arch_manifest.h"

using vllm::platforms::ArchIsCompiled;
using vllm::platforms::CompiledArch;
using vllm::platforms::ParseCompiledArchs;

TEST_CASE("THE RED CASE: a manifest without the device's arch answers NO") {
  // The defect this row exists to close. Before #1357 the predicate was
  // `return true` and this request was served with no SASS for the device: a
  // default build requests 121a alone, and the card is an sm_86.
  CHECK_FALSE(ArchIsCompiled("121a", 8, 6));

  // The same build on a GB10 is the positive control, and it is an EXACT match
  // including the arch-specific suffix. If this ever answers no, the change has
  // disabled FlashAttention-2 on the one device we measure.
  CHECK(ArchIsCompiled("121a", 12, 1));
}

TEST_CASE("an EMPTY manifest is the 'FA2 was not built' case and answers no") {
  // VT_FA2_ARCHS resolves empty when no requested arch provides the feature —
  // CudaArchFeatures.cmake reports it DISABLED. The predicate must agree rather
  // than fall through to a default yes.
  CHECK_FALSE(ArchIsCompiled("", 12, 1));
  CHECK_FALSE(ArchIsCompiled("", 8, 0));
  CHECK(ParseCompiledArchs("").empty());
}

TEST_CASE("the arch-specific suffix is load-bearing in BOTH directions") {
  // `mma.sync ... kind::mxf4nvf4` is rejected on base sm_121, which is why
  // VLLM_CPP_CUDA_ARCHITECTURES defaults to `121a`. An 'a' target is emitted for
  // its own arch alone, so neither direction may be treated as the other.
  CHECK(ArchIsCompiled("121a", 12, 1));
  CHECK(ArchIsCompiled("121", 12, 1));  // a base target serves the base arch
  // A BASE-only manifest does not license the arch-specific claim and an
  // arch-specific manifest does not license a base one; both are the same arch
  // number, and a matcher comparing numbers alone would pass both.
  const std::vector<CompiledArch> a_only = ParseCompiledArchs("121a");
  REQUIRE(a_only.size() == 1);
  CHECK(a_only[0].major == 12);
  CHECK(a_only[0].minor == 1);
  CHECK(a_only[0].suffix == 'a');

  const std::vector<CompiledArch> base_only = ParseCompiledArchs("121");
  REQUIRE(base_only.size() == 1);
  CHECK(base_only[0].suffix == '\0');
}

TEST_CASE("SASS minor-version compatibility: sm_80 serves sm_86, not the reverse") {
  // A base cubin runs on a later minor of the SAME major — the rule that lets an
  // sm_80 build serve an sm_86 card. It does NOT run backwards, and it does not
  // cross a major.
  CHECK(ArchIsCompiled("80", 8, 6));
  CHECK(ArchIsCompiled("80", 8, 9));
  CHECK(ArchIsCompiled("80", 8, 0));
  CHECK_FALSE(ArchIsCompiled("86", 8, 0));
  CHECK_FALSE(ArchIsCompiled("80", 9, 0));
  CHECK_FALSE(ArchIsCompiled("90", 8, 0));

  // An arch-SPECIFIC target does not inherit forwards, because it is emitted for
  // exactly its own arch.
  CHECK_FALSE(ArchIsCompiled("80a", 8, 6));
}

TEST_CASE("the ten-SM release bundle serves every architecture it names") {
  // .github/workflows/ci.yml ships this set, and the FA2 row narrows it to the
  // architectures with a kernel body. Whatever survives must serve its own card.
  const std::string release = "80,86,87,89,120a,121a";
  CHECK(ArchIsCompiled(release, 8, 0));
  CHECK(ArchIsCompiled(release, 8, 6));
  CHECK(ArchIsCompiled(release, 8, 7));
  CHECK(ArchIsCompiled(release, 8, 9));
  CHECK(ArchIsCompiled(release, 12, 0));
  CHECK(ArchIsCompiled(release, 12, 1));
  // 8.9 is the highest Ada minor in the set, so a later same-major card is
  // served by the 8.9 base cubin; a Hopper card is not served at all, which is
  // correct — the FA2 feature row names no 9.x arch.
  CHECK(ArchIsCompiled(release, 8, 10));
  CHECK_FALSE(ArchIsCompiled(release, 9, 0));
  CHECK_FALSE(ArchIsCompiled(release, 10, 0));
}

TEST_CASE("parsing accepts the generated forms and DROPS what it cannot read") {
  // The manifest is CMake CUDA_ARCHITECTURES form, comma separated. Whitespace
  // and a trailing separator are tolerated because a generator emitting a list
  // is allowed to be sloppy about them; an entry that cannot be parsed is
  // DROPPED rather than guessed, because a guessed entry is a claim.
  const std::vector<CompiledArch> ok = ParseCompiledArchs(" 80 , 121a ,");
  REQUIRE(ok.size() == 2);
  CHECK(ok[0].major == 8);
  CHECK(ok[0].minor == 0);
  CHECK(ok[1].major == 12);
  CHECK(ok[1].minor == 1);
  CHECK(ok[1].suffix == 'a');

  // Semicolons too: a raw CMake list may reach the generator unconverted.
  CHECK(ParseCompiledArchs("80;86").size() == 2);

  // Garbage contributes nothing, and it must not make a valid neighbour vanish.
  const std::vector<CompiledArch> mixed = ParseCompiledArchs("garbage,121a,,x9");
  REQUIRE(mixed.size() == 1);
  CHECK(mixed[0].major == 12);
  CHECK(mixed[0].suffix == 'a');
  CHECK_FALSE(ArchIsCompiled("garbage", 12, 1));
}
