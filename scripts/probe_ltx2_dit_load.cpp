// Does a REAL LTX-2.5 DiT resolve onto the L2 contract? — the bf16 arm, measured.
//
// A sibling of `scripts/probe_ltx2_text_encoder_load.cpp`, for the other half of
// the same question. That probe answered "does the bf16 TEXT ENCODER load", which
// is what #2140 fixed. This one answers "does the 42 GB bf16 DiT resolve", which
// nothing in this tree had asked: every LTX-2.5 render taken here loaded the
// NVFP4 or the FP8 transformer, so the dev bf16 file's own arm is unexercised
// past its header.
//
// ─── WHAT IT ESTABLISHES, AND WHAT IT CANNOT ────────────────────────────────
//
// `Ltx2LoadDitFromSafetensors` and `Ltx2StreamDitToDevice` share their whole
// prologue and differ only in what the per-tensor loop does with the bytes
// (`ltx2_loader.cpp:703-806`): plan the file, parse the geometry, adopt the
// declared config, build the contract, and refuse an unported family. THAT
// PROLOGUE IS WHERE EVERY DiT REFUSAL IN THIS TREE HAS HAPPENED — issue #1148's
// "the checkpoint carries modules this port does NOT carry" and `PlanDit`'s own
// arm resolution are both in it — and it touches only the 677,616-byte header.
//
// This probe runs that prologue's public equivalents and then walks the contract
// against the file's own header, checking, per tensor:
//
//   - the name the contract requires is IN the file, under the ComfyUI prefix;
//   - its stored shape is the logical one the contract asks for;
//   - its stored dtype is one this loader materializes;
//   - its byte count is exactly what that shape and dtype require, which is
//     `MaterializeDitTensor`'s own BF16 check (`ltx2_loader.cpp:499-506`).
//
// IT DOES NOT MATERIALIZE. A host load of this file is ~42 GB of bf16 and the
// device load is the same bytes uploaded one tensor at a time; neither fits on a
// CPU gate box and the second needs a GPU at all. So this probe cannot say the
// render works, and it does not claim to. What it CAN say is whether the arm is
// refused before a byte is read — which is the thing a GPU lease should not be
// spent discovering, and the thing the #2140 refusal turned out to be.
//
// ─── BUILD AND RUN (there is no CMake target; this is the recorded recipe) ───
// Deliberately not a target, for the same reason its three sibling probes are
// not: a probe should not charge every configure. Written down rather than
// implied, because a reviewer cannot re-run a probe whose compile line was never
// recorded.
//
//   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
//   ninja -C build vllm
//   g++ -O2 -std=c++20 -Iinclude -Ithird_party
//       scripts/probe_ltx2_dit_load.cpp build/libvllm.a -o /tmp/ltx2_dit -pthread
//   (one line; it is split here only because a trailing backslash inside a `//`
//   comment is -Wcomment, and this file is compiled warning-clean on purpose)
//   /tmp/ltx2_dit <transformer.safetensors>
//
// Exit 0 and a trailing `OK` is a clean resolution; exit 1 and `REFUSED: <msg>`
// is the loader's own refusal, printed rather than swallowed; exit 2 is a
// contract tensor this file cannot satisfy, listed by name.
#include <cstdio>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/ltx2.h"
#include "vllm/model_executor/models/ltx2_loader.h"

namespace {

std::string ShapeText(const std::vector<int64_t>& s) {
  std::string out = "[";
  for (size_t i = 0; i < s.size(); ++i) {
    out += (i != 0 ? ", " : "") + std::to_string(s[i]);
  }
  return out + "]";
}

// The width one stored element occupies, for the encodings this loader
// materializes. Anything else returns 0 and is reported rather than assumed.
size_t ElemBytes(const std::string& dtype) {
  if (dtype == "BF16" || dtype == "F16") return 2;
  if (dtype == "F32") return 4;
  if (dtype == "F8_E4M3" || dtype == "U8") return 1;
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <transformer.safetensors>\n", argv[0]);
    return 64;
  }
  const std::string path = argv[1];

  try {
    const vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(path);

    // Census first, from the file's own header, so the arm is a count and not a
    // claim. `PlanDit` decides `quant` off exactly these dtypes.
    std::map<std::string, int64_t> dtypes;
    int64_t sidecars = 0;
    for (const std::string& name : file.Names()) {
      ++dtypes[file.Get(name).dtype];
      if (name.size() >= 6 && name.compare(name.size() - 6, 6, "_scale") == 0) ++sidecars;
    }
    std::printf("file            %s\n", path.c_str());
    std::printf("tensors         %zu\n", file.Names().size());
    std::printf("dtypes         ");
    for (const auto& kv : dtypes) std::printf(" %s=%ld", kv.first.c_str(), kv.second);
    std::printf("\nscale_sidecars  %ld\n", sidecars);

    // The shared prologue, in the order both loaders run it.
    vllm::Ltx2DitQuant quant = vllm::Ltx2DitQuant::kFp8;  // never the expected value
    const vllm::Ltx2DitParams from_shapes =
        vllm::Ltx2ParseDitParamsFromCheckpoint(file, &quant);
    const char* arm = quant == vllm::Ltx2DitQuant::kNvfp4  ? "kNvfp4"
                      : quant == vllm::Ltx2DitQuant::kFp8  ? "kFp8"
                                                           : "kNone";
    std::printf("resolved_arm    %s\n", arm);
    std::printf("model_version   %s\n", vllm::Ltx2ReadCheckpointModelVersion(file).c_str());
    std::printf("geometry        layers=%ld inner=%ld audio_inner=%ld in_ch=%ld audio_in_ch=%ld\n",
                static_cast<long>(from_shapes.num_layers),
                static_cast<long>(from_shapes.inner_dim()),
                static_cast<long>(from_shapes.audio_inner_dim()),
                static_cast<long>(from_shapes.in_channels),
                static_cast<long>(from_shapes.audio_in_channels));

    // The declared config, which is what the engine adopts and what the SHAPES
    // cannot see. `Ltx2AdoptDeclaredDitParams` refuses a config describing
    // another checkpoint, so reaching past this line is itself a result.
    const nlohmann::json config = vllm::Ltx2ReadCheckpointConfig(file);
    const vllm::Ltx2DitParams declared = vllm::Ltx2AdoptDeclaredDitParams(
        config, from_shapes, "the probed DiT's own __metadata__[\"config\"]");
    std::printf("adopted_config  rope_f64=%d av_ca_mult=%ld prompt_adaln=%d keyframes=%d\n",
                declared.double_precision_rope ? 1 : 0,
                static_cast<long>(declared.av_ca_timestep_scale_multiplier),
                declared.use_prompt_adaln_single ? 1 : 0,
                declared.use_keyframes_abs_pos_embedding ? 1 : 0);

    // The contract the per-tensor loop will walk.
    const std::vector<vllm::Ltx2TensorSpec> contract = vllm::EnumerateLtx2DitTensors(declared);
    std::printf("contract        %zu tensors\n", contract.size());

    // The file's names, with the ComfyUI prefix stripped the way `PlanDit`
    // strips it, so the contract's bare names can be looked up directly.
    const std::string kPrefix = "model.diffusion_model.";
    std::map<std::string, std::string> bare_to_file;
    for (const std::string& name : file.Names()) {
      const std::string bare =
          name.rfind(kPrefix, 0) == 0 ? name.substr(kPrefix.size()) : name;
      bare_to_file[bare] = name;
    }

    // The walk. Every failure is COLLECTED rather than thrown on, so one run
    // reports the whole gap instead of the first name in header order.
    std::vector<std::string> missing;
    std::vector<std::string> mismatched;
    std::vector<std::string> unreadable;
    int64_t contract_bytes = 0;
    std::set<std::string> bound;
    for (const vllm::Ltx2TensorSpec& spec : contract) {
      auto it = bare_to_file.find(spec.name);
      if (it == bare_to_file.end()) {
        missing.push_back(spec.name + " " + ShapeText(spec.shape));
        continue;
      }
      bound.insert(it->second);
      const vllm::StTensor& t = file.Get(it->second);
      std::vector<int64_t> logical = t.shape;
      if (t.dtype == "U8" && logical.size() == 2) logical[1] *= 2;  // two values per byte
      if (logical != spec.shape) {
        mismatched.push_back(spec.name + " file " + ShapeText(logical) + " contract " +
                             ShapeText(spec.shape));
        continue;
      }
      const size_t elem = ElemBytes(t.dtype);
      if (elem == 0) {
        unreadable.push_back(spec.name + " dtype " + t.dtype);
        continue;
      }
      int64_t want = static_cast<int64_t>(elem);
      for (int64_t d : t.shape) want *= d;
      if (static_cast<int64_t>(t.nbytes) != want) {
        mismatched.push_back(spec.name + " holds " + std::to_string(t.nbytes) +
                             " bytes, its shape and " + t.dtype + " need " +
                             std::to_string(want));
        continue;
      }
      contract_bytes += static_cast<int64_t>(t.nbytes);
    }

    // Names the FILE carries that the contract does not bind. This is the other
    // direction, and it is the one `UnportedFamilies` reads: a family here is
    // what #1148 refused on. Reported as a count plus its distinct prefixes,
    // because the list itself can be thousands of names.
    std::set<std::string> unbound_prefixes;
    int64_t unbound = 0;
    for (const auto& kv : bare_to_file) {
      if (bound.count(kv.second) != 0) continue;
      ++unbound;
      const size_t dot = kv.first.find('.');
      unbound_prefixes.insert(dot == std::string::npos ? kv.first : kv.first.substr(0, dot));
    }

    std::printf("contract_bytes  %ld (%.2f GiB, what a load materializes)\n",
                static_cast<long>(contract_bytes),
                static_cast<double>(contract_bytes) / 1073741824.0);
    std::printf("bound           %zu of %zu file tensors\n", bound.size(), file.Names().size());
    std::printf("unbound         %ld tensors, top-level names:", static_cast<long>(unbound));
    for (const std::string& p : unbound_prefixes) std::printf(" %s", p.c_str());
    std::printf("\n");

    // THE VERDICT `RefuseUnported` WOULD REACH, stated rather than left for the
    // reader to infer from a count. `UnportedFamilies` (`ltx2_loader.cpp:618-631`)
    // skips a family for which `LoadedElsewhere` is true, and the two
    // `*_embeddings_connector` families are exactly those: they are outside the
    // DiT contract by design and `Ltx2LoadConnectorWeights` loads them, which
    // `RefuseUnported`'s own message says in as many words (`:654-656`). Every
    // OTHER unbound family is one this port does not carry, and without
    // `allow_unported_modules` the load refuses on it — which is what a render
    // would hit, since `ltx2-gen` passes the option only under `--allow-unported`.
    std::vector<std::string> would_refuse;
    for (const std::string& p : unbound_prefixes) {
      if (p == "video_embeddings_connector" || p == "audio_embeddings_connector") continue;
      would_refuse.push_back(p);
    }
    if (would_refuse.empty()) {
      std::printf("unported        none: the load does NOT need allow_unported_modules\n");
    } else {
      std::printf("unported        the load REFUSES without allow_unported_modules on:");
      for (const std::string& p : would_refuse) std::printf(" %s", p.c_str());
      std::printf("\n");
    }

    for (const std::string& m : missing) std::printf("MISSING     %s\n", m.c_str());
    for (const std::string& m : mismatched) std::printf("MISMATCH    %s\n", m.c_str());
    for (const std::string& m : unreadable) std::printf("UNREADABLE  %s\n", m.c_str());

    if (!missing.empty() || !mismatched.empty() || !unreadable.empty()) {
      std::printf("NOT RESOLVED: %zu missing, %zu mismatched, %zu unreadable\n", missing.size(),
                  mismatched.size(), unreadable.size());
      return 2;
    }
    // Stated rather than implied, because the difference between this and a load
    // is the whole reason the probe is cheap.
    std::printf(
        "NOT ESTABLISHED HERE: no byte was materialized and no device was touched. "
        "This says the arm is not refused before the copy, not that the render runs.\n");
    std::printf("OK\n");
    return 0;
  } catch (const std::exception& e) {
    std::printf("REFUSED: %s\n", e.what());
    return 1;
  }
}
