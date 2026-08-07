// MiniMax-H3 — the ORIGINAL bf16 DiT release: 13 safetensors shards, 66.3 GB.
//
// Every DiT loader before this one took a SINGLE file (one GGUF, or one NVFP4
// safetensors), which made the full-precision checkpoint the one thing we could
// not load — and therefore made "is quantization costing us render quality?"
// unanswerable. H3 is unusually quantization-sensitive (Q3_K_M -> Q4_K_M alone
// turned a murky lattice-covered silhouette into a photoreal close-up; ComfyUI PR
// 15298 traces it to the partial split-half RoPE producing channel-wise magnitude
// outliers that corrupt even INT8), so the comparison is worth the loader.
//
// This file owns the SHARD RESOLUTION half: the checkpoint's own
// `model.safetensors.index.json` weight map is used, never guessed around, and a
// tensor the index names but whose shard does not contain it throws BY NAME
// rather than being skipped (a skipped weight reads as zeros later, which is a
// plausible-looking render rather than an error). It mirrors the in-tree
// multi-shard template `LoadMiniMaxH3EncoderWeights(const
// std::vector<SafetensorsFile>&, ...)` in minimax_h3_vae_loader.cpp: one index
// over every shard, so a tensor is found wherever it lives.
//
// The STREAMING stager that a real 66.3 GB run uses lives next to its GGUF and
// NVFP4 twins in minimax_h3_device.cpp, because it shares their view binder.
#include <sys/stat.h>

#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/minimax_h3.h"
#include "vt/dtype.h"  // VT_CHECK

namespace vllm {
namespace {

// The two spellings a multi-shard release uses. The H3 DiT ships the first; the
// second is the diffusers convention for a pipeline sub-model. Nothing is
// discovered by SCANNING the directory — the index file is what maps a tensor to
// a shard, and guessing by filename is exactly the mistake that silently loads a
// stale or partial shard set.
const char* const kIndexNames[] = {
    "model.safetensors.index.json",
    "diffusion_pytorch_model.safetensors.index.json",
};

bool IsDir(const std::string& path) {
  struct stat st {};
  return ::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

bool IsFile(const std::string& path) {
  struct stat st {};
  return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

std::string StripTrailingSlash(const std::string& dir) {
  std::string out = dir;
  while (out.size() > 1 && out.back() == '/') out.pop_back();
  return out;
}

// The index path inside `dir`, or "" when the directory holds neither spelling.
std::string FindIndexPath(const std::string& dir) {
  for (const char* name : kIndexNames) {
    const std::string candidate = StripTrailingSlash(dir) + "/" + name;
    if (IsFile(candidate)) return candidate;
  }
  return std::string();
}

}  // namespace

// The fp32 ISLAND split, single-sourced (minimax_h3_transformer.py:85-101). All
// four staging paths (GGUF stream, NVFP4 bf16 stream, NVFP4 fp4 stream, and the
// sharded bf16 stream) must agree on it: `vt::MatmulBT` rejects a mixed (f32
// activation, bf16 weight) pair, so a tensor on the wrong side fails loudly at
// the first island GEMM rather than drifting numerically.
bool MiniMaxH3IsFp32IslandTensor(const std::string& n) {
  return n.rfind("video_patch_proj.", 0) == 0 || n.rfind("audio_patch_proj.", 0) == 0 ||
         n.rfind("time_embedder.", 0) == 0 || n.rfind("final_layer.video_out.", 0) == 0 ||
         n.rfind("final_layer.audio_out.", 0) == 0 || n == "rope.inv_freq";
}

struct MiniMaxH3ShardedCheckpoint::Impl {
  std::string dir;
  std::string index_path;
  std::vector<std::string> shard_files;  // index-first-seen order, deduplicated
  std::vector<SafetensorsFile> shards;   // parallel to shard_files
  std::vector<std::string> names;        // every tensor the index names
  std::map<std::string, size_t> shard_of;          // name -> slot in shard_files
  std::map<std::string, const StTensor*> tensors;  // name -> its entry in that shard
};

MiniMaxH3ShardedCheckpoint::MiniMaxH3ShardedCheckpoint() : impl_(std::make_unique<Impl>()) {}
MiniMaxH3ShardedCheckpoint::~MiniMaxH3ShardedCheckpoint() = default;
MiniMaxH3ShardedCheckpoint::MiniMaxH3ShardedCheckpoint(MiniMaxH3ShardedCheckpoint&&) noexcept =
    default;
MiniMaxH3ShardedCheckpoint& MiniMaxH3ShardedCheckpoint::operator=(
    MiniMaxH3ShardedCheckpoint&&) noexcept = default;

bool MiniMaxH3ShardedCheckpoint::IsShardedDir(const std::string& path) {
  return IsDir(path) && !FindIndexPath(path).empty();
}

MiniMaxH3ShardedCheckpoint MiniMaxH3ShardedCheckpoint::Open(const std::string& dir) {
  VT_CHECK(IsDir(dir), "minimax_h3 sharded: '" + dir + "' is not a directory");
  const std::string index_path = FindIndexPath(dir);
  VT_CHECK(!index_path.empty(), "minimax_h3 sharded: '" + StripTrailingSlash(dir) +
                                    "' holds no model.safetensors.index.json");

  MiniMaxH3ShardedCheckpoint out;
  Impl& impl = *out.impl_;
  impl.dir = StripTrailingSlash(dir);
  impl.index_path = index_path;

  // The index IS the map. LoadSafetensorsIndex already rejects a shard value that
  // is not a plain filename, so a hostile index cannot escape the directory.
  const std::map<std::string, std::string> weight_map = LoadSafetensorsIndex(index_path);
  VT_CHECK(!weight_map.empty(),
           "minimax_h3 sharded: '" + index_path + "' has an empty weight_map");

  std::map<std::string, size_t> slot_of_file;
  for (const auto& entry : weight_map) {
    if (slot_of_file.count(entry.second) != 0) continue;
    slot_of_file.emplace(entry.second, impl.shard_files.size());
    impl.shard_files.push_back(entry.second);
  }

  // Open every shard ONCE, before any tensor pointer is taken: the StTensor
  // addresses below point into these objects, so the vector must be final first.
  impl.shards.reserve(impl.shard_files.size());
  for (const std::string& file : impl.shard_files) {
    impl.shards.push_back(SafetensorsFile::Open(impl.dir + "/" + file));
  }

  // One index over every shard. A name the index promises but whose shard does
  // not contain it is a HARD error, reported with the tensor AND the shard.
  std::vector<std::set<std::string>> present;
  present.reserve(impl.shards.size());
  for (const SafetensorsFile& shard : impl.shards) {
    present.emplace_back(shard.Names().begin(), shard.Names().end());
  }
  impl.names.reserve(weight_map.size());
  for (const auto& entry : weight_map) {
    const std::string& name = entry.first;
    const size_t slot = slot_of_file.at(entry.second);
    VT_CHECK(present[slot].count(name) != 0,
             "minimax_h3 sharded: the index names tensor '" + name + "' in shard '" +
                 entry.second + "', but that shard does not contain it");
    impl.names.push_back(name);
    impl.shard_of.emplace(name, slot);
    impl.tensors.emplace(name, &impl.shards[slot].Get(name));
  }
  return out;
}

const std::vector<std::string>& MiniMaxH3ShardedCheckpoint::Names() const { return impl_->names; }

bool MiniMaxH3ShardedCheckpoint::Has(const std::string& name) const {
  return impl_->tensors.count(name) != 0;
}

const StTensor& MiniMaxH3ShardedCheckpoint::Get(const std::string& name) const {
  const auto it = impl_->tensors.find(name);
  VT_CHECK(it != impl_->tensors.end(),
           "minimax_h3 sharded: no tensor named '" + name + "' in " + impl_->index_path);
  return *it->second;
}

const std::string& MiniMaxH3ShardedCheckpoint::ShardOf(const std::string& name) const {
  const auto it = impl_->shard_of.find(name);
  VT_CHECK(it != impl_->shard_of.end(),
           "minimax_h3 sharded: no tensor named '" + name + "' in " + impl_->index_path);
  return impl_->shard_files[it->second];
}

const std::vector<std::string>& MiniMaxH3ShardedCheckpoint::ShardFiles() const {
  return impl_->shard_files;
}

const std::string& MiniMaxH3ShardedCheckpoint::IndexPath() const { return impl_->index_path; }

size_t MiniMaxH3ShardedCheckpoint::ShardCount() const { return impl_->shards.size(); }

std::vector<MiniMaxH3TensorSpec> EnumerateMiniMaxH3ShardedTensors(
    const MiniMaxH3ShardedCheckpoint& ckpt) {
  std::vector<MiniMaxH3TensorSpec> out;
  out.reserve(ckpt.Names().size());
  for (const std::string& name : ckpt.Names()) {
    const StTensor& t = ckpt.Get(name);
    // The bf16 release stores plain tensors; a packed quantized weight here would
    // mean the caller pointed a bf16 loader at a quantized checkpoint, and its
    // logical shape would be half its stored one — so refuse rather than derive
    // a silently halved geometry.
    VT_CHECK(t.dtype == "F32" || t.dtype == "BF16" || t.dtype == "F16",
             "minimax_h3 sharded: tensor '" + name + "' has unsupported dtype '" + t.dtype +
                 "' (expected F32/BF16/F16)");
    MiniMaxH3TensorSpec spec;
    spec.name = name;
    spec.shape = t.shape;
    spec.fp32 = MiniMaxH3IsFp32IslandTensor(name);
    out.push_back(std::move(spec));
  }
  return out;
}

MiniMaxH3GgufDit LoadMiniMaxH3DitFromShards(const MiniMaxH3ShardedCheckpoint& ckpt) {
  MiniMaxH3GgufDit out;
  const std::vector<MiniMaxH3TensorSpec> manifest = EnumerateMiniMaxH3ShardedTensors(ckpt);
  out.params = ParseMiniMaxH3DitParamsFromGgufManifest(manifest);
  for (const MiniMaxH3TensorSpec& spec : manifest) {
    out.storage[spec.name] = MiniMaxH3ReadSafetensorF32(ckpt.Get(spec.name));
    out.shapes[spec.name] = spec.shape;
  }
  BindMiniMaxH3DitViews(&out);
  return out;
}

}  // namespace vllm
