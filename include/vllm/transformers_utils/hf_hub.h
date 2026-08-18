// vllm.cpp original: the HuggingFace hub protocol, authentication and endpoint.
//
// vLLM reaches the hub through `huggingface_hub`, so the wire protocol itself
// has no vLLM source to mirror. The structural reference is the secondary
// oracle llama.cpp at stock tag `b10451` (commit
// `10bf611e533d81f739128304991c5e133c6aebd8`):
//   - endpoint resolution, `common/common.cpp:1530-1542`
//   - reference to commit, `common/hf-cache.cpp:229-289`
//   - recursive file listing, `common/hf-cache.cpp:291-360`
//   - byte address form, `common/hf-cache.cpp:347`
//
// Three integrity rules deliberately do NOT port llama.cpp, and they are the
// reason this file exists rather than a copy of that one. They are recorded in
// `.agents/specs/hf-model-download.md`. See `HfFile::oid` and
// `HubListRepoFiles`.
//
// ENG-HF-MODEL-DOWNLOAD, issue #1280.
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace vllm {
namespace transformers_utils {

// Everything the hub calls read from the environment, resolved once so a caller
// can override any field in a test without touching the environment.
struct HfHubOptions {
  // `HF_ENDPOINT`, else `https://huggingface.co/`. Always ends in '/'.
  std::string endpoint;
  // `HF_TOKEN`, else the first line of the file named by `HF_TOKEN_PATH`.
  std::string token;
  // `HF_HUB_OFFLINE`. Resolve from the cache and open no socket.
  bool offline = false;
  // Where `models--org--repo` lives. Defaults to `HfHubCacheDir()`.
  std::filesystem::path hub_dir;
  int connect_timeout_seconds = 10;
  int read_timeout_seconds = 30;
};

// Read `HF_ENDPOINT`, `HF_TOKEN`, `HF_TOKEN_PATH`, `HF_HUB_OFFLINE` and the
// cache directory.
HfHubOptions HfHubOptionsFromEnv();

// One entry of a recursive tree listing.
struct HfFile {
  // Repository-relative path, always a real subpath of the snapshot directory.
  std::string path;
  uint64_t size = 0;
  // The large-file-storage object identifier, or empty.
  //
  // EMPTY UNLESS THE LISTING REQUEST CARRIED A TOKEN. On 17 August 2026 the
  // tree API answered an unauthenticated caller on the gated repository
  // `Lightricks/LTX-2.5` with an `lfs.oid` of one character repeated 64 times,
  // identical for all 14 large-file-storage files. That value passes
  // llama.cpp's `is_valid_oid` at `common/hf-cache.cpp:161`, which accepts any
  // 40 or 64 character hexadecimal string. An unauthenticated identifier
  // therefore proves nothing and is dropped rather than trusted.
  std::string oid;
  // `{endpoint}{repo}/resolve/{commit}/{path}`.
  std::string url;
};

// GET `{endpoint}api/models/{repo}/refs`, and return the commit the reference
// names. `revision` empty means the default branch, which is `main` when the
// listing offers it. A 40 character hexadecimal `revision` is already a commit
// and is returned without a request.
//
// Throws std::runtime_error on a refusal. HTTP 401 and 403 name the repository
// and `HF_TOKEN`.
std::string HubResolveRefToCommit(const std::string& repo_id,
                                  const std::string& revision,
                                  const HfHubOptions& opts);

// GET `{endpoint}api/models/{repo}/tree/{commit}?recursive=true`.
//
// The listing is REFUSED, not filtered, when two distinct files carry the same
// object identifier. A hub that hands out one identifier for every file is
// answering something other than the truth about the repository, and a
// per-entry filter would let the rest of that answer through.
//
// Throws std::runtime_error on a refusal.
std::vector<HfFile> HubListRepoFiles(const std::string& repo_id,
                                     const std::string& commit,
                                     const HfHubOptions& opts);

// `{endpoint}{repo}/resolve/{commit}/{path}`.
std::string HubFileUrl(const std::string& endpoint, const std::string& repo_id,
                       const std::string& commit, const std::string& path);

// Resolve the reference to a commit ONCE and record it, so that every later
// call in the same run names that commit and a moving `main` cannot change what
// a second run loads.
//
// The cached `refs/{ref}` file is the first probe, so a warm cache issues no
// request at all. Offline never opens a socket, and a cold cache under offline
// throws a std::runtime_error naming the repository directory that was
// searched.
std::string HubResolveCommitCached(const std::string& repo_id,
                                   const std::string& revision,
                                   const HfHubOptions& opts);

// True when `repo_id` has the shape the hub accepts: base characters
// `[A-Za-z0-9_]`, the special characters `/.-` only between base characters,
// and exactly one '/'. Mirrors llama.cpp `common/hf-cache.cpp:121-142`.
bool IsValidHfRepoId(const std::string& repo_id);

}  // namespace transformers_utils
}  // namespace vllm
