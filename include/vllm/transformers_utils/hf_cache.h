// vllm.cpp original: HuggingFace local cache layout and offline resolution.
//
// The layout is HuggingFace's documented one, `{hub}/models--org--repo/` with
// `refs`, `blobs` and `snapshots/{commit}/{path}`, so a host that already holds
// a Python `huggingface_hub` cache gets a hit and downloads nothing.
//
// vLLM reaches this layout through `huggingface_hub` rather than implementing
// it, so the structural reference is the secondary oracle llama.cpp at stock
// tag `b10451` (commit `10bf611e533d81f739128304991c5e133c6aebd8`):
//   - cache directory resolution, `common/hf-cache.cpp:37-67`
//   - repository folder name, `common/hf-cache.cpp:79-86`
//   - snapshot entry, link then move then copy, `common/hf-cache.cpp:455-496`
//
// ENG-HF-MODEL-DOWNLOAD, issue #1280.
#pragma once

#include <filesystem>
#include <functional>
#include <string>
#include <system_error>

namespace vllm {
namespace transformers_utils {

// The directory that holds the `models--org--repo` folders. Mirrors
// `huggingface_hub`, and llama.cpp `common/hf-cache.cpp:37-67`, in this order:
// `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, `HF_HOME`/hub,
// `XDG_CACHE_HOME`/huggingface/hub, `HOME`/.cache/huggingface/hub, and then the
// passwd database (`common/hf-cache.cpp:56-62`), because a container started
// with `--user` and no HOME still has one. Returns an empty path only when none
// of the five is set AND the passwd database offers no home directory, which
// every caller reads as "this host has no cache" rather than as a directory
// named by the empty string. Upstream throws at `:63` instead; the divergence
// is argued at the return statement.
//
// The value is resolved on every call and never cached in a function-local
// static, because a process that changes `HF_HOME` must see the change. The
// upstream caches it, which freezes the first reader's environment for the
// whole process.
std::filesystem::path HfHubCacheDir();

// "org/repo" -> "models--org--repo". Every '/' becomes '--'.
std::string HfRepoFolderName(const std::string& repo_id);

// `{hub_dir}/models--org--repo`. Empty when `hub_dir` is empty.
std::filesystem::path HfRepoPath(const std::filesystem::path& hub_dir,
                                 const std::string& repo_id);

// `{repo_path}/refs/{ref}` read as a commit, or empty on a miss. Opens no
// socket, so this is the offline half of reference resolution.
std::string HfReadRef(const std::filesystem::path& repo_path,
                      const std::string& ref);

// Write `{repo_path}/refs/{ref}`. Creates the parent directories. Throws
// std::runtime_error when the write fails.
void HfWriteRef(const std::filesystem::path& repo_path, const std::string& ref,
                const std::string& commit);

// `{repo_path}/snapshots/{commit}` and `{repo_path}/blobs/{oid}`.
std::filesystem::path HfSnapshotPath(const std::filesystem::path& repo_path,
                                     const std::string& commit);
std::filesystem::path HfBlobPath(const std::filesystem::path& repo_path,
                                 const std::string& oid);

// The two file-system calls the snapshot-entry fallback can fail on.
// Production takes the defaults. A test overrides `create_symlink` to stand in
// for a file system that holds no symbolic link, which is not a hypothetical:
// `/cache` is a declared container volume and this fleet's shared storage is
// CIFS.
struct HfCacheFsHooks {
  std::function<void(const std::filesystem::path& target,
                     const std::filesystem::path& link, std::error_code& ec)>
      create_symlink;
  std::function<void(const std::filesystem::path& from,
                     const std::filesystem::path& to, std::error_code& ec)>
      rename;
};
const HfCacheFsHooks& DefaultHfCacheFsHooks();

// Place the snapshot entry for `blob_path` at `final_path`: a symbolic link
// first, then a move, then a copy. Once a link fails, the DIRECTORY holding
// `blob_path` stays in degraded mode and no later entry there tries again, and
// the switch is logged exactly once for that directory. The latch is keyed on
// the directory rather than on the process because the evidence is about a file
// system, and a directory sits on exactly one: a 300-shard repository still
// pays one failed system call, while a CIFS cache root can no longer degrade a
// second cache root on a disk that holds symbolic links.
// Returns false only when all three placements failed.
bool HfFinalizeSnapshotEntry(
    const std::filesystem::path& blob_path,
    const std::filesystem::path& final_path,
    const HfCacheFsHooks& hooks = DefaultHfCacheFsHooks());

// How many times this process logged the degraded-mode switch. The "log it one
// time" rule is a behavior, so it needs something a test can read.
int HfSymlinkFallbackLogCount();

// Clear every degraded-mode latch and the counter. This exists for tests: the
// latches and the counter outlive one case, so a case that must observe the
// FIRST fallback would otherwise depend on the order the runner happened to
// pick.
void HfResetSymlinkFallbackStateForTesting();

// Resolve a model path to a local snapshot directory, opening no socket.
//
// A `.gguf` file is returned unchanged, because it carries no config.json and
// would otherwise fall through to the cache search and be reported as missing.
// A directory that holds a config.json is returned unchanged. Anything else is
// read as a repository id, and the winner is the NEWEST entry under
// `{hub_dir}/models--org--repo/snapshots` that holds a config.json, with the
// greater path breaking a tie.
//
// The order is total on purpose. The DFlash draft path shipped with "the last
// entry `std::filesystem::directory_iterator` yielded", and that iterator does
// not order its entries, so a repository holding two revisions could resolve
// differently on two hosts and, after a re-write of the directory, on two runs
// of one command. "Newest" is also what the relocated comment always claimed
// the function did.
//
// Returns `path` unchanged on any miss, so the caller reports what the user
// typed. An empty `hub_dir` is a miss.
std::string ResolveCachedSnapshotDir(const std::string& path,
                                     const std::filesystem::path& hub_dir);

}  // namespace transformers_utils
}  // namespace vllm
