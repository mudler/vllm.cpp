// See include/vllm/transformers_utils/hf_cache.h for the layout this reads and
// for the llama.cpp `b10451` anchors it mirrors.
#include "vllm/transformers_utils/hf_cache.h"

#if !defined(_WIN32)
#include <pwd.h>
#include <unistd.h>
#endif

#include <atomic>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace vllm {
namespace transformers_utils {

namespace fs = std::filesystem;

namespace {

// True when `subpath` stays inside `base`. Mirrors llama.cpp
// `common/hf-cache.cpp:165-174 @ b10451`. A reference name and a listing entry
// both arrive from the network and both become path components, so neither may
// climb out of the repository directory.
bool IsContainedSubpath(const fs::path& base, const fs::path& subpath) {
  if (subpath.empty() || subpath.is_absolute()) return false;
  const fs::path b = fs::absolute(base).lexically_normal();
  const fs::path t = (b / subpath).lexically_normal();
  auto [b_end, unused] = std::mismatch(b.begin(), b.end(), t.begin(), t.end());
  (void)unused;
  return b_end == b.end();
}

// The environment variable is used only when it is set AND not empty. An empty
// value is how a container clears an inherited setting, and reading it as a
// directory rooted at "" would silently write the cache into the working
// directory.
const char* NonEmptyEnv(const char* name) {
  const char* value = std::getenv(name);
  return (value != nullptr && value[0] != '\0') ? value : nullptr;
}

// Degraded mode is keyed on the DIRECTORY the blob lives in, never on the
// process. The reasoning behind the latch is "a file system that refused one
// symbolic link refuses the next one too", and that reasoning holds per file
// system. A directory sits on exactly one file system, so the blobs directory
// of one repository is the largest scope the evidence covers: a 300-shard
// repository still pays one failed system call and prints one warning, while a
// CIFS mount can no longer degrade a second cache root on a disk that holds
// symbolic links perfectly well.
std::mutex g_symlink_state_mu;
std::set<std::string> g_symlink_disabled_dirs;
std::atomic<int> g_symlink_fallback_logs{0};

std::string SymlinkLatchKey(const fs::path& blob_path) {
  return blob_path.parent_path().lexically_normal().string();
}

bool SymlinksDisabledFor(const std::string& key) {
  const std::lock_guard<std::mutex> lock(g_symlink_state_mu);
  return g_symlink_disabled_dirs.count(key) != 0;
}

// True when this call is the one that switched `key` into degraded mode, which
// is the call that owns the single log line.
bool DisableSymlinksFor(const std::string& key) {
  const std::lock_guard<std::mutex> lock(g_symlink_state_mu);
  return g_symlink_disabled_dirs.insert(key).second;
}

}  // namespace

std::filesystem::path HfHubCacheDir() {
  // Mirrors `huggingface_hub`, and llama.cpp `common/hf-cache.cpp:37-67`.
  // llama.cpp's `LLAMA_CACHE` entry is that project's own name and has no
  // meaning here, so it is not read.
  struct Entry {
    const char* var;
    fs::path suffix;
  };
  const Entry entries[] = {
      {"HF_HUB_CACHE", fs::path()},
      {"HUGGINGFACE_HUB_CACHE", fs::path()},
      {"HF_HOME", fs::path("hub")},
      {"XDG_CACHE_HOME", fs::path("huggingface") / "hub"},
      {"HOME", fs::path(".cache") / "huggingface" / "hub"},
  };
  for (const Entry& entry : entries) {
    if (const char* value = NonEmptyEnv(entry.var)) {
      const fs::path base(value);
      return entry.suffix.empty() ? base : base / entry.suffix;
    }
  }
#if !defined(_WIN32)
  // llama.cpp `common/hf-cache.cpp:56-62 @ b10451`. A container started with
  // `--user` and no HOME still has a passwd entry, and huggingface_hub finds a
  // cache there, so refusing here would answer "this host has no cache" about a
  // host that has one.
  if (const struct passwd* pw = ::getpwuid(::getuid())) {
    if (pw->pw_dir != nullptr && pw->pw_dir[0] != '\0') {
      return fs::path(pw->pw_dir) / ".cache" / "huggingface" / "hub";
    }
  }
#endif
  // Upstream throws here (`:63`). This returns an empty path instead, which
  // every caller reads as "this host has no cache": the cache directory is
  // resolved eagerly by callers that may never need it, and a throw would turn
  // a passwd-less container into a failure on a code path that has no cache
  // work to do.
  return {};
}

std::string HfRepoFolderName(const std::string& repo_id) {
  std::string folder = "models--";
  for (const char c : repo_id) {
    if (c == '/') {
      folder += "--";
    } else {
      folder.push_back(c);
    }
  }
  return folder;
}

std::filesystem::path HfRepoPath(const std::filesystem::path& hub_dir,
                                 const std::string& repo_id) {
  if (hub_dir.empty()) return {};
  return hub_dir / HfRepoFolderName(repo_id);
}

std::filesystem::path HfSnapshotPath(const std::filesystem::path& repo_path,
                                     const std::string& commit) {
  return repo_path / "snapshots" / commit;
}

std::filesystem::path HfBlobPath(const std::filesystem::path& repo_path,
                                 const std::string& oid) {
  return repo_path / "blobs" / oid;
}

std::string HfReadRef(const std::filesystem::path& repo_path,
                      const std::string& ref) {
  if (repo_path.empty() || ref.empty()) return {};
  const fs::path refs_path = repo_path / "refs";
  if (!IsContainedSubpath(refs_path, ref)) return {};

  std::ifstream in(refs_path / ref, std::ios::binary);
  if (!in) return {};
  std::string commit;
  std::getline(in, commit);
  // A file another tool wrote can carry trailing whitespace, and the commit
  // goes straight into a request path.
  while (!commit.empty() &&
         (commit.back() == '\r' || commit.back() == '\n' ||
          commit.back() == ' ' || commit.back() == '\t')) {
    commit.pop_back();
  }
  return commit;
}

void HfWriteRef(const std::filesystem::path& repo_path, const std::string& ref,
                const std::string& commit) {
  if (repo_path.empty()) {
    throw std::runtime_error("vllm.cpp: cannot record reference '" + ref +
                             "': this host has no HuggingFace cache directory");
  }
  const fs::path refs_path = repo_path / "refs";
  if (!IsContainedSubpath(refs_path, ref)) {
    throw std::runtime_error("vllm.cpp: refusing reference name '" + ref +
                             "': it escapes " + refs_path.string());
  }
  const fs::path target = refs_path / ref;
  std::error_code ec;
  fs::create_directories(target.parent_path(), ec);
  if (ec) {
    throw std::runtime_error("vllm.cpp: cannot create " +
                             target.parent_path().string() + ": " + ec.message());
  }
  std::ofstream out(target, std::ios::binary | std::ios::trunc);
  out << commit;
  out.flush();
  if (!out) {
    throw std::runtime_error("vllm.cpp: cannot write " + target.string());
  }
}

const HfCacheFsHooks& DefaultHfCacheFsHooks() {
  static const HfCacheFsHooks hooks = [] {
    HfCacheFsHooks h;
    h.create_symlink = [](const fs::path& target, const fs::path& link,
                          std::error_code& ec) {
      fs::create_symlink(target, link, ec);
    };
    h.rename = [](const fs::path& from, const fs::path& to,
                  std::error_code& ec) { fs::rename(from, to, ec); };
    return h;
  }();
  return hooks;
}

bool HfFinalizeSnapshotEntry(const std::filesystem::path& blob_path,
                             const std::filesystem::path& final_path,
                             const HfCacheFsHooks& hooks) {
  std::error_code ec;
  if (blob_path == final_path || fs::exists(final_path, ec)) return true;
  if (!fs::exists(blob_path, ec)) return false;

  ec.clear();
  fs::create_directories(final_path.parent_path(), ec);
  if (ec) {
    std::cerr << "vllm.cpp: cannot create " << final_path.parent_path().string()
              << ": " << ec.message() << "\n";
    return false;
  }

  const std::string latch_key = SymlinkLatchKey(blob_path);
  std::error_code link_ec;
  if (!SymlinksDisabledFor(latch_key)) {
    // A RELATIVE target keeps the whole cache tree movable, which is how
    // huggingface_hub writes it.
    const fs::path target =
        fs::relative(blob_path, final_path.parent_path(), link_ec);
    if (!link_ec) hooks.create_symlink(target, final_path, link_ec);
    if (!link_ec) return true;
  }

  if (DisableSymlinksFor(latch_key)) {
    g_symlink_fallback_logs.fetch_add(1);
    std::cerr << "vllm.cpp: " << latch_key
              << " is on a file system that holds no symbolic link ("
              << link_ec.message()
              << "); moving or copying HuggingFace cache entries there "
                 "instead\n";
  }

  ec.clear();
  hooks.rename(blob_path, final_path, ec);
  if (!ec) return true;

  std::error_code copy_ec;
  fs::copy(blob_path, final_path, copy_ec);
  if (copy_ec) {
    std::cerr << "vllm.cpp: cannot place " << final_path.string()
              << ": move failed (" << ec.message() << ") and copy failed ("
              << copy_ec.message() << ")\n";
    return false;
  }
  return true;
}

int HfSymlinkFallbackLogCount() { return g_symlink_fallback_logs.load(); }

void HfResetSymlinkFallbackStateForTesting() {
  {
    const std::lock_guard<std::mutex> lock(g_symlink_state_mu);
    g_symlink_disabled_dirs.clear();
  }
  g_symlink_fallback_logs.store(0);
}

std::string ResolveCachedSnapshotDir(const std::string& path,
                                     const std::filesystem::path& hub_dir) {
  std::error_code ec;
  // A .gguf file carries no config.json, so it is probed first or it would fall
  // through to the cache search and be reported as missing.
  if (fs::is_regular_file(path, ec) && fs::path(path).extension() == ".gguf") {
    return path;
  }
  if (fs::exists(fs::path(path) / "config.json", ec)) return path;
  if (hub_dir.empty()) return path;

  const fs::path snapshots = HfRepoPath(hub_dir, path) / "snapshots";
  if (!fs::is_directory(snapshots, ec)) return path;
  // The NEWEST snapshot that holds a config.json wins, and the path breaks a
  // tie. `std::filesystem::directory_iterator` does not order its entries, so
  // the relocated walk's "the last one the iterator yielded" answered a
  // repository with two revisions differently on different hosts and could
  // answer one host differently on two runs. The order here is total, so two
  // runs of one command load one checkpoint.
  std::string best;
  fs::file_time_type best_time{};
  for (const fs::directory_entry& entry : fs::directory_iterator(snapshots, ec)) {
    std::error_code entry_ec;
    if (!fs::exists(entry.path() / "config.json", entry_ec)) continue;
    const fs::file_time_type written =
        fs::last_write_time(entry.path(), entry_ec);
    // An unreadable timestamp must not win over a readable one, and must still
    // be usable when it is the only candidate.
    const fs::file_time_type stamp =
        entry_ec ? fs::file_time_type::min() : written;
    const std::string candidate = entry.path().string();
    if (best.empty() || stamp > best_time ||
        (stamp == best_time && candidate > best)) {
      best = candidate;
      best_time = stamp;
    }
  }
  return best.empty() ? path : best;
}

}  // namespace transformers_utils
}  // namespace vllm
