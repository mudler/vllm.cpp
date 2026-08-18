// See include/vllm/transformers_utils/hf_cache.h for the layout this reads and
// for the llama.cpp `b10451` anchors it mirrors.
#include "vllm/transformers_utils/hf_cache.h"

#include <atomic>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
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

// Degraded mode is process-wide on purpose: a file system that refused one
// symbolic link refuses the next one too, and a 300-shard repository must not
// pay 300 failed system calls or print 300 identical warnings.
std::atomic<bool> g_symlinks_disabled{false};
std::atomic<int> g_symlink_fallback_logs{0};

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

  std::error_code link_ec;
  if (!g_symlinks_disabled.load()) {
    // A RELATIVE target keeps the whole cache tree movable, which is how
    // huggingface_hub writes it.
    const fs::path target =
        fs::relative(blob_path, final_path.parent_path(), link_ec);
    if (!link_ec) hooks.create_symlink(target, final_path, link_ec);
    if (!link_ec) return true;
  }

  if (!g_symlinks_disabled.exchange(true)) {
    g_symlink_fallback_logs.fetch_add(1);
    std::cerr << "vllm.cpp: this file system holds no symbolic link ("
              << link_ec.message()
              << "); moving or copying HuggingFace cache entries instead\n";
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
  g_symlinks_disabled.store(false);
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
  std::string best;
  for (const fs::directory_entry& entry : fs::directory_iterator(snapshots, ec)) {
    if (fs::exists(entry.path() / "config.json", ec)) best = entry.path().string();
  }
  return best.empty() ? path : best;
}

}  // namespace transformers_utils
}  // namespace vllm
