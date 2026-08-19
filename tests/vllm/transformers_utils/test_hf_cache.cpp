// ENG-HF-MODEL-DOWNLOAD W2 (#1280): the HuggingFace local cache layout.
//
// The layout under test is HuggingFace's documented one, so the fixtures below
// build a `models--org--repo/{refs,blobs,snapshots}` tree by hand and assert
// that the reader finds it. The structural reference is llama.cpp at stock tag
// `b10451` (commit `10bf611e533d81f739128304991c5e133c6aebd8`),
// `common/hf-cache.cpp:37-67` and `:455-496`.
//
// The DFlash cases are a RELOCATION gate. `ResolveDflashDraftDir` lived in
// `src/vllm/entrypoints/model_loader.cpp:279-303` and now calls this shared
// function, so these cases pin the behavior that must not move: the `.gguf`
// passthrough, the config.json passthrough, the `models--org--repo` slug, and
// the "return what the user typed" answer on every miss.
#include <doctest/doctest.h>

#if !defined(_WIN32)
#include <pwd.h>
#include <unistd.h>
#endif

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "support/process_id.h"
#include "support/test_env.h"
#include "vllm/transformers_utils/hf_cache.h"

namespace fs = std::filesystem;
using vllm::transformers_utils::HfBlobPath;
using vllm::transformers_utils::HfCacheFsHooks;
using vllm::transformers_utils::DefaultHfCacheFsHooks;
using vllm::transformers_utils::HfFinalizeSnapshotEntry;
using vllm::transformers_utils::HfHubCacheDir;
using vllm::transformers_utils::HfReadRef;
using vllm::transformers_utils::HfResetSymlinkFallbackStateForTesting;
using vllm::transformers_utils::HfRepoFolderName;
using vllm::transformers_utils::HfRepoPath;
using vllm::transformers_utils::HfSnapshotPath;
using vllm::transformers_utils::HfSymlinkFallbackLogCount;
using vllm::transformers_utils::HfWriteRef;
using vllm::transformers_utils::ResolveCachedSnapshotDir;

namespace {

// A unique directory removed on destruction, so a failing case leaves nothing
// behind for the next one to read as its own state.
class TempDir {
 public:
  TempDir() {
    static std::atomic<int> counter{0};
    path_ = fs::temp_directory_path() /
            ("vllm_hf_cache_test_" + std::to_string(vllm_test::ProcessId()) + "_" +
             std::to_string(counter.fetch_add(1)));
    fs::remove_all(path_);
    fs::create_directories(path_);
  }
  ~TempDir() {
    std::error_code ec;
    fs::remove_all(path_, ec);
  }
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;

  const fs::path& path() const { return path_; }

 private:
  fs::path path_;
};

void WriteFile(const fs::path& p, const std::string& body) {
  fs::create_directories(p.parent_path());
  std::ofstream out(p, std::ios::binary);
  out << body;
}

std::string ReadFile(const fs::path& p) {
  std::ifstream in(p, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

// Saves and restores every cache-root variable, so one case cannot leak its
// environment into the next.
class EnvGuard {
 public:
  EnvGuard() {
    for (const char* name : kNames) {
      const char* v = std::getenv(name);
      saved_.emplace_back(name, v == nullptr ? std::string() : std::string(v));
      vllm_test::UnsetEnv(name);
    }
  }
  ~EnvGuard() {
    for (const auto& [name, value] : saved_) {
      vllm_test::SetEnv(name.c_str(), value);
    }
  }

 private:
  static constexpr const char* kNames[] = {"HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE",
                                           "HF_HOME", "XDG_CACHE_HOME", "HOME"};
  std::vector<std::pair<std::string, std::string>> saved_;
};

}  // namespace

TEST_CASE("HfRepoFolderName mirrors the HuggingFace folder convention") {
  CHECK(HfRepoFolderName("z-lab/Qwen3.6-27B-DFlash") ==
        "models--z-lab--Qwen3.6-27B-DFlash");
  CHECK(HfRepoFolderName("Lightricks/LTX-2.5") == "models--Lightricks--LTX-2.5");
  // Every '/' becomes '--', not one '-'. The relocated implementation had a
  // discarded first attempt that produced one '-'; a repo id can carry more
  // than one slash, so this asserts the rule and not one example.
  CHECK(HfRepoFolderName("a/b/c") == "models--a--b--c");
}

TEST_CASE("HfRepoPath, HfSnapshotPath and HfBlobPath build the documented tree") {
  const fs::path hub = "/cache/hub";
  const fs::path repo = HfRepoPath(hub, "org/repo");
  CHECK(repo == fs::path("/cache/hub/models--org--repo"));
  CHECK(HfSnapshotPath(repo, "abc123") ==
        fs::path("/cache/hub/models--org--repo/snapshots/abc123"));
  CHECK(HfBlobPath(repo, "deadbeef") ==
        fs::path("/cache/hub/models--org--repo/blobs/deadbeef"));
  // An empty hub directory means "this host has no cache", and must not turn
  // into a relative path rooted at the working directory.
  CHECK(HfRepoPath(fs::path(), "org/repo").empty());
}

TEST_CASE("HfHubCacheDir follows the huggingface_hub resolution order") {
  EnvGuard guard;

  // Nothing set at all. llama.cpp `common/hf-cache.cpp:56-62 @ b10451` asks the
  // passwd database before it gives up, and a container started with `--user`
  // and no HOME is exactly the case that needs it. The precondition is read
  // from the passwd database directly, so the branch this case takes is decided
  // by the host and never by the answer under test.
#if defined(_WIN32)
  // No passwd database. The documented "this host has no cache" answer stands.
  CHECK(HfHubCacheDir().empty());
#else
  const struct passwd* pw = ::getpwuid(::getuid());
  const bool passwd_has_home =
      pw != nullptr && pw->pw_dir != nullptr && pw->pw_dir[0] != '\0';
  if (passwd_has_home) {
    REQUIRE_FALSE(HfHubCacheDir().empty());
    CHECK(HfHubCacheDir() ==
          fs::path(pw->pw_dir) / ".cache" / "huggingface" / "hub");
  } else {
    CHECK(HfHubCacheDir().empty());
  }
#endif

  vllm_test::SetEnv("HOME", "/home/u");
  CHECK(HfHubCacheDir() == fs::path("/home/u/.cache/huggingface/hub"));

  vllm_test::SetEnv("XDG_CACHE_HOME", "/xdg");
  CHECK(HfHubCacheDir() == fs::path("/xdg/huggingface/hub"));

  vllm_test::SetEnv("HF_HOME", "/cache");
  CHECK(HfHubCacheDir() == fs::path("/cache/hub"));

  vllm_test::SetEnv("HUGGINGFACE_HUB_CACHE", "/legacy-hub");
  CHECK(HfHubCacheDir() == fs::path("/legacy-hub"));

  vllm_test::SetEnv("HF_HUB_CACHE", "/hub");
  CHECK(HfHubCacheDir() == fs::path("/hub"));
}

TEST_CASE("an empty cache variable reads as unset, not as the working directory") {
  // Emptying a variable is how a container clears an inherited setting, and
  // `HF_HOME=` resolving to a path rooted at "" would write the cache into the
  // working directory. `vllm_test::SetEnv` DELETES on an empty value by design
  // (tests/support/test_env.h), so this case has to call setenv itself, and it
  // is POSIX-only because `_putenv_s(name, "")` deletes on Windows and a
  // defined-but-empty variable cannot exist there at all.
  EnvGuard guard;
  vllm_test::SetEnv("HOME", "/home/u");
#if !defined(_WIN32)
  REQUIRE(::setenv("HF_HOME", "", /*overwrite=*/1) == 0);
  REQUIRE(std::getenv("HF_HOME") != nullptr);
  CHECK(HfHubCacheDir() == fs::path("/home/u/.cache/huggingface/hub"));
  REQUIRE(::setenv("HF_HUB_CACHE", "", /*overwrite=*/1) == 0);
  CHECK(HfHubCacheDir() == fs::path("/home/u/.cache/huggingface/hub"));
#endif
  // The same variable with a value still wins, so the case above is measuring
  // the empty check and not a variable the resolution never reads.
  vllm_test::SetEnv("HF_HOME", "/cache");
  CHECK(HfHubCacheDir() == fs::path("/cache/hub"));
}

TEST_CASE("HfHubCacheDir is re-read, not frozen on the first call") {
  EnvGuard guard;
  vllm_test::SetEnv("HF_HOME", "/first");
  CHECK(HfHubCacheDir() == fs::path("/first/hub"));
  vllm_test::SetEnv("HF_HOME", "/second");
  // llama.cpp caches this in a function-local static, which would answer
  // "/first/hub" here for the rest of the process.
  CHECK(HfHubCacheDir() == fs::path("/second/hub"));
}

TEST_CASE("refs round trip, and a miss reads as empty") {
  TempDir tmp;
  const fs::path repo = HfRepoPath(tmp.path(), "org/repo");

  CHECK(HfReadRef(repo, "main").empty());

  const std::string commit = "0123456789abcdef0123456789abcdef01234567";
  HfWriteRef(repo, "main", commit);
  CHECK(fs::is_regular_file(repo / "refs" / "main"));
  CHECK(HfReadRef(repo, "main") == commit);
  CHECK(HfReadRef(repo, "other").empty());

  // Trailing whitespace written by another tool must not become part of the
  // commit, because the commit goes straight into a request path. The carriage
  // return and the space matter: std::getline already drops the newline on its
  // own, so a newline alone does not test the trim at all.
  WriteFile(repo / "refs" / "v1", commit + " \t\r\n");
  CHECK(HfReadRef(repo, "v1") == commit);
}

TEST_CASE("refs resolution refuses a reference that escapes the repo directory") {
  TempDir tmp;
  const fs::path repo = HfRepoPath(tmp.path(), "org/repo");
  const std::string commit = "0123456789abcdef0123456789abcdef01234567";
  HfWriteRef(repo, "main", commit);

  // A REAL file at the escaped location, so the case fails when the containment
  // check goes away. Asserting an escaping name against a location that holds
  // nothing proves only that the file is absent.
  const std::string outside = "89abcdef0123456789abcdef0123456789abcdef";
  WriteFile(tmp.path() / "outside-ref", outside);
  CHECK(fs::is_regular_file(repo / "refs" / ".." / ".." / "outside-ref"));
  CHECK(HfReadRef(repo, "../../outside-ref").empty());

  CHECK_THROWS_AS(HfWriteRef(repo, "../escape", commit), std::runtime_error);
  CHECK_FALSE(fs::exists(repo / ".." / "escape"));
}

TEST_CASE("the snapshot entry is a symbolic link when the file system allows one") {
  HfResetSymlinkFallbackStateForTesting();
  TempDir tmp;
  const fs::path repo = HfRepoPath(tmp.path(), "org/repo");
  const fs::path blob = HfBlobPath(repo, "deadbeef");
  WriteFile(blob, "weights");
  const fs::path final_path = HfSnapshotPath(repo, "c0ffee") / "model.safetensors";

  REQUIRE(HfFinalizeSnapshotEntry(blob, final_path));
  CHECK(fs::is_symlink(final_path));
  CHECK(ReadFile(final_path) == "weights");
  // The link is relative, so the whole cache tree stays movable.
  CHECK(fs::read_symlink(final_path).is_relative());
  // The blob is still the one copy on disk.
  CHECK(fs::is_regular_file(blob));
}

TEST_CASE("no symbolic link: the entry falls back and is a real file") {
  HfResetSymlinkFallbackStateForTesting();
  TempDir tmp;
  const fs::path repo = HfRepoPath(tmp.path(), "org/repo");

  // Stand in for a file system that holds no symbolic link. /cache is a
  // declared container volume and this fleet's shared storage is CIFS, so this
  // is a supported configuration and not a defensive branch.
  HfCacheFsHooks hooks = DefaultHfCacheFsHooks();
  hooks.create_symlink = [](const fs::path&, const fs::path&,
                            std::error_code& ec) {
    ec = std::make_error_code(std::errc::function_not_supported);
  };


  const fs::path blob_a = HfBlobPath(repo, "aaaa");
  WriteFile(blob_a, "first");
  const fs::path final_a = HfSnapshotPath(repo, "c0ffee") / "a.safetensors";
  REQUIRE(HfFinalizeSnapshotEntry(blob_a, final_a, hooks));
  CHECK(fs::is_regular_file(final_a));
  CHECK_FALSE(fs::is_symlink(final_a));
  CHECK(ReadFile(final_a) == "first");
  // The MOVE ran, so the blob is gone. Without this the case cannot tell a move
  // from the copy that follows it, and deleting the rename step leaves it green.
  CHECK_FALSE(fs::exists(blob_a));

  const fs::path blob_b = HfBlobPath(repo, "bbbb");
  WriteFile(blob_b, "second");
  const fs::path final_b = HfSnapshotPath(repo, "c0ffee") / "sub/b.safetensors";
  REQUIRE(HfFinalizeSnapshotEntry(blob_b, final_b, hooks));
  CHECK(fs::is_regular_file(final_b));
  CHECK_FALSE(fs::is_symlink(final_b));
  CHECK(ReadFile(final_b) == "second");
  CHECK_FALSE(fs::exists(blob_b));

  // Two fallbacks, ONE line. A per-file warning on a repository with 300 shards
  // is 300 lines that say the same thing.
  CHECK(HfSymlinkFallbackLogCount() == 1);
}

TEST_CASE("no symbolic link and no move: the entry is copied and the blob stays") {
  HfResetSymlinkFallbackStateForTesting();
  TempDir tmp;
  const fs::path repo = HfRepoPath(tmp.path(), "org/repo");

  HfCacheFsHooks hooks = DefaultHfCacheFsHooks();
  hooks.create_symlink = [](const fs::path&, const fs::path&,
                            std::error_code& ec) {
    ec = std::make_error_code(std::errc::function_not_supported);
  };
  hooks.rename = [](const fs::path&, const fs::path&, std::error_code& ec) {
    ec = std::make_error_code(std::errc::cross_device_link);
  };

  const fs::path blob = HfBlobPath(repo, "cccc");
  WriteFile(blob, "payload");
  const fs::path final_path = HfSnapshotPath(repo, "c0ffee") / "c.safetensors";
  REQUIRE(HfFinalizeSnapshotEntry(blob, final_path, hooks));
  CHECK(fs::is_regular_file(final_path));
  CHECK_FALSE(fs::is_symlink(final_path));
  CHECK(ReadFile(final_path) == "payload");
  // A copy leaves the blob in place; a move would have taken it.
  CHECK(fs::is_regular_file(blob));
}

TEST_CASE("one file system with no symbolic link does not degrade another") {
  // The latch is evidence about a FILE SYSTEM, and a directory lives on exactly
  // one of them. A process-wide latch generalizes one CIFS mount onto every
  // later cache root, including one that holds symbolic links perfectly well,
  // and never retries.
  HfResetSymlinkFallbackStateForTesting();
  TempDir cifs;
  TempDir local;

  HfCacheFsHooks refusing = DefaultHfCacheFsHooks();
  refusing.create_symlink = [](const fs::path&, const fs::path&,
                               std::error_code& ec) {
    ec = std::make_error_code(std::errc::function_not_supported);
  };

  const fs::path cifs_repo = HfRepoPath(cifs.path(), "org/repo");
  const fs::path cifs_blob = HfBlobPath(cifs_repo, "aaaa");
  WriteFile(cifs_blob, "first");
  const fs::path cifs_entry = HfSnapshotPath(cifs_repo, "c0ffee") / "a.safetensors";
  REQUIRE(HfFinalizeSnapshotEntry(cifs_blob, cifs_entry, refusing));
  CHECK_FALSE(fs::is_symlink(cifs_entry));
  CHECK(HfSymlinkFallbackLogCount() == 1);

  // A DIFFERENT cache root, with a file system that does hold symbolic links.
  const fs::path local_repo = HfRepoPath(local.path(), "org/repo");
  const fs::path local_blob = HfBlobPath(local_repo, "bbbb");
  WriteFile(local_blob, "second");
  const fs::path local_entry =
      HfSnapshotPath(local_repo, "c0ffee") / "b.safetensors";
  REQUIRE(HfFinalizeSnapshotEntry(local_blob, local_entry));
  CHECK(fs::is_symlink(local_entry));
  CHECK(ReadFile(local_entry) == "second");
  // The blob is still there: a symbolic link neither moves nor copies it.
  CHECK(fs::is_regular_file(local_blob));
  // ...and the second root logged nothing, because nothing was degraded there.
  CHECK(HfSymlinkFallbackLogCount() == 1);
}

TEST_CASE("finalizing an entry that already exists is a no-op that reports success") {
  HfResetSymlinkFallbackStateForTesting();
  TempDir tmp;
  const fs::path repo = HfRepoPath(tmp.path(), "org/repo");
  const fs::path blob = HfBlobPath(repo, "dddd");
  WriteFile(blob, "new");
  const fs::path final_path = HfSnapshotPath(repo, "c0ffee") / "d.safetensors";
  WriteFile(final_path, "already here");
  CHECK(HfFinalizeSnapshotEntry(blob, final_path));
  CHECK(ReadFile(final_path) == "already here");
}

// ── the relocated DFlash draft resolution ───────────────────────────────────

TEST_CASE("ResolveCachedSnapshotDir returns a .gguf draft unchanged") {
  TempDir tmp;
  const fs::path hub = tmp.path() / "hub";
  const fs::path gguf = tmp.path() / "draft.gguf";
  WriteFile(gguf, "GGUF");

  // A cache entry that the repository-id branch WOULD find, keyed by the same
  // string. Without it this case passes against any implementation that hands
  // the path back, including one that never looks at the extension. With it the
  // case answers the real question: which probe wins.
  //
  // A .gguf file carries no config.json, so it is probed BEFORE the config.json
  // test and before the cache search, or it would be reported as missing.
  WriteFile(hub / HfRepoFolderName(gguf.string()) / "snapshots" / "s" /
                "config.json",
            "{}");
  CHECK(ResolveCachedSnapshotDir(gguf.string(), hub) == gguf.string());
}

TEST_CASE("ResolveCachedSnapshotDir returns a local directory with config.json unchanged") {
  TempDir tmp;
  const fs::path hub = tmp.path() / "hub";
  const fs::path dir = tmp.path() / "local-draft";
  WriteFile(dir / "config.json", "{}");

  // As above: a cache entry keyed by the same string, so the case fails if the
  // local directory stops winning.
  WriteFile(hub / HfRepoFolderName(dir.string()) / "snapshots" / "s" /
                "config.json",
            "{}");
  CHECK(ResolveCachedSnapshotDir(dir.string(), hub) == dir.string());
}

TEST_CASE("ResolveCachedSnapshotDir maps a repo id to its cache snapshot") {
  TempDir tmp;
  const fs::path hub = tmp.path() / "hub";
  const std::string repo_id = "z-lab/Qwen3.6-27B-DFlash";
  const fs::path snapshot =
      hub / "models--z-lab--Qwen3.6-27B-DFlash" / "snapshots" / "abc123";
  WriteFile(snapshot / "config.json", "{}");
  CHECK(ResolveCachedSnapshotDir(repo_id, hub) == snapshot.string());
}

TEST_CASE("ResolveCachedSnapshotDir picks the newest snapshot") {
  // A repository can hold more than one snapshot, because a second revision is
  // fetched beside the first. The winner is the one written most recently,
  // which is what the DFlash comment at model_loader.cpp always claimed ("the
  // newest ... snapshots/<hash>/ dir") and what the relocated walk did not do:
  // it returned whatever `std::filesystem::directory_iterator` happened to
  // yield last, and that order is unspecified. This case swaps the timestamps
  // and asserts the answer follows them, so it cannot pass by name order or by
  // iterator order.
  TempDir tmp;
  const fs::path hub = tmp.path() / "hub";
  const std::string repo_id = "org/repo";
  const fs::path a = hub / "models--org--repo" / "snapshots" / "aaaa";
  const fs::path b = hub / "models--org--repo" / "snapshots" / "bbbb";
  WriteFile(a / "config.json", "{}");
  WriteFile(b / "config.json", "{}");

  const fs::file_time_type now = fs::file_time_type::clock::now();
  fs::last_write_time(a, now - std::chrono::hours(48));
  fs::last_write_time(b, now - std::chrono::hours(1));
  CHECK(ResolveCachedSnapshotDir(repo_id, hub) == b.string());

  fs::last_write_time(a, now);
  CHECK(ResolveCachedSnapshotDir(repo_id, hub) == a.string());

  // A snapshot with no config.json never wins, however new it is.
  const fs::path partial = hub / "models--org--repo" / "snapshots" / "cccc";
  fs::create_directories(partial);
  fs::last_write_time(partial, now + std::chrono::hours(1));
  CHECK(ResolveCachedSnapshotDir(repo_id, hub) == a.string());
}

TEST_CASE("two snapshots written at the same instant resolve to one stable answer") {
  // The timestamp cannot separate them, so the path does. The requirement is
  // that the answer is the SAME on every call: two runs of one command must not
  // load two different checkpoints.
  TempDir tmp;
  const fs::path hub = tmp.path() / "hub";
  const fs::path a = hub / "models--org--repo" / "snapshots" / "aaaa";
  const fs::path b = hub / "models--org--repo" / "snapshots" / "bbbb";
  WriteFile(a / "config.json", "{}");
  WriteFile(b / "config.json", "{}");
  const fs::file_time_type stamp = fs::file_time_type::clock::now();
  fs::last_write_time(a, stamp);
  fs::last_write_time(b, stamp);

  const std::string first = ResolveCachedSnapshotDir("org/repo", hub);
  CHECK(first == b.string());
  CHECK(ResolveCachedSnapshotDir("org/repo", hub) == first);
}

TEST_CASE("ResolveCachedSnapshotDir reports the original path on every miss") {
  TempDir tmp;
  const fs::path hub = tmp.path() / "hub";
  const std::string repo_id = "org/repo";

  // No cache at all.
  CHECK(ResolveCachedSnapshotDir(repo_id, hub) == repo_id);
  // An empty hub directory: the host has no cache root.
  CHECK(ResolveCachedSnapshotDir(repo_id, fs::path()) == repo_id);
  // A snapshots directory that holds no snapshot with a config.json.
  const fs::path snapshot = hub / "models--org--repo" / "snapshots" / "abc123";
  fs::create_directories(snapshot);
  CHECK(ResolveCachedSnapshotDir(repo_id, hub) == repo_id);
  // ...and it resolves as soon as the config.json appears.
  WriteFile(snapshot / "config.json", "{}");
  CHECK(ResolveCachedSnapshotDir(repo_id, hub) == snapshot.string());
}
