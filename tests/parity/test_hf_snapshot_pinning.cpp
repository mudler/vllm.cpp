// vllm.cpp original. GATE-PIN-UNPINNED-SNAPSHOTS (issue #471): the resolver
// gate. Proves, on any box, with no GPU and no real checkpoint, that a pinned
// accessor SELECTS its revision out of a cache that also holds a decoy of the
// SAME repo, and REFUSES (returns "" -> the caller's loud skip) rather than
// substituting when only the decoy is present.
//
// Why this test exists rather than a run on the gate host. The three DFlash
// gates check `HasCuda()` BEFORE they resolve anything, so on a CPU box they can
// never demonstrate selection, and on the GPU box they only demonstrate it while
// the cache happens to be in the two-revision state. Synthesising the cache
// makes the hazardous state available on demand and makes the proof independent
// of what anyone has downloaded.
//
// The hazard being modelled is real and specific: `unsloth/Qwen3.6-27B-NVFP4`
// caches @890bdef7 (NVFP4 W4A4, bf16 GDN tower) and @ccdaab7e (the same repo
// name, silently re-quantized to FP8 W8A8). MEASURED on the gate host, readdir
// yields @890bdef7 first today -- so an unpinned resolver gets the right answer
// there RIGHT NOW and would get the wrong one after any reordering. A test that
// only ran on that host would therefore have passed either way. This one cannot.
#include <doctest/doctest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "hf_snapshot.h"

namespace fs = std::filesystem;

namespace {

// A scratch $HOME that is restored on scope exit, so one case's cache layout can
// never leak into another's.
class ScopedHome {
 public:
  explicit ScopedHome(const fs::path& root) : root_(root) {
    const char* prev = std::getenv("HOME");
    had_prev_ = prev != nullptr;
    if (had_prev_) prev_ = prev;
    setenv("HOME", root_.string().c_str(), /*overwrite=*/1);
  }
  ~ScopedHome() {
    if (had_prev_)
      setenv("HOME", prev_.c_str(), /*overwrite=*/1);
    else
      unsetenv("HOME");
    std::error_code ec;
    fs::remove_all(root_, ec);
  }

  ScopedHome(const ScopedHome&) = delete;
  ScopedHome& operator=(const ScopedHome&) = delete;

 private:
  fs::path root_;
  std::string prev_;
  bool had_prev_ = false;
};

// A variable forced UNSET for the duration. The local_dir arm
// below is only reached when `VT_NEMOTRON35_SNAPSHOT` is unset, and a developer
// who exports it in their own shell would otherwise silently take a DIFFERENT
// branch than the one the case is about -- which reads as a pass.
class ScopedUnsetEnv {
 public:
  explicit ScopedUnsetEnv(const char* name) : name_(name) {
    const char* prev = std::getenv(name);
    had_prev_ = prev != nullptr;
    if (had_prev_) prev_ = prev;
    unsetenv(name_);
  }
  ~ScopedUnsetEnv() {
    if (had_prev_) setenv(name_, prev_.c_str(), /*overwrite=*/1);
  }

  ScopedUnsetEnv(const ScopedUnsetEnv&) = delete;
  ScopedUnsetEnv& operator=(const ScopedUnsetEnv&) = delete;

 private:
  const char* name_;
  std::string prev_;
  bool had_prev_ = false;
};

// Same for an env override, so an unset variable stays unset afterwards.
class ScopedEnv {
 public:
  ScopedEnv(const char* name, const std::string& value) : name_(name) {
    const char* prev = std::getenv(name);
    had_prev_ = prev != nullptr;
    if (had_prev_) prev_ = prev;
    setenv(name_, value.c_str(), /*overwrite=*/1);
  }
  ~ScopedEnv() {
    if (had_prev_)
      setenv(name_, prev_.c_str(), /*overwrite=*/1);
    else
      unsetenv(name_);
  }

  ScopedEnv(const ScopedEnv&) = delete;
  ScopedEnv& operator=(const ScopedEnv&) = delete;

 private:
  const char* name_;
  std::string prev_;
  bool had_prev_ = false;
};

fs::path ScratchRoot(const std::string& tag) {
  const fs::path root =
      fs::temp_directory_path() / ("vt_hf_snapshot_pin_" + tag);
  std::error_code ec;
  fs::remove_all(root, ec);
  return root;
}

// Materialise `<home>/.cache/huggingface/hub/<repo>/snapshots/<rev>/config.json`.
// `config.json` is exactly what HfSnapshot probes for, so this is the minimum
// that makes a revision look cached.
fs::path PlantRevision(const fs::path& home, const std::string& repo,
                       const std::string& rev) {
  const fs::path dir =
      home / ".cache/huggingface/hub" / repo / "snapshots" / rev;
  fs::create_directories(dir);
  std::ofstream(dir / "config.json") << "{}\n";
  return dir;
}

// The real second revision of the real repo. Not a placeholder: this is the FP8
// re-quant that shares `unsloth/Qwen3.6-27B-NVFP4`'s name.
constexpr const char* kUnsloth27bRepo = "models--unsloth--Qwen3.6-27B-NVFP4";
constexpr const char* kUnsloth27bDecoyRevision =
    "ccdaab7e68af2409599b8949a8f2685703c9bae5";

// ------------------------------------------------------------------------- //
// GATE-SNAPSHOT-CONTENT-PIN (issue #569): the `hf download --local-dir` arm.
// ------------------------------------------------------------------------- //

// One `.cache/huggingface/download/<file>.metadata` sidecar, in the layout
// huggingface_hub actually writes: commit hash, etag, timestamp, one per line
// (huggingface_hub/_local_folder.py module docstring, and
// `LocalDownloadFileMetadata` / `read_download_metadata` which parse it back in
// that order).
void PlantSidecar(const fs::path& dir, const std::string& file,
                  const std::string& commit) {
  const fs::path meta = dir / ".cache/huggingface/download";
  fs::create_directories(meta);
  std::ofstream(meta / (file + ".metadata"))
      << commit << "\n"
      << std::string(64, 'e') << "\n"
      << "1786554554.7491977\n";
}

// The per-revision tree listing `hf download --local-dir` leaves behind. The
// body is deliberately a bare `{}`: the DEMONSTRATED failure this row closes had
// an empty `touch`ed manifest, because nothing ever read the contents.
void PlantTreeManifest(const fs::path& dir, const std::string& revision) {
  const fs::path trees = dir / ".cache/huggingface/trees";
  fs::create_directories(trees);
  std::ofstream(trees / (revision + ".json")) << "{}\n";
}

// A staged `hf download --local-dir` tree: the payload files a gate loads, the
// tree manifest for `manifest_revision`, and one sidecar per payload file naming
// `sidecar_revision`. Passing two different revisions is the substitution the
// old existence check could not see.
void PlantLocalDir(const fs::path& dir, const std::string& manifest_revision,
                   const std::string& sidecar_revision) {
  fs::create_directories(dir);
  for (const char* file : {"config.json", "hf_quant_config.json",
                           "model-00001-of-00001.safetensors"}) {
    std::ofstream(dir / file) << "{}\n";
    PlantSidecar(dir, file, sidecar_revision);
  }
  PlantTreeManifest(dir, manifest_revision);
}

// `$CHECKPOINT_ROOT/<staged dir name>`, which is the ONE path the default arm
// resolves.
fs::path StagedCheckpoint(const fs::path& root) {
  return root / parity::kNemotron35LightningLocalDirName;
}

// A revision that is not the pin: the shape of "the publisher re-quantized and
// the same `hf download` command landed different bytes under the same path".
const std::string kOtherRevision(40, 'b');

}  // namespace

TEST_CASE("hf_snapshot: the pinned revision is selected out of a two-revision cache") {
  const fs::path home = ScratchRoot("select");
  ScopedHome guard(home);

  // Plant the decoy FIRST, so a resolver that took "whatever came back first"
  // from an insertion-ordered listing would pick the wrong one. Readdir order is
  // not insertion order, which is the whole point -- neither order may matter.
  PlantRevision(home, kUnsloth27bRepo, kUnsloth27bDecoyRevision);
  const fs::path want =
      PlantRevision(home, kUnsloth27bRepo, parity::kQwen27NvfP4Revision);

  const std::string got = parity::Qwen27NvfP4Snapshot();
  CHECK(got == want.string());
  // Stated separately from the equality above: an assertion that only compared
  // paths would still pass if BOTH constants were changed to the decoy.
  CHECK(got.find(parity::kQwen27NvfP4Revision) != std::string::npos);
  CHECK(got.find(kUnsloth27bDecoyRevision) == std::string::npos);
}

TEST_CASE("hf_snapshot: a cache holding only ANOTHER revision refuses, never substitutes") {
  const fs::path home = ScratchRoot("refuse");
  ScopedHome guard(home);

  PlantRevision(home, kUnsloth27bRepo, kUnsloth27bDecoyRevision);

  // The repo is cached. The pinned revision is not. The only correct answer is
  // "" -- which is what makes the caller emit its skip instead of gating the
  // wrong model. An unpinned `directory_iterator` returns the decoy here.
  CHECK(parity::Qwen27NvfP4Snapshot().empty());
}

TEST_CASE("hf_snapshot: every pinned accessor selects, refuses, and skips alike") {
  struct Arm {
    const char* name;
    const char* repo;
    const char* revision;
    std::string (*resolve)();
  };
  // Every accessor in the header. A new pin that forgets to appear here is
  // caught by test_snapshot_pins.py, which parses the header and this list.
  const Arm arms[] = {
      {"27B unsloth NVFP4", kUnsloth27bRepo, parity::kQwen27NvfP4Revision,
       &parity::Qwen27NvfP4Snapshot},
      {"35B-A3B nvidia NVFP4", "models--nvidia--Qwen3.6-35B-A3B-NVFP4",
       parity::kQwen36A3bNvfP4Revision, &parity::Qwen36A3bNvfP4Snapshot},
      {"27B z-lab DFlash draft", "models--z-lab--Qwen3.6-27B-DFlash",
       parity::kQwen27DFlashDraftRevision, &parity::Qwen27DFlashDraftSnapshot},
  };

  // A decoy revision that is not any real pin, used per-repo.
  const std::string decoy(40, 'a');

  for (const Arm& arm : arms) {
    CAPTURE(arm.name);

    {  // Nothing cached at all -> refuse.
      const fs::path home = ScratchRoot("empty");
      ScopedHome guard(home);
      CHECK(arm.resolve().empty());
    }
    {  // Only a decoy revision of the right repo -> refuse.
      const fs::path home = ScratchRoot("decoy");
      ScopedHome guard(home);
      PlantRevision(home, arm.repo, decoy);
      CHECK(arm.resolve().empty());
    }
    {  // Pinned revision present alongside the decoy -> select the pin.
      const fs::path home = ScratchRoot("both");
      ScopedHome guard(home);
      PlantRevision(home, arm.repo, decoy);
      const fs::path want = PlantRevision(home, arm.repo, arm.revision);
      CHECK(arm.resolve() == want.string());
    }
    {  // Directory present but no config.json -> refuse. An empty or partially
       // downloaded snapshot is not a checkpoint.
      const fs::path home = ScratchRoot("noconfig");
      ScopedHome guard(home);
      fs::create_directories(home / ".cache/huggingface/hub" / arm.repo /
                             "snapshots" / arm.revision);
      CHECK(arm.resolve().empty());
    }
  }
}

TEST_CASE("hf_snapshot: the env override is the ONLY way to gate another checkpoint") {
  const fs::path home = ScratchRoot("override");
  ScopedHome guard(home);

  // A deliberate different-checkpoint run: an explicit directory, named by a
  // human, outside the cache entirely.
  const fs::path elsewhere = home / "deliberate-checkpoint";
  fs::create_directories(elsewhere);
  std::ofstream(elsewhere / "config.json") << "{}\n";

  {
    ScopedEnv over("VT_QWEN27_SNAPSHOT", elsewhere.string());
    CHECK(parity::Qwen27NvfP4Snapshot() == elsewhere.string());
  }
  {
    // An override pointing somewhere without a config.json refuses rather than
    // falling back to the cache -- otherwise a typo'd override would silently
    // gate the pinned model and the run would be misattributed.
    PlantRevision(home, kUnsloth27bRepo, parity::kQwen27NvfP4Revision);
    ScopedEnv over("VT_QWEN27_SNAPSHOT", (home / "nonexistent").string());
    CHECK(parity::Qwen27NvfP4Snapshot().empty());
  }
  {
    // An empty override is not an override; the pin still applies.
    ScopedEnv over("VT_QWEN27_SNAPSHOT", "");
    CHECK(parity::Qwen27NvfP4Snapshot().find(parity::kQwen27NvfP4Revision) !=
          std::string::npos);
  }
}

// --------------------------------------------------------------------------- //
// GATE-SNAPSHOT-CONTENT-PIN (issue #569).
//
// The Nemotron-3.5-Lightning gate model is staged as a `hf download --local-dir`
// tree, so the revision is not in the PATH. The pin used to be the EXISTENCE of
// `.cache/huggingface/trees/<revision>.json`, which records "this revision was
// downloaded into this directory once" and never "these bytes are that
// revision". The two come apart, and huggingface_hub says so itself:
//
//   * `_tree_cache.py` (1.23.0/1.24.0, the versions that wrote the manifest on
//     the gate host) opens with "Because a commit hash is immutable, its tree
//     listing never changes and can be cached forever WITHOUT ANY INVALIDATION
//     LOGIC", and `write_tree_cache` only ever `os.replace`s `<commit>.json`.
//     Nothing in the library removes one. Manifests ACCUMULATE, so after a
//     re-download at a new revision the OLD manifest is still sitting there
//     vouching for the NEW bytes.
//   * The per-file sidecar does not accumulate. `file_download.py` calls
//     `write_download_metadata(..., commit_hash=...)` on EVERY outcome --
//     fresh download, cache copy, and the etag-already-matches early return --
//     so line 1 of `.cache/huggingface/download/<file>.metadata` always names
//     the revision the bytes on disk were fetched as.
//
// So the manifest is the wrong record and the sidecar is the right one. These
// cases pin that, and they are the reason W6 of `.agents/specs/nemotron-h-model.md`
// (#517) may be believed: a token-exact gate is precisely the instrument that
// CANNOT see a substituted checkpoint.
// --------------------------------------------------------------------------- //

TEST_CASE("hf_snapshot: a local_dir whose sidecars name ANOTHER revision refuses") {
  const fs::path root = ScratchRoot("nemotron_substituted");
  ScopedHome guard(root);  // also cleans `root` up on scope exit
  ScopedUnsetEnv no_override("VT_NEMOTRON35_SNAPSHOT");
  ScopedEnv checkpoint_root("CHECKPOINT_ROOT", root.string());

  // The DEMONSTRATED failure, staged exactly: the pinned revision's manifest is
  // present, and every file in the tree was actually downloaded at a different
  // one. An existence check on the manifest resolves this directory.
  const fs::path staged = StagedCheckpoint(root);
  PlantLocalDir(staged, parity::kNemotron35LightningNvfP4Revision,
                kOtherRevision);

  std::string why;
  const std::string got = parity::Nemotron35LightningSnapshot(&why);
  CHECK(got.empty());
  // The refusal has to SAY what was wrong, or the exit-77 banner the callers
  // print degenerates into "not staged" and the operator re-runs it. WHICH file
  // is named is readdir order and deliberately not asserted; that it names the
  // revision actually on disk is the part an operator needs.
  CHECK(why.find(kOtherRevision) != std::string::npos);
  CHECK(why.find("not " +
                 std::string(parity::kNemotron35LightningNvfP4Revision)) !=
        std::string::npos);
}

TEST_CASE("hf_snapshot: a RIGHT config beside WRONG weights refuses") {
  const fs::path root = ScratchRoot("nemotron_weights_only");
  ScopedHome guard(root);
  ScopedUnsetEnv no_override("VT_NEMOTRON35_SNAPSHOT");
  ScopedEnv checkpoint_root("CHECKPOINT_ROOT", root.string());

  // The case that decides config-only versus per-file, staged as the difference
  // between them: `config.json` is genuinely the pinned revision, the SHARD the
  // goldens actually depend on is not. `hf download <repo> <shard> --revision B`
  // into a tree fetched at A produces exactly this, and a resolver that read one
  // sidecar would hand the gate a mixed checkpoint and call it pinned.
  const fs::path staged = StagedCheckpoint(root);
  PlantLocalDir(staged, parity::kNemotron35LightningNvfP4Revision,
                parity::kNemotron35LightningNvfP4Revision);
  PlantSidecar(staged, "model-00001-of-00001.safetensors", kOtherRevision);

  std::string why;
  CHECK(parity::Nemotron35LightningSnapshot(&why).empty());
  CHECK(why.find("model-00001-of-00001.safetensors") != std::string::npos);
  CHECK(why.find(kOtherRevision) != std::string::npos);
}

TEST_CASE("hf_snapshot: a local_dir whose sidecars name the PIN resolves") {
  const fs::path root = ScratchRoot("nemotron_matching");
  ScopedHome guard(root);
  ScopedUnsetEnv no_override("VT_NEMOTRON35_SNAPSHOT");
  ScopedEnv checkpoint_root("CHECKPOINT_ROOT", root.string());

  const fs::path staged = StagedCheckpoint(root);
  PlantLocalDir(staged, parity::kNemotron35LightningNvfP4Revision,
                parity::kNemotron35LightningNvfP4Revision);

  std::string why;
  CHECK(parity::Nemotron35LightningSnapshot(&why) == staged.string());
  CHECK(why.empty());
  // The no-argument spelling every existing caller uses stays identical.
  CHECK(parity::Nemotron35LightningSnapshot() == staged.string());
}

TEST_CASE("hf_snapshot: an ACCUMULATED manifest does not vouch for newer bytes") {
  const fs::path root = ScratchRoot("nemotron_accumulated");
  ScopedHome guard(root);
  ScopedUnsetEnv no_override("VT_NEMOTRON35_SNAPSHOT");
  ScopedEnv checkpoint_root("CHECKPOINT_ROOT", root.string());

  // What `hf download --local-dir` leaves behind after fetching the pin and then
  // re-fetching the same repo at a later revision: BOTH manifests, and sidecars
  // rewritten in place to the later one. Nothing here is hypothetical -- it is
  // what the writer's own "cached forever without any invalidation logic" means.
  const fs::path staged = StagedCheckpoint(root);
  PlantLocalDir(staged, kOtherRevision, kOtherRevision);
  PlantTreeManifest(staged, parity::kNemotron35LightningNvfP4Revision);

  std::string why;
  CHECK(parity::Nemotron35LightningSnapshot(&why).empty());
  CHECK(why.find(kOtherRevision) != std::string::npos);
}

TEST_CASE("hf_snapshot: a local_dir file with NO sidecar refuses") {
  const fs::path root = ScratchRoot("nemotron_unrecorded");
  ScopedHome guard(root);
  ScopedUnsetEnv no_override("VT_NEMOTRON35_SNAPSHOT");
  ScopedEnv checkpoint_root("CHECKPOINT_ROOT", root.string());

  const fs::path staged = StagedCheckpoint(root);
  PlantLocalDir(staged, parity::kNemotron35LightningNvfP4Revision,
                parity::kNemotron35LightningNvfP4Revision);

  SUBCASE("a shard dropped in by hand carries no provenance at all") {
    // The check is symmetric on purpose. If only the files that HAVE a sidecar
    // were checked, substituting a checkpoint would cost one `rm` of the
    // sidecar -- and the file whose bytes the goldens depend on is exactly the
    // one an attacker or an accident replaces.
    std::ofstream(staged / "model-00002-of-00002.safetensors") << "{}\n";
    std::string why;
    CHECK(parity::Nemotron35LightningSnapshot(&why).empty());
    CHECK(why.find("model-00002-of-00002.safetensors") != std::string::npos);
    // The MESSAGE is asserted, not only the refusal. Falling through to the
    // wrong-revision branch also refuses -- "" never equals the pin -- so
    // without this the no-sidecar branch could be deleted outright and every
    // case here would still pass while the banner told the operator the file
    // "was downloaded at revision , not 29f2d174...", which is not what
    // happened and sends them looking for the wrong thing.
    CHECK(why.find("has no .cache/huggingface/download/") != std::string::npos);
  }
  SUBCASE("deleting a sidecar is not a way to pass") {
    fs::remove(staged / ".cache/huggingface/download/config.json.metadata");
    std::string why;
    CHECK(parity::Nemotron35LightningSnapshot(&why).empty());
    CHECK(why.find("config.json") != std::string::npos);
    CHECK(why.find("has no .cache/huggingface/download/") != std::string::npos);
  }
}

TEST_CASE("hf_snapshot: a tree with no staged FILE refuses rather than passing") {
  const fs::path root = ScratchRoot("nemotron_vacuous");
  ScopedHome guard(root);
  ScopedUnsetEnv no_override("VT_NEMOTRON35_SNAPSHOT");
  ScopedEnv checkpoint_root("CHECKPOINT_ROOT", root.string());

  // `config.json` as a DIRECTORY. `fs::exists` says yes, so the arm gets past
  // the staging probe, and the sweep then has zero regular files to check. A
  // sweep with nothing to check must not report success: "every file matches"
  // is vacuously true of no files, and that is the one way a content pin can go
  // quiet without anyone deleting a line of it.
  const fs::path staged = StagedCheckpoint(root);
  fs::create_directories(staged / "config.json");

  std::string why;
  CHECK(parity::Nemotron35LightningSnapshot(&why).empty());
  CHECK(why.find("no files staged") != std::string::npos);
}

TEST_CASE("hf_snapshot: the local_dir arm keeps the behaviors it already had") {
  const fs::path root = ScratchRoot("nemotron_preserved");
  ScopedHome guard(root);

  SUBCASE("no CHECKPOINT_ROOT and no override -> refuse") {
    ScopedUnsetEnv no_override("VT_NEMOTRON35_SNAPSHOT");
    ScopedUnsetEnv no_root("CHECKPOINT_ROOT");
    CHECK(parity::Nemotron35LightningSnapshot().empty());
  }
  SUBCASE("CHECKPOINT_ROOT set but nothing staged under it -> refuse") {
    ScopedUnsetEnv no_override("VT_NEMOTRON35_SNAPSHOT");
    ScopedEnv checkpoint_root("CHECKPOINT_ROOT", root.string());
    CHECK(parity::Nemotron35LightningSnapshot().empty());
  }
  SUBCASE("the override names a directory outright and wins over the root") {
    // Unchanged semantics: naming ONE directory is the deliberate
    // different-checkpoint run, and it is not revision-gated. Naming a ROOT is
    // not, which is why only the root arm carries the content pin.
    const fs::path elsewhere = root / "deliberate-checkpoint";
    fs::create_directories(elsewhere);
    std::ofstream(elsewhere / "config.json") << "{}\n";
    PlantLocalDir(StagedCheckpoint(root),
                  parity::kNemotron35LightningNvfP4Revision,
                  parity::kNemotron35LightningNvfP4Revision);
    ScopedEnv checkpoint_root("CHECKPOINT_ROOT", root.string());
    ScopedEnv over("VT_NEMOTRON35_SNAPSHOT", elsewhere.string());
    CHECK(parity::Nemotron35LightningSnapshot() == elsewhere.string());
  }
  SUBCASE("a set-but-wrong override refuses rather than falling back") {
    PlantLocalDir(StagedCheckpoint(root),
                  parity::kNemotron35LightningNvfP4Revision,
                  parity::kNemotron35LightningNvfP4Revision);
    ScopedEnv checkpoint_root("CHECKPOINT_ROOT", root.string());
    ScopedEnv over("VT_NEMOTRON35_SNAPSHOT", (root / "nonexistent").string());
    CHECK(parity::Nemotron35LightningSnapshot().empty());
  }
}
