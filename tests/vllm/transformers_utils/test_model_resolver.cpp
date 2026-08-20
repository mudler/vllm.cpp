// ENG-HF-MODEL-DOWNLOAD W4 (#1280): the `--model` grammar.
//
// Every case runs against an IN-PROCESS FAKE HUB, a real `httplib::Server` on
// an ephemeral port reached over plain hypertext transfer protocol through
// `HF_ENDPOINT`, following `tests/vllm/transformers_utils/test_hf_hub.cpp`.
//
// THE HUB RECORDS EVERY PATH IT WAS ASKED FOR, and two of the cases below are
// statements about a request that was NOT made: a local path resolves with the
// hub receiving nothing, and a snapshot fetch never asks for the decoy
// `original/model.safetensors`. A downloaded-file COUNT bounds neither: it
// cannot say which file was skipped, and it reads the same whether the decoy
// was skipped or a real shard was.
//
// Reachability is NOT proven here. This file calls the resolver directly, so it
// cannot see a deleted call site in `server_main.cpp`. That is
// `tests/vllm/entrypoints/openai/test_serve_hf_model.cpp`, which enters through
// `vllm_server_main`.
#include <doctest/doctest.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <httplib/httplib.h>
#include <nlohmann/json.hpp>

#include "support/process_id.h"
#include "support/test_env.h"
#include "vllm/gguf_builder.h"
#include "vllm/transformers_utils/model_resolver.h"

namespace fs = std::filesystem;
using vllm::transformers_utils::IsRepoRootFile;
using vllm::transformers_utils::ModelReference;
using vllm::transformers_utils::ModelResolveOptions;
using vllm::transformers_utils::ParsedModelReference;
using vllm::transformers_utils::ParseModelReference;
using vllm::transformers_utils::ResolveModelPath;
using vllm::transformers_utils::SelectWeightFiles;

namespace {

constexpr const char* kCommit = "1111111111111111111111111111111111111111";

class TempDir {
 public:
  TempDir() {
    static std::atomic<int> counter{0};
    path_ = fs::temp_directory_path() /
            ("vllm_model_resolver_test_" +
             std::to_string(vllm_test::ProcessId()) + "_" +
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

// A minimal but VALID GGUF: the structural proof runs the tree's own reader, so
// a fixture of random bytes would be refused for a reason no case is about.
std::string TinyGguf() {
  gguf_test::GgufModelBuilder builder;
  builder.AddKv(gguf_test::StrKv("general.architecture", "llama"));
  builder.AddTensor("token_embd.weight", {4, 2}, /*F32=*/0,
                    std::string(4 * 2 * 4, '\x01'));
  return builder.Build();
}

// A VALID single-tensor safetensors blob. The structural proof runs on every
// `.safetensors` the resolver fetches, so a fixture of arbitrary bytes would be
// refused for a reason none of these cases is about.
std::string TinySafetensors(const std::string& marker) {
  nlohmann::json header;
  header["t"] = {{"dtype", "U8"},
                 {"shape", {marker.size()}},
                 {"data_offsets", {0, marker.size()}}};
  std::string text = header.dump();
  while (text.size() % 8 != 0) text.push_back(' ');
  std::string out(8, '\0');
  for (int i = 0; i < 8; ++i) {
    out[static_cast<size_t>(i)] =
        static_cast<char>((text.size() >> (8 * i)) & 0xff);
  }
  return out + text + marker;
}

// The repository the fake hub serves. `files` maps a repository-relative path
// to its bytes.
class FakeHub {
 public:
  FakeHub() {
    server_.Get("/api/models/(.*)/refs",
                [this](const httplib::Request& req, httplib::Response& res) {
                  Record(req);
                  res.set_content(
                      std::string(
                          R"({"branches":[{"name":"main","targetCommit":")") +
                          kCommit + R"("}],"tags":[]})",
                      "application/json");
                });
    server_.Get("/api/models/(.*)/tree/(.*)",
                [this](const httplib::Request& req, httplib::Response& res) {
                  Record(req);
                  res.set_content(Tree(), "application/json");
                });
    server_.Get("/(.*)/resolve/(.*)",
                [this](const httplib::Request& req, httplib::Response& res) {
                  Record(req);
                  const std::string marker =
                      std::string("/resolve/") + kCommit + "/";
                  const size_t at = req.path.find(marker);
                  if (at == std::string::npos) {
                    res.status = 404;
                    return;
                  }
                  const std::string rel = req.path.substr(at + marker.size());
                  const auto it = files_.find(rel);
                  if (it == files_.end()) {
                    res.status = 404;
                    return;
                  }
                  res.set_header("Accept-Ranges", "bytes");
                  res.set_header("ETag", "\"etag\"");
                  if (req.method == "HEAD") {
                    res.set_header("Content-Length",
                                   std::to_string(it->second.size()));
                    res.status = 200;
                    return;
                  }
                  res.status = 200;
                  res.set_content(it->second, "application/octet-stream");
                });
    port_ = server_.bind_to_any_port("127.0.0.1");
    thread_ = std::thread([this] { server_.listen_after_bind(); });
    server_.wait_until_ready();
  }
  ~FakeHub() {
    server_.stop();
    if (thread_.joinable()) thread_.join();
  }
  FakeHub(const FakeHub&) = delete;
  FakeHub& operator=(const FakeHub&) = delete;

  std::string endpoint() const {
    return "http://127.0.0.1:" + std::to_string(port_) + "/";
  }
  void add(const std::string& path, std::string bytes) {
    files_[path] = std::move(bytes);
  }
  // Give one listed file an object identifier. The hub reports one for a
  // large-file entry and reports none for a small one, and which of the two a
  // file gets decides the name its cache blob takes.
  void add_oid(const std::string& path, std::string oid) {
    oids_[path] = std::move(oid);
  }
  int request_count() const { return requests_.load(); }
  std::vector<std::string> paths() const {
    const std::lock_guard<std::mutex> lock(mu_);
    return paths_;
  }
  bool asked_for(const std::string& needle) const {
    for (const std::string& path : paths()) {
      if (path.find(needle) != std::string::npos) return true;
    }
    return false;
  }

 private:
  std::string Tree() const {
    nlohmann::json out = nlohmann::json::array();
    for (const auto& entry : files_) {
      nlohmann::json item;
      item["type"] = "file";
      item["path"] = entry.first;
      item["size"] = entry.second.size();
      const auto oid = oids_.find(entry.first);
      if (oid != oids_.end()) item["oid"] = oid->second;
      out.push_back(item);
    }
    return out.dump();
  }
  void Record(const httplib::Request& req) {
    requests_.fetch_add(1);
    const std::lock_guard<std::mutex> lock(mu_);
    paths_.push_back(req.path);
  }

  httplib::Server server_;
  std::thread thread_;
  int port_ = 0;
  std::atomic<int> requests_{0};
  mutable std::mutex mu_;
  std::vector<std::string> paths_;
  std::map<std::string, std::string> files_;
  std::map<std::string, std::string> oids_;
};

// A scratch `HF_HOME` and `HF_ENDPOINT`, restored on the way out so one case
// cannot leak its hub into the next.
class HubEnvironment {
 public:
  HubEnvironment(const FakeHub& hub, const fs::path& home) {
    const char* endpoint = std::getenv("HF_ENDPOINT");
    const char* hf_home = std::getenv("HF_HOME");
    const char* offline = std::getenv("HF_HUB_OFFLINE");
    saved_endpoint_ = endpoint == nullptr ? std::string() : endpoint;
    saved_home_ = hf_home == nullptr ? std::string() : hf_home;
    saved_offline_ = offline == nullptr ? std::string() : offline;
    vllm_test::SetEnv("HF_ENDPOINT", hub.endpoint().c_str());
    vllm_test::SetEnv("HF_HOME", home.string().c_str());
    vllm_test::SetEnv("HF_HUB_OFFLINE", "");
    vllm_test::SetEnv("HF_TOKEN", "");
  }
  ~HubEnvironment() {
    vllm_test::SetEnv("HF_ENDPOINT", saved_endpoint_.c_str());
    vllm_test::SetEnv("HF_HOME", saved_home_.c_str());
    vllm_test::SetEnv("HF_HUB_OFFLINE", saved_offline_.c_str());
  }
  HubEnvironment(const HubEnvironment&) = delete;
  HubEnvironment& operator=(const HubEnvironment&) = delete;

 private:
  std::string saved_endpoint_;
  std::string saved_home_;
  std::string saved_offline_;
};

std::string ReadAll(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

std::string MessageOf(const std::function<void()>& body) {
  try {
    body();
  } catch (const std::exception& e) {
    return e.what();
  }
  return std::string();
}

// A repository shaped like a published safetensors checkpoint, DECOY included.
void PopulateSafetensorsRepo(FakeHub& hub) {
  hub.add("config.json", R"({"architectures":["LlamaForCausalLM"]})");
  hub.add("tokenizer.json", R"({"version":"1.0"})");
  hub.add("tokenizer_config.json", R"({"model_max_length":128})");
  hub.add("README.md", "# a model card nobody needs to fetch");
  hub.add("model.safetensors", TinySafetensors("SHARD-ONE-BYTES"));
  hub.add("model.safetensors.index.json",
          R"({"metadata":{"total_size":15},)"
          R"("weight_map":{"a":"model.safetensors","b":"model.safetensors"}})");
  // TWO DECOYS, because two independent guards keep them out and one decoy
  // would leave whichever guard it does not reach ungated.
  //
  // `original/` is a duplicate-format copy a published checkpoint carries. It
  // is kept out by the ROOT-ONLY rule, which is why the directory also holds a
  // `config.json`: phase one fetches every configuration file it sees, so a
  // rule that stopped filtering by depth would ask for this one.
  hub.add("original/model.safetensors",
          TinySafetensors("A-SECOND-COPY-NOBODY-WANTS"));
  hub.add("original/config.json", R"({"architectures":["LlamaForCausalLM"]})");
  // `consolidated.safetensors` sits at the ROOT beside the sharded weights, the
  // way Mistral publishes one, and only INDEX-DRIVEN selection keeps it out: it
  // matches `*.safetensors` and the `weight_map` does not name it.
  hub.add("consolidated.safetensors",
          TinySafetensors("THE-UNSHARDED-COPY-NOBODY-WANTS"));
}

}  // namespace

TEST_CASE("model resolver: the grammar, decided without a socket") {
  TempDir dir;
  fs::create_directories(dir.path() / "local-model");
  { std::ofstream out(dir.path() / "local.gguf", std::ios::binary); out << "GGUF"; }

  CHECK(ParseModelReference((dir.path() / "local-model").string()).kind ==
        ModelReference::kLocalDirectory);
  CHECK(ParseModelReference((dir.path() / "local.gguf").string()).kind ==
        ModelReference::kLocalGgufFile);

  const ParsedModelReference snapshot = ParseModelReference("org/repo");
  CHECK(snapshot.kind == ModelReference::kHubSnapshot);
  CHECK(snapshot.repo_id == "org/repo");

  const ParsedModelReference gguf = ParseModelReference("org/repo:Q4_K_M");
  CHECK(gguf.kind == ModelReference::kHubGgufFile);
  CHECK(gguf.repo_id == "org/repo");
  CHECK(gguf.tag == "Q4_K_M");

  // THE LAST COLON, which is why a Windows path is not read as a tag. Splitting
  // on the first colon would make `C` the repository and `\models\qwen` the
  // quantization.
  CHECK(ParseModelReference("C:\\models\\qwen").kind ==
        ModelReference::kUnrecognized);
  // `org/repo:x` is a valid repository plus a tag; `org/repo:x:Q8_0` splits at
  // the LAST colon, and `org/repo:x` is not a repository identifier, so the
  // whole value falls through rather than silently naming a different tag.
  CHECK(ParseModelReference("org/repo:x:Q8_0").kind ==
        ModelReference::kUnrecognized);

  CHECK(ParseModelReference("/no/such/path").kind ==
        ModelReference::kUnrecognized);
  CHECK(ParseModelReference("").kind == ModelReference::kUnrecognized);
  // Two slashes is not `org/repo`.
  CHECK(ParseModelReference("a/b/c").kind == ModelReference::kUnrecognized);
}

TEST_CASE("model resolver: a LOCAL path resolves and the hub receives NOTHING") {
  FakeHub hub;
  TempDir dir;
  const HubEnvironment env(hub, dir.path() / "home");
  PopulateSafetensorsRepo(hub);

  const fs::path local = dir.path() / "local-model";
  fs::create_directories(local);
  { std::ofstream out(local / "config.json"); out << "{}"; }

  ModelResolveOptions opts;
  CHECK(ResolveModelPath(local.string(), opts) == local.string());

  const fs::path gguf = dir.path() / "local.gguf";
  { std::ofstream out(gguf, std::ios::binary); out << TinyGguf(); }
  CHECK(ResolveModelPath(gguf.string(), opts) == gguf.string());

  // An unrecognized value comes back unchanged so the loader's existing error
  // still fires on what the user typed.
  CHECK(ResolveModelPath("/no/such/model", opts) == "/no/such/model");

  // ZERO requests. A local path can never be shadowed by a network call.
  CHECK(hub.request_count() == 0);
}

TEST_CASE("model resolver: a snapshot fetch NEVER asks for the decoy under original/") {
  FakeHub hub;
  TempDir dir;
  const HubEnvironment env(hub, dir.path() / "home");
  PopulateSafetensorsRepo(hub);

  ModelResolveOptions opts;
  const std::string snapshot = ResolveModelPath("org/repo", opts);

  std::string joined;
  for (const std::string& path : hub.paths()) joined += path + "\n";
  INFO("paths:\n" << joined);

  // THE ASSERTIONS THIS CASE EXISTS FOR, one per guard. A count of downloaded
  // files would read the same whether the skipped file was a decoy or a real
  // shard, and it could not say which guard did the skipping.
  //
  // The root-only rule keeps the duplicate-format directory out.
  CHECK_FALSE(hub.asked_for("original"));
  // Index-driven selection keeps the root-level second copy out. Removing it
  // and globbing `*.safetensors` fetches this file too.
  CHECK_FALSE(hub.asked_for("consolidated"));

  // And the snapshot is complete and usable.
  CHECK(fs::is_regular_file(fs::path(snapshot) / "config.json"));
  CHECK(fs::is_regular_file(fs::path(snapshot) / "tokenizer.json"));
  CHECK(fs::is_regular_file(fs::path(snapshot) / "model.safetensors"));
  CHECK(ReadAll(fs::path(snapshot) / "model.safetensors") ==
        TinySafetensors("SHARD-ONE-BYTES"));
  CHECK(fs::path(snapshot).filename().string() == kCommit);

  // A SECOND run over the warm cache asks for no bytes at all: only the tree
  // listing, because the reference is already recorded.
  const size_t before = hub.paths().size();
  const std::string again = ResolveModelPath("org/repo", opts);
  CHECK(again == snapshot);
  int byte_requests = 0;
  const std::vector<std::string> after = hub.paths();
  for (size_t i = before; i < after.size(); ++i) {
    if (after[i].find("/resolve/") != std::string::npos) byte_requests += 1;
  }
  CHECK(byte_requests == 0);
}

TEST_CASE("model resolver: a repository with no config.json fails BEFORE the weights") {
  FakeHub hub;
  TempDir dir;
  const HubEnvironment env(hub, dir.path() / "home");
  hub.add("README.md", "# a dataset, not a model");
  hub.add("model.safetensors", TinySafetensors(std::string(4096, 'W')));

  ModelResolveOptions opts;
  const std::string message =
      MessageOf([&] { ResolveModelPath("org/repo", opts); });
  INFO("refusal: " << message);
  REQUIRE_FALSE(message.empty());
  CHECK(message.find("config.json") != std::string::npos);
  // THE POINT OF THE TWO PHASES: the weights were never asked for. A one-phase
  // fetch would have paid for them before finding out.
  CHECK_FALSE(hub.asked_for("model.safetensors"));
}

TEST_CASE("model resolver: org/repo:Q4_K_M returns ONE gguf file") {
  FakeHub hub;
  TempDir dir;
  const HubEnvironment env(hub, dir.path() / "home");
  const std::string gguf = TinyGguf();
  hub.add("README.md", "# quantizations");
  hub.add("Model-Q4_K_M.gguf", gguf);
  hub.add("Model-Q8_0.gguf", gguf);

  ModelResolveOptions opts;
  const std::string path = ResolveModelPath("org/repo:Q4_K_M", opts);
  CHECK(fs::path(path).filename().string() == "Model-Q4_K_M.gguf");
  CHECK(fs::is_regular_file(path));
  CHECK(ReadAll(path) == gguf);
  // The other quantization was never fetched. `org/repo:TAG` is ONE file.
  CHECK_FALSE(hub.asked_for("Q8_0"));
  CHECK_FALSE(hub.asked_for("README"));
}

TEST_CASE("model resolver: an unknown tag names the tags the listing HOLDS") {
  FakeHub hub;
  TempDir dir;
  const HubEnvironment env(hub, dir.path() / "home");
  hub.add("Model-Q4_K_M.gguf", TinyGguf());
  hub.add("Model-Q8_0.gguf", TinyGguf());

  ModelResolveOptions opts;
  const std::string message =
      MessageOf([&] { ResolveModelPath("org/repo:IQ2_XXS", opts); });
  INFO("refusal: " << message);
  REQUIRE_FALSE(message.empty());
  CHECK(message.find("IQ2_XXS") != std::string::npos);
  // A refusal that does not say what IS there costs the reader a second run.
  CHECK(message.find("q4_k_m") != std::string::npos);
  CHECK(message.find("q8_0") != std::string::npos);
}

TEST_CASE("model resolver: --download-dir places the cache where it says") {
  FakeHub hub;
  TempDir dir;
  const HubEnvironment env(hub, dir.path() / "home");
  PopulateSafetensorsRepo(hub);

  ModelResolveOptions opts;
  opts.download_dir = dir.path() / "elsewhere";
  const std::string snapshot = ResolveModelPath("org/repo", opts);
  CHECK(snapshot.rfind((dir.path() / "elsewhere").string(), 0) == 0);
  CHECK(fs::is_regular_file(fs::path(snapshot) / "config.json"));
  // And NOT under HF_HOME, which is what the flag exists to override.
  CHECK_FALSE(fs::exists(dir.path() / "home" / "hub" / "models--org--repo"));
}

TEST_CASE("model resolver: --revision names a branch, and there is no @rev syntax") {
  FakeHub hub;
  TempDir dir;
  const HubEnvironment env(hub, dir.path() / "home");
  PopulateSafetensorsRepo(hub);

  ModelResolveOptions opts;
  opts.revision = "main";
  CHECK_FALSE(ResolveModelPath("org/repo", opts).empty());

  // `org/repo@abc` is NOT a form this row invented. vLLM spells the revision as
  // its own flag (`arg_utils.py:839`), so the value falls through unchanged.
  CHECK(ParseModelReference("org/repo@abc").kind ==
        ModelReference::kUnrecognized);

  ModelResolveOptions missing;
  missing.revision = "no-such-branch";
  const std::string message =
      MessageOf([&] { ResolveModelPath("org/repo", missing); });
  INFO("refusal: " << message);
  CHECK(message.find("no-such-branch") != std::string::npos);
}

TEST_CASE("model resolver: what an https endpoint does, on this build's TLS state") {
  // BOTH ARMS ASSERT, for the reason recorded on the matching case in
  // `test_downloader.cpp`: an `#ifndef` around the whole body turns this case
  // into `assertions: 0` the moment W5 defines the macro, and a zero-assertion
  // case is a skip wearing a pass. The endpoint is LOOPBACK on a port nothing
  // listens on so that a TLS build fails to connect instead of reaching the
  // network.
  TempDir dir;
  vllm_test::SetEnv("HF_ENDPOINT", "https://127.0.0.1:1/");
  vllm_test::SetEnv("HF_HOME", (dir.path() / "home").string().c_str());
  vllm_test::SetEnv("HF_HUB_OFFLINE", "");
  ModelResolveOptions opts;
  const std::string message =
      MessageOf([&] { ResolveModelPath("org/repo", opts); });
  vllm_test::SetEnv("HF_ENDPOINT", "");
  vllm_test::SetEnv("HF_HOME", "");
  INFO("refusal: " << message);
  REQUIRE_FALSE(message.empty());
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
  // A build that speaks https reports the transport, not a rebuild instruction.
  CHECK(message.find("cannot speak HTTPS") == std::string::npos);
  CHECK(message.find("VLLM_CPP_OPENSSL") == std::string::npos);
  CHECK(message.find("VLLM_CPP_BUILD_BORINGSSL") == std::string::npos);
#else
  // Not a connection error that reads like a network fault.
  CHECK(message.find("VLLM_CPP_HF_DOWNLOAD") != std::string::npos);
  CHECK(message.find("VLLM_CPP_OPENSSL") != std::string::npos);
  CHECK(message.find("VLLM_CPP_BUILD_BORINGSSL") != std::string::npos);
#endif
}

TEST_CASE("model resolver: SelectWeightFiles is index driven, then format ordered") {
  const std::vector<std::string> listed = {
      "config.json", "model-00001-of-00002.safetensors",
      "model-00002-of-00002.safetensors", "pytorch_model.bin",
      "original/model.safetensors"};

  // INDEX WINS. The `weight_map` names exactly what to fetch, so the decoy and
  // the second format never enter the answer.
  const std::vector<std::string> map = {"model-00001-of-00002.safetensors",
                                        "model-00002-of-00002.safetensors"};
  CHECK(SelectWeightFiles(listed, map) == map);

  // With NO index, the FIRST matching pattern wins: safetensors before bin.
  const std::vector<std::string> globbed = SelectWeightFiles(listed, {});
  REQUIRE(globbed.size() == 2);
  CHECK(globbed[0] == "model-00001-of-00002.safetensors");
  CHECK(globbed[1] == "model-00002-of-00002.safetensors");

  // A repository with ONLY `.bin` gets `.bin`.
  const std::vector<std::string> bin_only =
      SelectWeightFiles({"config.json", "pytorch_model.bin"}, {});
  REQUIRE(bin_only.size() == 1);
  CHECK(bin_only[0] == "pytorch_model.bin");

  // Nothing at the root means nothing to fetch, whatever a subdirectory holds.
  CHECK(SelectWeightFiles({"original/model.safetensors"}, {}).empty());

  CHECK(IsRepoRootFile("config.json"));
  CHECK_FALSE(IsRepoRootFile("original/model.safetensors"));
  CHECK_FALSE(IsRepoRootFile(""));
}

// F6 of the fifth fresh review: the object-identifier PREFERENCE in
// `BlobNameFor` was unpinned. Dropping it left every case green, because both
// names are valid and nothing observable changed for a caller. What it does
// change is the cache file name, which is the thing `docs/USAGE.md` promises
// `--verbose` reports, so it is pinned here rather than recorded as owed.
TEST_CASE("model resolver: a blob is named by the object identifier, else by the commit and the path") {
  FakeHub hub;
  TempDir dir;
  const fs::path home = dir.path() / "home";
  HubEnvironment env(hub, home);
  // The identifier is only TRUSTED under a token, because an anonymous listing
  // for a gated repository fabricates one. So the case that wants to see it
  // used has to set one.
  vllm_test::SetEnv("HF_TOKEN", "hf_a_test_token");

  PopulateSafetensorsRepo(hub);
  // Forty hexadecimal characters, not one repeated, so neither integrity rule
  // fires on it.
  const std::string kOid = "0123456789abcdef0123456789abcdef01234567";
  hub.add_oid("model.safetensors", kOid);

  const ModelResolveOptions opts;
  const std::string resolved = ResolveModelPath("org/repo", opts);
  vllm_test::SetEnv("HF_TOKEN", "");
  REQUIRE_FALSE(resolved.empty());

  const fs::path blobs = home / "hub" / "models--org--repo" / "blobs";
  // The listing carried an identifier for the shard, so the identifier IS the
  // blob name: it is a content hash, so two repositories holding one byte
  // sequence share one blob.
  CHECK(fs::is_regular_file(blobs / kOid));
  // It carried none for `config.json`, so the fallback is the commit and the
  // flattened path, which names exactly one byte sequence for the same reason a
  // content hash would.
  CHECK(fs::is_regular_file(blobs / (std::string(kCommit) + "--config.json")));
  // And the identifier was not ALSO used as a second name for the same bytes.
  CHECK_FALSE(fs::exists(blobs /
                         (std::string(kCommit) + "--model.safetensors")));
}
