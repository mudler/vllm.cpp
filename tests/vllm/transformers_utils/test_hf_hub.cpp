// ENG-HF-MODEL-DOWNLOAD W2 (#1280): the HuggingFace hub protocol.
//
// Every case runs against an IN-PROCESS FAKE HUB: a real `httplib::Server` on
// an ephemeral port, reached through `HF_ENDPOINT` over plain hypertext
// transfer protocol. The fixture follows
// `tests/vllm/entrypoints/openai/test_api_server.cpp`, which already starts a
// server inside a test. There is no TLS here on purpose; TLS has its own
// instruments in W5, and a hermetic test that speaks plain HTTP proves protocol
// and refusal, not transport security.
//
// The fake hub COUNTS its requests. "Zero requests" is the only honest way to
// state a cache hit, and a mock that merely records the last call cannot say it.
#include <doctest/doctest.h>

#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <httplib/httplib.h>
#include <nlohmann/json.hpp>

#include "support/process_id.h"
#include "support/test_env.h"
#include "vllm/transformers_utils/hf_cache.h"
#include "vllm/transformers_utils/hf_hub.h"

namespace fs = std::filesystem;
using vllm::transformers_utils::HfHubOptions;
using vllm::transformers_utils::HfHubOptionsFromEnv;
using vllm::transformers_utils::HfFile;
using vllm::transformers_utils::HfReadRef;
using vllm::transformers_utils::HfRepoPath;
using vllm::transformers_utils::HubFileUrl;
using vllm::transformers_utils::HubListRepoFiles;
using vllm::transformers_utils::HubResolveCommitCached;
using vllm::transformers_utils::HubResolveRefToCommit;
using vllm::transformers_utils::IsValidHfRepoId;

namespace {

constexpr const char* kCommit = "1111111111111111111111111111111111111111";
constexpr const char* kToken = "hf_testtokentesttokentesttokentesttoken";

// A well-formed 64 character identifier that is NOT one repeated character.
// A fixture meaning "a real content hash" cannot be spelled `std::string(64,
// '2')` any more, because that is the shape the degeneracy rule refuses, and a
// fixture written that way would be refused for a reason the case is not about.
std::string Sha256Like(const std::string& seed) {
  std::string oid;
  while (oid.size() < 64) oid += "0123456789abcdef";
  oid.resize(64);
  for (size_t i = 0; i < seed.size() && i < oid.size(); ++i) oid[i] = seed[i];
  return oid;
}

class TempDir {
 public:
  TempDir() {
    static std::atomic<int> counter{0};
    path_ = fs::temp_directory_path() /
            ("vllm_hf_hub_test_" + std::to_string(vllm_test::ProcessId()) + "_" +
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

// A real HTTP server standing in for huggingface.co. It answers the three
// protocol calls, counts every request it received, and records the
// Authorization header it observed.
class FakeHub {
 public:
  FakeHub() {
    server_.Get("/api/models/(.*)/refs", [this](const httplib::Request& req,
                                                httplib::Response& res) {
      Record(req);
      if (!redirect_.empty()) {
        res.status = 302;
        res.set_header("Location", redirect_);
        return;
      }
      if (gated_ && AuthHeader(req).empty()) {
        res.status = 401;
        res.set_content(R"({"error":"Invalid credentials in Authorization header"})",
                        "application/json");
        return;
      }
      res.set_content(refs_body_, "application/json");
    });
    server_.Get("/api/models/(.*)/tree/(.*)", [this](const httplib::Request& req,
                                                     httplib::Response& res) {
      Record(req);
      if (gated_ && AuthHeader(req).empty()) {
        res.status = 401;
        res.set_content(R"({"error":"Invalid credentials in Authorization header"})",
                        "application/json");
        return;
      }
      res.set_content(AuthHeader(req).empty() ? tree_anonymous_ : tree_authenticated_,
                      "application/json");
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

  int request_count() const { return requests_.load(); }

  std::vector<std::string> paths() const {
    std::lock_guard<std::mutex> lock(mu_);
    return paths_;
  }

  std::string last_authorization() const {
    std::lock_guard<std::mutex> lock(mu_);
    return last_authorization_;
  }

  void set_gated(bool gated) { gated_ = gated; }
  // Answer the refs call with a 302 to `url` instead of a body.
  void set_redirect(std::string url) { redirect_ = std::move(url); }
  void set_refs_body(std::string body) { refs_body_ = std::move(body); }
  void set_tree_anonymous(std::string body) { tree_anonymous_ = std::move(body); }
  void set_tree_authenticated(std::string body) {
    tree_authenticated_ = std::move(body);
  }
  // Both arms at once, for a listing whose shape does not depend on the caller.
  void set_tree(const std::string& body) {
    tree_anonymous_ = body;
    tree_authenticated_ = body;
  }

 private:
  static std::string AuthHeader(const httplib::Request& req) {
    return req.has_header("Authorization") ? req.get_header_value("Authorization")
                                           : std::string();
  }

  void Record(const httplib::Request& req) {
    requests_.fetch_add(1);
    std::lock_guard<std::mutex> lock(mu_);
    paths_.push_back(req.target);
    last_authorization_ = AuthHeader(req);
  }

  httplib::Server server_;
  std::thread thread_;
  int port_ = 0;
  std::atomic<int> requests_{0};
  mutable std::mutex mu_;
  std::vector<std::string> paths_;
  std::string last_authorization_;
  bool gated_ = false;
  std::string redirect_;
  std::string refs_body_ =
      std::string(R"({"branches":[{"name":"main","ref":"refs/heads/main",)") +
      R"("targetCommit":")" + kCommit + R"("}],"tags":[]})";
  std::string tree_anonymous_ =
      R"([{"type":"file","path":"config.json","size":12},)"
      R"({"type":"file","path":"model.safetensors","size":4096,)"
      R"("lfs":{"oid":"a1234567890abcdef0123456789abcdef0123456789abcdef0123456789abcde"}}])";
  std::string tree_authenticated_ = tree_anonymous_;
};

// A SECOND server, on its own port and therefore a different authority. It
// answers every path with a usable refs body and records what it was sent, so a
// case can ask the one question that matters about a redirect: did the bearer
// token leave for another host.
class TokenTrap {
 public:
  TokenTrap() {
    server_.Get(".*", [this](const httplib::Request& req,
                             httplib::Response& res) {
      {
        const std::lock_guard<std::mutex> lock(mu_);
        requests_ += 1;
        authorization_ = req.has_header("Authorization")
                             ? req.get_header_value("Authorization")
                             : std::string();
      }
      res.set_content(
          std::string(R"({"branches":[{"name":"main","targetCommit":")") +
              kCommit + R"("}]})",
          "application/json");
    });
    port_ = server_.bind_to_any_port("127.0.0.1");
    thread_ = std::thread([this] { server_.listen_after_bind(); });
    server_.wait_until_ready();
  }
  ~TokenTrap() {
    server_.stop();
    if (thread_.joinable()) thread_.join();
  }
  TokenTrap(const TokenTrap&) = delete;
  TokenTrap& operator=(const TokenTrap&) = delete;

  std::string url() const {
    return "http://127.0.0.1:" + std::to_string(port_) + "/redirected";
  }
  int request_count() const {
    const std::lock_guard<std::mutex> lock(mu_);
    return requests_;
  }
  std::string authorization() const {
    const std::lock_guard<std::mutex> lock(mu_);
    return authorization_;
  }

 private:
  httplib::Server server_;
  std::thread thread_;
  int port_ = 0;
  mutable std::mutex mu_;
  int requests_ = 0;
  std::string authorization_;
};

// Options aimed at a fake hub and a scratch cache, with nothing read from the
// environment, so a case states exactly the configuration it exercises.
HfHubOptions OptionsFor(const FakeHub& hub, const fs::path& hub_dir) {
  HfHubOptions opts;
  opts.endpoint = hub.endpoint();
  opts.hub_dir = hub_dir;
  opts.connect_timeout_seconds = 5;
  opts.read_timeout_seconds = 5;
  return opts;
}

// A tree in which every large-file-storage entry carries the SAME identifier.
// This is the measured shape, not an invented one: on 17 August 2026 the tree
// API answered an unauthenticated caller on the gated repository
// `Lightricks/LTX-2.5` with an `lfs.oid` of one character repeated 64 times,
// identical for all 14 large-file-storage files.
//
// The two shards carry DIFFERENT sizes, which is also what a real sharded
// repository looks like. Both integrity rules therefore have something to fire
// on, and the cases below assert each rule on a fixture that isolates it, so
// neither rule can hide the other's absence.
std::string FabricatedOidTree() {
  const std::string oid(64, 'a');
  return std::string(R"([{"type":"file","path":"model-00001-of-00002.safetensors",)") +
         R"("size":4096,"lfs":{"oid":")" + oid + R"("}},)" +
         R"({"type":"file","path":"model-00002-of-00002.safetensors",)" +
         R"("size":2048,"lfs":{"oid":")" + oid + R"("}}])";
}

// The same repeated-identifier shape with a WELL-FORMED identifier, so only the
// size disagreement is left to catch it.
std::string SharedOidTree(const std::string& oid, uint64_t first_size,
                          uint64_t second_size) {
  return std::string(R"([{"type":"file","path":"model-00001-of-00002.safetensors",)") +
         R"("size":)" + std::to_string(first_size) + R"(,"lfs":{"oid":")" + oid +
         R"("}},)" +
         R"({"type":"file","path":"model-00002-of-00002.safetensors",)" +
         R"("size":)" + std::to_string(second_size) + R"(,"lfs":{"oid":")" + oid +
         R"("}}])";
}

}  // namespace

TEST_CASE("IsValidHfRepoId accepts a repository id and refuses everything else") {
  CHECK(IsValidHfRepoId("org/repo"));
  CHECK(IsValidHfRepoId("z-lab/Qwen3.6-27B-DFlash"));
  CHECK(IsValidHfRepoId("Lightricks/LTX-2.5"));
  CHECK_FALSE(IsValidHfRepoId(""));
  CHECK_FALSE(IsValidHfRepoId("norepo"));
  CHECK_FALSE(IsValidHfRepoId("a/b/c"));
  CHECK_FALSE(IsValidHfRepoId("/repo"));
  CHECK_FALSE(IsValidHfRepoId("org/"));
  CHECK_FALSE(IsValidHfRepoId("org/../etc"));
  CHECK_FALSE(IsValidHfRepoId("org/re po"));
}

TEST_CASE("HubFileUrl is the byte address form") {
  CHECK(HubFileUrl("https://huggingface.co/", "org/repo", kCommit,
                   "model.safetensors") ==
        std::string("https://huggingface.co/org/repo/resolve/") + kCommit +
            "/model.safetensors");
}

TEST_CASE("HfHubOptionsFromEnv reads the documented environment") {
  TempDir tmp;
  const char* kNames[] = {"HF_ENDPOINT", "HF_TOKEN", "HF_TOKEN_PATH",
                          "HF_HUB_OFFLINE", "HF_HUB_CACHE"};
  std::vector<std::pair<std::string, std::string>> saved;
  for (const char* name : kNames) {
    const char* v = std::getenv(name);
    saved.emplace_back(name, v == nullptr ? std::string() : std::string(v));
    vllm_test::UnsetEnv(name);
  }

  {
    const HfHubOptions defaults = HfHubOptionsFromEnv();
    CHECK(defaults.endpoint == "https://huggingface.co/");
    CHECK(defaults.token.empty());
    CHECK_FALSE(defaults.offline);
  }

  // A host name with no trailing slash still composes into a request path.
  vllm_test::SetEnv("HF_ENDPOINT", "http://mirror.internal");
  vllm_test::SetEnv("HF_TOKEN", kToken);
  vllm_test::SetEnv("HF_HUB_OFFLINE", "1");
  vllm_test::SetEnv("HF_HUB_CACHE", tmp.path().string());
  {
    const HfHubOptions opts = HfHubOptionsFromEnv();
    CHECK(opts.endpoint == "http://mirror.internal/");
    CHECK(opts.token == kToken);
    CHECK(opts.offline);
    CHECK(opts.hub_dir == tmp.path());
  }

  // HF_TOKEN_PATH is the fallback, and its trailing newline is not part of the
  // token: a token with a newline in it produces a malformed request header.
  vllm_test::UnsetEnv("HF_TOKEN");
  const fs::path token_file = tmp.path() / "token";
  {
    // The carriage return, the space and the tab matter: `std::getline` already
    // drops the newline on its own, so a file ending in "\n" alone does not
    // reach the trim at all. The sibling case at
    // tests/vllm/transformers_utils/test_hf_cache.cpp writes the same shape for
    // the same reason.
    std::ofstream out(token_file, std::ios::binary);
    out << kToken << " \t\r\n";
  }
  vllm_test::SetEnv("HF_TOKEN_PATH", token_file.string());
  CHECK(HfHubOptionsFromEnv().token == kToken);

  for (const auto& [name, value] : saved) vllm_test::SetEnv(name.c_str(), value);
}

TEST_CASE("an empty hub variable reads as unset") {
  // Emptying a variable is how a container clears an inherited setting.
  // `HF_ENDPOINT=` must not become the endpoint "" and `HF_TOKEN=` must not
  // become an `Authorization: Bearer ` header with nothing after it.
  // `vllm_test::SetEnv` DELETES on an empty value by design
  // (tests/support/test_env.h), so this case calls setenv itself, and it is
  // POSIX-only because a defined-but-empty variable cannot exist on Windows.
  const char* kNames[] = {"HF_ENDPOINT", "HF_TOKEN", "HF_TOKEN_PATH",
                          "HF_HUB_OFFLINE"};
  std::vector<std::pair<std::string, std::string>> saved;
  for (const char* name : kNames) {
    const char* v = std::getenv(name);
    saved.emplace_back(name, v == nullptr ? std::string() : std::string(v));
    vllm_test::UnsetEnv(name);
  }

#if !defined(_WIN32)
  REQUIRE(::setenv("HF_ENDPOINT", "", /*overwrite=*/1) == 0);
  REQUIRE(::setenv("HF_TOKEN", "", /*overwrite=*/1) == 0);
  REQUIRE(::setenv("HF_HUB_OFFLINE", "", /*overwrite=*/1) == 0);
  REQUIRE(std::getenv("HF_ENDPOINT") != nullptr);
  const HfHubOptions opts = HfHubOptionsFromEnv();
  CHECK(opts.endpoint == "https://huggingface.co/");
  CHECK(opts.token.empty());
  CHECK_FALSE(opts.offline);
#endif

  // The same variables with values still land, so the case above measures the
  // empty check and not a variable nothing reads.
  vllm_test::SetEnv("HF_ENDPOINT", "http://mirror.internal/");
  vllm_test::SetEnv("HF_TOKEN", kToken);
  const HfHubOptions set = HfHubOptionsFromEnv();
  CHECK(set.endpoint == "http://mirror.internal/");
  CHECK(set.token == kToken);

  for (const auto& [name, value] : saved) vllm_test::SetEnv(name.c_str(), value);
}

TEST_CASE("a bracketed IPv6 authority is split into a host and a port") {
  // llama.cpp `common/http.h:75-83 @ b10451` splits it, and W1's anchor
  // correction is the row that found that line. The colon inside an IPv6
  // literal is not the port separator, so the bracket arm has to run before the
  // colon search or `std::stoi` is handed ":1" and throws a std::invalid_argument
  // from inside the parser.
  TempDir tmp;
  HfHubOptions opts;
  opts.hub_dir = tmp.path();
  opts.connect_timeout_seconds = 2;
  opts.read_timeout_seconds = 2;

  // Port 1 on the IPv6 loopback: nothing listens there, and a host with no IPv6
  // at all fails to resolve it, so both hosts reach the same transport refusal.
  // What neither host may produce is a parse failure.
  opts.endpoint = "http://[::1]:1/";
  std::string message;
  try {
    HubResolveRefToCommit("org/repo", "", opts);
    FAIL("a dead port must refuse");
  } catch (const std::runtime_error& e) {
    message = e.what();
  }
  INFO("message: ", message);
  CHECK(message.find("cannot reach") != std::string::npos);

  // An unterminated literal is named for what it is, rather than reaching the
  // network with a host that is missing its closing bracket.
  opts.endpoint = "http://[::1/";
  std::string unterminated;
  try {
    HubResolveRefToCommit("org/repo", "", opts);
    FAIL("an unterminated IPv6 host must refuse");
  } catch (const std::runtime_error& e) {
    unterminated = e.what();
  }
  CHECK(unterminated.find("unterminated IPv6 host") != std::string::npos);
}

TEST_CASE("the reference resolves to a commit over the fake hub") {
  FakeHub hub;
  TempDir tmp;
  const HfHubOptions opts = OptionsFor(hub, tmp.path());

  CHECK(HubResolveRefToCommit("org/repo", "", opts) == kCommit);
  REQUIRE(hub.request_count() == 1);
  CHECK(hub.paths().at(0) == "/api/models/org/repo/refs");
}

TEST_CASE("a revision that is already a commit costs no request") {
  FakeHub hub;
  TempDir tmp;
  const HfHubOptions opts = OptionsFor(hub, tmp.path());
  CHECK(HubResolveRefToCommit("org/repo", kCommit, opts) == kCommit);
  CHECK(hub.request_count() == 0);
}

TEST_CASE("main wins over another branch, whatever order the listing has") {
  FakeHub hub;
  TempDir tmp;
  const std::string other(40, '3');
  hub.set_refs_body(std::string(R"({"branches":[)") +
                    R"({"name":"dev","ref":"refs/heads/dev","targetCommit":")" +
                    other + R"("},)" +
                    R"({"name":"main","ref":"refs/heads/main","targetCommit":")" +
                    kCommit + R"("}]})");
  CHECK(HubResolveRefToCommit("org/repo", "", OptionsFor(hub, tmp.path())) ==
        kCommit);
}

TEST_CASE("a named revision selects that branch") {
  FakeHub hub;
  TempDir tmp;
  const std::string other(40, '3');
  hub.set_refs_body(std::string(R"({"branches":[)") +
                    R"({"name":"dev","ref":"refs/heads/dev","targetCommit":")" +
                    other + R"("},)" +
                    R"({"name":"main","ref":"refs/heads/main","targetCommit":")" +
                    kCommit + R"("}]})");
  CHECK(HubResolveRefToCommit("org/repo", "dev", OptionsFor(hub, tmp.path())) ==
        other);
  CHECK_THROWS_AS(HubResolveRefToCommit("org/repo", "absent", OptionsFor(hub, tmp.path())),
                  std::runtime_error);
}

TEST_CASE("a gated repository refuses with the repository and HF_TOKEN named") {
  FakeHub hub;
  TempDir tmp;
  hub.set_gated(true);
  HfHubOptions opts = OptionsFor(hub, tmp.path());

  std::string message;
  try {
    HubResolveRefToCommit("org/gated", "", opts);
    FAIL("an unauthenticated call on a gated repository must refuse");
  } catch (const std::runtime_error& e) {
    message = e.what();
  }
  CHECK(message.find("org/gated") != std::string::npos);
  CHECK(message.find("HF_TOKEN") != std::string::npos);
  CHECK(message.find("401") != std::string::npos);
}

TEST_CASE("the token is sent and the hub observes it") {
  FakeHub hub;
  TempDir tmp;
  hub.set_gated(true);
  HfHubOptions opts = OptionsFor(hub, tmp.path());
  opts.token = kToken;

  CHECK(HubResolveRefToCommit("org/gated", "", opts) == kCommit);
  CHECK(hub.last_authorization() == std::string("Bearer ") + kToken);
}

TEST_CASE("no token means no Authorization header at all") {
  FakeHub hub;
  TempDir tmp;
  CHECK(HubResolveRefToCommit("org/repo", "", OptionsFor(hub, tmp.path())) ==
        kCommit);
  CHECK(hub.last_authorization().empty());
}

TEST_CASE("the second run is a cache hit and issues zero requests") {
  FakeHub hub;
  TempDir tmp;
  const HfHubOptions opts = OptionsFor(hub, tmp.path());

  CHECK(HubResolveCommitCached("org/repo", "", opts) == kCommit);
  const int after_first = hub.request_count();
  REQUIRE(after_first == 1);
  // The commit was recorded, so a moving `main` cannot change what a second run
  // loads.
  CHECK(HfReadRef(HfRepoPath(tmp.path(), "org/repo"), "main") == kCommit);

  CHECK(HubResolveCommitCached("org/repo", "", opts) == kCommit);
  CHECK(hub.request_count() == after_first);
}

TEST_CASE("offline with a warm cache succeeds and opens no socket") {
  FakeHub hub;
  TempDir tmp;
  HfHubOptions opts = OptionsFor(hub, tmp.path());
  REQUIRE(HubResolveCommitCached("org/repo", "", opts) == kCommit);
  const int warm = hub.request_count();

  opts.offline = true;
  CHECK(HubResolveCommitCached("org/repo", "", opts) == kCommit);
  CHECK(hub.request_count() == warm);
}

TEST_CASE("offline with a cold cache refuses and names the directory it searched") {
  FakeHub hub;
  TempDir tmp;
  HfHubOptions opts = OptionsFor(hub, tmp.path());
  opts.offline = true;

  std::string message;
  try {
    HubResolveCommitCached("org/repo", "", opts);
    FAIL("offline with a cold cache must refuse");
  } catch (const std::runtime_error& e) {
    message = e.what();
  }
  CHECK(hub.request_count() == 0);
  CHECK(message.find(HfRepoPath(tmp.path(), "org/repo").string()) !=
        std::string::npos);
  CHECK(message.find("HF_HUB_OFFLINE") != std::string::npos);
}

TEST_CASE("the recursive listing carries paths, sizes and the byte address") {
  FakeHub hub;
  TempDir tmp;
  const std::vector<HfFile> files =
      HubListRepoFiles("org/repo", kCommit, OptionsFor(hub, tmp.path()));

  REQUIRE(files.size() == 2);
  CHECK(files.at(0).path == "config.json");
  CHECK(files.at(0).size == 12);
  CHECK(files.at(1).path == "model.safetensors");
  CHECK(files.at(1).size == 4096);
  CHECK(files.at(1).url == hub.endpoint() + "org/repo/resolve/" + kCommit +
                               "/model.safetensors");
  const std::vector<std::string> paths = hub.paths();
  REQUIRE(paths.size() == 1);
  CHECK(paths.at(0) ==
        std::string("/api/models/org/repo/tree/") + kCommit + "?recursive=true");
}

TEST_CASE("an unauthenticated listing carries NO object identifier") {
  FakeHub hub;
  TempDir tmp;
  const std::vector<HfFile> files =
      HubListRepoFiles("org/repo", kCommit, OptionsFor(hub, tmp.path()));
  REQUIRE(files.size() == 2);
  // The hub served an lfs.oid. Without a token it is not evidence about the
  // repository, so it is dropped rather than used as a blob name.
  for (const HfFile& file : files) CHECK(file.oid.empty());
}

TEST_CASE("an authenticated listing keeps distinct object identifiers") {
  FakeHub hub;
  TempDir tmp;
  HfHubOptions opts = OptionsFor(hub, tmp.path());
  opts.token = kToken;
  hub.set_tree_authenticated(
      std::string(R"([{"type":"file","path":"a.safetensors","size":1,)") +
      R"("lfs":{"oid":")" + Sha256Like("a2") + R"("}},)" +
      R"({"type":"file","path":"b.safetensors","size":2,)" +
      R"("lfs":{"oid":")" + Sha256Like("b3") + R"("}}])");

  const std::vector<HfFile> files = HubListRepoFiles("org/repo", kCommit, opts);
  REQUIRE(files.size() == 2);
  CHECK(files.at(0).oid == Sha256Like("a2"));
  CHECK(files.at(1).oid == Sha256Like("b3"));
}

TEST_CASE("the measured degenerate identifier is refused by the degeneracy rule") {
  // The regression test for the 17 August 2026 event. llama.cpp's is_valid_oid
  // at common/hf-cache.cpp:161 accepts any 40 or 64 character hexadecimal
  // string, so a verbatim port of that function turns this case red.
  //
  // The refusal asserted here is the DEGENERACY rule, named in the message. The
  // fixture also disagrees on size, so the other rule would catch it too, and
  // asserting the rule by name is what stops one rule from masking the other's
  // deletion.
  FakeHub hub;
  TempDir tmp;
  HfHubOptions opts = OptionsFor(hub, tmp.path());
  opts.token = kToken;
  hub.set_tree(FabricatedOidTree());

  std::string message;
  try {
    HubListRepoFiles("org/repo", kCommit, opts);
    FAIL("a listing carrying a degenerate object identifier must be refused");
  } catch (const std::runtime_error& e) {
    message = e.what();
  }
  INFO("message: ", message);
  CHECK(message.find(std::string(64, 'a')) != std::string::npos);
  CHECK(message.find("characters are all 'a'") != std::string::npos);
  CHECK(message.find("model-00001-of-00002.safetensors") != std::string::npos);
  // It is refused on the FIRST such file, before a duplicate can even be seen.
  CHECK(message.find("bytes") == std::string::npos);
}

TEST_CASE("a degenerate identifier is refused whether or not a token was sent") {
  // The rule that must never depend on the token. An identifier that is one
  // character repeated is never real, so an anonymous caller has exactly as
  // much reason to refuse it. The old rule sat behind the token and produced
  // the inversion this repair removes: a repository that loaded anonymously
  // started failing the moment a token was set.
  FakeHub hub;
  TempDir tmp;
  hub.set_tree(FabricatedOidTree());

  std::string message;
  try {
    HubListRepoFiles("org/repo", kCommit, OptionsFor(hub, tmp.path()));
    FAIL("a degenerate identifier must be refused with no token too");
  } catch (const std::runtime_error& e) {
    message = e.what();
  }
  INFO("message: ", message);
  CHECK(message.find("characters are all 'a'") != std::string::npos);
}

TEST_CASE("one identifier on two files of different sizes is refused") {
  // The size rule, on a WELL-FORMED identifier, so the degeneracy rule has
  // nothing to fire on and this case measures one rule alone. A content hash
  // that named two different sizes would be a broken instrument, whatever the
  // repository holds.
  FakeHub hub;
  TempDir tmp;
  HfHubOptions opts = OptionsFor(hub, tmp.path());
  opts.token = kToken;
  const std::string oid =
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
  hub.set_tree(SharedOidTree(oid, 4096, 2048));

  std::string message;
  try {
    HubListRepoFiles("org/repo", kCommit, opts);
    FAIL("one identifier naming two sizes must be refused");
  } catch (const std::runtime_error& e) {
    message = e.what();
  }
  INFO("message: ", message);
  CHECK(message.find(oid) != std::string::npos);
  CHECK(message.find("model-00001-of-00002.safetensors") != std::string::npos);
  CHECK(message.find("model-00002-of-00002.safetensors") != std::string::npos);
  CHECK(message.find("4096") != std::string::npos);
  CHECK(message.find("2048") != std::string::npos);
  CHECK(message.find("two different sizes") != std::string::npos);
}

TEST_CASE("one identifier spelled in two cases on two sizes is still refused") {
  // THE BYPASS. `IsHexString` accepts 'A' through 'F' as well as 'a' through
  // 'f', so a hub that spells one identifier `ab234567...` on the first entry
  // and `AB234567...` on the second used to land two keys in the owner map and
  // the size rule compared nothing. The measured pair is `a.bin` at 4096 bytes
  // and `b.bin` at 2048 bytes, which the same rule refuses the moment both
  // spellings agree on case.
  //
  // `HF_ENDPOINT` names the host, so the listing is input the hub does not have
  // to answer truthfully. A rule an attacker turns off by changing one letter's
  // case is not a rule.
  FakeHub hub;
  TempDir tmp;
  HfHubOptions opts = OptionsFor(hub, tmp.path());
  opts.token = kToken;
  const std::string lower =
      "ab234567890abcdef0123456789abcdef0123456789abcdef0123456789abcde";
  std::string upper = lower;
  for (char& c : upper) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  REQUIRE(lower != upper);
  hub.set_tree(std::string(R"([{"type":"file","path":"a.bin","size":4096,)") +
               R"("lfs":{"oid":")" + lower + R"("}},)" +
               R"({"type":"file","path":"b.bin","size":2048,)" +
               R"("lfs":{"oid":")" + upper + R"("}}])");

  std::string message;
  try {
    HubListRepoFiles("org/repo", kCommit, opts);
    FAIL("one identifier spelled in two cases on two sizes must be refused");
  } catch (const std::runtime_error& e) {
    message = e.what();
  }
  INFO("message: ", message);
  CHECK(message.find("a.bin") != std::string::npos);
  CHECK(message.find("b.bin") != std::string::npos);
  CHECK(message.find("4096") != std::string::npos);
  CHECK(message.find("2048") != std::string::npos);
  CHECK(message.find("two different sizes") != std::string::npos);
  // The message quotes the identifier in the folded spelling, so a reader is
  // not told two identifiers collided when one did.
  CHECK(message.find(lower) != std::string::npos);
}

TEST_CASE("an identifier is folded to lower case before it becomes a blob name") {
  // The same raw-string assumption reaches past the owner map. `HfFile::oid` is
  // what `HfBlobPath` will name a cache file with, and a name that keeps the
  // case the listing chose splits one blob across two files on a case-sensitive
  // file system while a case-insensitive one, such as the CIFS mounts this
  // project reads checkpoints from, collapses them. Hexadecimal is
  // case-insensitive by definition and the hub and git both emit lower case, so
  // the canonical spelling is the lower-case one.
  FakeHub hub;
  TempDir tmp;
  HfHubOptions opts = OptionsFor(hub, tmp.path());
  opts.token = kToken;
  const std::string lower =
      "ab234567890abcdef0123456789abcdef0123456789abcdef0123456789abcde";
  std::string upper = lower;
  for (char& c : upper) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  hub.set_tree(std::string(R"([{"type":"file","path":"a.bin","size":4096,)") +
               R"("lfs":{"oid":")" + upper + R"("}}])");

  const std::vector<HfFile> files = HubListRepoFiles("org/repo", kCommit, opts);
  REQUIRE(files.size() == 1);
  CHECK(files.at(0).oid == lower);
}

TEST_CASE("two cases of one identifier agreeing on size are still accepted") {
  // The negative control for the fold. Folding must not convert the legitimate
  // duplicate-content shape into a refusal, and it must not leave the two
  // spellings looking like two different identifiers either: they are one, and
  // both entries load.
  FakeHub hub;
  TempDir tmp;
  HfHubOptions opts = OptionsFor(hub, tmp.path());
  opts.token = kToken;
  const std::string lower =
      "ab234567890abcdef0123456789abcdef0123456789abcdef0123456789abcde";
  std::string upper = lower;
  for (char& c : upper) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  hub.set_tree(std::string(R"([{"type":"file","path":"a.bin","size":4096,)") +
               R"("lfs":{"oid":")" + lower + R"("}},)" +
               R"({"type":"file","path":"b.bin","size":4096,)" +
               R"("lfs":{"oid":")" + upper + R"("}}])");

  const std::vector<HfFile> files = HubListRepoFiles("org/repo", kCommit, opts);
  REQUIRE(files.size() == 2);
  CHECK(files.at(0).oid == lower);
  CHECK(files.at(1).oid == lower);
}

TEST_CASE("an upper case degenerate identifier is refused by the degeneracy rule") {
  // The degeneracy rule asks whether every character is the same one, which no
  // change of case can defeat on its own, and the fold cannot break it either
  // because folding a repeated character leaves it repeated. This case pins
  // that, so the fold cannot silently move the rule.
  FakeHub hub;
  TempDir tmp;
  HfHubOptions opts = OptionsFor(hub, tmp.path());
  opts.token = kToken;
  hub.set_tree(std::string(R"([{"type":"file","path":"a.bin","size":4096,)") +
               R"("lfs":{"oid":")" + std::string(64, 'A') + R"("}}])");

  std::string message;
  try {
    HubListRepoFiles("org/repo", kCommit, opts);
    FAIL("an upper case degenerate identifier must be refused");
  } catch (const std::runtime_error& e) {
    message = e.what();
  }
  INFO("message: ", message);
  CHECK(message.find("characters are all 'a'") != std::string::npos);
}

TEST_CASE("two files that share an identifier AND a size are accepted") {
  // THE FALSE POSITIVE the old rule produced, and the reason it had to change.
  // `lfs.oid` is the sha256 of the contents and the plain `oid` is the git blob
  // sha1, so two byte-identical files in one repository share one identifier by
  // construction. A repository that ships the same tokenizer file twice, or the
  // same shard under two names, is legitimate and must load.
  FakeHub hub;
  TempDir tmp;
  HfHubOptions opts = OptionsFor(hub, tmp.path());
  opts.token = kToken;
  const std::string oid =
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
  hub.set_tree(SharedOidTree(oid, 4096, 4096));

  const std::vector<HfFile> files = HubListRepoFiles("org/repo", kCommit, opts);
  REQUIRE(files.size() == 2);
  CHECK(files.at(0).path == "model-00001-of-00002.safetensors");
  CHECK(files.at(1).path == "model-00002-of-00002.safetensors");
  CHECK(files.at(0).oid == oid);
  CHECK(files.at(1).oid == oid);
  CHECK(files.at(0).size == 4096);
  CHECK(files.at(1).size == 4096);
}

TEST_CASE("the content size is read from lfs.size when the top level omits it") {
  // The size the rule compares has to be the CONTENT size. The tree API reports
  // it at the top level for every file and repeats it in `lfs.size`, so the top
  // level is read first and `lfs.size` is the fallback. `lfs.pointerSize` is the
  // size of the pointer file, is around 135 bytes for every shard, and would
  // make two different shards look equal.
  FakeHub hub;
  TempDir tmp;
  HfHubOptions opts = OptionsFor(hub, tmp.path());
  opts.token = kToken;
  const std::string oid = Sha256Like("a7");
  hub.set_tree(
      std::string(R"([{"type":"file","path":"a.safetensors",)") +
      R"("lfs":{"oid":")" + oid + R"(","size":4096,"pointerSize":135}},)" +
      R"({"type":"file","path":"b.safetensors",)" +
      R"("lfs":{"oid":")" + Sha256Like("b8") +
      R"(","size":2048,"pointerSize":135}}])");

  const std::vector<HfFile> files = HubListRepoFiles("org/repo", kCommit, opts);
  REQUIRE(files.size() == 2);
  CHECK(files.at(0).size == 4096);
  CHECK(files.at(1).size == 2048);
}

TEST_CASE("an untokenized identifier is dropped rather than used as a blob name") {
  // A well-formed identifier read without a token. It never becomes a blob
  // name, because an unauthenticated answer about a gated repository is not
  // evidence. That drop is a decision about USE, and it is the one thing the
  // token still governs: both integrity rules run whoever asked.
  FakeHub hub;
  TempDir tmp;
  hub.set_tree(SharedOidTree(
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef", 4096,
      4096));
  const std::vector<HfFile> files =
      HubListRepoFiles("org/repo", kCommit, OptionsFor(hub, tmp.path()));
  REQUIRE(files.size() == 2);
  for (const HfFile& file : files) CHECK(file.oid.empty());
  // The size is read whether or not the identifier is kept.
  CHECK(files.at(0).size == 4096);
}

TEST_CASE("the listing skips directories and refuses a path that escapes the snapshot") {
  FakeHub hub;
  TempDir tmp;
  hub.set_tree(
      R"([{"type":"directory","path":"original"},)"
      R"({"type":"file","path":"../escape.bin","size":1},)"
      R"({"type":"file","path":"/absolute.bin","size":1},)"
      R"({"type":"file","path":"config.json","size":12}])");
  const std::vector<HfFile> files =
      HubListRepoFiles("org/repo", kCommit, OptionsFor(hub, tmp.path()));
  REQUIRE(files.size() == 1);
  CHECK(files.at(0).path == "config.json");
}

TEST_CASE("an https endpoint is refused by name when the build cannot speak TLS") {
  // W5 owns the TLS build options themselves. This case exists in W2 because
  // the DEFAULT endpoint is https, so a build with no TLS would otherwise fail
  // with a transport error that says nothing about why. Both arms are asserted,
  // so the case keeps counting once W5 turns TLS on rather than compiling away.
  TempDir tmp;
  HfHubOptions opts;
  // Port 1 refuses immediately, so the TLS-capable arm does not wait on a real
  // host and reaches no network.
  opts.endpoint = "https://127.0.0.1:1/";
  opts.hub_dir = tmp.path();
  opts.connect_timeout_seconds = 2;
  opts.read_timeout_seconds = 2;

  std::string message;
  try {
    HubResolveRefToCommit("org/repo", "", opts);
    FAIL("an https endpoint on a dead port must refuse");
  } catch (const std::runtime_error& e) {
    message = e.what();
  }

#ifndef CPPHTTPLIB_OPENSSL_SUPPORT
  CHECK(message.find("VLLM_CPP_HF_DOWNLOAD") != std::string::npos);
  CHECK(message.find("VLLM_CPP_OPENSSL") != std::string::npos);
  CHECK(message.find("VLLM_CPP_BUILD_BORINGSSL") != std::string::npos);
#else
  // With TLS in the build the refusal is a transport failure, and it must NOT
  // send the reader to rebuild something that is already there.
  CHECK(message.find("VLLM_CPP_OPENSSL") == std::string::npos);
  CHECK(message.find("127.0.0.1") != std::string::npos);
#endif
}

TEST_CASE("a redirected API answer is refused and the token does not follow it") {
  // `set_follow_location(true)` forwards the whole request, headers included,
  // to wherever the answer points: third_party/httplib/httplib.h:7774 copies
  // the request and :13537 hands a cross-host redirect to
  // `create_redirect_client`. The bearer token would then reach a host the
  // caller never named. llama.cpp's `api_get` at `common/hf-cache.cpp:198-226 @
  // b10451` does not set it either. The redirect that this row does want to
  // follow is the content-delivery-network address for the BYTES, which is W3
  // and a different client.
  TokenTrap trap;
  FakeHub hub;
  TempDir tmp;
  hub.set_redirect(trap.url());
  HfHubOptions opts = OptionsFor(hub, tmp.path());
  opts.token = kToken;

  std::string message;
  try {
    HubResolveRefToCommit("org/repo", "", opts);
    FAIL("a redirected API answer must refuse rather than be followed");
  } catch (const std::runtime_error& e) {
    message = e.what();
  }
  INFO("message: ", message);
  CHECK(message.find("302") != std::string::npos);
  // The one assertion that matters: nothing was sent to the other authority.
  CHECK(trap.request_count() == 0);
  CHECK(trap.authorization().empty());
}

TEST_CASE("offline opens no socket in the reference call") {
  // The spec's `## Scope` says HF_HUB_OFFLINE resolves from the cache and opens
  // no socket. `HubResolveCommitCached` checked it; the two calls that speak to
  // the hub did not, so any caller reaching them directly went to the network
  // under a setting that forbids it.
  FakeHub hub;
  TempDir tmp;
  HfHubOptions opts = OptionsFor(hub, tmp.path());
  opts.offline = true;

  std::string message;
  try {
    HubResolveRefToCommit("org/repo", "", opts);
    FAIL("offline must refuse before a request is made");
  } catch (const std::runtime_error& e) {
    message = e.what();
  }
  CHECK(hub.request_count() == 0);
  CHECK(message.find("HF_HUB_OFFLINE") != std::string::npos);
  CHECK(message.find("org/repo") != std::string::npos);
}

TEST_CASE("offline opens no socket in the tree listing") {
  FakeHub hub;
  TempDir tmp;
  HfHubOptions opts = OptionsFor(hub, tmp.path());
  opts.offline = true;

  std::string message;
  try {
    HubListRepoFiles("org/repo", kCommit, opts);
    FAIL("offline must refuse before a request is made");
  } catch (const std::runtime_error& e) {
    message = e.what();
  }
  CHECK(hub.request_count() == 0);
  CHECK(message.find("HF_HUB_OFFLINE") != std::string::npos);
}

TEST_CASE("the tree listing refuses a revision that is not a commit") {
  // Every call after the reference resolution names the commit, so that a
  // moving `main` cannot change what a second run loads. A listing asked for
  // "main" would defeat that, and it must not reach the hub to find out.
  FakeHub hub;
  TempDir tmp;
  const HfHubOptions opts = OptionsFor(hub, tmp.path());

  std::string message;
  try {
    HubListRepoFiles("org/repo", "main", opts);
    FAIL("a tree listing must be asked for a commit");
  } catch (const std::runtime_error& e) {
    message = e.what();
  }
  CHECK(message.find("is not a commit") != std::string::npos);
  CHECK(hub.request_count() == 0);

  // A 64 character identifier is a large-file-storage object, not a commit.
  CHECK_THROWS_AS(HubListRepoFiles("org/repo", std::string(64, 'a'), opts),
                  std::runtime_error);
  // ...and a 40 character non-hexadecimal string is not one either.
  CHECK_THROWS_AS(HubListRepoFiles("org/repo", std::string(40, 'z'), opts),
                  std::runtime_error);
  CHECK(hub.request_count() == 0);
}

TEST_CASE("a malformed object identifier is dropped and the file is kept") {
  // The other arm of "no hexadecimal string is proof of anything". The refusal
  // arm covers a listing that repeats ONE identifier; this one covers a value
  // that is not an identifier at all. The file stays, because it is still
  // addressable by path and only content addressing is lost.
  FakeHub hub;
  TempDir tmp;
  HfHubOptions opts = OptionsFor(hub, tmp.path());
  opts.token = kToken;
  hub.set_tree(
      std::string(R"([{"type":"file","path":"short.safetensors","size":1,)") +
      R"("lfs":{"oid":"abc123"}},)" +
      R"({"type":"file","path":"notes.txt","size":2,)" +
      R"("lfs":{"oid":")" + std::string(64, 'z') + R"("}},)" +
      R"({"type":"file","path":"good.safetensors","size":3,)" +
      R"("lfs":{"oid":")" + Sha256Like("c4") + R"("}}])");

  const std::vector<HfFile> files = HubListRepoFiles("org/repo", kCommit, opts);
  REQUIRE(files.size() == 3);
  CHECK(files.at(0).path == "short.safetensors");
  CHECK(files.at(0).oid.empty());
  CHECK(files.at(1).path == "notes.txt");
  CHECK(files.at(1).oid.empty());
  // The well-formed one is untouched, so the case measures the drop and not a
  // listing that carries no identifiers at all.
  CHECK(files.at(2).oid == Sha256Like("c4"));
}

TEST_CASE("ONE path listed twice at an agreed size is accepted") {
  // The size rule refuses a DISAGREEMENT, not a repetition. A redundant
  // listing that repeats one entry unchanged says nothing contradictory, so it
  // stays usable. Its pair, "ONE path listed twice with disagreeing sizes is
  // refused", serves this same repeated path at two sizes, so the two cases
  // measure the rule's boundary rather than asserting that a repeated path is
  // exempt from it.
  FakeHub hub;
  TempDir tmp;
  HfHubOptions opts = OptionsFor(hub, tmp.path());
  opts.token = kToken;
  const std::string oid = Sha256Like("d5");
  hub.set_tree(
      std::string(R"([{"type":"file","path":"model.safetensors","size":4096,)") +
      R"("lfs":{"oid":")" + oid + R"("}},)" +
      R"({"type":"file","path":"model.safetensors","size":4096,)" +
      R"("lfs":{"oid":")" + oid + R"("}}])");

  const std::vector<HfFile> files = HubListRepoFiles("org/repo", kCommit, opts);
  REQUIRE(files.size() == 2);
  CHECK(files.at(0).path == "model.safetensors");
  CHECK(files.at(1).path == "model.safetensors");
  CHECK(files.at(0).oid == oid);
  CHECK(files.at(1).oid == oid);
}

TEST_CASE("an entry that reports no size cannot disarm the size rule") {
  // The rule keeps ONE owner per identifier, and an owner whose size the
  // listing never reported can neither agree nor disagree with a later entry.
  // Keeping such an entry as the owner would silence the rule for that
  // identifier for the rest of the listing. `HF_ENDPOINT` is user
  // configurable, so a mirror that omits `size` on ONE entry would otherwise
  // buy every later entry under that identifier an unconditional pass.
  //
  // The first entry here carries `lfs.pointerSize` and no size, which is the
  // shape that reads as a size only to a caller that reads the wrong field.
  FakeHub hub;
  TempDir tmp;
  HfHubOptions opts = OptionsFor(hub, tmp.path());
  opts.token = kToken;
  const std::string oid = Sha256Like("e6");
  hub.set_tree(
      std::string(R"([{"type":"file","path":"a.bin",)") +
      R"("lfs":{"oid":")" + oid + R"(","pointerSize":135}},)" +
      R"({"type":"file","path":"b.bin","size":4096,)" +
      R"("lfs":{"oid":")" + oid + R"("}},)" +
      R"({"type":"file","path":"c.bin","size":2048,)" +
      R"("lfs":{"oid":")" + oid + R"("}}])");

  std::string message;
  try {
    HubListRepoFiles("org/repo", kCommit, opts);
    FAIL("an entry with no size must not disarm the size rule");
  } catch (const std::runtime_error& e) {
    message = e.what();
  }
  INFO("message: ", message);
  CHECK(message.find(oid) != std::string::npos);
  CHECK(message.find("b.bin") != std::string::npos);
  CHECK(message.find("c.bin") != std::string::npos);
  CHECK(message.find("4096") != std::string::npos);
  CHECK(message.find("2048") != std::string::npos);
  CHECK(message.find("two different sizes") != std::string::npos);
}

TEST_CASE("ONE path listed twice with disagreeing sizes is refused") {
  // The rule reads "a shared identifier whose entries disagree on a known size
  // is refused", and it carries no exemption for a repeated path. One path
  // that is 4096 bytes and 2048 bytes in the same listing is self
  // contradictory whichever field is believed, and it is exactly the shape a
  // hub answering something other than the truth produces.
  FakeHub hub;
  TempDir tmp;
  HfHubOptions opts = OptionsFor(hub, tmp.path());
  opts.token = kToken;
  const std::string oid = Sha256Like("f7");
  hub.set_tree(
      std::string(R"([{"type":"file","path":"model.safetensors","size":4096,)") +
      R"("lfs":{"oid":")" + oid + R"("}},)" +
      R"({"type":"file","path":"model.safetensors","size":2048,)" +
      R"("lfs":{"oid":")" + oid + R"("}}])");

  std::string message;
  try {
    HubListRepoFiles("org/repo", kCommit, opts);
    FAIL("one path listed twice at two sizes must be refused");
  } catch (const std::runtime_error& e) {
    message = e.what();
  }
  INFO("message: ", message);
  CHECK(message.find(oid) != std::string::npos);
  CHECK(message.find("model.safetensors") != std::string::npos);
  CHECK(message.find("4096") != std::string::npos);
  CHECK(message.find("2048") != std::string::npos);
  CHECK(message.find("two different sizes") != std::string::npos);
}

TEST_CASE("a listing that reports no size is not a zero-byte file") {
  // `HfFile::size` has to be able to say UNKNOWN. A zero-byte file and an
  // entry whose size the listing never reported are different facts, and a
  // plain `uint64_t` spells both of them `0`. W3 sizes a byte range and a
  // resume offset from this field, so the ambiguity would be inherited by the
  // downloader rather than staying inside the listing.
  FakeHub hub;
  TempDir tmp;
  HfHubOptions opts = OptionsFor(hub, tmp.path());
  opts.token = kToken;
  hub.set_tree(
      std::string(R"([{"type":"file","path":"empty.txt","size":0},)") +
      R"({"type":"file","path":"unsized.bin",)" +
      R"("lfs":{"oid":")" + Sha256Like("a9") + R"(","pointerSize":135}}])");

  const std::vector<HfFile> files = HubListRepoFiles("org/repo", kCommit, opts);
  REQUIRE(files.size() == 2);
  CHECK(files.at(0).path == "empty.txt");
  REQUIRE(files.at(0).size.has_value());
  CHECK(files.at(0).size.value() == 0);
  CHECK(files.at(1).path == "unsized.bin");
  // `lfs.pointerSize` is not a content size, so this entry has none.
  CHECK_FALSE(files.at(1).size.has_value());
}
