// ENG-HF-MODEL-DOWNLOAD W4 (#1280): THE REACHABILITY GATE.
//
// `.agents/reachability.md` asks two questions, and this file answers the
// second one. `tests/vllm/transformers_utils/test_model_resolver.cpp` proves
// the resolver WORKS; it constructs the call itself, so it would stay green
// with the production call site deleted. This file proves the resolver is
// REACHED: it enters at `VllmServerMain(argc, argv)`, which is what the C ABI's
// `vllm_server_main` and the `vllm-server` binary both call, hands it
// `--model org/repo` with `HF_ENDPOINT` aimed at an in-process fake hub, and
// then asks the running server to complete a request.
//
// Deleting the `ResolveModelArgument` call in `server_main.cpp` turns this red:
// the loader then opens `tiny/llama` as a relative directory, finds nothing,
// and the server never binds.
//
// WHY A CHILD PROCESS. `VllmServerMain` blocks in `listen`, and `ParseArgs`
// reports a bad argument through `Usage()`, which calls `std::exit`. The
// pattern is the one in `test_serve_recipe_args.cpp` and
// `test_serve_residency_config.cpp`, with `fork` plus `execv` instead of
// `popen` so the parent holds the child's process id and can stop the server
// once it has answered. The FAKE HUB LIVES IN THE PARENT, which is why the
// child is a fresh `execv` rather than a forked copy: a forked child would
// inherit no listening thread and there would be nothing to fetch from.
#include <doctest/doctest.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <httplib/httplib.h>
#include <nlohmann/json.hpp>

#include "support/process_id.h"
#include "vllm/entrypoints/openai/server_main.h"

namespace fs = std::filesystem;
using nlohmann::json;

namespace {

constexpr const char* kCommit = "1111111111111111111111111111111111111111";
// The DEFAULT branch's commit, and the hub serves NOTHING under it. The
// checkpoint lives on `kRevision` instead, which is what makes `--revision`
// load bearing: a run that does not carry the flag resolves `main`, gets this
// commit, lists an empty tree and never finds a config.json, so the server does
// not bind. Without that the flag would be reached but unpinned, because the
// fake hub answered `main` and `--revision main` identically.
constexpr const char* kDefaultCommit =
    "2222222222222222222222222222222222222222";
constexpr const char* kRevision = "serving";
constexpr const char* kRepoId = "tiny/llama";

std::string ReadAll(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

// The committed `llama_embed_e2e` checkpoint is a real, tiny, loadable
// safetensors file, and it is 154 KB rather than a fabricated tensor set. It
// carries the `LlamaModel` (pooling) tensor names, so every name gains the
// `model.` prefix the `LlamaForCausalLM` loader reads. The DATA SECTION is
// copied verbatim and the offsets are unchanged, because they are relative to
// the start of that section and the header length is not part of them.
std::string CausalLmSafetensors(const fs::path& source) {
  const std::string bytes = ReadAll(source);
  REQUIRE(bytes.size() > 8);
  uint64_t header_len = 0;
  for (int i = 7; i >= 0; --i) {
    header_len = (header_len << 8) |
                 static_cast<unsigned char>(bytes[static_cast<size_t>(i)]);
  }
  REQUIRE(header_len + 8 <= bytes.size());
  const json header = json::parse(bytes.substr(8, header_len));
  json renamed = json::object();
  for (const auto& item : header.items()) {
    if (item.key() == "__metadata__") continue;
    renamed["model." + item.key()] = item.value();
  }
  std::string text = renamed.dump();
  while (text.size() % 8 != 0) text.push_back(' ');
  std::string out(8, '\0');
  for (int i = 0; i < 8; ++i) {
    out[static_cast<size_t>(i)] =
        static_cast<char>((text.size() >> (8 * i)) & 0xff);
  }
  out += text;
  out += bytes.substr(8 + header_len);
  return out;
}

// The shape of the committed fixture, spelled as a generation config. The
// tensor sizes come from the checkpoint and the loader refuses a mismatch, so
// these numbers are pinned by the file rather than chosen here.
std::string CausalLmConfig() {
  json config;
  config["architectures"] = json::array({"LlamaForCausalLM"});
  config["model_type"] = "llama";
  config["hidden_size"] = 64;
  config["num_hidden_layers"] = 2;
  config["num_attention_heads"] = 4;
  config["num_key_value_heads"] = 2;
  config["head_dim"] = 16;
  config["intermediate_size"] = 128;
  config["rms_norm_eps"] = 1e-05;
  config["rope_theta"] = 500000.0;
  config["vocab_size"] = 32;
  config["max_position_embeddings"] = 128;
  config["torch_dtype"] = "bfloat16";
  // No `lm_head.weight` in the checkpoint, so the output head aliases the
  // embedding table. The loader skips the tensor on this setting.
  config["tie_word_embeddings"] = true;
  config["attention_bias"] = false;
  return config.dump();
}

// The fake hub: refs, a recursive tree, and the bytes. It records the paths it
// was asked for, so the decoy assertion is a statement about the server.
class FakeHub {
 public:
  FakeHub() {
    server_.Get("/api/models/(.*)/refs",
                [this](const httplib::Request& req, httplib::Response& res) {
                  Record(req);
                  res.set_content(
                      std::string(
                          R"({"branches":[{"name":"main","targetCommit":")") +
                          kDefaultCommit + R"("},{"name":")" + kRevision +
                          R"(","targetCommit":")" + kCommit +
                          R"("}],"tags":[]})",
                      "application/json");
                });
    server_.Get("/api/models/(.*)/tree/(.*)",
                [this](const httplib::Request& req, httplib::Response& res) {
                  Record(req);
                  // Only `kCommit` holds the checkpoint. The default branch's
                  // commit lists nothing, so a run that lost `--revision`
                  // fetches nothing and the server never binds.
                  if (req.path.find(kCommit) == std::string::npos) {
                    res.set_content("[]", "application/json");
                    return;
                  }
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
                  const auto it = files_.find(req.path.substr(at + marker.size()));
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
    json out = json::array();
    for (const auto& entry : files_) {
      json item;
      item["type"] = "file";
      item["path"] = entry.first;
      item["size"] = entry.second.size();
      out.push_back(item);
    }
    return out.dump();
  }
  void Record(const httplib::Request& req) {
    const std::lock_guard<std::mutex> lock(mu_);
    paths_.push_back(req.path);
  }

  httplib::Server server_;
  std::thread thread_;
  int port_ = 0;
  mutable std::mutex mu_;
  std::vector<std::string> paths_;
  std::map<std::string, std::string> files_;
};

// A port nothing is listening on, AND THE PROBE SOCKET IS CLOSED before the
// number is returned.
//
// The first version of this function bound with an `httplib::Server` and called
// `probe.stop()`. That released nothing. `Server::stop()`
// (`third_party/httplib/httplib.h:11460`) is guarded by `if (is_running_)`, this
// function never calls `listen_after_bind()`, so `is_running_` was false and the
// socket was never shut down, and `Server::~Server()` is `= default` and does not
// close it either. httplib sets `SO_REUSEPORT` (`httplib.h:9455`), so the forked
// child bound the SAME port successfully and the kernel then load balanced
// inbound connections between the orphan socket, which nobody ever accepted on,
// and the real server.
//
// Measured on this box with `ss -ltnp` during an unmutated run at head
// `e25d3d344`: two LISTEN rows on one port, one owned by the parent and one by
// the child, the parent's carrying `Recv-Q 1`, which is the hung request sitting
// in a backlog nobody drains. Three runs took 90.4 s, 240.4 s and 180.4 s, and
// the 240.4 s one FAILED. Every stall was an exact multiple of a client read
// timeout rather than an intermediate value, which is what a contended box would
// have given.
//
// A plain POSIX socket is used rather than an httplib one because the close has
// to be unconditional.
int FreePort() {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  REQUIRE(fd >= 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = ::htonl(INADDR_LOOPBACK);
  addr.sin_port = 0;
  REQUIRE(::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0);
  socklen_t length = sizeof(addr);
  REQUIRE(::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &length) == 0);
  const int port = static_cast<int>(::ntohs(addr.sin_port));
  REQUIRE(::close(fd) == 0);
  return port;
}

std::vector<std::string> SplitOnSpaces(const std::string& text) {
  std::vector<std::string> out;
  std::string current;
  for (const char c : text) {
    if (c == ' ') {
      if (!current.empty()) out.push_back(current);
      current.clear();
    } else {
      current.push_back(c);
    }
  }
  if (!current.empty()) out.push_back(current);
  return out;
}

}  // namespace

// THE CHILD. Skip-decorated, so a normal run never executes it; it runs only
// when the parent below re-execs this binary by name.
TEST_CASE("serve_hf_model_child" * doctest::skip()) {
  const char* raw = std::getenv("VLLM_TEST_SERVE_ARGS");
  REQUIRE(raw != nullptr);
  std::vector<std::string> args{"vllm-server"};
  for (std::string& token : SplitOnSpaces(raw)) args.push_back(std::move(token));
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (std::string& arg : args) argv.push_back(arg.data());
  const int rc = vllm::entrypoints::openai::VllmServerMain(
      static_cast<int>(argv.size()), argv.data());
  std::cout << "SERVE_RC=" << rc << "\n" << std::flush;
  std::exit(rc);
}

TEST_CASE("serve: --model org/repo FETCHES the checkpoint and completes a request") {
  const fs::path fixture(VLLM_LLAMA_EMBED_FIXTURE_DIR);
  REQUIRE(fs::is_regular_file(fixture / "model.safetensors"));

  FakeHub hub;
  const std::string weights = CausalLmSafetensors(fixture / "model.safetensors");
  hub.add("config.json", CausalLmConfig());
  hub.add("tokenizer.json", ReadAll(fixture / "tokenizer.json"));
  hub.add("model.safetensors", weights);
  hub.add("model.safetensors.index.json",
          R"({"metadata":{"total_size":1},"weight_map":)"
          R"({"model.embed_tokens.weight":"model.safetensors"}})");
  // The duplicate-format decoy a published checkpoint carries. Nothing must ask
  // for it.
  hub.add("original/model.safetensors", weights);

  const fs::path home =
      fs::temp_directory_path() /
      ("vllm_serve_hf_model_" + std::to_string(vllm_test::ProcessId()));
  fs::remove_all(home);
  fs::create_directories(home);

  // `--download-dir` IS the directory that holds the `models--org--repo`
  // folders, which is how vLLM hands it to `snapshot_download(cache_dir=...)`
  // (`config/model.py:183`). It is a sibling of `hub/`, never inside it, so the
  // assertions at the end can tell the two apart.
  const fs::path download_dir = home / "dl";
  fs::create_directories(download_dir);

  const int port = FreePort();
  // `--max-num-seqs 4` is a HARNESS setting, not a product one: the HTTP worker
  // pool is sized from it plus four (`ApiServer::kControlWorkerHeadroom`), so
  // the default 32 starts a 36-thread pool to serve one four-token request.
  // It is NOT why this case used to be slow. That was a leaked listening socket
  // in `FreePort`, and the note there records the measurement.
  //
  // `--revision` and `--download-dir` are here because they are PRODUCTION
  // FLAGS with a production effect, and until they were on this line the only
  // thing that reached `ParseArgs` for either of them was nothing at all: both
  // `ParseArgs` branches could be deleted with every suite still green, because
  // the unit cases set `ModelResolveOptions` by hand. They are load bearing
  // now: the checkpoint is published on a non-default branch, and the cache
  // lands under `--download-dir` rather than under `HF_HOME`.
  const std::string serve_args = std::string("--model ") + kRepoId +
                                 " --revision " + kRevision +
                                 " --download-dir " + download_dir.string() +
                                 " --port " + std::to_string(port) +
                                 " --host 127.0.0.1 --max-model-len 64"
                                 " --block-size 16 --max-num-seqs 4"
                                 " --disable-log-requests";

  char exe[4096];
  const ssize_t n = ::readlink("/proc/self/exe", exe, sizeof(exe) - 1);
  REQUIRE(n > 0);
  exe[n] = '\0';

  ::setenv("VLLM_TEST_SERVE_ARGS", serve_args.c_str(), 1);
  ::setenv("HF_ENDPOINT", hub.endpoint().c_str(), 1);
  ::setenv("HF_HOME", home.c_str(), 1);
  ::unsetenv("HF_HUB_OFFLINE");
  ::unsetenv("HF_TOKEN");

  const pid_t pid = ::fork();
  REQUIRE(pid >= 0);
  if (pid == 0) {
    const char* child_argv[] = {exe, "--no-skip",
                                "--test-case=serve_hf_model_child", nullptr};
    ::execv(exe, const_cast<char* const*>(child_argv));
    std::_Exit(127);
  }

  // Wait for the server to bind. A checkpoint this small loads in well under a
  // second on any host, and the ceiling exists so a failure is a failure rather
  // than a hung suite.
  httplib::Client client("http://127.0.0.1:" + std::to_string(port));
  client.set_connection_timeout(1, 0);
  client.set_read_timeout(30, 0);
  bool healthy = false;
  for (int attempt = 0; attempt < 600 && !healthy; ++attempt) {
    int status = 0;
    if (::waitpid(pid, &status, WNOHANG) == pid) break;  // the child died
    const httplib::Result health = client.Get("/health");
    healthy = health && health->status == 200;
    if (!healthy) std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  std::string completion_body;
  std::string completion_error;
  int completion_status = 0;
  if (healthy) {
    const json request = {{"model", kRepoId},
                          {"prompt", "abc"},
                          {"max_tokens", 4},
                          {"temperature", 0}};
    // ONE ATTEMPT. There WAS a retry loop here, three attempts whenever the
    // exchange did not complete, and it was absorbing the leaked probe socket
    // that `FreePort` above now closes. With the leak gone the completion
    // answers in 0.2 to 0.4 s and there is nothing left for a retry to absorb,
    // so a second attempt could only hide the next defect the way this one hid
    // that one. Sixty seconds of dead time read as a green run with no output.
    //
    // A FRESH client rather than the health poll's. That poll leaves a
    // keep-alive socket the server may close on its own timeout, and a reused
    // dead socket reports as a transport error that reads exactly like a
    // refusal.
    httplib::Client caller("http://127.0.0.1:" + std::to_string(port));
    caller.set_connection_timeout(5, 0);
    caller.set_read_timeout(60, 0);
    const httplib::Result answer =
        caller.Post("/v1/completions", request.dump(), "application/json");
    if (answer) {
      completion_status = answer->status;
      completion_body = answer->body;
    } else {
      completion_error = httplib::to_string(answer.error());
    }
  }

  ::kill(pid, SIGKILL);
  int status = 0;
  ::waitpid(pid, &status, 0);

  std::string joined;
  for (const std::string& path : hub.paths()) joined += path + "\n";
  INFO("hub paths:\n" << joined);
  INFO("completion: " << completion_body);
  INFO("transport error: " << completion_error);

  // THE SERVER BOOTED FROM A CHECKPOINT IT FETCHED. Nothing was on disk when
  // the process started: `HF_HOME` was an empty scratch directory.
  REQUIRE(healthy);
  REQUIRE(completion_status == 200);
  const json completion = json::parse(completion_body);
  REQUIRE(completion.contains("choices"));
  REQUIRE(completion["choices"].size() == 1);
  CHECK(completion["choices"][0]["finish_reason"] == "length");
  CHECK(completion["usage"]["completion_tokens"] == 4);
  // The served name is what the user typed, not the commit directory the cache
  // happens to hold.
  CHECK(completion["model"] == kRepoId);

  // The fetch went through the hub, index driven, and never touched the decoy.
  CHECK(hub.asked_for("/api/models/tiny/llama/refs"));
  CHECK(hub.asked_for("/resolve/" + std::string(kCommit) + "/model.safetensors"));
  CHECK_FALSE(hub.asked_for("original"));

  // And the cache holds the HuggingFace layout, so a second run is offline.
  //
  // IT LANDED UNDER `--download-dir`, NOT UNDER `HF_HOME`, which is the
  // statement that `--download-dir` reached the resolver rather than being
  // parsed and dropped. Deleting its `ParseArgs` branch puts the tree back
  // under `HF_HOME/hub` and turns both of these red.
  CHECK(fs::is_regular_file(download_dir / "models--tiny--llama" / "snapshots" /
                            kCommit / "config.json"));
  CHECK_FALSE(fs::exists(home / "hub" / "models--tiny--llama"));
  // And it is the `--revision` commit. `main` names `kDefaultCommit`, under
  // which this hub lists nothing, so a run that lost the flag never gets this
  // far: it fails at `REQUIRE(healthy)` above.
  CHECK_FALSE(fs::exists(download_dir / "models--tiny--llama" / "snapshots" /
                         kDefaultCommit));

  fs::remove_all(home);
}
