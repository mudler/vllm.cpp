// ENG-HF-MODEL-DOWNLOAD W3 (#1280): the byte transport.
//
// Every case runs against an IN-PROCESS FAKE HUB, a real `httplib::Server` on
// an ephemeral port reached over plain hypertext transfer protocol, following
// `tests/vllm/transformers_utils/test_hf_hub.cpp`. There is no TLS here on
// purpose: TLS has its own instruments in W5, and a hermetic test that speaks
// plain HTTP proves resume, truncation and integrity, not transport security.
//
// The hub RECORDS the request headers it saw. "The Range header was sent" and
// "the decoy was never requested" are both statements about what the server
// received, and a client-side mock cannot make either of them.
#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <httplib/httplib.h>
#include <nlohmann/json.hpp>

#include "support/process_id.h"
#include "vllm/transformers_utils/downloader.h"
#include "vllm/transformers_utils/hf_hub.h"

namespace fs = std::filesystem;
using vllm::transformers_utils::HfDownloadOptions;
using vllm::transformers_utils::HfDownloadResult;
using vllm::transformers_utils::HfFileShape;
using vllm::transformers_utils::HfHubOptions;
using vllm::transformers_utils::HfInstallDownloadInterruptHandler;
using vllm::transformers_utils::HfRemoteFile;
using vllm::transformers_utils::HfRepoLock;
using vllm::transformers_utils::HfResetDownloadInterrupt;
using vllm::transformers_utils::HfShapeForPath;
using vllm::transformers_utils::HfVerifyFileShape;
using vllm::transformers_utils::HubDownloadFile;
using vllm::transformers_utils::HubProbeFile;

namespace {

class TempDir {
 public:
  TempDir() {
    static std::atomic<int> counter{0};
    path_ = fs::temp_directory_path() /
            ("vllm_downloader_test_" + std::to_string(vllm_test::ProcessId()) +
             "_" + std::to_string(counter.fetch_add(1)));
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

// A valid single-tensor safetensors blob: 8 length bytes, the header, then
// exactly `data_end` payload bytes. `extra_payload` appends bytes the header
// does not account for, which is the shape the data-end rule refuses.
std::string SafetensorsBlob(size_t payload_bytes, size_t extra_payload = 0) {
  nlohmann::json header;
  header["t"] = {{"dtype", "U8"},
                 {"shape", {static_cast<uint64_t>(payload_bytes)}},
                 {"data_offsets", {0, static_cast<uint64_t>(payload_bytes)}}};
  std::string text = header.dump();
  while (text.size() % 8 != 0) text.push_back(' ');
  std::string out(8, '\0');
  for (int i = 0; i < 8; ++i) {
    out[static_cast<size_t>(i)] =
        static_cast<char>((text.size() >> (8 * i)) & 0xff);
  }
  out += text;
  for (size_t i = 0; i < payload_bytes; ++i) {
    out.push_back(static_cast<char>('A' + static_cast<int>(i % 26)));
  }
  out.append(extra_payload, 'Z');
  return out;
}

// The fake hub, serving ONE body at ONE address, with the three answers a
// transport has to survive.
class ByteHub {
 public:
  ByteHub() {
    // httplib routes HEAD to the GET handler
    // (third_party/httplib/httplib.h:12183) and writes no body for it, so both
    // verbs are answered here and the method is read from the request.
    server_.Get("/file", [this](const httplib::Request& req,
                                httplib::Response& res) {
      Record(req);
      res.set_header("ETag", "\"" + etag_ + "\"");
      res.set_header("Accept-Ranges", accepts_ranges_ ? "bytes" : "none");
      if (req.method == "HEAD") {
        res.set_header("Content-Length", std::to_string(body_.size()));
        res.status = 200;
        return;
      }

      if (truncate_after_ > 0) {
        // A body SHORTER than the length its own answer declares. httplib
        // reports some of these as a read error and others as a short success,
        // so this is the shape the byte count in the downloader exists for.
        const std::string prefix = body_.substr(0, truncate_after_);
        const size_t declared = body_.size();
        res.set_content_provider(
            declared, "application/octet-stream",
            [prefix](size_t offset, size_t, httplib::DataSink& sink) {
              if (offset == 0) {
                sink.write(prefix.data(), prefix.size());
                return true;
              }
              return false;  // abort: the connection closes early.
            });
        return;
      }

      const bool ranged = req.has_header("Range") && !ignore_range_;
      // THE FULL BODY IS SET IN BOTH ARMS. httplib slices it itself when the
      // request carried a range AND the status is 206
      // (`Server::apply_ranges`, third_party/httplib/httplib.h:12231), and it
      // writes the `Content-Range` header from that slice. A handler that
      // pre-sliced would be sliced a SECOND time, which is a fixture defect
      // that reads exactly like a resume bug in the client: this suite
      // measured a 12 byte tail where 24 were due.
      res.status = ranged ? 206 : 200;
      res.set_content(body_, "application/octet-stream");
    });
    port_ = server_.bind_to_any_port("127.0.0.1");
    thread_ = std::thread([this] { server_.listen_after_bind(); });
    server_.wait_until_ready();
  }
  ~ByteHub() {
    server_.stop();
    if (thread_.joinable()) thread_.join();
  }
  ByteHub(const ByteHub&) = delete;
  ByteHub& operator=(const ByteHub&) = delete;

  std::string url() const {
    return "http://127.0.0.1:" + std::to_string(port_) + "/file";
  }
  void set_body(std::string body) { body_ = std::move(body); }
  void set_ignore_range(bool value) { ignore_range_ = value; }
  void set_accepts_ranges(bool value) { accepts_ranges_ = value; }
  void set_truncate_after(size_t value) { truncate_after_ = value; }
  const std::string& body() const { return body_; }

  int request_count() const { return requests_.load(); }
  std::vector<std::string> range_headers() const {
    const std::lock_guard<std::mutex> lock(mu_);
    return ranges_;
  }

 private:
  void Record(const httplib::Request& req) {
    requests_.fetch_add(1);
    const std::lock_guard<std::mutex> lock(mu_);
    if (req.has_header("Range")) ranges_.push_back(req.get_header_value("Range"));
  }

  httplib::Server server_;
  std::thread thread_;
  int port_ = 0;
  std::atomic<int> requests_{0};
  mutable std::mutex mu_;
  std::vector<std::string> ranges_;
  std::string body_ = "0123456789abcdefghijklmnopqrstuvwxyz";
  std::string etag_ = "abc123";
  bool ignore_range_ = false;
  bool accepts_ranges_ = true;
  size_t truncate_after_ = 0;
};

HfHubOptions OptionsFor() {
  HfHubOptions opts;
  opts.connect_timeout_seconds = 5;
  opts.read_timeout_seconds = 5;
  return opts;
}

HfDownloadOptions DownloadOptionsFor() {
  HfDownloadOptions opts;
  opts.hub = OptionsFor();
  return opts;
}

std::string ReadAll(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

void WriteAll(const fs::path& path, const std::string& bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

std::string MessageOf(const std::function<void()>& body) {
  try {
    body();
  } catch (const std::exception& e) {
    return e.what();
  }
  return std::string();
}

}  // namespace

TEST_CASE("downloader: a HEAD probe reads the size, the entity tag and range support") {
  ByteHub hub;
  const HfRemoteFile info = HubProbeFile(hub.url(), OptionsFor());
  REQUIRE(info.size.has_value());
  CHECK(*info.size == hub.body().size());
  // The quotes are transport spelling and must not reach a blob file name.
  CHECK(info.etag == "abc123");
  CHECK(info.accepts_ranges);

  hub.set_accepts_ranges(false);
  CHECK_FALSE(HubProbeFile(hub.url(), OptionsFor()).accepts_ranges);
}

TEST_CASE("downloader: a complete transfer lands the bytes and leaves no .incomplete") {
  ByteHub hub;
  TempDir dir;
  const fs::path dest = dir.path() / "blob";

  const HfDownloadResult result =
      HubDownloadFile(hub.url(), dest, hub.body().size(), HfFileShape::kOpaque,
                      DownloadOptionsFor());
  CHECK(result.bytes_written == hub.body().size());
  CHECK_FALSE(result.already_present);
  CHECK(ReadAll(dest) == hub.body());
  // The `.incomplete` name is what a partial transfer occupies. Its absence is
  // the proof the rename happened rather than a copy.
  CHECK_FALSE(fs::exists(fs::path(dest.string() + ".incomplete")));

  // A SECOND call opens no socket at all. That is what makes a warm cache free.
  const int before = hub.request_count();
  const HfDownloadResult again =
      HubDownloadFile(hub.url(), dest, hub.body().size(), HfFileShape::kOpaque,
                      DownloadOptionsFor());
  CHECK(again.already_present);
  CHECK(again.bytes_written == 0);
  CHECK(hub.request_count() == before);
}

TEST_CASE("downloader: an interrupted transfer RESUMES with a Range header") {
  ByteHub hub;
  TempDir dir;
  const fs::path dest = dir.path() / "blob";
  const fs::path partial = fs::path(dest.string() + ".incomplete");

  // Stand in for a transfer that stopped a third of the way through.
  const size_t already = 12;
  WriteAll(partial, hub.body().substr(0, already));

  const HfDownloadResult result =
      HubDownloadFile(hub.url(), dest, hub.body().size(), HfFileShape::kOpaque,
                      DownloadOptionsFor());
  CHECK(result.resumed);
  // The TAIL was transferred, not the whole file. A test that only checked the
  // final bytes would pass on a client that threw the partial file away.
  CHECK(result.bytes_written == hub.body().size() - already);

  const std::vector<std::string> ranges = hub.range_headers();
  REQUIRE_FALSE(ranges.empty());
  CHECK(ranges.back() == "bytes=12-");
  // And the FINAL bytes are the whole body, not the tail written twice.
  CHECK(ReadAll(dest) == hub.body());
}

TEST_CASE("downloader: a 200 answer to a range request is REFUSED, never appended") {
  // THE DELIBERATE DIVERGENCE FROM llama.cpp `common/download.cpp:222-235 @
  // b10451`, which warns and continues. Appending a whole body onto a partial
  // file writes the first N bytes twice, and a token gate cannot see a corrupt
  // weight: the model still emits tokens.
  ByteHub hub;
  hub.set_ignore_range(true);
  TempDir dir;
  const fs::path dest = dir.path() / "blob";
  const fs::path partial = fs::path(dest.string() + ".incomplete");
  WriteAll(partial, hub.body().substr(0, 12));

  const std::string message = MessageOf([&] {
    HubDownloadFile(hub.url(), dest, hub.body().size(), HfFileShape::kOpaque,
                    DownloadOptionsFor());
  });
  INFO("refusal: " << message);
  REQUIRE_FALSE(message.empty());
  CHECK(message.find("HTTP 200") != std::string::npos);
  CHECK(message.find("206") != std::string::npos);
  // The destination name was never taken, so a later run cannot read the
  // wreckage as a cache hit.
  CHECK_FALSE(fs::exists(dest));
  // And the partial file is still only the 12 bytes it started as: nothing was
  // appended before the refusal.
  CHECK(fs::file_size(partial) == 12);
}

TEST_CASE("downloader: a body shorter than its Content-Length is refused") {
  ByteHub hub;
  hub.set_truncate_after(10);
  TempDir dir;
  const fs::path dest = dir.path() / "blob";

  const std::string message = MessageOf([&] {
    HubDownloadFile(hub.url(), dest, hub.body().size(), HfFileShape::kOpaque,
                    DownloadOptionsFor());
  });
  INFO("refusal: " << message);
  REQUIRE_FALSE(message.empty());
  // The refusal must come from the BYTE COUNT, not from whatever the transport
  // happened to report. Measured: with the count checked after httplib's own
  // verdict, deleting the count left this case green, because the generic
  // transport message satisfied "it refused".
  CHECK(message.find("truncated") != std::string::npos);
  CHECK(message.find("delivered 10") != std::string::npos);
  CHECK_FALSE(fs::exists(dest));
}

TEST_CASE("downloader: a safetensors whose data end misses the file size is refused") {
  // `8 + header_len + max(data_offsets[1]) == file_size`. The body below is a
  // well-formed header over 64 payload bytes with 16 bytes nobody accounts for
  // appended, which is what a resumed-onto-a-200 transfer leaves behind and
  // what a length field cannot see.
  ByteHub hub;
  hub.set_body(SafetensorsBlob(64, /*extra_payload=*/16));
  TempDir dir;
  const fs::path dest = dir.path() / "model.safetensors";

  CHECK(HfShapeForPath("model.safetensors") == HfFileShape::kSafetensors);
  const std::string message = MessageOf([&] {
    HubDownloadFile(hub.url(), dest, hub.body().size(),
                    HfFileShape::kSafetensors, DownloadOptionsFor());
  });
  INFO("refusal: " << message);
  REQUIRE_FALSE(message.empty());
  CHECK(message.find("safetensors") != std::string::npos);
  // The transfer matched every length the transport reported. Only the
  // structural rule can refuse it, which is why the row does not accept an
  // opaque remote field as proof.
  CHECK_FALSE(fs::exists(dest));

  // The same blob WITHOUT the unaccounted tail is accepted.
  ByteHub good;
  good.set_body(SafetensorsBlob(64));
  const fs::path ok = dir.path() / "good.safetensors";
  HubDownloadFile(good.url(), ok, good.body().size(), HfFileShape::kSafetensors,
                  DownloadOptionsFor());
  CHECK(fs::exists(ok));
}

TEST_CASE("downloader: the shape a path declares by its extension") {
  CHECK(HfShapeForPath("model.safetensors") == HfFileShape::kSafetensors);
  CHECK(HfShapeForPath("dir/Model-Q4_K_M.gguf") == HfFileShape::kGguf);
  CHECK(HfShapeForPath("config.json") == HfFileShape::kOpaque);
  CHECK(HfShapeForPath("tokenizer.model") == HfFileShape::kOpaque);
}

TEST_CASE("downloader: what an https address does, on this build's TLS state") {
  // BOTH ARMS ASSERT. The first version of this case wrapped its whole body in
  // `#ifndef CPPHTTPLIB_OPENSSL_SUPPORT`, so the moment W5 defines that macro
  // the case would have compiled to nothing and reported `assertions: 0`, which
  // this tree records as a skip wearing a pass. The preprocessor now selects
  // WHICH statement is made, never whether one is made.
  //
  // The address is LOOPBACK on a port nothing listens on, not
  // `huggingface.co`. Under a no-TLS build the scheme is refused before any
  // socket opens, so the host never mattered; under a TLS build it would be a
  // real network call, and a hermetic suite must not make one.
  TempDir dir;
  const std::string message = MessageOf([&] {
    HubProbeFile("https://127.0.0.1:1/org/repo/resolve/main/config.json",
                 OptionsFor());
  });
  INFO("refusal: " << message);
  REQUIRE_FALSE(message.empty());
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
  // This build SPEAKS https, so the refusal is the connection failing and must
  // NOT name the build options: a message telling a working build to rebuild
  // itself sends the reader after the wrong thing.
  CHECK(message.find("cannot speak HTTPS") == std::string::npos);
  CHECK(message.find("VLLM_CPP_OPENSSL") == std::string::npos);
  CHECK(message.find("VLLM_CPP_BUILD_BORINGSSL") == std::string::npos);
#else
  CHECK(message.find("VLLM_CPP_HF_DOWNLOAD") != std::string::npos);
  CHECK(message.find("VLLM_CPP_OPENSSL") != std::string::npos);
  CHECK(message.find("VLLM_CPP_BUILD_BORINGSSL") != std::string::npos);
#endif
}

TEST_CASE("downloader: HF_HUB_OFFLINE opens no socket") {
  ByteHub hub;
  TempDir dir;
  HfDownloadOptions opts = DownloadOptionsFor();
  opts.hub.offline = true;

  const int before = hub.request_count();
  const std::string message = MessageOf([&] {
    HubDownloadFile(hub.url(), dir.path() / "blob", std::nullopt,
                    HfFileShape::kOpaque, opts);
  });
  INFO("refusal: " << message);
  CHECK(message.find("HF_HUB_OFFLINE") != std::string::npos);
  CHECK(hub.request_count() == before);
}

TEST_CASE("downloader: SIGINT cancels the transfer and KEEPS the partial file") {
  ByteHub hub;
  TempDir dir;
  const fs::path dest = dir.path() / "blob";
  // The runner installs its OWN SIGINT handler and reports a chained signal as
  // a crashed case, and the production handler chains on purpose so that a
  // second Ctrl-C still ends the process. Parking SIGINT on SIG_IGN for the
  // duration makes the chain a no-op, so the REAL handler is the thing under
  // test and the runner is not signalled. The previous handler is put back
  // before the case returns.
  void (*runner_handler)(int) = std::signal(SIGINT, SIG_IGN);
  HfInstallDownloadInterruptHandler();
  std::raise(SIGINT);
  CHECK(vllm::transformers_utils::HfDownloadInterrupted());

  const std::string message = MessageOf([&] {
    HubDownloadFile(hub.url(), dest, hub.body().size(), HfFileShape::kOpaque,
                    DownloadOptionsFor());
  });
  HfResetDownloadInterrupt();
  std::signal(SIGINT, runner_handler);
  INFO("refusal: " << message);
  REQUIRE_FALSE(message.empty());
  CHECK(message.find("SIGINT") != std::string::npos);
  CHECK_FALSE(fs::exists(dest));
  // The partial file survives, so the next run resumes rather than starting
  // again. That is the whole point of cancelling rather than deleting.
  CHECK(fs::exists(fs::path(dest.string() + ".incomplete")));
}

TEST_CASE("downloader: the per-repository lock is taken beside the repository") {
  TempDir dir;
  const fs::path repo = dir.path() / "models--org--repo";
  {
    const HfRepoLock lock(repo);
    CHECK(lock.held());
    CHECK(lock.path() == fs::path(repo.string() + ".lock"));
    // It is NOT inside the repository directory, so the cache walk never has to
    // filter it out of a snapshot listing.
    CHECK(lock.path().parent_path() == dir.path());
  }
  // An empty repository path is a host with no cache. It takes no lock rather
  // than locking a file named by the empty string.
  const HfRepoLock none{fs::path()};
  CHECK_FALSE(none.held());
}

TEST_CASE("downloader: HfVerifyFileShape refuses a truncated safetensors on disk") {
  TempDir dir;
  const fs::path file = dir.path() / "x.safetensors";
  const std::string blob = SafetensorsBlob(64);
  WriteAll(file, blob.substr(0, blob.size() - 8));
  const std::string message =
      MessageOf([&] { HfVerifyFileShape(file, HfFileShape::kSafetensors); });
  INFO("refusal: " << message);
  REQUIRE_FALSE(message.empty());

  WriteAll(file, blob);
  HfVerifyFileShape(file, HfFileShape::kSafetensors);  // no throw
  // An opaque file is never opened, whatever it holds.
  WriteAll(file, "not a safetensors at all");
  HfVerifyFileShape(file, HfFileShape::kOpaque);
}
