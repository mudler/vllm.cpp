// See include/vllm/transformers_utils/downloader.h for the llama.cpp `b10451`
// anchors this mirrors and for the ONE behavior it deliberately does not port.
#include "vllm/transformers_utils/downloader.h"

#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

#include <httplib/httplib.h>
#include <nlohmann/json.hpp>

#include "vllm/model_executor/model_loader/gguf_reader.h"

namespace vllm {
namespace transformers_utils {

namespace fs = std::filesystem;
using nlohmann::json;

namespace {

// A transfer follows at most this many redirects. HuggingFace answers a
// `resolve` address with one hop to the content delivery network; the budget
// exists so a mirror that loops cannot spin forever.
constexpr int kMaxRedirects = 5;

// Progress is printed at most this often, in transferred bytes. A line per
// chunk would drown the terminal on a 60 GB shard.
constexpr uint64_t kProgressStrideBytes = 8ull * 1024 * 1024;

// `SIGINT` cancellation. `volatile sig_atomic_t` is the only thing a handler may
// touch, so the flag is that and nothing else; every decision the flag drives is
// taken by the transfer loop, outside the handler.
volatile std::sig_atomic_t g_interrupted = 0;
bool g_handler_installed = false;
void (*g_previous_handler)(int) = nullptr;

extern "C" void HfDownloadSignalHandler(int signal_number) {
  g_interrupted = 1;
  // Chain to whatever was installed before, so `Ctrl-C` still ends the process
  // rather than only ending the download. A handler that swallowed the signal
  // would leave the operator pressing it a second time to no effect.
  if (g_previous_handler != nullptr && g_previous_handler != SIG_DFL &&
      g_previous_handler != SIG_IGN) {
    g_previous_handler(signal_number);
  }
}

std::string HeaderOrEmpty(const httplib::Response& res, const char* name) {
  return res.has_header(name) ? res.get_header_value(name) : std::string();
}

// `"abc"` and `W/"abc"` both name the entity `abc`. The quotes and the weakness
// marker are transport spelling, and a blob file must not be named after them.
std::string NormalizeEtag(std::string etag) {
  if (etag.rfind("W/", 0) == 0) etag = etag.substr(2);
  if (etag.size() >= 2 && etag.front() == '"' && etag.back() == '"') {
    etag = etag.substr(1, etag.size() - 2);
  }
  return etag;
}

std::optional<uint64_t> ParseLength(const std::string& text) {
  if (text.empty()) return std::nullopt;
  for (const char c : text) {
    if (c < '0' || c > '9') return std::nullopt;
  }
  try {
    return static_cast<uint64_t>(std::stoull(text));
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

// `bytes 100-4095/4096` -> first byte 100, total 4096. Either field is empty
// when the header did not carry it or did not parse.
struct ContentRange {
  std::optional<uint64_t> first;
  std::optional<uint64_t> total;
};

ContentRange ParseContentRange(const std::string& value) {
  ContentRange out;
  const size_t space = value.find(' ');
  if (space == std::string::npos) return out;
  const std::string spec = value.substr(space + 1);
  const size_t dash = spec.find('-');
  const size_t slash = spec.find('/');
  if (dash == std::string::npos || slash == std::string::npos || dash > slash) {
    return out;
  }
  out.first = ParseLength(spec.substr(0, dash));
  const std::string total = spec.substr(slash + 1);
  if (total != "*") out.total = ParseLength(total);
  return out;
}

// One client aimed at `url`'s authority, with the offline and no-TLS refusals
// already applied. `path` receives the request target that goes with it.
httplib::Client MakeClient(const std::string& url, const HfHubOptions& opts,
                           std::string* path) {
  HfRefuseHttpsWithoutTls(url);
  const HfParsedUrl parts = HfParseUrl(url);
  *path = parts.path;
  httplib::Client client(parts.scheme + "://" + HfFormatHost(parts.host) + ":" +
                         std::to_string(parts.port));
  // NO `set_follow_location`. httplib copies the whole request, headers
  // included, when it follows a redirect, so following one here would hand the
  // bearer token to whatever host the answer names. The hops are taken by hand
  // below and the token is dropped at the first of them.
  client.set_connection_timeout(opts.connect_timeout_seconds, 0);
  client.set_read_timeout(opts.read_timeout_seconds, 0);
  return client;
}

httplib::Headers BaseHeaders(const HfHubOptions& opts, bool send_token) {
  httplib::Headers headers = {{"User-Agent", "vllm.cpp"}};
  if (send_token && !opts.token.empty()) {
    headers.emplace("Authorization", "Bearer " + opts.token);
  }
  return headers;
}

bool IsRedirect(int status) {
  return status == 301 || status == 302 || status == 303 || status == 307 ||
         status == 308;
}

void RefuseOffline(const std::string& url, const HfHubOptions& opts) {
  if (!opts.offline) return;
  throw std::runtime_error(
      "vllm.cpp: HF_HUB_OFFLINE is set, so " + url +
      " cannot be fetched. Unset HF_HUB_OFFLINE, or point HF_HOME at a cache "
      "that already holds the file.");
}

void RefuseStatus(int status, const std::string& url) {
  if (status == 401 || status == 403) {
    throw std::runtime_error(
        "vllm.cpp: HuggingFace refused " + url + " with HTTP " +
        std::to_string(status) +
        ". The repository is private or gated. Set HF_TOKEN (or HF_TOKEN_PATH) "
        "to a token that has been granted access to it.");
  }
  if (status == 404) {
    throw std::runtime_error("vllm.cpp: HuggingFace answered HTTP 404 for " +
                             url +
                             ". The repository, the revision or the file does "
                             "not exist.");
  }
  throw std::runtime_error("vllm.cpp: HuggingFace answered HTTP " +
                           std::to_string(status) + " for " + url);
}

std::string HumanBytes(uint64_t bytes) {
  static const char* kUnits[] = {"B", "KiB", "MiB", "GiB", "TiB"};
  double value = static_cast<double>(bytes);
  int unit = 0;
  while (value >= 1024.0 && unit < 4) {
    value /= 1024.0;
    unit += 1;
  }
  std::ostringstream out;
  out << std::fixed << std::setprecision(unit == 0 ? 0 : 1) << value << ' '
      << kUnits[unit];
  return out.str();
}

// The safetensors completeness proof, and the whole reason this row does not
// accept a remote length field as evidence: this arithmetic is answered by the
// file itself.
void VerifySafetensors(const fs::path& file) {
  std::ifstream in(file, std::ios::binary);
  if (!in) {
    throw std::runtime_error("vllm.cpp: cannot read " + file.string() +
                             " to check that it is a complete safetensors file");
  }
  std::error_code ec;
  const uint64_t file_size = static_cast<uint64_t>(fs::file_size(file, ec));
  if (ec) {
    throw std::runtime_error("vllm.cpp: cannot size " + file.string() + ": " +
                             ec.message());
  }
  if (file_size < 8) {
    throw std::runtime_error("vllm.cpp: " + file.string() + " is " +
                             std::to_string(file_size) +
                             " bytes, which is shorter than a safetensors "
                             "header length field");
  }
  unsigned char length_bytes[8] = {0};
  in.read(reinterpret_cast<char*>(length_bytes), 8);
  uint64_t header_len = 0;
  for (int i = 7; i >= 0; --i) {
    header_len = (header_len << 8) | static_cast<uint64_t>(length_bytes[i]);
  }
  if (header_len > file_size - 8) {
    throw std::runtime_error(
        "vllm.cpp: " + file.string() + " declares a " +
        std::to_string(header_len) + " byte safetensors header, which does not "
        "fit in its " + std::to_string(file_size) + " bytes");
  }
  std::string header(static_cast<size_t>(header_len), '\0');
  in.read(header.data(), static_cast<std::streamsize>(header_len));
  if (!in) {
    throw std::runtime_error("vllm.cpp: " + file.string() +
                             " ends inside its safetensors header");
  }
  json doc;
  try {
    doc = json::parse(header);
  } catch (const json::exception& e) {
    throw std::runtime_error("vllm.cpp: " + file.string() +
                             " has a safetensors header that is not JSON: " +
                             e.what());
  }
  if (!doc.is_object()) {
    throw std::runtime_error("vllm.cpp: " + file.string() +
                             " has a safetensors header that is not an object");
  }
  uint64_t data_end = 0;
  for (const auto& [name, entry] : doc.items()) {
    if (name == "__metadata__" || !entry.is_object()) continue;
    if (!entry.contains("data_offsets") || !entry["data_offsets"].is_array() ||
        entry["data_offsets"].size() != 2 ||
        !entry["data_offsets"][1].is_number_unsigned()) {
      continue;
    }
    data_end = std::max(data_end, entry["data_offsets"][1].get<uint64_t>());
  }
  const uint64_t expected = 8 + header_len + data_end;
  if (expected != file_size) {
    throw std::runtime_error(
        "vllm.cpp: " + file.string() + " is not a complete safetensors file: 8 + " +
        std::to_string(header_len) + " header bytes + " +
        std::to_string(data_end) + " data bytes is " +
        std::to_string(expected) + ", and the file is " +
        std::to_string(file_size) + " bytes");
  }
}

// The GGUF proof runs through `vllm::GgufFile::Open`, which reads the magic and
// the version, caps the counts before it allocates, and validates EVERY tensor
// span against the real file size. That is the data-end check, and it is
// already the tree's one GGUF header reader. A second parser here would be a
// second thing to keep correct.
void VerifyGguf(const fs::path& file) {
  try {
    const vllm::GgufFile gguf = vllm::GgufFile::Open(file.string());
    if (gguf.Tensors().empty()) {
      throw std::runtime_error("it declares no tensor");
    }
  } catch (const std::exception& e) {
    throw std::runtime_error("vllm.cpp: " + file.string() +
                             " is not a complete GGUF file: " + e.what());
  }
}

}  // namespace

void HfInstallDownloadInterruptHandler() {
  if (g_handler_installed) return;
  g_previous_handler = std::signal(SIGINT, HfDownloadSignalHandler);
  g_handler_installed = true;
}

bool HfDownloadInterrupted() { return g_interrupted != 0; }

void HfResetDownloadInterrupt() { g_interrupted = 0; }

HfFileShape HfShapeForPath(const std::string& path) {
  const fs::path p(path);
  const std::string extension = p.extension().string();
  if (extension == ".safetensors") return HfFileShape::kSafetensors;
  if (extension == ".gguf") return HfFileShape::kGguf;
  return HfFileShape::kOpaque;
}

void HfVerifyFileShape(const fs::path& file, HfFileShape shape) {
  switch (shape) {
    case HfFileShape::kSafetensors:
      VerifySafetensors(file);
      return;
    case HfFileShape::kGguf:
      VerifyGguf(file);
      return;
    case HfFileShape::kOpaque:
      return;
  }
}

HfRepoLock::HfRepoLock(const fs::path& repo_path) {
  if (repo_path.empty()) return;
#if !defined(_WIN32)
  std::error_code ec;
  fs::create_directories(repo_path.parent_path(), ec);
  path_ = repo_path;
  path_ += ".lock";
  fd_ = ::open(path_.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0666);
  if (fd_ < 0) return;
  // BLOCKING, deliberately. vLLM's `weight_utils.py:506` blocks too, and the
  // alternative is a second process deciding the model is unavailable when it
  // is merely being fetched by the first.
  if (::flock(fd_, LOCK_EX) != 0) {
    ::close(fd_);
    fd_ = -1;
  }
#else
  // No advisory-lock path on this platform yet. `held()` reports false and the
  // download still runs, because refusing to fetch a model because a lock could
  // not be taken is worse than the race the lock prevents.
  (void)repo_path;
#endif
}

HfRepoLock::~HfRepoLock() {
#if !defined(_WIN32)
  if (fd_ >= 0) {
    ::flock(fd_, LOCK_UN);
    ::close(fd_);
  }
#endif
}

HfRemoteFile HubProbeFile(const std::string& url, const HfHubOptions& opts) {
  RefuseOffline(url, opts);

  HfRemoteFile info;
  std::string current = url;
  bool send_token = true;
  for (int hop = 0; hop <= kMaxRedirects; ++hop) {
    std::string path;
    httplib::Client client = MakeClient(current, opts, &path);
    const httplib::Result res = client.Head(path, BaseHeaders(opts, send_token));
    if (!res) {
      throw std::runtime_error("vllm.cpp: cannot reach " + current + ": " +
                               httplib::to_string(res.error()));
    }
    // HuggingFace answers the `resolve` address with a redirect and puts the
    // content entity tag on THAT answer, so the first tag seen wins and a later
    // hop cannot overwrite it with the tag of a storage object.
    if (info.etag.empty()) {
      std::string etag = HeaderOrEmpty(*res, "X-Linked-Etag");
      if (etag.empty()) etag = HeaderOrEmpty(*res, "ETag");
      info.etag = NormalizeEtag(etag);
    }
    if (!info.size.has_value()) {
      std::string length = HeaderOrEmpty(*res, "X-Linked-Size");
      if (length.empty()) length = HeaderOrEmpty(*res, "Content-Length");
      info.size = ParseLength(length);
    }
    if (HeaderOrEmpty(*res, "Accept-Ranges") == "bytes") {
      info.accepts_ranges = true;
    }
    if (IsRedirect(res->status)) {
      const std::string location = HeaderOrEmpty(*res, "Location");
      if (location.empty()) {
        throw std::runtime_error("vllm.cpp: " + current + " answered HTTP " +
                                 std::to_string(res->status) +
                                 " with no Location header");
      }
      current = location;
      // The credential stops at the host the caller named.
      send_token = false;
      continue;
    }
    if (res->status != 200) RefuseStatus(res->status, current);
    return info;
  }
  throw std::runtime_error("vllm.cpp: " + url + " redirected more than " +
                           std::to_string(kMaxRedirects) + " times");
}

HfDownloadResult HubDownloadFile(const std::string& url, const fs::path& dest,
                                 const std::optional<uint64_t>& expected_size,
                                 HfFileShape shape,
                                 const HfDownloadOptions& opts) {
  HfDownloadResult result;
  std::error_code ec;

  // A file already at the destination name is a cache hit ONLY when it proves
  // itself. The name was given to it by an earlier run that renamed it after
  // the same proof, so this re-check is cheap and it catches an entry a third
  // party truncated under the cache.
  if (fs::is_regular_file(dest, ec)) {
    const uint64_t size = static_cast<uint64_t>(fs::file_size(dest, ec));
    if (!ec && (!expected_size.has_value() || *expected_size == size)) {
      HfVerifyFileShape(dest, shape);
      result.file_size = size;
      result.already_present = true;
      return result;
    }
  }

  RefuseOffline(url, opts.hub);
  fs::create_directories(dest.parent_path(), ec);

  fs::path temp = dest;
  temp += ".incomplete";

  uint64_t offset = 0;
  if (fs::is_regular_file(temp, ec)) {
    const uint64_t partial = static_cast<uint64_t>(fs::file_size(temp, ec));
    if (!ec) offset = partial;
  }

  const HfRemoteFile probe = HubProbeFile(url, opts.hub);
  std::optional<uint64_t> total = probe.size.has_value() ? probe.size : expected_size;
  if (probe.size.has_value() && expected_size.has_value() &&
      *probe.size != *expected_size) {
    throw std::runtime_error(
        "vllm.cpp: the tree listing gives " + url + " a size of " +
        std::to_string(*expected_size) + " bytes and the file answers " +
        std::to_string(*probe.size) +
        " bytes. The two disagree, so neither can be used to prove the "
        "transfer complete.");
  }

  if (offset > 0) {
    // A partial file that is already the whole file needs no request, and a
    // partial file the server cannot resume is started again rather than
    // appended to.
    const bool complete = total.has_value() && offset == *total;
    if (!complete && (!probe.accepts_ranges || (total.has_value() && offset > *total))) {
      fs::remove(temp, ec);
      offset = 0;
    }
  }

  const bool resume = offset > 0;
  result.resumed = resume;
  result.etag = probe.etag;

  if (!total.has_value() || offset < *total) {
    std::ofstream out(temp, std::ios::binary |
                                (resume ? std::ios::app : std::ios::trunc));
    if (!out) {
      throw std::runtime_error("vllm.cpp: cannot open " + temp.string() +
                               " for writing");
    }

    std::string current = url;
    bool send_token = true;
    bool done = false;
    for (int hop = 0; hop <= kMaxRedirects && !done; ++hop) {
      std::string path;
      httplib::Client client = MakeClient(current, opts.hub, &path);
      httplib::Headers headers = BaseHeaders(opts.hub, send_token);
      if (resume) {
        headers.emplace("Range", "bytes=" + std::to_string(offset) + "-");
      }

      int status = 0;
      std::string location;
      std::optional<uint64_t> body_length;
      ContentRange range;
      uint64_t received = 0;
      uint64_t next_report = kProgressStrideBytes;
      bool cancelled = false;

      const httplib::Result res = client.Get(
          path, headers,
          [&](const httplib::Response& response) {
            status = response.status;
            location = HeaderOrEmpty(response, "Location");
            body_length = ParseLength(HeaderOrEmpty(response, "Content-Length"));
            range = ParseContentRange(HeaderOrEmpty(response, "Content-Range"));
            // A RESUMED transfer accepts ONLY 206, and the refusal is taken
            // HERE rather than after the call, because returning true would
            // hand the body to the receiver below and the first `offset` bytes
            // would already be on disk twice by the time the throw ran.
            // Measured: this suite saw a 12 byte partial file grow to 48 bytes
            // under a refusal that still reported the right reason.
            if (resume) return status == 206;
            return status == 200;
          },
          [&](const char* data, size_t length) {
            if (HfDownloadInterrupted()) {
              cancelled = true;
              return false;
            }
            out.write(data, static_cast<std::streamsize>(length));
            if (!out) return false;
            received += length;
            if (opts.verbose && received >= next_report) {
              next_report = received + kProgressStrideBytes;
              std::cerr << "download: " << dest.filename().string() << ' '
                        << HumanBytes(offset + received);
              if (total.has_value()) {
                std::cerr << " / " << HumanBytes(*total);
              }
              std::cerr << std::endl;
            }
            return true;
          });

      if (cancelled) {
        out.close();
        throw std::runtime_error(
            "vllm.cpp: the download of " + url +
            " was cancelled by SIGINT. The partial file is kept at " +
            temp.string() + ", so the next run resumes rather than starting "
            "again.");
      }

      if (IsRedirect(status)) {
        if (location.empty()) {
          throw std::runtime_error("vllm.cpp: " + current + " answered HTTP " +
                                   std::to_string(status) +
                                   " with no Location header");
        }
        current = location;
        send_token = false;
        continue;
      }

      // THE DIVERGENCE FROM llama.cpp `common/download.cpp:222-235 @ b10451`,
      // which warns and continues here. A 200 answer to a range request carries
      // the WHOLE body, and appending a whole body to a partial file writes the
      // first `offset` bytes twice. See downloader.h.
      if (resume && status == 200) {
        out.close();
        throw std::runtime_error(
            "vllm.cpp: " + current + " was asked for bytes " +
            std::to_string(offset) +
            "- and answered HTTP 200 with the whole file instead of HTTP 206. "
            "Appending that body to the partial file at " + temp.string() +
            " would write the first " + std::to_string(offset) +
            " bytes twice and silently corrupt the weight, so the transfer is "
            "refused. Delete that file and run again to fetch it from the "
            "start.");
      }
      if (status != 200 && status != 206) {
        out.close();
        RefuseStatus(status, current);
      }
      if (resume && range.first.has_value() && *range.first != offset) {
        out.close();
        throw std::runtime_error(
            "vllm.cpp: " + current + " was asked for bytes " +
            std::to_string(offset) + "- and answered with a range starting at " +
            std::to_string(*range.first) + ". The two do not line up, so the "
            "transfer is refused rather than written to the wrong offset.");
      }
      if (range.total.has_value() && !total.has_value()) total = range.total;

      // THE BYTE COUNT COMES FIRST, ahead of the transport's own verdict. A
      // body shorter than the length its own answer declared is a TRUNCATED
      // transfer, and httplib reports some of these as a read error and
      // returns others as a successful short body. With this check placed
      // AFTER `!res`, the transport's generic message won every truncation
      // this suite could construct, and deleting the check left the truncation
      // case GREEN, so the guarantee had no test. Ordered this way the specific
      // diagnosis wins and the case measures the check rather than httplib.
      if (body_length.has_value() && received != *body_length) {
        out.close();
        throw std::runtime_error(
            "vllm.cpp: " + current + " declared a body of " +
            std::to_string(*body_length) + " bytes and delivered " +
            std::to_string(received) +
            ". The transfer is truncated and is refused.");
      }
      if (!res) {
        out.close();
        throw std::runtime_error(
            "vllm.cpp: the transfer of " + current + " ended after " +
            std::to_string(offset + received) + " bytes: " +
            httplib::to_string(res.error()));
      }
      if (!out) {
        out.close();
        throw std::runtime_error("vllm.cpp: writing " + temp.string() +
                                 " failed after " +
                                 std::to_string(offset + received) + " bytes");
      }
      result.bytes_written = received;
      done = true;
    }

    out.close();
    if (!done) {
      throw std::runtime_error("vllm.cpp: " + url + " redirected more than " +
                               std::to_string(kMaxRedirects) + " times");
    }
  }

  const uint64_t written = static_cast<uint64_t>(fs::file_size(temp, ec));
  if (ec) {
    throw std::runtime_error("vllm.cpp: cannot size " + temp.string() + ": " +
                             ec.message());
  }
  if (total.has_value() && written != *total) {
    throw std::runtime_error(
        "vllm.cpp: " + url + " is " + std::to_string(*total) +
        " bytes and the transfer left " + std::to_string(written) +
        " bytes in " + temp.string() + ". The transfer is refused.");
  }

  // The STRUCTURAL proof, before the rename and never after it. The destination
  // name is what a later run reads as a cache hit, so a file may only take that
  // name once it has proven itself from its own bytes.
  HfVerifyFileShape(temp, shape);

  fs::remove(dest, ec);
  fs::rename(temp, dest, ec);
  if (ec) {
    throw std::runtime_error("vllm.cpp: cannot move " + temp.string() + " to " +
                             dest.string() + ": " + ec.message());
  }
  result.file_size = written;
  return result;
}

}  // namespace transformers_utils
}  // namespace vllm
