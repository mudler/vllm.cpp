// vllm.cpp original: byte transport for a HuggingFace file. Probe, resume,
// structural proof, cross-process lock and progress.
//
// vLLM reaches the bytes through `huggingface_hub`, so the wire behavior has no
// vLLM source to mirror and the structural reference is the secondary oracle
// llama.cpp at stock tag `b10451` (commit
// `10bf611e533d81f739128304991c5e133c6aebd8`):
//   - size, entity tag and range probe, `common/download.cpp:321-349`
//   - range resume, `common/download.cpp:222-235`
// The cross-process lock mirrors vLLM `weight_utils.py:506` at pin
// `5559679229`, which takes one lock per model before it downloads.
//
// ONE BEHAVIOR DELIBERATELY DIVERGES FROM llama.cpp, and it is the reason this
// file is not a port of that one. See `HubDownloadFile`.
//
// ENG-HF-MODEL-DOWNLOAD W3, issue #1280.
#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

#include "vllm/transformers_utils/hf_hub.h"

namespace vllm {
namespace transformers_utils {

// What a `HEAD` probe learned about one remote file. Mirrors llama.cpp
// `common/download.cpp:321-349 @ b10451`, which asks the same three questions.
struct HfRemoteFile {
  // `Content-Length`, and EMPTY when the answer carried none. It is an optional
  // for the reason `HfFile::size` is one: a zero-byte file and an unanswered
  // size are different facts, and `0` spells both. Every caller that sizes a
  // byte range, a resume offset or a completeness check from it has to be able
  // to tell them apart.
  std::optional<uint64_t> size;
  // The entity tag with its quotes and any `W/` prefix removed, or empty.
  std::string etag;
  // `Accept-Ranges: bytes`. A server that does not say so is never asked to
  // resume, because a range it ignores is the failure mode below.
  bool accepts_ranges = false;
};

// `HEAD {url}`, following a redirect to the content delivery network. The
// bearer token is sent to the host the caller named and to NO redirect target,
// for the reason `hf_hub.cpp`'s client does not set `set_follow_location`.
//
// Throws std::runtime_error on a refusal, and refuses before opening a socket
// when `opts.offline` is set.
HfRemoteFile HubProbeFile(const std::string& url, const HfHubOptions& opts);

// The three file shapes this row can prove complete STRUCTURALLY, which is the
// only proof it accepts. An opaque remote field is not one: a hub that is not
// answering the truth about a repository can answer a length field too.
enum class HfFileShape {
  // Nothing about the interior is known. Only the byte count is checked.
  kOpaque,
  // `8 + header_len + max(data_offsets[1]) == file_size`.
  kSafetensors,
  // Magic, version, tensor count, and every tensor span inside the data end.
  kGguf,
};

// The shape a repository-relative path declares by its extension.
HfFileShape HfShapeForPath(const std::string& path);

// Prove `file` complete for `shape`. Throws std::runtime_error naming the file
// and the part that did not add up. `kOpaque` returns without reading a byte.
void HfVerifyFileShape(const std::filesystem::path& file, HfFileShape shape);

struct HfDownloadOptions {
  HfHubOptions hub;
  // Progress to standard error. The server spells this `--verbose`.
  bool verbose = false;
};

// What one `HubDownloadFile` call did. `bytes_written` counts what THIS call
// transferred, so a resumed transfer reports the tail and a cache hit reports
// zero. That distinction is the only honest way for a test to state either.
struct HfDownloadResult {
  uint64_t bytes_written = 0;
  uint64_t file_size = 0;
  bool resumed = false;
  bool already_present = false;
  std::string etag;
};

// Fetch `url` into `dest`.
//
// The bytes land in `{dest}.incomplete` and the file is renamed to `dest` only
// after `HfVerifyFileShape` passes, so a partial or malformed transfer never
// occupies the name a later run reads as a cache hit.
//
// An existing `{dest}.incomplete` is RESUMED with `Range: bytes=N-`.
//
// THE DIVERGENCE. llama.cpp warns and continues when a range request is
// answered `200` instead of `206` (`common/download.cpp:222-235 @ b10451`,
// "server did not respond with 206 ... restarting download"; the fork's
// variant logs and keeps the handle). We THROW. A `200` carries the WHOLE body,
// and appending a whole body onto a partial file produces a file whose bytes
// are the first N twice. That file has the wrong length, so the structural
// check catches it here, but the same shape on a format with no structural
// proof is a silently corrupt weight, and a token gate cannot see a corrupt
// weight: it still emits tokens. A refusal that names the cause costs one
// re-run. A corrupt shard costs a debugging session that starts at the model.
//
// `expected_size` is the size the TREE LISTING reported, and it is an optional
// because the listing may report none. It is checked against the transferred
// byte count when it has a value and is not invented when it does not.
//
// Cancels on `SIGINT` once `HfInstallDownloadInterruptHandler` has been called:
// the partial file is left in place, so the next run resumes rather than
// starting again.
//
// Throws std::runtime_error on any refusal, and opens no socket under
// `opts.hub.offline`.
HfDownloadResult HubDownloadFile(const std::string& url,
                                 const std::filesystem::path& dest,
                                 const std::optional<uint64_t>& expected_size,
                                 HfFileShape shape,
                                 const HfDownloadOptions& opts);

// One lock per REPOSITORY, held across processes, mirroring vLLM
// `weight_utils.py:506`. Two `vllm-server` processes started at once against
// one cache must not write one blob twice.
//
// The lock file sits BESIDE the repository directory rather than inside it, so
// it is never mistaken for a snapshot entry and never has to be filtered out of
// the cache walk. An empty `repo_path` takes no lock, because a host with no
// cache directory has nothing to serialize on.
class HfRepoLock {
 public:
  explicit HfRepoLock(const std::filesystem::path& repo_path);
  ~HfRepoLock();
  HfRepoLock(const HfRepoLock&) = delete;
  HfRepoLock& operator=(const HfRepoLock&) = delete;

  // True when this object holds a file lock. False when the host has no cache
  // directory, or when the platform has no advisory lock: the download still
  // runs, because refusing to fetch a model because a lock could not be taken
  // is worse than the race the lock prevents.
  bool held() const { return fd_ >= 0; }
  const std::filesystem::path& path() const { return path_; }

 private:
  int fd_ = -1;
  std::filesystem::path path_;
};

// Install the `SIGINT` handler that cancels an in-flight transfer. Idempotent.
// The previous handler is remembered and re-raised, so `Ctrl-C` still ends the
// process rather than only ending the download.
void HfInstallDownloadInterruptHandler();

// True once `SIGINT` arrived after the handler was installed.
bool HfDownloadInterrupted();

// Clear the flag. For tests, and for a caller that handled the cancellation.
void HfResetDownloadInterrupt();

}  // namespace transformers_utils
}  // namespace vllm
