// vllm.cpp original: the HuggingFace hub protocol, authentication and endpoint.
//
// vLLM reaches the hub through `huggingface_hub`, so the wire protocol itself
// has no vLLM source to mirror. The structural reference is the secondary
// oracle llama.cpp at stock tag `b10451` (commit
// `10bf611e533d81f739128304991c5e133c6aebd8`):
//   - endpoint resolution, `common/common.cpp:1530-1542`
//   - reference to commit, `common/hf-cache.cpp:229-289`
//   - recursive file listing, `common/hf-cache.cpp:291-360`
//   - byte address form, `common/hf-cache.cpp:347`
//
// Three integrity rules deliberately do NOT port llama.cpp, and they are the
// reason this file exists rather than a copy of that one. They are recorded in
// `.agents/specs/hf-model-download.md`. See `HfFile::oid` and
// `HubListRepoFiles`.
//
// ENG-HF-MODEL-DOWNLOAD, issue #1280.
#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace vllm {
namespace transformers_utils {

// Everything the hub calls read from the environment, resolved once so a caller
// can override any field in a test without touching the environment.
struct HfHubOptions {
  // `HF_ENDPOINT`, else `https://huggingface.co/`. Always ends in '/'.
  std::string endpoint;
  // `HF_TOKEN`, else the first line of the file named by `HF_TOKEN_PATH`.
  std::string token;
  // `HF_HUB_OFFLINE`. Resolve from the cache and open no socket: every call in
  // this header refuses rather than reaching the hub, and names the variable.
  bool offline = false;
  // Where `models--org--repo` lives. Defaults to `HfHubCacheDir()`.
  std::filesystem::path hub_dir;
  int connect_timeout_seconds = 10;
  int read_timeout_seconds = 30;
};

// Read `HF_ENDPOINT`, `HF_TOKEN`, `HF_TOKEN_PATH`, `HF_HUB_OFFLINE` and the
// cache directory.
HfHubOptions HfHubOptionsFromEnv();

// One entry of a recursive tree listing.
struct HfFile {
  // Repository-relative path, always a real subpath of the snapshot directory.
  std::string path;
  // The CONTENT size, read from the entry's top-level `size` and, when the
  // listing omits that, from `lfs.size`. Never `lfs.pointerSize`, which is the
  // size of the pointer file and is nearly equal for every shard.
  //
  // EMPTY when the listing reported no size at all. It is an optional rather
  // than a zero, because a zero-byte file and an unreported size are different
  // facts and `0` spells both. A caller that sizes a byte range, a resume
  // offset, or a completeness check from this field has to be able to tell
  // them apart, and the size rule on `HubListRepoFiles` compares only the
  // sizes the listing actually reported.
  std::optional<uint64_t> size;
  // The large-file-storage object identifier, or empty.
  //
  // EMPTY UNLESS THE LISTING REQUEST CARRIED A TOKEN. On 17 August 2026 the
  // tree API answered an unauthenticated caller on the gated repository
  // `Lightricks/LTX-2.5` with an `lfs.oid` of one character repeated 64 times,
  // identical for all 14 large-file-storage files. That value passes
  // llama.cpp's `is_valid_oid` at `common/hf-cache.cpp:161`, which accepts any
  // 40 or 64 character hexadecimal string. An unauthenticated identifier
  // therefore proves nothing and is dropped rather than used as a blob name.
  //
  // That drop is a decision about USE, and it is the only thing the token
  // governs. The two integrity rules on `HubListRepoFiles` run whoever asked.
  //
  // LOWER CASE whatever the listing spelled, so a value that becomes a blob
  // file name names one file on a case-sensitive file system and on a
  // case-insensitive one alike.
  std::string oid;
  // `{endpoint}{repo}/resolve/{commit}/{path}`.
  std::string url;
};

// GET `{endpoint}api/models/{repo}/refs`, and return the commit the reference
// names. `revision` empty means the default branch, which is `main` when the
// listing offers it. A 40 character hexadecimal `revision` is already a commit
// and is returned without a request.
//
// A redirect is NOT followed. See the client in hf_hub.cpp: httplib forwards
// the request headers to the redirect target, and the bearer token must not
// reach a host the caller never named.
//
// Throws std::runtime_error on a refusal. HTTP 401 and 403 name the repository
// and `HF_TOKEN`, and `opts.offline` refuses before any socket is opened.
std::string HubResolveRefToCommit(const std::string& repo_id,
                                  const std::string& revision,
                                  const HfHubOptions& opts);

// GET `{endpoint}api/models/{repo}/tree/{commit}?recursive=true`.
//
// `commit` must be a 40 character hexadecimal commit, because every call after
// the reference resolution names one; asking for a branch here would let a
// moving `main` change what a second run loads.
//
// Two integrity rules REFUSE the listing rather than filter it, because a hub
// answering something other than the truth about one file is not a source the
// rest of the answer can be trusted from. Neither depends on the token.
//
//  1. Two entries carry one object identifier and disagree on size. A content
//     hash cannot name two sizes, so this is always a broken instrument. The
//     rule asks nothing about the PATH: one path listed twice at two different
//     sizes is self contradictory whichever entry is believed, and it is
//     refused too.
//  2. An identifier is one character repeated, for example 'a' 64 times. No
//     content hash produces that, and it is the value measured on gated
//     `Lightricks/LTX-2.5` on 17 August 2026.
//
// Two entries that share an identifier AND a size are ACCEPTED. That is
// duplicate content, which is legitimate: `lfs.oid` is the sha256 of the
// contents and the plain `oid` is the git blob sha1, so two byte-identical
// files in one repository share one identifier by construction. Refusing that
// rejected a valid repository, which is why the earlier any-duplicate rule was
// replaced. See `.agents/specs/hf-model-download.md`.
//
// An identifier that is neither 40 nor 64 hexadecimal characters is dropped and
// its file is kept, because the file is still addressable by path.
//
// Every identifier is FOLDED to lower case before either rule reads it and
// before it reaches `HfFile::oid`. Hexadecimal is case-insensitive and the hub
// and git both emit lower case, so `AB23...` and `ab23...` are one value. A
// listing that spelled them differently used to slip past rule 1, because the
// two spellings were two keys.
//
// Throws std::runtime_error on a refusal.
std::vector<HfFile> HubListRepoFiles(const std::string& repo_id,
                                     const std::string& commit,
                                     const HfHubOptions& opts);

// `{endpoint}{repo}/resolve/{commit}/{path}`.
std::string HubFileUrl(const std::string& endpoint, const std::string& repo_id,
                       const std::string& commit, const std::string& path);

// Resolve the reference to a commit ONCE and record it, so that every later
// call in the same run names that commit and a moving `main` cannot change what
// a second run loads.
//
// The cached `refs/{ref}` file is the first probe, so a warm cache issues no
// request at all. Offline never opens a socket, and a cold cache under offline
// throws a std::runtime_error naming the repository directory that was
// searched.
std::string HubResolveCommitCached(const std::string& repo_id,
                                   const std::string& revision,
                                   const HfHubOptions& opts);

// A parsed hypertext address, and the two checks every call in this file and
// in `downloader.h` runs before it opens a socket. They are declared HERE so
// the byte transport speaks ONE address grammar with the protocol half rather
// than growing a second one that drifts.
//
// Mirrors llama.cpp `common/http.h:33-98 @ b10451`, narrowed to what a hub
// address needs: no user information, because a hub endpoint carries none.
struct HfParsedUrl {
  std::string scheme;
  std::string host;
  int port = 0;
  std::string path;  // always begins with '/'
};

// Throws std::runtime_error naming `url` on a missing scheme, an unterminated
// bracketed IPv6 authority, an unsupported scheme, or an empty host.
HfParsedUrl HfParseUrl(const std::string& url);

// `[host]` for an IPv6 literal, `host` otherwise. What httplib's scheme-host-port
// constructor needs.
std::string HfFormatHost(const std::string& host);

// Throw when `url` is `https` and this build carries no transport layer
// security, with a message that NAMES the three build options. A build where
// the option resolved OFF otherwise fails with a connection error that reads
// like a network fault.
void HfRefuseHttpsWithoutTls(const std::string& url);

// True when `repo_id` has the shape the hub accepts: base characters
// `[A-Za-z0-9_]`, the special characters `/.-` only between base characters,
// and exactly one '/'. Mirrors llama.cpp `common/hf-cache.cpp:121-142`.
bool IsValidHfRepoId(const std::string& repo_id);

}  // namespace transformers_utils
}  // namespace vllm
