// See include/vllm/transformers_utils/hf_hub.h for the protocol this speaks and
// for the llama.cpp `b10451` anchors it mirrors, and for the three integrity
// rules it deliberately does not port.
#include "vllm/transformers_utils/hf_hub.h"

#include <cstdlib>
#include <fstream>
#include <map>
#include <stdexcept>
#include <utility>

#include <httplib/httplib.h>
#include <nlohmann/json.hpp>

#include "vllm/http_transport_abi.h"
#include "vllm/transformers_utils/hf_cache.h"

namespace vllm {
namespace transformers_utils {

namespace fs = std::filesystem;
using nlohmann::json;

namespace {

const char* NonEmptyEnv(const char* name) {
  const char* value = std::getenv(name);
  return (value != nullptr && value[0] != '\0') ? value : nullptr;
}

// Both cases are accepted, which mirrors llama.cpp's `is_valid_oid`
// (`common/hf-cache.cpp:161 @ b10451`). An OBJECT IDENTIFIER is therefore
// folded to one case by its caller before anything keys on it, because two
// spellings of one identifier are one identifier. A COMMIT is deliberately not
// folded: nothing keys an integrity rule on it, so a second spelling costs at
// worst a second snapshot directory rather than a rule that stops firing, and
// neither reference implementation folds it either.
bool IsHexString(const std::string& s, size_t length) {
  if (s.size() != length) return false;
  for (const char c : s) {
    const bool hex = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') ||
                     (c >= 'A' && c <= 'F');
    if (!hex) return false;
  }
  return true;
}

// True when every character of `oid` is the same one, for example 'a' repeated
// 64 times. No content hash produces that value. This is the shape the
// HuggingFace tree API actually served on 17 August 2026 for the gated
// repository `Lightricks/LTX-2.5`, identically for all 14 large-file-storage
// files, and llama.cpp's `is_valid_oid` at `common/hf-cache.cpp:161 @ b10451`
// accepts it because it only asks whether the characters are hexadecimal.
bool IsDegenerateOid(const std::string& oid) {
  return !oid.empty() &&
         oid.find_first_not_of(oid.front()) == std::string::npos;
}

bool IsAlphanumeric(char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
         (c >= '0' && c <= '9');
}

// An entry path arrives from the network and becomes a path component under the
// snapshot directory. It must be relative, must carry no "." or ".." component,
// and must carry no empty component.
bool IsSafeEntryPath(const std::string& path) {
  if (path.empty()) return false;
  const fs::path p(path);
  if (p.is_absolute()) return false;
  for (const fs::path& part : p) {
    const std::string s = part.string();
    if (s.empty() || s == "." || s == "..") return false;
  }
  return true;
}

// RFC 3986 section 3.1: `scheme = ALPHA *( ALPHA / DIGIT / "+" / "-" / "." )`
// followed by ':'. A reference that begins with one is absolute and is used as
// it stands; everything else is resolved against the address that sent it.
bool HasScheme(const std::string& reference) {
  if (reference.empty()) return false;
  const unsigned char first = static_cast<unsigned char>(reference.front());
  const bool alpha = (first >= 'A' && first <= 'Z') || (first >= 'a' && first <= 'z');
  if (!alpha) return false;
  for (size_t i = 1; i < reference.size(); ++i) {
    const char c = reference[i];
    if (c == ':') return true;
    const bool valid = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                       (c >= '0' && c <= '9') || c == '+' || c == '-' || c == '.';
    if (!valid) return false;
  }
  return false;
}

// RFC 3986 section 5.2.4, transcribed. `..` walks a segment off the output
// buffer, which is why the algorithm cannot be written as a single pass over
// the input: `/a/b/../../c` has to end at `/c` and not at `/a/c`.
std::string RemoveDotSegments(const std::string& path) {
  std::string in = path;
  std::string out;
  const auto drop_last_segment = [&out]() {
    const size_t slash = out.rfind('/');
    out = slash == std::string::npos ? std::string() : out.substr(0, slash);
  };
  while (!in.empty()) {
    if (in.rfind("../", 0) == 0) {
      in.erase(0, 3);
    } else if (in.rfind("./", 0) == 0) {
      in.erase(0, 2);
    } else if (in.rfind("/./", 0) == 0) {
      in.replace(0, 3, "/");
    } else if (in == "/.") {
      in = "/";
    } else if (in.rfind("/../", 0) == 0) {
      in.replace(0, 4, "/");
      drop_last_segment();
    } else if (in == "/..") {
      in = "/";
      drop_last_segment();
    } else if (in == "." || in == "..") {
      in.clear();
    } else {
      const size_t next = in.find('/', in.front() == '/' ? 1 : 0);
      if (next == std::string::npos) {
        out += in;
        in.clear();
      } else {
        out += in.substr(0, next);
        in.erase(0, next);
      }
    }
  }
  return out;
}

// GET a JSON document from the hub. `repo_id` appears in every refusal, because
// a message that does not name the repository cannot be acted on.
json ApiGet(const HfHubOptions& opts, const std::string& relative_path,
            const std::string& repo_id) {
  // The offline check sits HERE, on the one function that opens a socket,
  // rather than on each caller. `HubResolveCommitCached` has its own offline
  // refusal, which names the cache directory it searched and fires before this
  // one; every other path through the hub used to reach the network under a
  // setting that forbids it.
  if (opts.offline) {
    throw std::runtime_error(
        "vllm.cpp: HF_HUB_OFFLINE is set, so repository '" + repo_id +
        "' cannot be read from " + opts.endpoint +
        ". Unset HF_HUB_OFFLINE, or point HF_HOME at a cache that already "
        "holds the repository.");
  }

  const HfParsedUrl url = HfParseUrl(opts.endpoint, "HF_ENDPOINT");

  HfRefuseHttpsWithoutTls(opts.endpoint);

  httplib::Client client(url.scheme + "://" + HfFormatHost(url.host) + ":" +
                         std::to_string(url.port));
  // NO `set_follow_location(true)`. httplib copies the whole request, headers
  // included, when it follows a redirect (third_party/httplib/httplib.h:7774)
  // and routes a cross-host redirect through `create_redirect_client` with it
  // (:13537), so following one here would hand the bearer token to whatever
  // host the answer names. llama.cpp does not set it on its API client either
  // (`common/hf-cache.cpp:198-226 @ b10451`). The redirect this row does follow
  // is the content-delivery-network address for the BYTES, which W3 owns and
  // which carries no credential.
  client.set_connection_timeout(opts.connect_timeout_seconds, 0);
  client.set_read_timeout(opts.read_timeout_seconds, 0);

  httplib::Headers headers = {{"User-Agent", "vllm.cpp"},
                              {"Accept", "application/json"}};
  // vLLM hands `huggingface_hub` whatever token the environment holds and lets
  // the hub judge it. llama.cpp additionally requires an `hf_` prefix at
  // `common/hf-cache.cpp:144-155`, which silently drops a valid token issued by
  // a self-hosted mirror, so that check is not ported.
  if (!opts.token.empty()) {
    headers.emplace("Authorization", "Bearer " + opts.token);
  }

  const std::string request_path = url.path + relative_path;
  const httplib::Result res = client.Get(request_path, headers);
  if (!res) {
    throw std::runtime_error("vllm.cpp: cannot reach " + opts.endpoint +
                             " for repository '" + repo_id +
                             "': " + httplib::to_string(res.error()));
  }

  if (res->status == 401 || res->status == 403) {
    throw std::runtime_error(
        "vllm.cpp: HuggingFace refused repository '" + repo_id + "' with HTTP " +
        std::to_string(res->status) +
        ". The repository is private or gated. Set HF_TOKEN (or HF_TOKEN_PATH) "
        "to a token that has been granted access to it.");
  }
  if (res->status == 404) {
    throw std::runtime_error("vllm.cpp: HuggingFace answered HTTP 404 for '" +
                             request_path + "'. Repository '" + repo_id +
                             "' or the revision it was asked for does not "
                             "exist.");
  }
  if (res->status != 200) {
    throw std::runtime_error("vllm.cpp: HuggingFace answered HTTP " +
                             std::to_string(res->status) + " for repository '" +
                             repo_id + "'");
  }

  try {
    return json::parse(res->body);
  } catch (const json::exception& e) {
    throw std::runtime_error("vllm.cpp: HuggingFace answered '" + request_path +
                             "' with a body that is not JSON: " + e.what());
  }
}

// Collect (name, commit) pairs from the "branches" and "tags" arrays, skipping
// any entry whose name would escape the refs directory or whose commit is not a
// commit. `--revision` names a branch or a tag, so both are read.
std::vector<std::pair<std::string, std::string>> CollectRefs(const json& doc) {
  std::vector<std::pair<std::string, std::string>> refs;
  for (const char* key : {"branches", "tags"}) {
    if (!doc.is_object() || !doc.contains(key) || !doc[key].is_array()) continue;
    for (const json& item : doc[key]) {
      if (!item.is_object() || !item.contains("name") ||
          !item["name"].is_string() || !item.contains("targetCommit") ||
          !item["targetCommit"].is_string()) {
        continue;
      }
      const std::string name = item["name"].get<std::string>();
      const std::string commit = item["targetCommit"].get<std::string>();
      if (!IsSafeEntryPath(name) || !IsHexString(commit, 40)) continue;
      refs.emplace_back(name, commit);
    }
  }
  return refs;
}

}  // namespace

HfParsedUrl HfParseUrl(const std::string& url, const std::string& role) {
  HfParsedUrl parts;
  const size_t scheme_end = url.find("://");
  if (scheme_end == std::string::npos) {
    throw std::runtime_error("vllm.cpp: " + role + " '" + url +
                             "' has no scheme; expected http:// or https://");
  }
  parts.scheme = url.substr(0, scheme_end);
  std::string rest = url.substr(scheme_end + 3);

  const size_t slash = rest.find('/');
  if (slash != std::string::npos) {
    parts.host = rest.substr(0, slash);
    parts.path = rest.substr(slash);
  } else {
    parts.host = rest;
    parts.path = "/";
  }

  std::string port_text;
  if (!parts.host.empty() && parts.host.front() == '[') {
    const size_t close = parts.host.find(']');
    if (close == std::string::npos) {
      throw std::runtime_error("vllm.cpp: " + role + " '" + url +
                               "' has an unterminated IPv6 host");
    }
    const std::string after = parts.host.substr(close + 1);
    if (!after.empty() && after.front() == ':') port_text = after.substr(1);
    parts.host = parts.host.substr(1, close - 1);
  } else {
    const size_t colon = parts.host.find(':');
    if (colon != std::string::npos) {
      port_text = parts.host.substr(colon + 1);
      parts.host = parts.host.substr(0, colon);
    }
  }

  if (!port_text.empty()) {
    parts.port = std::stoi(port_text);
  } else if (parts.scheme == "http") {
    parts.port = 80;
  } else if (parts.scheme == "https") {
    parts.port = 443;
  } else {
    throw std::runtime_error("vllm.cpp: " + role + " '" + url +
                             "' uses the unsupported scheme '" + parts.scheme +
                             "'");
  }
  if (parts.host.empty()) {
    throw std::runtime_error("vllm.cpp: " + role + " '" + url +
                             "' has no host");
  }
  return parts;
}

std::string HfResolveUrl(const std::string& base, const std::string& location) {
  // An absolute URL is already resolved. The scheme test is RFC 3986 section
  // 3.1's grammar rather than a search for "://", so that a `Location` of
  // `mailto:x` or a Windows-looking `c:/x` is not mistaken for a path.
  if (HasScheme(location)) return location;

  const size_t scheme_end = base.find("://");
  if (scheme_end == std::string::npos) {
    throw std::runtime_error("vllm.cpp: cannot resolve the redirect target '" +
                             location + "' because the address that sent it, '" +
                             base + "', is not absolute");
  }
  const std::string scheme = base.substr(0, scheme_end);
  const size_t authority_start = scheme_end + 3;
  size_t authority_end = base.find_first_of("/?#", authority_start);
  if (authority_end == std::string::npos) authority_end = base.size();
  // The authority is taken as TEXT rather than re-serialised from a parse, so
  // an address that named no port keeps naming none and the address in a later
  // refusal is the one the hub actually sent.
  const std::string origin =
      scheme + "://" + base.substr(authority_start, authority_end - authority_start);

  if (location.empty()) return base;
  // A network-path reference keeps the base's scheme and nothing else.
  if (location.rfind("//", 0) == 0) return scheme + ":" + location;

  const size_t location_split = location.find_first_of("?#");
  const std::string location_path = location.substr(0, location_split);
  const std::string location_tail =
      location_split == std::string::npos ? std::string()
                                          : location.substr(location_split);

  // An absolute-path reference replaces the base's path outright. THE FORM
  // huggingface.co SENDS.
  if (location.front() == '/') {
    return origin + RemoveDotSegments(location_path) + location_tail;
  }

  std::string base_path = base.substr(authority_end);
  const size_t base_split = base_path.find_first_of("?#");
  if (base_split != std::string::npos) base_path = base_path.substr(0, base_split);
  if (base_path.empty()) base_path = "/";

  // A reference with an empty path keeps the base's path and replaces only what
  // it does carry, per RFC 3986 section 5.3's `defined(R.query)` arm.
  if (location_path.empty()) return origin + base_path + location_tail;

  // A relative-path reference is merged onto the base's DIRECTORY, which is the
  // base path up to and including its last slash.
  const size_t last_slash = base_path.rfind('/');
  const std::string directory =
      last_slash == std::string::npos ? std::string("/")
                                      : base_path.substr(0, last_slash + 1);
  return origin + RemoveDotSegments(directory + location_path) + location_tail;
}

std::string HfFormatHost(const std::string& host) {
  return host.find(':') != std::string::npos ? "[" + host + "]" : host;
}

void HfRefuseHttpsWithoutTls(const std::string& url) {
#ifndef CPPHTTPLIB_OPENSSL_SUPPORT
  if (url.rfind("https://", 0) == 0) {
    throw std::runtime_error(
        "vllm.cpp: this build cannot speak HTTPS, so it cannot reach " + url +
        ". Rebuild with -DVLLM_CPP_HF_DOWNLOAD=ON and one of "
        "-DVLLM_CPP_OPENSSL=ON (default, needs the OpenSSL development files) "
        "or -DVLLM_CPP_BUILD_BORINGSSL=ON, or set HF_ENDPOINT to an http:// "
        "mirror.");
  }
#else
  (void)url;
#endif
}

bool IsValidHfRepoId(const std::string& repo_id) {
  // Mirrors llama.cpp `common/hf-cache.cpp:121-142 @ b10451`: base characters
  // [A-Za-z0-9_] are always valid, the special characters [/.-] must sit
  // between two base characters, and exactly one '/' is required.
  if (repo_id.empty() || repo_id.length() > 256) return false;
  int slashes = 0;
  bool previous_was_special = true;
  for (const char c : repo_id) {
    if (IsAlphanumeric(c) || c == '_') {
      previous_was_special = false;
    } else if (c == '/' || c == '.' || c == '-') {
      if (previous_was_special) return false;
      slashes += (c == '/');
      previous_was_special = true;
    } else {
      return false;
    }
  }
  return !previous_was_special && slashes == 1;
}

HfHubOptions HfHubOptionsFromEnv() {
  HfHubOptions opts;

  // Mirrors llama.cpp `common/common.cpp:1530-1542 @ b10451`, minus that
  // project's own MODEL_ENDPOINT name.
  opts.endpoint = "https://huggingface.co/";
  if (const char* endpoint = NonEmptyEnv("HF_ENDPOINT")) {
    opts.endpoint = endpoint;
    if (opts.endpoint.back() != '/') opts.endpoint += '/';
  }

  if (const char* token = NonEmptyEnv("HF_TOKEN")) {
    opts.token = token;
  } else if (const char* token_path = NonEmptyEnv("HF_TOKEN_PATH")) {
    std::ifstream in(token_path, std::ios::binary);
    std::getline(in, opts.token);
    while (!opts.token.empty() &&
           (opts.token.back() == '\r' || opts.token.back() == '\n' ||
            opts.token.back() == ' ' || opts.token.back() == '\t')) {
      opts.token.pop_back();
    }
  }

  if (const char* offline = NonEmptyEnv("HF_HUB_OFFLINE")) {
    const std::string value(offline);
    opts.offline = value != "0" && value != "false" && value != "False";
  }

  opts.hub_dir = HfHubCacheDir();
  return opts;
}

std::string HubFileUrl(const std::string& endpoint, const std::string& repo_id,
                       const std::string& commit, const std::string& path) {
  return endpoint + repo_id + "/resolve/" + commit + "/" + path;
}

std::string HubResolveRefToCommit(const std::string& repo_id,
                                  const std::string& revision,
                                  const HfHubOptions& opts) {
  if (!IsValidHfRepoId(repo_id)) {
    throw std::runtime_error("vllm.cpp: '" + repo_id +
                             "' is not a HuggingFace repository identifier; "
                             "expected the form org/repo");
  }
  // A revision that is already a commit names itself, and costs no request.
  if (IsHexString(revision, 40)) return revision;

  const json doc = ApiGet(opts, "api/models/" + repo_id + "/refs", repo_id);
  const std::vector<std::pair<std::string, std::string>> refs = CollectRefs(doc);
  if (refs.empty()) {
    throw std::runtime_error("vllm.cpp: repository '" + repo_id +
                             "' offered no usable branch or tag");
  }

  if (!revision.empty()) {
    for (const auto& [name, commit] : refs) {
      if (name == revision) return commit;
    }
    throw std::runtime_error("vllm.cpp: repository '" + repo_id +
                             "' has no branch or tag named '" + revision + "'");
  }

  for (const auto& [name, commit] : refs) {
    if (name == "main") return commit;
  }
  return refs.front().second;
}

std::vector<HfFile> HubListRepoFiles(const std::string& repo_id,
                                     const std::string& commit,
                                     const HfHubOptions& opts) {
  if (!IsValidHfRepoId(repo_id)) {
    throw std::runtime_error("vllm.cpp: '" + repo_id +
                             "' is not a HuggingFace repository identifier; "
                             "expected the form org/repo");
  }
  if (!IsHexString(commit, 40)) {
    throw std::runtime_error("vllm.cpp: '" + commit +
                             "' is not a commit; the reference must be resolved "
                             "before the tree is listed");
  }

  const json doc = ApiGet(
      opts, "api/models/" + repo_id + "/tree/" + commit + "?recursive=true",
      repo_id);
  if (!doc.is_array()) {
    throw std::runtime_error("vllm.cpp: the tree listing for repository '" +
                             repo_id + "' is not an array");
  }

  // Whether the identifier is USED as a blob name still depends on the token.
  // See HfFile::oid. The two integrity rules below do NOT: an identifier that
  // is broken on its face is broken whoever asked for it, and gating them on
  // the token was an inversion in which a repository that loaded anonymously
  // began failing the moment a token was set.
  const bool trust_oids = !opts.token.empty();

  // The first entry of an identifier whose size the listing REPORTED. An entry
  // that reports no size is never an owner, because it can neither agree nor
  // disagree with a later entry, and keeping it would silence the size rule for
  // that identifier for the rest of the listing.
  //
  // ONE owner is enough, rather than a list of every known-size entry, because
  // size equality is transitive: every later known size is compared against the
  // first one, so two entries that disagree with each other cannot both agree
  // with it. The list would cost memory and buy no refusal.
  struct OidOwner {
    std::string path;
    uint64_t size = 0;
  };

  std::vector<HfFile> files;
  std::map<std::string, OidOwner> oid_owner;
  for (const json& item : doc) {
    if (!item.is_object() || !item.contains("type") || !item["type"].is_string() ||
        item["type"].get<std::string>() != "file" || !item.contains("path") ||
        !item["path"].is_string()) {
      continue;
    }

    HfFile file;
    file.path = item["path"].get<std::string>();
    if (!IsSafeEntryPath(file.path)) continue;

    // The tree API reports the CONTENT size at the top level for every file,
    // including a large-file-storage file, whose `lfs.size` repeats it. The
    // top level is therefore read first and `lfs.size` is the fallback for a
    // listing that omits it. `lfs.pointerSize` is the size of the pointer file
    // and is never the content size, so it is not read.
    // `HfFile::size` stays empty when the listing reports no size, so that an
    // unreported size and a zero-byte file stay different facts.
    if (item.contains("size") && item["size"].is_number_unsigned()) {
      file.size = item["size"].get<uint64_t>();
    } else if (item.contains("lfs") && item["lfs"].is_object() &&
               item["lfs"].contains("size") &&
               item["lfs"]["size"].is_number_unsigned()) {
      file.size = item["lfs"]["size"].get<uint64_t>();
    }

    // Read the identifier whatever the token says, because the checks below
    // have to see it.
    std::string oid;
    if (item.contains("lfs") && item["lfs"].is_object() &&
        item["lfs"].contains("oid") && item["lfs"]["oid"].is_string()) {
      oid = item["lfs"]["oid"].get<std::string>();
    } else if (item.contains("oid") && item["oid"].is_string()) {
      oid = item["oid"].get<std::string>();
    }
    // A malformed identifier is dropped, not used, and the file is kept: it is
    // still addressable by path, and only content addressing is lost.
    if (!oid.empty() && !IsHexString(oid, 40) && !IsHexString(oid, 64)) {
      oid.clear();
    }
    // FOLD the identifier to one case. `IsHexString` accepts 'A' through 'F' as
    // well as 'a' through 'f', so without this a listing that spells one
    // identifier two ways lands TWO keys in the owner map below and the size
    // rule compares nothing: `a.bin` at 4096 bytes under `ab23...` beside
    // `b.bin` at 2048 bytes under `AB23...` was ACCEPTED, where the same pair
    // in one case is refused. `HF_ENDPOINT` names the host, so the listing is
    // input the hub does not have to answer truthfully, and a rule that one
    // letter's case switches off is not a rule.
    //
    // FOLDED rather than REFUSED, because case carries no information here.
    // Hexadecimal is case-insensitive by definition and both git and the hub
    // emit lower case, so an upper-case spelling is another spelling of one
    // value rather than a claim about a second object. Refusing it would reject
    // a mirror over a difference that means nothing, which is the false
    // positive the earlier any-duplicate rule was replaced for.
    //
    // The fold is applied to `oid` ITSELF and not only to the map key, because
    // the identifier also leaves this function in `HfFile::oid` and becomes a
    // cache file name through `HfBlobPath`. A name that kept the listing's case
    // would split one blob across two files on a case-sensitive file system
    // while a case-insensitive one, such as the CIFS mounts this project reads
    // checkpoints from, collapsed them, so the same listing would populate a
    // different cache on two hosts.
    //
    // `IsDegenerateOid` needs no separate repair: folding a repeated character
    // leaves it repeated, so the fold cannot move that verdict, and it only
    // makes the refusal quote the canonical spelling.
    for (char& c : oid) {
      if (c >= 'A' && c <= 'F') c = static_cast<char>(c - 'A' + 'a');
    }

    if (!oid.empty()) {
      // RULE 2, the degenerate identifier. Refused, and never conditional on a
      // token, because no content hash is one character repeated.
      if (IsDegenerateOid(oid)) {
        throw std::runtime_error(
            "vllm.cpp: the tree listing for repository '" + repo_id +
            "' gives '" + file.path + "' the object identifier " + oid +
            ", whose " + std::to_string(oid.size()) +
            " characters are all '" + std::string(1, oid.front()) +
            "'. No content hash produces that value, so the hub is not "
            "answering the truth about this repository. The listing is not "
            "usable.");
      }
      // RULE 1, the size disagreement. Two files sharing an identifier is
      // LEGITIMATE: `lfs.oid` is the sha256 of the contents and the plain
      // `oid` is the git blob sha1, so two byte-identical files in one
      // repository share one identifier by construction. What no content hash
      // can do is name two different sizes.
      //
      // The rule asks nothing about the PATH. One path listed twice at two
      // different sizes is self contradictory whichever entry is believed, and
      // exempting it would exempt the exact shape this rule exists to catch.
      const auto it = oid_owner.find(oid);
      if (it == oid_owner.end()) {
        if (file.size.has_value()) {
          oid_owner.emplace(oid, OidOwner{file.path, *file.size});
        }
      } else if (file.size.has_value() && it->second.size != *file.size) {
        throw std::runtime_error(
            "vllm.cpp: the tree listing for repository '" + repo_id +
            "' gives object identifier " + oid + " to '" + it->second.path +
            "' at " + std::to_string(it->second.size) + " bytes and to '" +
            file.path + "' at " + std::to_string(*file.size) +
            " bytes. One content hash cannot name two different sizes. The "
            "listing is not usable.");
      }
    }

    if (trust_oids) file.oid = oid;

    file.url = HubFileUrl(opts.endpoint, repo_id, commit, file.path);
    files.push_back(std::move(file));
  }
  return files;
}

std::string HubResolveCommitCached(const std::string& repo_id,
                                   const std::string& revision,
                                   const HfHubOptions& opts) {
  if (!IsValidHfRepoId(repo_id)) {
    throw std::runtime_error("vllm.cpp: '" + repo_id +
                             "' is not a HuggingFace repository identifier; "
                             "expected the form org/repo");
  }
  if (IsHexString(revision, 40)) return revision;

  const std::string ref = revision.empty() ? "main" : revision;
  const fs::path repo_path = HfRepoPath(opts.hub_dir, repo_id);

  if (!repo_path.empty()) {
    const std::string cached = HfReadRef(repo_path, ref);
    if (!cached.empty()) return cached;
  }

  if (opts.offline) {
    throw std::runtime_error(
        "vllm.cpp: HF_HUB_OFFLINE is set and repository '" + repo_id +
        "' has no recorded '" + ref + "' reference under " +
        (repo_path.empty() ? std::string("(this host has no HuggingFace cache "
                                         "directory)")
                           : repo_path.string()) +
        ". Fetch it once with HF_HUB_OFFLINE unset, or point HF_HOME at a cache "
        "that already holds it.");
  }

  const std::string commit = HubResolveRefToCommit(repo_id, revision, opts);
  if (!repo_path.empty()) HfWriteRef(repo_path, ref, commit);
  return commit;
}

}  // namespace transformers_utils

// The FETCHER half of the one-definition-rule instrument declared in
// `include/vllm/http_transport_abi.h`. It is defined HERE, in the translation
// unit that actually opens the hub connection, so the reading is that unit's
// own and not a shared constant compiled once.
HttpTransportAbi HubHttpTransportAbi() {
  HttpTransportAbi abi;
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
  abi.tls = true;
#else
  abi.tls = false;
#endif
  abi.result_size = sizeof(httplib::Result);
  abi.client_connection_size = sizeof(httplib::ClientConnection);
  return abi;
}

}  // namespace vllm
