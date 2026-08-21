// ENG-HF-MODEL-DOWNLOAD W5 (#1280): transport layer security, and the
// one-definition-rule hazard that comes with it.
//
// THREE THINGS ARE PINNED HERE, and each is a separate failure this row can
// ship.
//
//  1. ALL THREE library translation units that include the vendored httplib
//     header agree on what `CPPHTTPLIB_OPENSSL_SUPPORT` was: the listener
//     (`api_server.cpp`), the protocol client (`hf_hub.cpp`) and the transfer
//     loop (`downloader.cpp`). That define changes the layout of
//     `httplib::Result`, so a build that set it per FILE rather than per TARGET
//     links cleanly and then reads a response object sixteen bytes short. No
//     compiler diagnoses it and no other test in this tree would notice.
//     The transfer unit is read because a review MEASURED its absence:
//     undefining the macro for `downloader.cpp` alone left this suite at 5 of
//     5 cases and 17 of 17 assertions green.
//  2. The OpenAI listener is still plain hypertext transfer protocol. The same
//     define makes `httplib::SSLServer` compilable, and this row deliberately
//     does not enable a listener.
//  3. The hub client actually SPEAKS https. Every other suite in this row runs
//     against a plain-HTTP fake hub and therefore proves nothing about
//     transport security. This one starts a REAL `httplib::SSLServer` on
//     loopback with a self-signed certificate, points `HF_ENDPOINT` at it, and
//     drives the production `HubResolveRefToCommit` through a genuine
//     handshake. It opens no network connection.
//
// BOTH ARMS OF CASE 3 ASSERT. On a build with no transport layer security the
// case makes the opposite statement -- the refusal names the build options --
// rather than compiling to nothing. A case that compiles to nothing reports
// `assertions: 0`, which this repository records as a skip wearing a pass.
#include <doctest/doctest.h>

#include <atomic>
#include <cstddef>
#include <filesystem>
#include <string>
#include <thread>

#include <httplib/httplib.h>

#include "support/process_id.h"
#include "support/test_env.h"
#include "vllm/http_transport_abi.h"
#include "vllm/transformers_utils/hf_hub.h"

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
#include <openssl/asn1.h>
#include <openssl/bio.h>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/x509.h>
#include <openssl/x509v3.h>
#endif

namespace fs = std::filesystem;

using vllm::ApiServerHttpTransportAbi;
using vllm::ApiServerListenerIsTls;
using vllm::DownloaderHttpTransportAbi;
using vllm::HttpTransportAbi;
using vllm::HttpTransportAbiMismatch;
using vllm::HubHttpTransportAbi;
using vllm::transformers_utils::HfHubOptions;

namespace {

constexpr const char* kCommit = "2222222222222222222222222222222222222222";

class TempDir {
 public:
  TempDir() {
    static std::atomic<int> counter{0};
    path_ = fs::temp_directory_path() /
            ("vllm_tls_transport_test_" +
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

std::string MessageOf(const std::function<void()>& body) {
  try {
    body();
  } catch (const std::exception& e) {
    return e.what();
  }
  return std::string();
}

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT

// Write a self-signed certificate for the loopback address, plus its key.
//
// `CA:TRUE` so the certificate can be its own trust anchor: the client trusts
// it by name through `SSL_CERT_FILE`, which is what
// `SSL_CTX_set_default_verify_paths` reads and which is the path cpp-httplib
// takes when no explicit certificate authority is configured
// (`third_party/httplib/httplib.h:17125`). `IP:127.0.0.1` in the subject
// alternative name because httplib verifies the peer identity against the host
// it dialled and handles an address entry at `httplib.h:17564`.
//
// An ELLIPTIC CURVE key, not RSA: a 2048 bit RSA generation costs a
// noticeable fraction of a second and this runs on every invocation of the
// suite.
bool WriteLoopbackCertificate(const fs::path& cert_path,
                              const fs::path& key_path) {
  EVP_PKEY* key = EVP_EC_gen("P-256");
  if (key == nullptr) return false;
  X509* cert = X509_new();
  if (cert == nullptr) {
    EVP_PKEY_free(key);
    return false;
  }

  bool ok = X509_set_version(cert, 2) == 1;
  ok = ok && ASN1_INTEGER_set(X509_get_serialNumber(cert), 1) == 1;
  ok = ok && X509_gmtime_adj(X509_getm_notBefore(cert), 0) != nullptr;
  ok = ok && X509_gmtime_adj(X509_getm_notAfter(cert), 3600) != nullptr;
  ok = ok && X509_set_pubkey(cert, key) == 1;

  X509_NAME* name = X509_get_subject_name(cert);
  ok = ok && X509_NAME_add_entry_by_txt(
                 name, "CN", MBSTRING_ASC,
                 reinterpret_cast<const unsigned char*>("vllm.cpp test"), -1,
                 -1, 0) == 1;
  ok = ok && X509_set_issuer_name(cert, name) == 1;

  X509V3_CTX ctx;
  X509V3_set_ctx_nodb(&ctx);
  X509V3_set_ctx(&ctx, cert, cert, nullptr, nullptr, 0);
  for (const auto& [nid, value] :
       {std::pair<int, const char*>{NID_subject_alt_name, "IP:127.0.0.1"},
        std::pair<int, const char*>{NID_basic_constraints, "critical,CA:TRUE"}}) {
    X509_EXTENSION* ext = X509V3_EXT_conf_nid(nullptr, &ctx, nid, value);
    if (ext == nullptr) {
      ok = false;
      break;
    }
    ok = ok && X509_add_ext(cert, ext, -1) == 1;
    X509_EXTENSION_free(ext);
  }

  ok = ok && X509_sign(cert, key, EVP_sha256()) != 0;

  if (ok) {
    BIO* cert_bio = BIO_new_file(cert_path.string().c_str(), "wb");
    ok = cert_bio != nullptr && PEM_write_bio_X509(cert_bio, cert) == 1;
    if (cert_bio != nullptr) BIO_free(cert_bio);
  }
  if (ok) {
    BIO* key_bio = BIO_new_file(key_path.string().c_str(), "wb");
    ok = key_bio != nullptr &&
         PEM_write_bio_PrivateKey(key_bio, key, nullptr, nullptr, 0, nullptr,
                                  nullptr) == 1;
    if (key_bio != nullptr) BIO_free(key_bio);
  }

  X509_free(cert);
  EVP_PKEY_free(key);
  return ok;
}

// The fake hub of the other suites in this row, over TLS. It answers the one
// protocol call this case needs and counts what it received, so "the handshake
// happened" is measured on the server side too rather than inferred from a
// value the client could have produced from a cache.
class TlsHub {
 public:
  TlsHub(const fs::path& cert_path, const fs::path& key_path)
      : server_(cert_path.string().c_str(), key_path.string().c_str()) {
    server_.Get("/api/models/(.*)/refs",
                [this](const httplib::Request&, httplib::Response& res) {
                  requests_.fetch_add(1);
                  res.set_content(
                      std::string(R"({"branches":[{"name":"main",)") +
                          R"("ref":"refs/heads/main","targetCommit":")" +
                          kCommit + R"("}],"tags":[]})",
                      "application/json");
                });
    port_ = server_.bind_to_any_port("127.0.0.1");
    thread_ = std::thread([this] { server_.listen_after_bind(); });
    server_.wait_until_ready();
  }
  ~TlsHub() {
    server_.stop();
    if (thread_.joinable()) thread_.join();
  }
  TlsHub(const TlsHub&) = delete;
  TlsHub& operator=(const TlsHub&) = delete;

  bool valid() const { return server_.is_valid() && port_ > 0; }
  int request_count() const { return requests_.load(); }
  std::string endpoint() const {
    return "https://127.0.0.1:" + std::to_string(port_) + "/";
  }

 private:
  httplib::SSLServer server_;
  std::thread thread_;
  int port_ = 0;
  std::atomic<int> requests_{0};
};

#endif  // CPPHTTPLIB_OPENSSL_SUPPORT

}  // namespace

TEST_CASE(
    "tls transport: every library translation unit agrees on the httplib "
    "layout") {
  const HttpTransportAbi server = ApiServerHttpTransportAbi();
  const HttpTransportAbi fetcher = HubHttpTransportAbi();
  const HttpTransportAbi downloader = DownloaderHttpTransportAbi();

  INFO("server tls=" << server.tls << " result=" << server.result_size
                     << " conn=" << server.client_connection_size);
  INFO("fetcher tls=" << fetcher.tls << " result=" << fetcher.result_size
                      << " conn=" << fetcher.client_connection_size);
  INFO("downloader tls=" << downloader.tls
                         << " result=" << downloader.result_size
                         << " conn=" << downloader.client_connection_size);

  CHECK(server.tls == fetcher.tls);
  CHECK(server.result_size == fetcher.result_size);
  CHECK(server.client_connection_size == fetcher.client_connection_size);
  // The transfer unit, compared against the SERVER rather than against
  // `hf_hub.cpp`, so the two fetcher files cannot agree with each other and
  // outvote the listener.
  CHECK(server.tls == downloader.tls);
  CHECK(server.result_size == downloader.result_size);
  CHECK(server.client_connection_size == downloader.client_connection_size);
  // A zero would mean the reading never ran, which would make the comparisons
  // above agree about nothing.
  CHECK(server.result_size > 0);
  CHECK(fetcher.client_connection_size > 0);
  CHECK(downloader.result_size > 0);
  CHECK(downloader.client_connection_size > 0);
  // The production refusal `VllmServerMain` makes before it binds.
  const std::string mismatch = HttpTransportAbiMismatch();
  INFO("mismatch: " << mismatch);
  CHECK(mismatch.empty());
}

TEST_CASE(
    "tls transport: a configuration that asked for TLS actually got it") {
  // The expectation comes from the BUILD CONFIGURATION -- the option was on and
  // this host has the OpenSSL development files -- and never from the macro
  // under test. A build where `VLLM_CPP_HF_DOWNLOAD` silently resolved OFF, or
  // where the define reached the library but not its consumers, fails HERE
  // rather than quietly running the no-TLS arm of every other case and
  // reporting SUCCESS.
#ifdef VLLM_CPP_TEST_EXPECT_TLS
  CHECK(HubHttpTransportAbi().tls);
  CHECK(ApiServerHttpTransportAbi().tls);
  CHECK(DownloaderHttpTransportAbi().tls);
#ifndef CPPHTTPLIB_OPENSSL_SUPPORT
  FAIL_CHECK(
      "the build was configured with VLLM_CPP_HF_DOWNLOAD=ON and "
      "VLLM_CPP_OPENSSL=ON on a host that has OpenSSL, and this consumer was "
      "still compiled without CPPHTTPLIB_OPENSSL_SUPPORT");
#endif
#else
  // No expectation was recorded, so the only statement available is that the
  // build is internally consistent about not having TLS.
  CHECK(HubHttpTransportAbi().tls == ApiServerHttpTransportAbi().tls);
  CHECK(DownloaderHttpTransportAbi().tls == ApiServerHttpTransportAbi().tls);
#endif
}

TEST_CASE(
    "tls transport: this test binary agrees with the library it links") {
  // The THIRD opinion. The two readings above are both produced inside
  // `libvllm`, so a build that handed the define to the library and not to its
  // consumers would still make them agree while every test and every example
  // laid `httplib::Result` out differently. This case is compiled in the
  // consumer, so it measures the seam the other case cannot see.
  const HttpTransportAbi library = HubHttpTransportAbi();
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
  CHECK(library.tls);
#else
  CHECK_FALSE(library.tls);
#endif
  CHECK(library.result_size == sizeof(httplib::Result));
  CHECK(library.client_connection_size == sizeof(httplib::ClientConnection));
}

TEST_CASE("tls transport: the OpenAI listener stays plain HTTP") {
  // ENG-HF-MODEL-DOWNLOAD adds transport layer security to the hub CLIENT. It
  // does not add a TLS LISTENER, and it documents none. Enabling one is a
  // separate decision with its own certificate, key and flags, and shipping it
  // as an unannounced side effect of a download feature would change what
  // `vllm-server` binds by default.
  CHECK_FALSE(ApiServerListenerIsTls());
}

TEST_CASE("tls transport: what an https hub does on this build") {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
  // A REAL handshake against a REAL `httplib::SSLServer` on loopback. Nothing
  // leaves the machine.
  TempDir dir;
  const fs::path cert = dir.path() / "loopback.crt";
  const fs::path key = dir.path() / "loopback.key";
  REQUIRE(WriteLoopbackCertificate(cert, key));

  TlsHub hub(cert, key);
  REQUIRE(hub.valid());

  // `SSL_CERT_FILE` is what OpenSSL's default verify paths read, and cpp-httplib
  // installs those paths on a client that configured no certificate authority
  // of its own. This is therefore the ONLY way to trust the loopback
  // certificate without weakening the production client, which is the thing
  // under test.
  vllm_test::SetEnv("SSL_CERT_FILE", cert.string().c_str());

  HfHubOptions opts;
  opts.endpoint = hub.endpoint();
  opts.hub_dir = dir.path() / "hub";
  opts.connect_timeout_seconds = 10;
  opts.read_timeout_seconds = 20;

  const int before = hub.request_count();
  std::string commit;
  const std::string failure = MessageOf([&] {
    commit = vllm::transformers_utils::HubResolveRefToCommit("tiny/llama",
                                                             "main", opts);
  });
  vllm_test::UnsetEnv("SSL_CERT_FILE");

  INFO("failure: " << failure);
  INFO("endpoint: " << opts.endpoint);
  CHECK(failure.empty());
  CHECK(commit == kCommit);
  // Measured on the SERVER, so the commit above cannot have come from anywhere
  // but a completed TLS request.
  CHECK(hub.request_count() == before + 1);
#else
  // No transport layer security in this build. The production refusal has to
  // name the build options, because an `https` endpoint would otherwise fail
  // with something that reads like a network fault.
  TempDir dir;
  HfHubOptions opts;
  opts.endpoint = "https://127.0.0.1:1/";
  opts.hub_dir = dir.path() / "hub";
  const std::string failure = MessageOf([&] {
    (void)vllm::transformers_utils::HubResolveRefToCommit("tiny/llama", "main",
                                                          opts);
  });
  INFO("refusal: " << failure);
  REQUIRE_FALSE(failure.empty());
  CHECK(failure.find("VLLM_CPP_HF_DOWNLOAD") != std::string::npos);
  CHECK(failure.find("VLLM_CPP_OPENSSL") != std::string::npos);
  CHECK(failure.find("VLLM_CPP_BUILD_BORINGSSL") != std::string::npos);
#endif
}
