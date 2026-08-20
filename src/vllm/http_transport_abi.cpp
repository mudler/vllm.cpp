// The comparison half of include/vllm/http_transport_abi.h. The three READINGS
// are produced by the server, protocol and transfer translation units
// themselves; this file only holds them side by side, so it deliberately does
// NOT include <httplib/httplib.h> and cannot contribute a fourth opinion.
//
// ENG-HF-MODEL-DOWNLOAD W5, issue #1280.
#include "vllm/http_transport_abi.h"

#include <string>

namespace vllm {

namespace {

std::string Describe(const HttpTransportAbi& abi) {
  return std::string("tls=") + (abi.tls ? "yes" : "no") +
         " sizeof(httplib::Result)=" + std::to_string(abi.result_size) +
         " sizeof(httplib::ClientConnection)=" +
         std::to_string(abi.client_connection_size);
}

bool Agree(const HttpTransportAbi& a, const HttpTransportAbi& b) {
  return a.tls == b.tls && a.result_size == b.result_size &&
         a.client_connection_size == b.client_connection_size;
}

}  // namespace

std::string HttpTransportAbiMismatch() {
  const HttpTransportAbi server = ApiServerHttpTransportAbi();
  const HttpTransportAbi hub = HubHttpTransportAbi();
  const HttpTransportAbi downloader = DownloaderHttpTransportAbi();
  // Every reading is compared against the server's, so a disagreement in ANY
  // one of the three is named. Comparing neighbours pairwise would let two
  // agreeing halves outvote the third and report nothing.
  if (Agree(server, hub) && Agree(server, downloader)) {
    return std::string();
  }
  return "vllm.cpp: this binary was built with CPPHTTPLIB_OPENSSL_SUPPORT "
         "defined for some translation units and not for others, which is a "
         "one-definition-rule violation that links cleanly and corrupts every "
         "HTTP response object passed between them. The OpenAI server half "
         "reports [" +
         Describe(server) + "], the HuggingFace protocol half reports [" +
         Describe(hub) + "] and the file transfer half reports [" +
         Describe(downloader) +
         "]. Set the define on the target, never per source file, and rebuild "
         "from a clean build directory.";
}

}  // namespace vllm
