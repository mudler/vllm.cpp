// The comparison half of include/vllm/http_transport_abi.h. The two READINGS
// are produced by the server and fetcher translation units themselves; this
// file only holds them side by side, so it deliberately does NOT include
// <httplib/httplib.h> and cannot contribute a third opinion.
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

}  // namespace

std::string HttpTransportAbiMismatch() {
  const HttpTransportAbi server = ApiServerHttpTransportAbi();
  const HttpTransportAbi fetcher = HubHttpTransportAbi();
  if (server.tls == fetcher.tls &&
      server.result_size == fetcher.result_size &&
      server.client_connection_size == fetcher.client_connection_size) {
    return std::string();
  }
  return "vllm.cpp: this binary was built with CPPHTTPLIB_OPENSSL_SUPPORT "
         "defined for some translation units and not for others, which is a "
         "one-definition-rule violation that links cleanly and corrupts every "
         "HTTP response object passed between them. The OpenAI server half "
         "reports [" +
         Describe(server) + "] and the HuggingFace fetcher half reports [" +
         Describe(fetcher) +
         "]. Set the define on the target, never per source file, and rebuild "
         "from a clean build directory.";
}

}  // namespace vllm
