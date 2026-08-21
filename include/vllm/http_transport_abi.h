// vllm.cpp original: what each translation unit that includes the vendored
// cpp-httplib header believes about transport layer security.
//
// WHY THIS FILE EXISTS. `CPPHTTPLIB_OPENSSL_SUPPORT` is a WHOLE-HEADER switch.
// It changes the layout of `httplib::Result` and `httplib::ClientImpl`, it adds
// members to `httplib::ClientConnection`, and it selects a different socket
// path. This repository includes that one header from THREE library
// translation units: the server
// (`src/vllm/entrypoints/openai/api_server.cpp`), the protocol client
// (`src/vllm/transformers_utils/hf_hub.cpp`) and the transfer loop
// (`src/vllm/transformers_utils/downloader.cpp`). All three link into ONE
// library. A build that defined the macro for some of
// those translation units and not for the others is a one-definition-rule
// violation: it LINKS CLEANLY, because the mangled names do not carry the
// layout, and it misbehaves at run time when one half writes a `Result` the
// other half reads sixteen bytes short.
//
// The build system prevents that by setting the define on the TARGET rather
// than per file (`CMakeLists.txt`, the `VLLM_CPP_HF_DOWNLOAD` block), so every
// consumer of `vllm` inherits it. This header is the INSTRUMENT that proves the
// prevention held. Each of the three reports what its own preprocessor saw and
// the size its own compiler laid out, and `HttpTransportAbiMismatch` compares
// all three.
//
// THE THIRD READING IS NOT DECORATION. The first version of this instrument
// read only the server and the protocol client, and a review measured what that
// missed: undefining the macro for `downloader.cpp` ALONE compiled at exit 0,
// left `test_tls_transport` at 5 of 5 cases and 17 of 17 assertions green, and
// let the production binary start and answer. The transfer loop hands an
// `httplib::Result` across the same seam as the other two, so a comparison that
// skips it measures two thirds of the hazard and claims all of it.
//
// `src/vllm/entrypoints/openai/server_main.cpp` calls that comparison before it
// binds anything, so a mismatched binary refuses to serve instead of serving
// wrongly. `tests/vllm/transformers_utils/test_tls_transport.cpp` pins it.
//
// ENG-HF-MODEL-DOWNLOAD W5, issue #1280.
#pragma once

#include <cstddef>
#include <string>

namespace vllm {

// What ONE translation unit saw. Every field is filled from that unit's own
// preprocessor and its own `sizeof`, never from a shared constant, because a
// shared constant would be compiled once and could not disagree.
struct HttpTransportAbi {
  // `CPPHTTPLIB_OPENSSL_SUPPORT` as this translation unit was compiled. When
  // this is false the process cannot open an `https` connection at all.
  bool tls = false;
  // `sizeof(httplib::Result)` in this translation unit. The type gains an
  // `int` and a `uint64_t` under the macro
  // (`third_party/httplib/httplib.h:2045-2062`), so the two states lay it out
  // at two different sizes and a disagreement is measurable rather than merely
  // asserted.
  std::size_t result_size = 0;
  // `sizeof(httplib::ClientConnection)` in this translation unit. It gains a
  // session pointer under the macro (`httplib.h:2066-2100`).
  std::size_t client_connection_size = 0;
};

// Reported by the PROTOCOL translation unit, `hf_hub.cpp`.
HttpTransportAbi HubHttpTransportAbi();

// Reported by the TRANSFER translation unit, `downloader.cpp`. It runs the
// `HEAD`, the ranged `GET` and the resume, so it builds and destroys more
// `httplib::Result` objects than either other half.
HttpTransportAbi DownloaderHttpTransportAbi();

// Reported by the SERVER translation unit, `api_server.cpp`.
HttpTransportAbi ApiServerHttpTransportAbi();

// True when the OpenAI server's listener is a transport-layer-security
// listener. This row deliberately does NOT enable one: the same define that
// gives the hub client `https` also makes `httplib::SSLServer` compilable, and
// a listener is a separate decision with its own certificate, key and
// documentation. The value is computed from the TYPE the server actually
// constructs, so swapping that member flips it.
bool ApiServerListenerIsTls();

// Empty when all three readings agree. Otherwise a message naming every
// reading, suitable for refusing to start.
std::string HttpTransportAbiMismatch();

}  // namespace vllm
