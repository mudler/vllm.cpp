// Ported from: vllm/entrypoints/openai/api_server.py @ e24d1b24 (the route
// shapes) + the per-endpoint api_router modules
//   - completion/api_router.py:34    POST /v1/completions
//   - chat_completion/api_router.py:40 POST /v1/chat/completions
//   - models/api_router.py:20        GET  /v1/models
//   - serve/instrumentator/health.py:22 GET /health
//   - serve/instrumentator/basic.py:53  GET /version
//
// SCOPE (M3.1 Task 4 / T0): the thin HTTP transport over the Task-2 serving
// handlers. The OpenAI-protocol logic (parse → SamplingParams → engine →
// response/SSE) already lives in serving_{completion,chat}; this file only
// binds it to routes over the vendored cpp-httplib (third_party/httplib.h).
//
// DEPENDENCY DEVIATION (recorded): cpp-httplib is a header-only MIT HTTP
// TRANSPORT library (the same choice as llama.cpp's server) — NOT a compute/ML
// dependency, so it is consistent with the no-pytorch / no-ggml rule. It is
// gated behind the CMake option VLLM_CPP_SERVER.
//
// The route DISPATCH (parse body → handler → status + body / SSE chunks) is
// decoupled from the socket via the handle_* methods below, so the request
// handling is unit-testable WITHOUT binding a port; listen() wires the same
// methods onto an httplib::Server.
#ifndef VLLM_ENTRYPOINTS_OPENAI_API_SERVER_H_
#define VLLM_ENTRYPOINTS_OPENAI_API_SERVER_H_

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "vllm/entrypoints/openai/serving_chat.h"
#include "vllm/entrypoints/openai/serving_completion.h"
#include "vllm/entrypoints/openai/serving_models.h"

namespace vllm::entrypoints::openai {

// The HTTP api_server. Holds non-owning references to the serving handlers +
// model registry (constructed + owned by the caller; they outlive it).
class ApiServer {
 public:
  enum class HttpWorkerPoolMode {
    kCapacityFixed,
    // Diagnostic same-binary fallback for A/B attribution only.
    kLegacyDynamic,
  };

  // A live SSE response occupies one cpp-httplib worker while it waits on its
  // collector. Keep enough fixed workers for every scheduler-visible stream,
  // plus a small bounded reserve for health/discovery/control requests.
  static constexpr size_t kDefaultMaxConcurrentStreams = 8;
  static constexpr size_t kControlWorkerHeadroom = 4;

  ApiServer(OpenAIServingCompletion& completion, OpenAIServingChat& chat,
            OpenAIServingModels& models, std::string version,
            size_t max_concurrent_streams = kDefaultMaxConcurrentStreams,
            HttpWorkerPoolMode worker_pool_mode =
                HttpWorkerPoolMode::kCapacityFixed);
  ~ApiServer();

  ApiServer(const ApiServer&) = delete;
  ApiServer& operator=(const ApiServer&) = delete;

  // The socket-free result of dispatching one request (shared by the HTTP layer
  // and unit tests). Streaming uses either the legacy precomputed vector or
  // W2's live pull source.
  struct DispatchResult {
    int status = 200;
    bool streaming = false;
    std::string content_type = "application/json";
    std::string body;                     // populated when !streaming
    std::vector<std::string> sse_chunks;  // populated when streaming
    std::shared_ptr<SseStream> sse_stream;  // live AsyncLLM streaming path
  };

  // Route dispatch (parse the JSON body → the Task-1 request → the Task-2
  // handler → a DispatchResult). On a malformed body → 400; an unknown model →
  // 404; an internal failure → 500 — each carrying the OpenAI ErrorResponse
  // JSON. These are what listen()'s handlers call, and what the tests exercise
  // without a socket.
  DispatchResult handle_completions(const std::string& request_body);
  DispatchResult handle_chat_completions(const std::string& request_body);
  DispatchResult handle_models() const;
  DispatchResult handle_health() const;
  DispatchResult handle_version() const;

  // Bind host:port, register the routes and serve until stop() (blocking).
  // Returns false if the bind fails.
  bool listen(const std::string& host, int port);

  // Bind to an OS-assigned ephemeral port (port 0) on `host`; returns the bound
  // port (or -1 on failure). Follow with serve() to run the loop. Used by the
  // smoke test to avoid a fixed-port race.
  int bind_to_any_port(const std::string& host);
  // Run the accept loop after a successful bind_to_any_port (blocking).
  bool serve();
  // Signal the accept loop to stop (thread-safe); listen()/serve() then return.
  void stop();
  // True once the server is accepting connections (poll before issuing
  // requests against a bind_to_any_port + serve() server on another thread).
  bool is_running() const;
  // Configured fixed worker count. Exposed for startup diagnostics and the
  // transport-capacity regression; it is max_concurrent_streams + headroom,
  // or zero when the diagnostic legacy-dynamic mode is selected.
  size_t http_worker_count() const;

 private:
  OpenAIServingCompletion& completion_;
  OpenAIServingChat& chat_;
  OpenAIServingModels& models_;
  std::string version_;

  // Opaque httplib::Server (pimpl keeps third_party/httplib.h out of this
  // header — only api_server.cpp and the smoke test pull it in).
  struct Impl;
  std::unique_ptr<Impl> impl_;

  void register_routes();
};

}  // namespace vllm::entrypoints::openai

#endif  // VLLM_ENTRYPOINTS_OPENAI_API_SERVER_H_
