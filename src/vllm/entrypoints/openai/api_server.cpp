// Ported from: vllm/entrypoints/openai/api_server.py @ e24d1b24 + the
// per-endpoint api_router modules. See api_server.h for scope + the cpp-httplib
// dependency deviation.
#include "vllm/entrypoints/openai/api_server.h"

#include <exception>
#include <utility>

#include <httplib/httplib.h>
#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"

namespace vllm::entrypoints::openai {

// Opaque httplib::Server (pimpl — keeps httplib.h out of api_server.h).
struct ApiServer::Impl {
  httplib::Server server;
};

namespace {

// Build the OpenAI ErrorResponse JSON body for a failed request
// (serve/utils/error_response.py::create_error_response). `code` == the HTTP
// status code (upstream ErrorInfo.code carries it).
ApiServer::DispatchResult MakeError(int status, const std::string& type,
                                    const std::string& message) {
  ErrorResponse err;
  err.error.message = message;
  err.error.type = type;
  err.error.code = status;
  ApiServer::DispatchResult r;
  r.status = status;
  r.content_type = "application/json";
  r.body = nlohmann::json(err).dump();
  return r;
}

}  // namespace

ApiServer::ApiServer(OpenAIServingCompletion& completion,
                     OpenAIServingChat& chat, OpenAIServingModels& models,
                     std::string version)
    : completion_(completion),
      chat_(chat),
      models_(models),
      version_(std::move(version)),
      impl_(std::make_unique<Impl>()) {}

ApiServer::~ApiServer() = default;

ApiServer::DispatchResult ApiServer::handle_completions(
    const std::string& request_body) {
  // completion/api_router.py:46 (create_completion): parse → check_model →
  // handler → JSON (non-stream) or text/event-stream (stream).
  nlohmann::json body;
  try {
    body = nlohmann::json::parse(request_body);
  } catch (const std::exception& e) {
    return MakeError(400, "BadRequestError",
                     std::string("Invalid JSON body: ") + e.what());
  }
  CompletionRequest request;
  try {
    from_json(body, request);
  } catch (const std::exception& e) {
    return MakeError(400, "BadRequestError",
                     std::string("Invalid request: ") + e.what());
  }
  if (!models_.check_model(request.model)) {
    return MakeError(404, "NotFoundError",
                     "The model `" + request.model.value_or("") +
                         "` does not exist.");
  }

  CompletionResult result;
  try {
    result = completion_.create_completion(request);
  } catch (const std::exception& e) {
    return MakeError(500, "InternalServerError", e.what());
  }

  DispatchResult out;
  if (result.streaming) {
    out.streaming = true;
    out.content_type = "text/event-stream";
    out.sse_chunks = std::move(result.sse_chunks);
  } else {
    out.status = 200;
    out.content_type = "application/json";
    out.body = nlohmann::json(*result.response).dump();
  }
  return out;
}

ApiServer::DispatchResult ApiServer::handle_chat_completions(
    const std::string& request_body) {
  // chat_completion/api_router.py:53 (create_chat_completion).
  nlohmann::json body;
  try {
    body = nlohmann::json::parse(request_body);
  } catch (const std::exception& e) {
    return MakeError(400, "BadRequestError",
                     std::string("Invalid JSON body: ") + e.what());
  }
  ChatCompletionRequest request;
  try {
    from_json(body, request);
  } catch (const std::exception& e) {
    return MakeError(400, "BadRequestError",
                     std::string("Invalid request: ") + e.what());
  }
  if (!models_.check_model(request.model)) {
    return MakeError(404, "NotFoundError",
                     "The model `" + request.model.value_or("") +
                         "` does not exist.");
  }

  ChatCompletionResult result;
  try {
    result = chat_.create_chat_completion(request);
  } catch (const std::exception& e) {
    return MakeError(500, "InternalServerError", e.what());
  }

  DispatchResult out;
  if (result.streaming) {
    out.streaming = true;
    out.content_type = "text/event-stream";
    out.sse_chunks = std::move(result.sse_chunks);
  } else {
    out.status = 200;
    out.content_type = "application/json";
    out.body = nlohmann::json(*result.response).dump();
  }
  return out;
}

ApiServer::DispatchResult ApiServer::handle_models() const {
  // models/api_router.py:21 (show_available_models).
  DispatchResult out;
  out.status = 200;
  out.content_type = "application/json";
  out.body = nlohmann::json(models_.show_available_models()).dump();
  return out;
}

ApiServer::DispatchResult ApiServer::handle_health() const {
  // serve/instrumentator/health.py:22 — empty 200 (T0: always healthy).
  DispatchResult out;
  out.status = 200;
  out.content_type = "text/plain";
  out.body.clear();
  return out;
}

ApiServer::DispatchResult ApiServer::handle_version() const {
  // serve/instrumentator/basic.py:53 — {"version": <ver>}.
  DispatchResult out;
  out.status = 200;
  out.content_type = "application/json";
  out.body = nlohmann::json{{"version", version_}}.dump();
  return out;
}

void ApiServer::register_routes() {
  httplib::Server& server = impl_->server;

  // Write a DispatchResult onto an httplib::Response — either the full JSON body
  // or a chunked text/event-stream (SSE), matching upstream's JSONResponse vs
  // StreamingResponse(media_type="text/event-stream").
  auto write = [](const DispatchResult& result, httplib::Response& res) {
    res.status = result.status;
    if (result.streaming) {
      // Each `data: {...}\n\n` chunk written via cpp-httplib's chunked content
      // provider (StreamingResponse). The chunks are precomputed by the Task-2
      // handler (T0 engine runs synchronously); the provider streams them out.
      // httplib calls the provider with a cumulative BYTE offset (not a chunk
      // index) and re-invokes until done() is signalled — so emit every
      // precomputed chunk as its own sink.write() (one HTTP chunk each) in a
      // single pass, then done().
      auto chunks = std::make_shared<std::vector<std::string>>(result.sse_chunks);
      res.set_chunked_content_provider(
          result.content_type,
          [chunks](size_t /*offset*/, httplib::DataSink& sink) -> bool {
            for (const std::string& chunk : *chunks) {
              if (!sink.write(chunk.data(), chunk.size())) return false;
            }
            sink.done();
            return true;
          });
    } else {
      res.set_content(result.body, result.content_type);
    }
  };

  server.Post("/v1/completions",
              [this, write](const httplib::Request& req, httplib::Response& res) {
                write(handle_completions(req.body), res);
              });
  server.Post("/v1/chat/completions",
              [this, write](const httplib::Request& req, httplib::Response& res) {
                write(handle_chat_completions(req.body), res);
              });
  server.Get("/v1/models",
             [this, write](const httplib::Request&, httplib::Response& res) {
               write(handle_models(), res);
             });
  server.Get("/health",
             [this, write](const httplib::Request&, httplib::Response& res) {
               write(handle_health(), res);
             });
  server.Get("/version",
             [this, write](const httplib::Request&, httplib::Response& res) {
               write(handle_version(), res);
             });
}

bool ApiServer::listen(const std::string& host, int port) {
  register_routes();
  return impl_->server.listen(host, port);
}

int ApiServer::bind_to_any_port(const std::string& host) {
  register_routes();
  return impl_->server.bind_to_any_port(host);
}

bool ApiServer::serve() { return impl_->server.listen_after_bind(); }

void ApiServer::stop() { impl_->server.stop(); }

bool ApiServer::is_running() const { return impl_->server.is_running(); }

}  // namespace vllm::entrypoints::openai
