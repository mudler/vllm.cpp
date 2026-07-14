// Ported from: vllm/entrypoints/openai/api_server.py @ e24d1b24 + the
// per-endpoint api_router modules. See api_server.h for scope + the cpp-httplib
// dependency deviation.
#include "vllm/entrypoints/openai/api_server.h"

#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

#include <httplib/httplib.h>
#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"

namespace vllm::entrypoints::openai {

namespace {

size_t HttpWorkerCount(size_t max_concurrent_streams) {
  if (max_concurrent_streams == 0) {
    throw std::invalid_argument("max_concurrent_streams must be positive");
  }
  if (max_concurrent_streams >
      std::numeric_limits<size_t>::max() -
          ApiServer::kControlWorkerHeadroom) {
    throw std::invalid_argument("max_concurrent_streams is too large");
  }
  return max_concurrent_streams + ApiServer::kControlWorkerHeadroom;
}

size_t HttpWorkerCount(size_t max_concurrent_streams,
                       ApiServer::HttpWorkerPoolMode mode) {
  const size_t fixed_count = HttpWorkerCount(max_concurrent_streams);
  return mode == ApiServer::HttpWorkerPoolMode::kCapacityFixed ? fixed_count
                                                               : 0;
}

}  // namespace

// Opaque httplib::Server (pimpl — keeps httplib.h out of api_server.h).
struct ApiServer::Impl {
  Impl(size_t max_concurrent_streams, HttpWorkerPoolMode mode)
      : http_worker_count(HttpWorkerCount(max_concurrent_streams, mode)) {
    // cpp-httplib's default pool starts at hardware_concurrency()-1 and only
    // grows if idle_thread_count_ is exactly zero at enqueue. A burst can queue
    // accepted sockets while that counter is stale-positive; long-lived SSE
    // jobs then prevent the queued sockets from ever being read. A fixed floor
    // derived from the configured stream capacity removes that race and makes
    // resource use reproducible.
    if (http_worker_count != 0) {
      server.new_task_queue = [workers = http_worker_count]() {
        return new httplib::ThreadPool(workers);
      };
    }
  }

  httplib::Server server;
  size_t http_worker_count;
  // The legacy LLMEngine serving constructors remain for small synthetic
  // tests and embedding compatibility. Unlike AsyncLLM, that engine is driven
  // synchronously by its caller, so retain one shared lock for that seam only.
  // Production handlers use AsyncLLM and never take this request-level lock.
  std::mutex legacy_engine_mutex;
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
                     std::string version, size_t max_concurrent_streams,
                     HttpWorkerPoolMode worker_pool_mode)
    : completion_(completion),
      chat_(chat),
      models_(models),
      version_(std::move(version)),
      impl_(std::make_unique<Impl>(max_concurrent_streams, worker_pool_mode)) {}

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
    std::unique_lock<std::mutex> legacy_lock(impl_->legacy_engine_mutex,
                                             std::defer_lock);
    if (!completion_.uses_async_engine()) legacy_lock.lock();
    result = completion_.create_completion(request);
  } catch (const std::exception& e) {
    // DISCRIMINATOR: attribute a 500 to its endpoint + model + raw cause so a
    // benchmark driver that only sees the generic HTTP body can still recover
    // the true failure. std::cerr only (survives SIGKILL escalation).
    std::cerr << "api-server: 500 endpoint=/v1/completions model="
              << request.model.value_or("") << " what=" << e.what() << "\n";
    return MakeError(500, "InternalServerError", e.what());
  }

  DispatchResult out;
  if (result.streaming) {
    out.streaming = true;
    out.content_type = "text/event-stream";
    out.sse_chunks = std::move(result.sse_chunks);
    out.sse_stream = std::move(result.sse_stream);
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
    std::unique_lock<std::mutex> legacy_lock(impl_->legacy_engine_mutex,
                                             std::defer_lock);
    if (!chat_.uses_async_engine()) legacy_lock.lock();
    result = chat_.create_chat_completion(request);
  } catch (const std::exception& e) {
    std::cerr << "api-server: 500 endpoint=/v1/chat/completions model="
              << request.model.value_or("") << " what=" << e.what() << "\n";
    return MakeError(500, "InternalServerError", e.what());
  }

  DispatchResult out;
  if (result.streaming) {
    out.streaming = true;
    out.content_type = "text/event-stream";
    out.sse_chunks = std::move(result.sse_chunks);
    out.sse_stream = std::move(result.sse_stream);
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
  // Upstream calls engine_client.check_health() before returning an empty 200.
  // This bounded server currently exposes process liveness only.
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
      if (result.sse_stream != nullptr) {
        // W2 live StreamingResponse: one provider invocation pulls one
        // per-request collector output. A slow/disconnected client occupies
        // only its httplib worker; AsyncLLM keeps batching other requests.
        std::shared_ptr<SseStream> stream = result.sse_stream;
        res.set_chunked_content_provider(
            result.content_type,
            [stream](size_t /*offset*/, httplib::DataSink& sink) -> bool {
              try {
                std::string chunk;
                if (!stream->next(chunk)) {
                  sink.done();
                  return true;
                }
                if (!sink.write(chunk.data(), chunk.size())) {
                  stream->abort();
                  return false;
                }
                return true;
              } catch (...) {
                // MID-FLIGHT WITNESS: a provider exception here means the live
                // stream died after headers were already sent (the client sees
                // a truncated body, not a 500). Rethrow-to-inspect so the raw
                // cause reaches stderr before the abort.
                try {
                  std::rethrow_exception(std::current_exception());
                } catch (const std::exception& e) {
                  std::cerr << "sse: stream aborted mid-flight: " << e.what()
                            << "\n";
                } catch (...) {
                  std::cerr << "sse: stream aborted mid-flight: unknown error\n";
                }
                stream->abort();
                return false;
              }
            },
            [stream](bool success) {
              if (!success) stream->abort();
            });
        return;
      }

      // Legacy synchronous compatibility/test seam: write the precomputed
      // chunks exactly as before.
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

size_t ApiServer::http_worker_count() const {
  return impl_->http_worker_count;
}

}  // namespace vllm::entrypoints::openai
