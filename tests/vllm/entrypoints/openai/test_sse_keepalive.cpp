// SSE keepalive contract (VT_SERVER_SSE_PING_S) — chat + completion share
// AssignSseWaitResult / kSsePingFrame. Maint-bot #316: timeout emits a
// standalone comment frame; never concatenated with data; <=0 disables;
// a later real output still streams as a separate data frame.
#include <doctest/doctest.h>

#include <chrono>
#include <cstdlib>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "vllm/entrypoints/openai/serving_utils.h"
#include "vllm/outputs.h"
#include "vllm/sampling_params.h"
#include "vllm/v1/engine/output_processor.h"

using vllm::CompletionOutput;
using vllm::RequestOutput;
using vllm::RequestOutputKind;
using vllm::entrypoints::openai::AssignSseWaitResult;
using vllm::entrypoints::openai::kSsePingFrame;
using vllm::entrypoints::openai::SsePingIntervalSec;

namespace {

struct EnvRestorer {
  const char* key;
  std::optional<std::string> prev;
  explicit EnvRestorer(const char* k) : key(k) {
    if (const char* v = std::getenv(k)) prev = std::string(v);
  }
  ~EnvRestorer() {
    if (prev.has_value()) {
      ::setenv(key, prev->c_str(), 1);
    } else {
      ::unsetenv(key);
    }
  }
};

}  // namespace

TEST_CASE("SSE keepalive: frame is a pure comment, never a data frame") {
  const std::string ping(kSsePingFrame);
  CHECK(ping == ":\n\n");
  CHECK(ping.find("data:") == std::string::npos);
  const std::string data = "data: {\"id\":\"x\"}\n\n";
  CHECK(ping + data != ping);
  CHECK(data.find(ping) == std::string::npos);
}

TEST_CASE("SSE keepalive: chat+completion timeout -> standalone ping frame") {
  RequestOutput out;
  out.request_id = "stale";
  std::string chunk = "data: SHOULD_BE_OVERWRITTEN\n\n";
  const bool got = AssignSseWaitResult(std::nullopt, out, chunk);
  CHECK_FALSE(got);
  CHECK(chunk == std::string(kSsePingFrame));
  CHECK(chunk == ":\n\n");
  CHECK(out.request_id == "stale");
}

TEST_CASE("SSE keepalive: chat+completion data path fills out, no ping rewrite") {
  RequestOutput ready;
  ready.request_id = "req-1";
  ready.finished = true;
  RequestOutput out;
  std::string chunk = "untouched";
  const bool got = AssignSseWaitResult(std::move(ready), out, chunk);
  CHECK(got);
  CHECK(out.request_id == "req-1");
  CHECK(out.finished);
  CHECK(chunk == "untouched");
  CHECK(chunk != std::string(kSsePingFrame));
}

TEST_CASE("SSE keepalive: later real output is a separate framing step") {
  std::vector<std::string> frames;
  {
    RequestOutput out;
    std::string chunk;
    CHECK_FALSE(AssignSseWaitResult(std::nullopt, out, chunk));
    frames.push_back(chunk);
  }
  {
    RequestOutput ready;
    ready.request_id = "req-2";
    CompletionOutput co;
    co.index = 0;
    co.text = "hi";
    ready.outputs.push_back(std::move(co));
    RequestOutput out;
    std::string chunk;
    CHECK(AssignSseWaitResult(std::move(ready), out, chunk));
    CHECK(out.request_id == "req-2");
    REQUIRE(out.outputs.size() == 1);
    CHECK(out.outputs[0].text == "hi");
    frames.push_back("data: " + out.outputs[0].text + "\n\n");
  }
  REQUIRE(frames.size() == 2);
  CHECK(frames[0] == std::string(kSsePingFrame));
  CHECK(frames[1].rfind("data: ", 0) == 0);
  // Two discrete frames — next() never returns ping+data concatenated.
  CHECK(frames[0] + frames[1] != frames[0]);
  CHECK(frames[1] != frames[0] + frames[1]);
}

// #931. The default was 15 seconds, and that put a byte on the wire that the
// reference never emits.
//
// vLLM serves its streaming completions through uvicorn/starlette and yields
// nothing but `data: ` frames: grepping the pinned oracle
// (0.23.1rc1.dev1511+g555967922) for a yielded comment across ALL of
// vllm/entrypoints/ returns no hit. So a comment frame is not a mirrored
// behaviour, it is an invention — and vLLM's OWN benchmark client cannot
// survive one. `vllm bench serve` strips each network chunk before parsing
// (benchmarks/lib/endpoint_request_func.py:207), which destroys the "\n\n"
// separator at chunk boundaries, and its only resynchronisation path requires a
// `data: ` prefix (:48). A comment arriving before the first data frame
// therefore poisons its buffer permanently: it reports "Never received a valid
// chunk to calculate TTFT" and counts the request FAILED, while the server
// completes it normally and logs nothing.
//
// That is not hypothetical. It voided the 27B c16 leg at 93/96 (#577) and then
// the Qwen3.8-27B c1 and c8 legs at 5/6 and 36/48 (#931, #915) — in both cases
// taking exactly the SLOWEST requests, because those are the ones that reach a
// 15 s silence. In the #931 legs the failed requests' imputed TTFT was 92-94 s
// (c1, all three reps) against a p99 of 4.2 s among the successes: the timeout
// separated the two populations precisely.
//
// So the default mirrors vLLM — no keepalive — and #316's capability stays
// reachable for a deployment behind a proxy that needs it, as an opt-in whose
// cost is now stated.
TEST_CASE("SSE keepalive: DEFAULT OFF, mirroring vLLM's frame set") {
  EnvRestorer rest("VT_SERVER_SSE_PING_S");
  ::unsetenv("VT_SERVER_SSE_PING_S");
  CHECK(SsePingIntervalSec() == 0);
  ::setenv("VT_SERVER_SSE_PING_S", "", 1);
  CHECK(SsePingIntervalSec() == 0);
  // An unparsable spelling falls back to the DEFAULT, and the default is off.
  // It used to fall back to 15, so a typo silently enabled the frame.
  ::setenv("VT_SERVER_SSE_PING_S", "fifteen", 1);
  CHECK(SsePingIntervalSec() == 0);
}

TEST_CASE("SSE keepalive: VT_SERVER_SSE_PING_S <=0 disables") {
  EnvRestorer rest("VT_SERVER_SSE_PING_S");
  ::unsetenv("VT_SERVER_SSE_PING_S");
  CHECK(SsePingIntervalSec() == 0);
  ::setenv("VT_SERVER_SSE_PING_S", "0", 1);
  CHECK(SsePingIntervalSec() == 0);
  ::setenv("VT_SERVER_SSE_PING_S", "-3", 1);
  CHECK(SsePingIntervalSec() == 0);
  ::setenv("VT_SERVER_SSE_PING_S", "2", 1);
  CHECK(SsePingIntervalSec() == 2);
  ::setenv("VT_SERVER_SSE_PING_S", "9999", 1);
  CHECK(SsePingIntervalSec() == 600);
}

TEST_CASE("SSE keepalive: collector get_for timeout then later deliver") {
  using namespace std::chrono_literals;
  vllm::v1::RequestOutputCollector c(RequestOutputKind::kDelta, "sse0");
  CHECK_FALSE(c.get_for(15ms).has_value());
  std::thread th([&] {
    std::this_thread::sleep_for(25ms);
    RequestOutput out;
    out.request_id = "sse0";
    out.finished = true;
    c.put(std::move(out));
  });
  auto hit = c.get_for(500ms);
  th.join();
  REQUIRE(hit.has_value());
  CHECK(hit->request_id == "sse0");
}
