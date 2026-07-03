// Tests for the FCFS RequestQueue port
// (vllm/v1/core/sched/request_queue.py @ e24d1b24).
//
// Upstream has no dedicated request_queue unit test; these behaviors are the
// ones the scheduler relies on and that upstream's test_scheduler.py exercises
// indirectly (FCFS admission order, and preemption re-queue to the FRONT via
// scheduler.py::_preempt_request -> self.waiting.prepend_request). Ported as
// direct queue tests.
#include <doctest/doctest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "vllm/sampling_params.h"
#include "vllm/v1/core/sched/request_queue.h"
#include "vllm/v1/request.h"

using vllm::SamplingParams;
using vllm::v1::create_request_queue;
using vllm::v1::FCFSRequestQueue;
using vllm::v1::Request;
using vllm::v1::RequestQueue;
using vllm::v1::SchedulingPolicy;

namespace {

// Build a minimal Request (the queue only ever stores/returns the pointer).
Request MakeRequest(const std::string& id) {
  return Request(id, /*prompt_token_ids=*/{1, 2, 3}, SamplingParams{},
                 /*arrival_time=*/0.0);
}

}  // namespace

TEST_CASE("create_request_queue(kFCFS) returns an FCFS queue") {
  std::unique_ptr<RequestQueue> queue =
      create_request_queue(SchedulingPolicy::kFCFS);
  CHECK(queue != nullptr);
  CHECK(queue->empty());
  CHECK(queue->size() == 0);
}

TEST_CASE("create_request_queue(kPriority) throws (deferred to T1)") {
  CHECK_THROWS(create_request_queue(SchedulingPolicy::kPriority));
}

TEST_CASE("FCFS add_request / pop_request preserve arrival order") {
  FCFSRequestQueue queue;
  Request a = MakeRequest("A");
  Request b = MakeRequest("B");
  Request c = MakeRequest("C");

  queue.add_request(&a);
  queue.add_request(&b);
  queue.add_request(&c);

  CHECK(queue.size() == 3);
  CHECK_FALSE(queue.empty());

  // add A, B, C -> pop A, B, C.
  CHECK(queue.pop_request() == &a);
  CHECK(queue.pop_request() == &b);
  CHECK(queue.pop_request() == &c);
  CHECK(queue.empty());
}

TEST_CASE("FCFS pop_request on an empty queue throws") {
  FCFSRequestQueue queue;
  CHECK_THROWS(queue.pop_request());
}

TEST_CASE("FCFS peek_request returns the front without removing it") {
  FCFSRequestQueue queue;
  Request a = MakeRequest("A");
  Request b = MakeRequest("B");
  queue.add_request(&a);
  queue.add_request(&b);

  CHECK(queue.peek_request() == &a);
  CHECK(queue.peek_request() == &a);  // idempotent — no removal
  CHECK(queue.size() == 2);
  CHECK(queue.pop_request() == &a);
}

TEST_CASE("FCFS peek_request on an empty queue throws") {
  FCFSRequestQueue queue;
  CHECK_THROWS(queue.peek_request());
}

TEST_CASE("FCFS prepend_request puts a preempted request at the FRONT") {
  // Mirrors scheduler.py::_preempt_request re-queueing to self.waiting so the
  // preempted request is retried before the rest of the waiting queue.
  FCFSRequestQueue queue;
  Request a = MakeRequest("A");
  Request b = MakeRequest("B");
  Request x = MakeRequest("X");  // the preempted request
  queue.add_request(&a);
  queue.add_request(&b);

  queue.prepend_request(&x);

  // prepend X -> pop X first, then A, B.
  CHECK(queue.pop_request() == &x);
  CHECK(queue.pop_request() == &a);
  CHECK(queue.pop_request() == &b);
}

TEST_CASE("FCFS prepend_requests prepends in reverse order (extendleft)") {
  FCFSRequestQueue queue;
  Request a = MakeRequest("A");
  queue.add_request(&a);

  FCFSRequestQueue other;
  Request x = MakeRequest("X");
  Request y = MakeRequest("Y");
  Request z = MakeRequest("Z");
  other.add_request(&x);
  other.add_request(&y);
  other.add_request(&z);

  queue.prepend_requests(other);

  // extendleft([X, Y, Z]) onto [A] -> [Z, Y, X, A].
  CHECK(queue.pop_request() == &z);
  CHECK(queue.pop_request() == &y);
  CHECK(queue.pop_request() == &x);
  CHECK(queue.pop_request() == &a);
}

TEST_CASE("FCFS remove_request removes a specific request") {
  FCFSRequestQueue queue;
  Request a = MakeRequest("A");
  Request b = MakeRequest("B");
  Request c = MakeRequest("C");
  queue.add_request(&a);
  queue.add_request(&b);
  queue.add_request(&c);

  queue.remove_request(&b);

  CHECK(queue.size() == 2);
  CHECK(queue.pop_request() == &a);
  CHECK(queue.pop_request() == &c);
}

TEST_CASE("FCFS remove_request on a missing request throws") {
  FCFSRequestQueue queue;
  Request a = MakeRequest("A");
  Request missing = MakeRequest("missing");
  queue.add_request(&a);
  CHECK_THROWS(queue.remove_request(&missing));
}

TEST_CASE("FCFS remove_requests removes a set, preserving order") {
  FCFSRequestQueue queue;
  Request a = MakeRequest("A");
  Request b = MakeRequest("B");
  Request c = MakeRequest("C");
  Request d = MakeRequest("D");
  queue.add_request(&a);
  queue.add_request(&b);
  queue.add_request(&c);
  queue.add_request(&d);

  queue.remove_requests({&b, &d});

  CHECK(queue.size() == 2);
  CHECK(queue.pop_request() == &a);
  CHECK(queue.pop_request() == &c);
}

TEST_CASE("FCFS size / empty track the queue") {
  FCFSRequestQueue queue;
  CHECK(queue.empty());
  CHECK(queue.size() == 0);

  Request a = MakeRequest("A");
  queue.add_request(&a);
  CHECK_FALSE(queue.empty());
  CHECK(queue.size() == 1);

  queue.pop_request();
  CHECK(queue.empty());
  CHECK(queue.size() == 0);
}

TEST_CASE("FCFS iteration (begin/end + ToList) is in FCFS order") {
  FCFSRequestQueue queue;
  Request a = MakeRequest("A");
  Request b = MakeRequest("B");
  Request c = MakeRequest("C");
  queue.add_request(&a);
  queue.add_request(&b);
  queue.add_request(&c);

  std::vector<Request*> via_iter;
  for (Request* r : queue) {
    via_iter.push_back(r);
  }
  CHECK(via_iter == std::vector<Request*>{&a, &b, &c});
  CHECK(queue.ToList() == std::vector<Request*>{&a, &b, &c});
  // Iteration does not consume the queue.
  CHECK(queue.size() == 3);
}
