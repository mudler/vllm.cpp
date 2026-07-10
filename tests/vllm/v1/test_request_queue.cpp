// Tests for the RequestQueue port
// (vllm/v1/core/sched/request_queue.py @ e24d1b24).
//
// FCFS: upstream has no dedicated FCFS request_queue unit test; these behaviors
// are the ones the scheduler relies on and that upstream's test_scheduler.py
// exercises indirectly (FCFS admission order, and preemption re-queue to the
// FRONT via scheduler.py::_preempt_request -> self.waiting.prepend_request).
//
// PRIORITY (W4 / ENG-PRIORITY-SCHED): the PriorityRequestQueue heap-ordering
// behavior + the seeded random ordering/heap-property test ported from
// tests/v1/core/test_priority_scheduler_random.py @ e24d1b24 (reduced to the
// queue-level heap-property/ordering property per the async-serving spec's
// "Tests to port"). Ordering mirrors Request.__lt__ ((priority, arrival_time,
// request_id, identity)); lower priority value = popped first.
#include <doctest/doctest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "vllm/sampling_params.h"
#include "vllm/v1/core/sched/request_queue.h"
#include "vllm/v1/request.h"

using vllm::SamplingParams;
using vllm::v1::create_request_queue;
using vllm::v1::FCFSRequestQueue;
using vllm::v1::PriorityRequestQueue;
using vllm::v1::Request;
using vllm::v1::RequestPriorityLess;
using vllm::v1::RequestQueue;
using vllm::v1::SchedulingPolicy;

namespace {

// Build a minimal Request (the queue only ever stores/returns the pointer).
Request MakeRequest(const std::string& id) {
  return Request(id, /*prompt_token_ids=*/{1, 2, 3}, SamplingParams{},
                 /*arrival_time=*/0.0);
}

// Build a Request carrying a scheduling priority + arrival time (the priority
// queue orders by (priority, arrival_time, request_id)).
Request MakePriorityRequest(const std::string& id, int priority,
                            double arrival) {
  return Request(id, /*prompt_token_ids=*/{1, 2, 3}, SamplingParams{}, arrival,
                 /*block_hasher=*/nullptr, priority);
}

}  // namespace

TEST_CASE("create_request_queue(kFCFS) returns an FCFS queue") {
  std::unique_ptr<RequestQueue> queue =
      create_request_queue(SchedulingPolicy::kFCFS);
  CHECK(queue != nullptr);
  CHECK(queue->empty());
  CHECK(queue->size() == 0);
}

TEST_CASE("create_request_queue(kPriority) returns a priority queue") {
  std::unique_ptr<RequestQueue> queue =
      create_request_queue(SchedulingPolicy::kPriority);
  CHECK(queue != nullptr);
  CHECK(queue->empty());
  CHECK(queue->size() == 0);
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

// ===========================================================================
// PriorityRequestQueue (W4 / ENG-PRIORITY-SCHED).
// Ported from vllm/v1/core/sched/request_queue.py::PriorityRequestQueue and the
// heap-property/ordering assertions in test_priority_scheduler_random.py +
// test_scheduler.py::test_priority_scheduling_heap_property. Ordering mirrors
// Request.__lt__: lower `priority` first, then earlier arrival_time, then
// smaller request_id.
// ===========================================================================

TEST_CASE("Priority pop order: lower priority value is served first") {
  PriorityRequestQueue queue;
  // Add in non-priority order (priorities 2, 0, 1); distinct arrival times.
  Request a = MakePriorityRequest("A", /*priority=*/2, /*arrival=*/1.0);
  Request b = MakePriorityRequest("B", /*priority=*/0, /*arrival=*/2.0);
  Request c = MakePriorityRequest("C", /*priority=*/1, /*arrival=*/3.0);
  queue.add_request(&a);
  queue.add_request(&b);
  queue.add_request(&c);

  CHECK(queue.size() == 3);
  // Popped in priority order: B(0), C(1), A(2).
  CHECK(queue.pop_request() == &b);
  CHECK(queue.pop_request() == &c);
  CHECK(queue.pop_request() == &a);
  CHECK(queue.empty());
}

TEST_CASE("Priority tiebreak: equal priority -> earlier arrival first") {
  PriorityRequestQueue queue;
  Request a = MakePriorityRequest("A", /*priority=*/1, /*arrival=*/3.0);
  Request b = MakePriorityRequest("B", /*priority=*/1, /*arrival=*/1.0);
  Request c = MakePriorityRequest("C", /*priority=*/1, /*arrival=*/2.0);
  queue.add_request(&a);
  queue.add_request(&b);
  queue.add_request(&c);

  // Same priority -> arrival order: B(1.0), C(2.0), A(3.0).
  CHECK(queue.pop_request() == &b);
  CHECK(queue.pop_request() == &c);
  CHECK(queue.pop_request() == &a);
}

TEST_CASE("Priority tiebreak: equal priority+arrival -> smaller request_id") {
  PriorityRequestQueue queue;
  Request a = MakePriorityRequest("2", /*priority=*/1, /*arrival=*/1.0);
  Request b = MakePriorityRequest("0", /*priority=*/1, /*arrival=*/1.0);
  Request c = MakePriorityRequest("1", /*priority=*/1, /*arrival=*/1.0);
  queue.add_request(&a);
  queue.add_request(&b);
  queue.add_request(&c);

  // Ids compared lexicographically: "0", "1", "2".
  CHECK(queue.pop_request() == &b);
  CHECK(queue.pop_request() == &c);
  CHECK(queue.pop_request() == &a);
}

TEST_CASE("Priority peek returns the min without removing (idempotent)") {
  PriorityRequestQueue queue;
  Request a = MakePriorityRequest("A", 3, 1.0);
  Request b = MakePriorityRequest("B", 0, 2.0);
  queue.add_request(&a);
  queue.add_request(&b);

  CHECK(queue.peek_request() == &b);  // priority 0 is the min
  CHECK(queue.peek_request() == &b);  // idempotent
  CHECK(queue.size() == 2);
  CHECK(queue.pop_request() == &b);
}

TEST_CASE("Priority pop/peek on an empty queue throws") {
  PriorityRequestQueue queue;
  CHECK_THROWS(queue.pop_request());
  CHECK_THROWS(queue.peek_request());
}

TEST_CASE("Priority prepend_request is a heap insert (no front jump)") {
  // Upstream: a priority queue has no concept of prepending to the front;
  // prepend_request just inserts by (priority, arrival_time).
  PriorityRequestQueue queue;
  Request hi = MakePriorityRequest("hi", 0, 1.0);
  Request lo = MakePriorityRequest("lo", 5, 2.0);
  queue.add_request(&hi);

  // A low-priority "prepended" request does NOT jump ahead of hi.
  Request mid = MakePriorityRequest("mid", 3, 3.0);
  queue.prepend_request(&mid);
  queue.add_request(&lo);
  CHECK(queue.pop_request() == &hi);   // priority 0
  CHECK(queue.pop_request() == &mid);  // priority 3
  CHECK(queue.pop_request() == &lo);   // priority 5
}

TEST_CASE("Priority prepend_requests inserts all by priority order") {
  PriorityRequestQueue queue;
  Request a = MakePriorityRequest("A", 2, 1.0);
  queue.add_request(&a);

  FCFSRequestQueue other;  // ToList() gives FCFS order; priority re-derives it.
  Request x = MakePriorityRequest("X", 5, 2.0);
  Request y = MakePriorityRequest("Y", 0, 3.0);
  Request z = MakePriorityRequest("Z", 1, 4.0);
  other.add_request(&x);
  other.add_request(&y);
  other.add_request(&z);

  queue.prepend_requests(other);

  // All merged by priority: Y(0), Z(1), A(2), X(5).
  CHECK(queue.pop_request() == &y);
  CHECK(queue.pop_request() == &z);
  CHECK(queue.pop_request() == &a);
  CHECK(queue.pop_request() == &x);
}

TEST_CASE("Priority remove_request removes a specific request, keeps order") {
  PriorityRequestQueue queue;
  Request a = MakePriorityRequest("A", 0, 1.0);
  Request b = MakePriorityRequest("B", 1, 2.0);
  Request c = MakePriorityRequest("C", 2, 3.0);
  queue.add_request(&a);
  queue.add_request(&b);
  queue.add_request(&c);

  queue.remove_request(&b);

  CHECK(queue.size() == 2);
  CHECK(queue.pop_request() == &a);  // still priority order
  CHECK(queue.pop_request() == &c);
}

TEST_CASE("Priority remove_request on a missing request throws") {
  PriorityRequestQueue queue;
  Request a = MakePriorityRequest("A", 0, 1.0);
  Request missing = MakePriorityRequest("missing", 0, 2.0);
  queue.add_request(&a);
  CHECK_THROWS(queue.remove_request(&missing));
}

TEST_CASE("Priority remove_requests removes a set, keeps heap order") {
  PriorityRequestQueue queue;
  Request a = MakePriorityRequest("A", 0, 1.0);
  Request b = MakePriorityRequest("B", 1, 2.0);
  Request c = MakePriorityRequest("C", 2, 3.0);
  Request d = MakePriorityRequest("D", 3, 4.0);
  queue.add_request(&a);
  queue.add_request(&b);
  queue.add_request(&c);
  queue.add_request(&d);

  queue.remove_requests({&b, &d});

  CHECK(queue.size() == 2);
  CHECK(queue.pop_request() == &a);
  CHECK(queue.pop_request() == &c);
}

TEST_CASE("Priority ToList returns a snapshot in priority (pop) order") {
  PriorityRequestQueue queue;
  Request a = MakePriorityRequest("A", 3, 1.0);
  Request b = MakePriorityRequest("B", 1, 2.0);
  Request c = MakePriorityRequest("C", 2, 3.0);
  Request d = MakePriorityRequest("D", 0, 4.0);
  queue.add_request(&a);
  queue.add_request(&b);
  queue.add_request(&c);
  queue.add_request(&d);

  // ToList (upstream __iter__ = heappop of a copy) is sorted by priority.
  CHECK(queue.ToList() == std::vector<Request*>{&d, &b, &c, &a});
  // ToList does not consume the queue.
  CHECK(queue.size() == 4);
  CHECK(queue.pop_request() == &d);
}

// test_priority_scheduling_heap_property (queue level): add in random priority
// order, pop all, the popped priorities come out sorted ascending.
TEST_CASE("Priority queue pops requests in sorted priority order") {
  PriorityRequestQueue queue;
  const std::vector<int> priorities = {5, 1, 8, 3, 2, 7, 4, 6};
  std::vector<std::unique_ptr<Request>> storage;
  for (std::size_t i = 0; i < priorities.size(); ++i) {
    storage.push_back(std::make_unique<Request>(MakePriorityRequest(
        std::to_string(i), priorities[i], static_cast<double>(i))));
    queue.add_request(storage.back().get());
  }

  std::vector<int> popped;
  while (!queue.empty()) {
    popped.push_back(queue.pop_request()->priority);
  }
  std::vector<int> expected = priorities;
  std::sort(expected.begin(), expected.end());
  CHECK(popped == expected);
}

// Ported from test_priority_scheduler_random.py (reduced to the queue-level
// heap-property/ordering property): under random push/remove interleavings the
// queue always pops in the RequestPriorityLess total order, and peek == the
// next pop. Runs several fixed seeds for reproducibility.
TEST_CASE("Priority queue random property: pops in total order (seeded)") {
  for (unsigned seed : {1u, 7u, 42u, 1234u, 99999u}) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> prio(-3, 3);
    std::uniform_real_distribution<double> arr(0.0, 1.0);

    // Build a pool of distinct requests (unique request ids).
    const int n = 200;
    std::vector<std::unique_ptr<Request>> storage;
    storage.reserve(n);
    for (int i = 0; i < n; ++i) {
      storage.push_back(std::make_unique<Request>(
          MakePriorityRequest("req-" + std::to_string(i), prio(rng), arr(rng))));
    }

    PriorityRequestQueue queue;
    // Push all in pool order, then remove a random subset before draining, to
    // exercise remove_request's re-heapify alongside push/pop.
    for (auto& r : storage) queue.add_request(r.get());

    std::vector<Request*> removed;
    for (auto& r : storage) {
      if ((rng() & 7u) == 0u) {  // ~1/8 removed
        queue.remove_request(r.get());
        removed.push_back(r.get());
      }
    }
    const std::size_t remaining = storage.size() - removed.size();
    CHECK(queue.size() == remaining);

    // Drain: each popped element must be <= the next in the total order, and
    // peek must equal the element about to be popped.
    Request* prev = nullptr;
    std::size_t count = 0;
    while (!queue.empty()) {
      Request* peeked = queue.peek_request();
      Request* got = queue.pop_request();
      CHECK(peeked == got);
      if (prev != nullptr) {
        // Non-decreasing: the previous pop must NOT be greater than this one,
        // i.e. got is not strictly-less than prev.
        CHECK_FALSE(RequestPriorityLess(got, prev));
      }
      prev = got;
      ++count;
    }
    CHECK(count == remaining);
  }
}
