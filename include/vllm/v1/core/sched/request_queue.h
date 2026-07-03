// Ported from: vllm/v1/core/sched/request_queue.py @ e24d1b24
//
// Scope (M1.4 Task 1): the FCFS RequestQueue the V1 Scheduler drives — the
// waiting queue that holds admitted-but-not-yet-running requests and, via
// prepend_request, the re-queue target for FCFS-preempted requests. Behavioral
// only.
//
// Ownership: upstream the queue holds Request objects directly (deque[Request]);
// here the Scheduler owns the Request objects and the queue holds raw
// Request* pointers (non-owning). Lifetime is the Scheduler's responsibility,
// exactly as upstream where the queue and the running list alias the same
// Request instances.
//
// FCFS semantics (mirrored 1:1):
//   add_request     -> append (back)      : new arrivals join the tail
//   pop_request     -> popleft (front)    : next to schedule is the oldest
//   peek_request    -> queue[0]           : front without removing (throws if
//                                           empty, like Python's IndexError)
//   prepend_request -> appendleft (front) : a PREEMPTED request is re-queued to
//                                           the FRONT so it is retried first
//                                           (scheduler.py::_preempt_request ->
//                                            self.waiting.prepend_request)
//   prepend_requests-> extendleft         : prepends in REVERSE order of the
//                                           other queue (deque.extendleft)
//
// DEFERRED (T1): PriorityRequestQueue (the heap ordered by (priority,
// arrival_time)). create_request_queue(kPriority) throws — priority scheduling
// is not ported in T0.
#pragma once

#include <cstddef>
#include <deque>
#include <memory>
#include <vector>

namespace vllm::v1 {

struct Request;

// Enum for scheduling policies (mirrors upstream SchedulingPolicy(Enum)).
enum class SchedulingPolicy {
  kFCFS,      // "fcfs"
  kPriority,  // "priority" — DEFERRED (T1).
};

// Abstract base class for request queues.
class RequestQueue {
 public:
  virtual ~RequestQueue() = default;

  // Add a request to the queue according to the policy.
  virtual void add_request(Request* request) = 0;
  // Pop a request from the queue according to the policy.
  virtual Request* pop_request() = 0;
  // Peek at the request at the front of the queue without removing it.
  virtual Request* peek_request() const = 0;
  // Prepend a request to the front of the queue.
  virtual void prepend_request(Request* request) = 0;
  // Prepend all requests from another queue to the front of this queue.
  virtual void prepend_requests(const RequestQueue& requests) = 0;
  // Remove a specific request from the queue.
  virtual void remove_request(Request* request) = 0;
  // Remove multiple specific requests from the queue.
  virtual void remove_requests(const std::vector<Request*>& requests) = 0;

  // __len__ / __bool__.
  virtual std::size_t size() const = 0;
  virtual bool empty() const = 0;

  // __iter__: a snapshot of the queue in policy order (front-to-back for FCFS).
  // C++ stand-in for Python `list(queue)` — used by prepend_requests and by
  // the scheduler's remove-loops.
  virtual std::vector<Request*> ToList() const = 0;
};

// A first-come-first-served queue backed by a std::deque<Request*>.
class FCFSRequestQueue final : public RequestQueue {
 public:
  void add_request(Request* request) override;
  Request* pop_request() override;
  Request* peek_request() const override;
  void prepend_request(Request* request) override;
  void prepend_requests(const RequestQueue& requests) override;
  void remove_request(Request* request) override;
  void remove_requests(const std::vector<Request*>& requests) override;
  std::size_t size() const override;
  bool empty() const override;
  std::vector<Request*> ToList() const override;

  // Deque-style iteration in FCFS order (front = next to pop).
  using const_iterator = std::deque<Request*>::const_iterator;
  const_iterator begin() const { return queue_.begin(); }
  const_iterator end() const { return queue_.end(); }

 private:
  std::deque<Request*> queue_;
};

// Create a request queue based on the scheduling policy.
std::unique_ptr<RequestQueue> create_request_queue(SchedulingPolicy policy);

}  // namespace vllm::v1
