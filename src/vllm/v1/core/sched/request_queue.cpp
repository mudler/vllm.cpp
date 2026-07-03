// Ported from: vllm/v1/core/sched/request_queue.py @ e24d1b24
// See include/vllm/v1/core/sched/request_queue.h for the ownership /
// FCFS-semantics / deferred-priority notes.
#include "vllm/v1/core/sched/request_queue.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>

#include "vllm/v1/request.h"

namespace vllm::v1 {

void FCFSRequestQueue::add_request(Request* request) {
  // FCFS: append to the back.
  queue_.push_back(request);
}

Request* FCFSRequestQueue::pop_request() {
  // popleft(): raises IndexError on an empty deque.
  if (queue_.empty()) {
    throw std::out_of_range("pop from an empty queue");
  }
  Request* request = queue_.front();
  queue_.pop_front();
  return request;
}

Request* FCFSRequestQueue::peek_request() const {
  if (queue_.empty()) {
    throw std::out_of_range("peek from an empty queue");
  }
  return queue_.front();
}

void FCFSRequestQueue::prepend_request(Request* request) {
  // appendleft(): a preempted request is re-queued to the FRONT.
  queue_.push_front(request);
}

void FCFSRequestQueue::prepend_requests(const RequestQueue& requests) {
  // extendleft(requests): each element is pushed to the front in turn, so the
  // requests end up in REVERSE order of their appearance in `requests`.
  for (Request* request : requests.ToList()) {
    queue_.push_front(request);
  }
}

void FCFSRequestQueue::remove_request(Request* request) {
  // deque.remove(request): removes the first matching occurrence (raises
  // ValueError if not present).
  auto it = std::find(queue_.begin(), queue_.end(), request);
  if (it == queue_.end()) {
    throw std::invalid_argument("request not in queue");
  }
  queue_.erase(it);
}

void FCFSRequestQueue::remove_requests(
    const std::vector<Request*>& requests) {
  const std::unordered_set<Request*> to_remove(requests.begin(),
                                               requests.end());
  // deque does not support in-place filtering, so rebuild in FCFS order.
  std::deque<Request*> filtered;
  for (Request* request : queue_) {
    if (to_remove.find(request) == to_remove.end()) {
      filtered.push_back(request);
    }
  }
  queue_.swap(filtered);
}

std::size_t FCFSRequestQueue::size() const { return queue_.size(); }

bool FCFSRequestQueue::empty() const { return queue_.empty(); }

std::vector<Request*> FCFSRequestQueue::ToList() const {
  return std::vector<Request*>(queue_.begin(), queue_.end());
}

std::unique_ptr<RequestQueue> create_request_queue(SchedulingPolicy policy) {
  switch (policy) {
    case SchedulingPolicy::kFCFS:
      return std::make_unique<FCFSRequestQueue>();
    case SchedulingPolicy::kPriority:
      // DEFERRED (T1): PriorityRequestQueue is not ported in T0.
      throw std::invalid_argument(
          "priority scheduling policy is not implemented (deferred to M1.4 "
          "T1)");
  }
  throw std::invalid_argument("Unknown scheduling policy");
}

}  // namespace vllm::v1
