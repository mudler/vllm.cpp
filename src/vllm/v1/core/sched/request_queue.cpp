// Ported from: vllm/v1/core/sched/request_queue.py @ e24d1b24
// See include/vllm/v1/core/sched/request_queue.h for the ownership /
// FCFS-semantics / deferred-priority notes.
#include "vllm/v1/core/sched/request_queue.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>

#include "vllm/v1/request.h"

namespace vllm::v1 {

namespace {

// The std::*_heap algorithms build a MAX-heap under the given comparator, so
// heap_.front() is the element for which `comp(front, x)` is false for all x —
// i.e. the "greatest" per comp. We want front() to be the SMALLEST element per
// RequestPriorityLess (the next request to schedule), so we pass a comparator
// that is the reverse of RequestPriorityLess: HeapComp(a, b) is true iff
// `a` should be ordered AFTER `b` (a > b per __lt__ == RequestPriorityLess(b, a)).
struct HeapComp {
  bool operator()(const Request* a, const Request* b) const {
    return RequestPriorityLess(b, a);
  }
};

}  // namespace

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

// ---------------------------------------------------------------------------
// PriorityRequestQueue — heap by RequestPriorityLess (Request.__lt__).
// ---------------------------------------------------------------------------

void PriorityRequestQueue::add_request(Request* request) {
  // heapq.heappush(self._heap, request).
  heap_.push_back(request);
  std::push_heap(heap_.begin(), heap_.end(), HeapComp{});
}

Request* PriorityRequestQueue::pop_request() {
  // heapq.heappop: raises IndexError on an empty heap.
  if (heap_.empty()) {
    throw std::out_of_range("pop from empty heap");
  }
  std::pop_heap(heap_.begin(), heap_.end(), HeapComp{});
  Request* request = heap_.back();
  heap_.pop_back();
  return request;
}

Request* PriorityRequestQueue::peek_request() const {
  // self._heap[0]: raises IndexError on an empty heap.
  if (heap_.empty()) {
    throw std::out_of_range("peek from empty heap");
  }
  return heap_.front();
}

void PriorityRequestQueue::prepend_request(Request* request) {
  // Upstream: a priority queue has no front to prepend to — this is a plain
  // heap insert (order re-derived from (priority, arrival_time)).
  add_request(request);
}

void PriorityRequestQueue::prepend_requests(const RequestQueue& requests) {
  // Upstream: add every request according to the priority policy.
  for (Request* request : requests.ToList()) {
    add_request(request);
  }
}

void PriorityRequestQueue::remove_request(Request* request) {
  // self._heap.remove(request); heapq.heapify(self._heap). list.remove raises
  // ValueError when the element is absent.
  auto it = std::find(heap_.begin(), heap_.end(), request);
  if (it == heap_.end()) {
    throw std::invalid_argument("request not in queue");
  }
  heap_.erase(it);
  std::make_heap(heap_.begin(), heap_.end(), HeapComp{});
}

void PriorityRequestQueue::remove_requests(
    const std::vector<Request*>& requests) {
  const std::unordered_set<Request*> to_remove(requests.begin(),
                                               requests.end());
  std::vector<Request*> filtered;
  filtered.reserve(heap_.size());
  for (Request* request : heap_) {
    if (to_remove.find(request) == to_remove.end()) {
      filtered.push_back(request);
    }
  }
  heap_.swap(filtered);
  std::make_heap(heap_.begin(), heap_.end(), HeapComp{});
}

std::size_t PriorityRequestQueue::size() const { return heap_.size(); }

bool PriorityRequestQueue::empty() const { return heap_.empty(); }

std::vector<Request*> PriorityRequestQueue::ToList() const {
  // __iter__ yields heappop of a copy -> the elements in priority (pop) order.
  // RequestPriorityLess is a strict total order, so sorting a copy by it yields
  // exactly that sequence (and matches upstream heapq's extraction order).
  std::vector<Request*> snapshot(heap_.begin(), heap_.end());
  std::sort(snapshot.begin(), snapshot.end(),
            [](const Request* a, const Request* b) {
              return RequestPriorityLess(a, b);
            });
  return snapshot;
}

std::unique_ptr<RequestQueue> create_request_queue(SchedulingPolicy policy) {
  switch (policy) {
    case SchedulingPolicy::kFCFS:
      return std::make_unique<FCFSRequestQueue>();
    case SchedulingPolicy::kPriority:
      return std::make_unique<PriorityRequestQueue>();
  }
  throw std::invalid_argument("Unknown scheduling policy");
}

}  // namespace vllm::v1
