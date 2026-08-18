// vllm.cpp original — the device-free half of the graph dedup structural signature.
//
// Row ENG-CUDAGRAPH-DEDUP, issues #1162 and #1184, spec
// .agents/specs/eng-cudagraph-dedup.md.
// Ported from SGLang's dedup mixin at pin f63458b5be:
// python/sglang/srt/model_executor/runner_backend/cuda_graph_dedup_mixin.py:139-179.
//
// WHY THIS IS SPLIT OUT OF graph_dedup_runtime.h. The topology walk — Kahn ordering,
// the topological re-index, the sorted edge emission, the child-graph depth bound and
// the four degradation escapes — is ordinary graph code with no CUDA in it. Leaving it
// inside a header that includes <cuda_runtime.h> made it reachable by NO test on any
// tier: `cuda-fat-build` proved it compiled and nothing proved it was right. That gap is
// what the spec recorded under `## Risks/decisions` and what this file closes; the
// device-shaped part (node types, kernel and memcpy and memset parameters, the runtime's
// own node and edge queries) stays behind the `Rt` policy and is still device-only.
//
// THE POLICY CONTRACT. `Rt` supplies:
//
//   using Graph = ...;   // a handle type comparable against nullptr
//   using Node  = ...;   // a handle type usable as an unordered_map key
//   static bool GetNodes(Graph, std::vector<Node>* out);
//   static bool GetEdges(Graph, std::vector<Node>* from, std::vector<Node>* to);
//   static bool AppendNodePayload(Node, std::string* out, Graph* child);
//
// Each returns false on a runtime failure the walk is allowed to degrade around, and
// `AppendNodePayload` sets `*child` non-null exactly for a child-graph node, which the
// walk recurses into subject to `kMaxChildDepth`. Nothing here interprets a payload; the
// signature is a LOOKUP KEY and cudaGraphExecUpdate is the authority, so a key that is
// too coarse costs a wasted probe and can never make a replay wrong.
#ifndef VT_GRAPH_DEDUP_SIGNATURE_H_
#define VT_GRAPH_DEDUP_SIGNATURE_H_

#include <algorithm>
#include <cstddef>
#include <functional>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vt::graph_dedup_sig {

// Bounded: a capture nested past this depth degrades to a coarser key, which the probe
// still guards. Unbounded recursion over driver-owned structure is not a risk worth
// taking inside a capture path.
inline constexpr int kMaxChildDepth = 4;

inline void AppendNumber(std::string* out, long long value) {
  out->append(std::to_string(value));
  out->push_back(',');
}

// cuda_graph_dedup_mixin.py:139-179. Node order as the runtime reports it is not a
// contract, so the nodes are re-indexed by a deterministic topological order (Kahn,
// always taking the lowest available index, exactly as the upstream heapq does) and the
// edge set is emitted in that order's terms.
template <class Rt>
inline void AppendGraphSignature(typename Rt::Graph graph, std::string* out, int depth) {
  using Node = typename Rt::Node;
  using Graph = typename Rt::Graph;

  std::vector<Node> nodes;
  if (!Rt::GetNodes(graph, &nodes)) {
    out->append("nodes?;");
    return;
  }
  const std::size_t num_nodes = nodes.size();

  std::vector<Node> from;
  std::vector<Node> to;
  if (!Rt::GetEdges(graph, &from, &to)) {
    out->append("edges?;");
    return;
  }
  const std::size_t num_edges = std::min(from.size(), to.size());

  std::unordered_map<Node, int> index;
  index.reserve(num_nodes * 2);
  for (std::size_t i = 0; i < num_nodes; ++i) index[nodes[i]] = static_cast<int>(i);

  std::vector<std::vector<int>> children(num_nodes);
  std::vector<int> indegree(num_nodes, 0);
  std::vector<std::pair<int, int>> edges;
  edges.reserve(num_edges);
  for (std::size_t e = 0; e < num_edges; ++e) {
    const auto src = index.find(from[e]);
    const auto dst = index.find(to[e]);
    if (src == index.end() || dst == index.end()) {
      out->append("edge?;");
      return;
    }
    children[static_cast<std::size_t>(src->second)].push_back(dst->second);
    ++indegree[static_cast<std::size_t>(dst->second)];
    edges.emplace_back(src->second, dst->second);
  }

  std::priority_queue<int, std::vector<int>, std::greater<int>> ready;
  for (std::size_t i = 0; i < num_nodes; ++i) {
    if (indegree[i] == 0) ready.push(static_cast<int>(i));
  }
  std::vector<int> order;
  order.reserve(num_nodes);
  while (!ready.empty()) {
    const int current = ready.top();
    ready.pop();
    order.push_back(current);
    for (int child : children[static_cast<std::size_t>(current)]) {
      if (--indegree[static_cast<std::size_t>(child)] == 0) ready.push(child);
    }
  }
  if (order.size() != num_nodes) {
    // A cycle is impossible in a captured graph; if the runtime ever reports one, the
    // key degrades rather than the process aborting inside capture.
    out->append("cycle?;");
    return;
  }

  std::vector<int> topo(num_nodes, 0);
  for (std::size_t i = 0; i < order.size(); ++i) {
    topo[static_cast<std::size_t>(order[i])] = static_cast<int>(i);
  }

  out->push_back('[');
  for (int node_index : order) {
    Graph child = nullptr;
    if (!Rt::AppendNodePayload(nodes[static_cast<std::size_t>(node_index)], out, &child)) {
      out->append("node?;");
      return;
    }
    if (child != nullptr && depth < kMaxChildDepth) {
      AppendGraphSignature<Rt>(child, out, depth + 1);
    }
    out->push_back(';');
  }
  out->push_back(']');

  std::vector<std::pair<int, int>> topo_edges;
  topo_edges.reserve(edges.size());
  for (const auto& edge : edges) {
    topo_edges.emplace_back(topo[static_cast<std::size_t>(edge.first)],
                            topo[static_cast<std::size_t>(edge.second)]);
  }
  std::sort(topo_edges.begin(), topo_edges.end());
  for (const auto& edge : topo_edges) {
    AppendNumber(out, edge.first);
    AppendNumber(out, edge.second);
    out->push_back(';');
  }
}

template <class Rt>
inline std::string Signature(typename Rt::Graph graph) {
  std::string out;
  out.reserve(4096);
  AppendGraphSignature<Rt>(graph, &out, 0);
  return out;
}

}  // namespace vt::graph_dedup_sig

#endif  // VT_GRAPH_DEDUP_SIGNATURE_H_
