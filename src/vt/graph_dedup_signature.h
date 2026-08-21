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

// ===================================================================================
// WHICH FIELDS A NODE PAYLOAD CARRIES — the key's coarseness, decided here rather than
// inside the device header so a CPU test drives the SAME function production does.
//
// Row ENG-CUDAGRAPH-DEDUP, issue #1226. The device gate of 2026-08-18 measured the fold
// never happening: `N == M` in every `VT_CUDA_GRAPH_DEDUP=1` cell over two and three
// DISTINCT padded decode buckets. The cause was the key, not the driver. A kernel node's
// payload carried `gridDim` and `blockDim` and a memcpy node's carried the copy extent,
// the padded batch dimension lives in exactly those fields, so two decode buckets never
// shared a signature, no candidate group ever formed and `cudaGraphExecUpdate` was NEVER
// ATTEMPTED. The row's premise — "the decode graphs of two padded batch sizes are
// usually the same node topology with different parameters" — was refuted by its own key.
//
// THE HYPOTHESIS THESE FUNCTIONS EXIST TO TEST: the key is stricter than the operation it
// guards. `cudaGraphExecUpdate` requires the TOPOLOGY to match and is designed to permit
// PARAMETER changes; a kernel node's launch configuration is a parameter. Under the
// coarse key the extent-shaped fields leave the payload, two buckets group, and the probe
// decides. That is a MEASUREMENT, and `VT_CUDA_GRAPH_DEDUP_COARSE_KEY` is default OFF
// until it has one.
//
// WHY COARSENING IS SAFE TO TRY AT ALL, and this is the whole argument: the signature is
// a LOOKUP KEY and never an authority. `GraphDedupRegistry::Register` probes every
// candidate with the real driver update on a THROWAWAY executable before it folds, and a
// refusal gives that capture its own executable. A key that groups two graphs the driver
// then rejects costs one wasted probe, never a wrong replay. Do not weaken that property
// to make this experiment succeed; it is what makes the experiment cheap.
//
// WHAT IS DROPPED, AND WHAT IS DELIBERATELY KEPT:
//
//  * DROPPED — `gridDim` and `blockDim`. These are the fields that carry the padded batch
//    dimension, and they are pure launch configuration: `cudaGraphExecKernelNodeSetParams`
//    exists to change them, and the update contract's kernel restrictions are about the
//    function's CONTEXT and its use of device-side launch, not about its geometry.
//  * DROPPED — the memcpy `extent`. The gate named it alongside the launch dimensions as
//    the second place the batch size enters the key; leaving it in would separate exactly
//    the buckets dropping the launch dimensions is meant to join.
//  * DROPPED — the memset `width`, for the same reason and no other. A memset that clears
//    a per-sequence buffer scales its width with the batch.
//  * KEPT — `func`. Pointer identity is load-bearing and strictly stronger than the
//    demangled name SGLang reads; two graphs that run DIFFERENT kernels are not a fold
//    the probe should ever be asked about, and dropping this would turn the key into "a
//    graph with N nodes" and make every capture in the process one candidate group.
//  * KEPT — `sharedMemBytes`, and this is the argued one rather than the swept one. It is
//    a kernel-node parameter like the dimensions are, so the same reasoning would drop it;
//    two facts say do not. First, it is NOT where the batch dimension lives — decode
//    dynamic shared memory is sized by head dimension and block geometry — so dropping it
//    buys the hypothesis nothing. Second, a dynamic shared-memory size above the 48 KiB
//    static limit is only legal for a function that opted in through
//    `cudaFuncAttributeMaxDynamicSharedMemorySize`, an attribute of the FUNCTION and not
//    of the node, so a changed size is the field most likely to make the driver
//    re-instantiate rather than re-point. Keeping it preserves a real discriminator at no
//    cost to the thing being measured, and it keeps this experiment to ONE variable.
//  * KEPT — the memcpy `kind` and the memset `elementSize` and `height`. The update
//    contract explicitly refuses a changed memcpy memory type and refuses to change any
//    memset that is not 1-D, so these three are fields the driver itself treats as
//    identity. A key that dropped them would manufacture probe refusals.
//
// The node TYPE is emitted by the caller before any of these run, so a kernel payload and
// a memcpy payload can never collide however few fields either one carries.

// (func, gridDim, blockDim, sharedMemBytes) — cuda_graph_dedup_mixin.py:105-114 minus the
// driver-API attribute tuple. Under `coarse`, (func, sharedMemBytes).
inline void AppendKernelFields(std::string* out, long long func, unsigned grid_x,
                               unsigned grid_y, unsigned grid_z, unsigned block_x,
                               unsigned block_y, unsigned block_z, unsigned shared_bytes,
                               bool coarse) {
  AppendNumber(out, func);
  if (!coarse) {
    AppendNumber(out, grid_x);
    AppendNumber(out, grid_y);
    AppendNumber(out, grid_z);
    AppendNumber(out, block_x);
    AppendNumber(out, block_y);
    AppendNumber(out, block_z);
  }
  AppendNumber(out, shared_bytes);
}

// (kind, extent) — under `coarse`, (kind).
inline void AppendMemcpyFields(std::string* out, long long kind, long long width,
                               long long height, long long depth, bool coarse) {
  AppendNumber(out, kind);
  if (coarse) return;
  AppendNumber(out, width);
  AppendNumber(out, height);
  AppendNumber(out, depth);
}

// (elementSize, width, height) — under `coarse`, (elementSize, height).
inline void AppendMemsetFields(std::string* out, long long element_size, long long width,
                               long long height, bool coarse) {
  AppendNumber(out, element_size);
  if (!coarse) AppendNumber(out, width);
  AppendNumber(out, height);
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
