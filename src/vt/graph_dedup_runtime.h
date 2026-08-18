// vllm.cpp original — the CUDA / HIP binding for the graph-executable dedup registry.
//
// Row ENG-CUDAGRAPH-DEDUP, issue #1162, spec .agents/specs/eng-cudagraph-dedup.md.
// Ported from SGLang's dedup mixin at pin f63458b5be:
// python/sglang/srt/model_executor/runner_backend/cuda_graph_dedup_mixin.py
//   :27-37   dedup_update            -> Update
//   :105-114 kernel_node_payload     -> AppendKernelPayload
//   :117-136 graph_node_payload      -> AppendNodePayload
//   :139-179 graph_signature         -> AppendGraphSignature
//
// ONE source, two runtimes. The CUDA and HIP graph APIs differ only by symbol prefix
// and by two call shapes, so binding them with an alias block keeps this a single path
// instead of the hand-written parallel path AGENTS.md forbids. Include it from a .cu
// for the CUDA leg, or define VT_GRAPH_DEDUP_HIP first and include it from a .hip.
//
// DELIBERATE ADAPTATIONS, both recorded in the spec's `## Upstream chain`:
//  * kernel identity is the host function POINTER (cudaKernelNodeParams::func) rather
//    than the demangled name SGLang reads through cuKernelGetName / cuFuncGetName. The
//    runtime API hands us the pointer directly, no driver-API dependency is added, and
//    pointer identity is strictly stronger than name identity — two instantiations of
//    one template can share a name prefix but never an address.
//  * SGLang's launch-attribute probing (cuGraphKernelNodeGetAttribute, :58-102) is
//    driver-API-only and is dropped. Dropping a signature component can only produce a
//    false HIT, which Register probes and rejects, never a false miss that silently
//    shares an executable.
#ifndef VT_GRAPH_DEDUP_RUNTIME_H_
#define VT_GRAPH_DEDUP_RUNTIME_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#include "vt/graph_dedup.h"

#if defined(VT_GRAPH_DEDUP_HIP)
#include <hip/hip_runtime.h>
#define VTGD_FN(name) hip##name
#define VTGD_SUCCESS hipSuccess
#define VTGD_NODE_KERNEL hipGraphNodeTypeKernel
#define VTGD_NODE_MEMCPY hipGraphNodeTypeMemcpy
#define VTGD_NODE_MEMSET hipGraphNodeTypeMemset
#define VTGD_NODE_GRAPH hipGraphNodeTypeGraph
namespace vt::graph_dedup_rt {
using Graph = hipGraph_t;
using GraphExec = hipGraphExec_t;
using Node = hipGraphNode_t;
using NodeType = hipGraphNodeType;
using KernelNodeParams = hipKernelNodeParams;
using MemcpyNodeParams = hipMemcpy3DParms;
using MemsetNodeParams = hipMemsetParams;
using Stream = hipStream_t;
}  // namespace vt::graph_dedup_rt
#else
#include <cuda_runtime.h>
#define VTGD_FN(name) cuda##name
#define VTGD_SUCCESS cudaSuccess
#define VTGD_NODE_KERNEL cudaGraphNodeTypeKernel
#define VTGD_NODE_MEMCPY cudaGraphNodeTypeMemcpy
#define VTGD_NODE_MEMSET cudaGraphNodeTypeMemset
#define VTGD_NODE_GRAPH cudaGraphNodeTypeGraph
namespace vt::graph_dedup_rt {
using Graph = cudaGraph_t;
using GraphExec = cudaGraphExec_t;
using Node = cudaGraphNode_t;
using NodeType = cudaGraphNodeType;
using KernelNodeParams = cudaKernelNodeParams;
using MemcpyNodeParams = cudaMemcpy3DParms;
using MemsetNodeParams = cudaMemsetParams;
using Stream = cudaStream_t;
}  // namespace vt::graph_dedup_rt
#endif

namespace vt::graph_dedup_rt {

// cudaGraphGetEdges gained a fifth `cudaGraphEdgeData*` parameter in CUDA 13; the
// four-argument form is what CUDA 12 and HIP ship. Both shapes are bound here rather
// than at the call sites so the topology walk below stays one piece of code. This is
// the one place a local compile against CUDA 12 headers could not have caught, and did
// not: the `cuda-fat-build` job on nvidia/cuda:13.3.0 is what reported it.
inline bool GetEdges(Graph graph, Node* from, Node* to, std::size_t* num_edges) {
#if defined(VT_GRAPH_DEDUP_HIP)
  return hipGraphGetEdges(graph, from, to, num_edges) == hipSuccess;
#elif CUDART_VERSION >= 13000
  // Queried with every pointer null; filled with a real buffer, because passing null
  // edge data to the filling call is not a shape the documentation promises. The data
  // is discarded: an edge's annotation is not part of the identity this key needs.
  std::vector<cudaGraphEdgeData> edge_data;
  cudaGraphEdgeData* data = nullptr;
  if (from != nullptr) {
    edge_data.resize(*num_edges);
    data = edge_data.data();
  }
  return cudaGraphGetEdges(graph, from, to, data, num_edges) == cudaSuccess;
#else
  return cudaGraphGetEdges(graph, from, to, num_edges) == cudaSuccess;
#endif
}

inline void AppendNumber(std::string* out, long long value) {
  out->append(std::to_string(value));
  out->push_back(',');
}

inline void AppendGraphSignature(Graph graph, std::string* out, int depth);

// (func, gridDim, blockDim, sharedMemBytes) — cuda_graph_dedup_mixin.py:105-114 minus
// the driver-API attribute tuple, per the header note.
inline bool AppendKernelPayload(Node node, std::string* out) {
  KernelNodeParams params{};
  if (VTGD_FN(GraphKernelNodeGetParams)(node, &params) != VTGD_SUCCESS) return false;
  AppendNumber(out, static_cast<long long>(reinterpret_cast<std::uintptr_t>(params.func)));
  AppendNumber(out, params.gridDim.x);
  AppendNumber(out, params.gridDim.y);
  AppendNumber(out, params.gridDim.z);
  AppendNumber(out, params.blockDim.x);
  AppendNumber(out, params.blockDim.y);
  AppendNumber(out, params.blockDim.z);
  AppendNumber(out, params.sharedMemBytes);
  return true;
}

// cuda_graph_dedup_mixin.py:117-136, the same five cases in the same order.
inline bool AppendNodePayload(Node node, std::string* out, int depth) {
  NodeType type{};
  if (VTGD_FN(GraphNodeGetType)(node, &type) != VTGD_SUCCESS) return false;
  AppendNumber(out, static_cast<long long>(type));
  switch (type) {
    case VTGD_NODE_KERNEL:
      return AppendKernelPayload(node, out);
    case VTGD_NODE_MEMCPY: {
      MemcpyNodeParams params{};
      if (VTGD_FN(GraphMemcpyNodeGetParams)(node, &params) != VTGD_SUCCESS) return false;
      AppendNumber(out, static_cast<long long>(params.kind));
      AppendNumber(out, static_cast<long long>(params.extent.width));
      AppendNumber(out, static_cast<long long>(params.extent.height));
      AppendNumber(out, static_cast<long long>(params.extent.depth));
      return true;
    }
    case VTGD_NODE_MEMSET: {
      MemsetNodeParams params{};
      if (VTGD_FN(GraphMemsetNodeGetParams)(node, &params) != VTGD_SUCCESS) return false;
      AppendNumber(out, static_cast<long long>(params.elementSize));
      AppendNumber(out, static_cast<long long>(params.width));
      AppendNumber(out, static_cast<long long>(params.height));
      return true;
    }
    case VTGD_NODE_GRAPH: {
      Graph child = nullptr;
      if (VTGD_FN(GraphChildGraphNodeGetGraph)(node, &child) != VTGD_SUCCESS) return false;
      // Bounded: a capture nested past this depth degrades to a coarser key, which the
      // probe still guards. Unbounded recursion over driver-owned structure is not a
      // risk worth taking inside a capture path.
      if (depth >= 4) return true;
      AppendGraphSignature(child, out, depth + 1);
      return true;
    }
    default:
      return true;
  }
}

// cuda_graph_dedup_mixin.py:139-179. Node order as the runtime reports it is not a
// contract, so the nodes are re-indexed by a deterministic topological order (Kahn,
// always taking the lowest available index, exactly as the upstream heapq does) and the
// edge set is emitted in that order's terms.
inline void AppendGraphSignature(Graph graph, std::string* out, int depth) {
  std::size_t num_nodes = 0;
  if (VTGD_FN(GraphGetNodes)(graph, nullptr, &num_nodes) != VTGD_SUCCESS) {
    out->append("nodes?;");
    return;
  }
  std::vector<Node> nodes(num_nodes);
  if (num_nodes > 0 &&
      VTGD_FN(GraphGetNodes)(graph, nodes.data(), &num_nodes) != VTGD_SUCCESS) {
    out->append("nodes?;");
    return;
  }

  std::size_t num_edges = 0;
  if (!GetEdges(graph, nullptr, nullptr, &num_edges)) {
    out->append("edges?;");
    return;
  }
  std::vector<Node> from(num_edges);
  std::vector<Node> to(num_edges);
  if (num_edges > 0 && !GetEdges(graph, from.data(), to.data(), &num_edges)) {
    out->append("edges?;");
    return;
  }

  std::unordered_map<const void*, int> index;
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
    if (!AppendNodePayload(nodes[static_cast<std::size_t>(node_index)], out, depth)) {
      out->append("node?;");
      return;
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

inline std::string Signature(void* raw_graph) {
  std::string out;
  out.reserve(4096);
  AppendGraphSignature(static_cast<Graph>(raw_graph), &out, 0);
  return out;
}

inline void* Instantiate(void* raw_graph) {
  GraphExec exec = nullptr;
#if defined(VT_GRAPH_DEDUP_HIP)
  if (hipGraphInstantiate(&exec, static_cast<Graph>(raw_graph), nullptr, nullptr, 0) !=
      hipSuccess) {
    return nullptr;
  }
#else
  if (cudaGraphInstantiate(&exec, static_cast<Graph>(raw_graph), 0) != cudaSuccess) {
    return nullptr;
  }
#endif
  return reinterpret_cast<void*>(exec);
}

// cuda_graph_dedup_mixin.py:27-37. False on any non-success, including the case where
// the call succeeds but the result is not "updated", which is the outcome that matters
// and the one a bare error check would miss.
inline bool Update(void* exec, void* raw_graph, std::string* detail) {
  auto handle = reinterpret_cast<GraphExec>(exec);
  auto graph = static_cast<Graph>(raw_graph);
#if defined(VT_GRAPH_DEDUP_HIP)
  hipGraphNode_t error_node = nullptr;
  hipGraphExecUpdateResult result = hipGraphExecUpdateSuccess;
  const hipError_t status = hipGraphExecUpdate(handle, graph, &error_node, &result);
  const bool ok = status == hipSuccess && result == hipGraphExecUpdateSuccess;
  if (!ok && detail != nullptr) {
    *detail = "err=" + std::to_string(static_cast<int>(status)) +
              " result=" + std::to_string(static_cast<int>(result));
  }
  return ok;
#elif CUDART_VERSION >= 12000
  cudaGraphExecUpdateResultInfo info{};
  const cudaError_t status = cudaGraphExecUpdate(handle, graph, &info);
  const bool ok = status == cudaSuccess && info.result == cudaGraphExecUpdateSuccess;
  if (!ok && detail != nullptr) {
    *detail = "err=" + std::to_string(static_cast<int>(status)) +
              " result=" + std::to_string(static_cast<int>(info.result));
  }
  return ok;
#else
  cudaGraphNode_t error_node = nullptr;
  cudaGraphExecUpdateResult result = cudaGraphExecUpdateSuccess;
  const cudaError_t status = cudaGraphExecUpdate(handle, graph, &error_node, &result);
  const bool ok = status == cudaSuccess && result == cudaGraphExecUpdateSuccess;
  if (!ok && detail != nullptr) {
    *detail = "err=" + std::to_string(static_cast<int>(status)) +
              " result=" + std::to_string(static_cast<int>(result));
  }
  return ok;
#endif
}

inline void DestroyExec(void* exec) {
  if (exec != nullptr) VTGD_FN(GraphExecDestroy)(reinterpret_cast<GraphExec>(exec));
}

inline void DestroyGraph(void* raw_graph) {
  if (raw_graph != nullptr) VTGD_FN(GraphDestroy)(static_cast<Graph>(raw_graph));
}

inline void Launch(void* exec, void* stream) {
  VT_CHECK(VTGD_FN(GraphLaunch)(reinterpret_cast<GraphExec>(exec),
                                static_cast<Stream>(stream)) == VTGD_SUCCESS,
           "graph dedup: graph launch failed");
}

inline const GraphDedupOps& Ops() {
  static const GraphDedupOps ops = [] {
    GraphDedupOps table;
    table.signature = &Signature;
    table.instantiate = &Instantiate;
    table.update = &Update;
    table.destroy_exec = &DestroyExec;
    table.destroy_graph = &DestroyGraph;
    table.launch = &Launch;
    return table;
  }();
  return ops;
}

}  // namespace vt::graph_dedup_rt

#endif  // VT_GRAPH_DEDUP_RUNTIME_H_
