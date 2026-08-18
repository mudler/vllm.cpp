// vllm.cpp original — the CUDA / HIP binding for the graph-executable dedup registry.
//
// Row ENG-CUDAGRAPH-DEDUP, issues #1162 and #1184, spec
// .agents/specs/eng-cudagraph-dedup.md.
// Ported from SGLang's dedup mixin at pin f63458b5be:
// python/sglang/srt/model_executor/runner_backend/cuda_graph_dedup_mixin.py
//   :27-37   dedup_update            -> Update
//   :105-114 kernel_node_payload     -> AppendKernelPayload
//   :117-136 graph_node_payload      -> Runtime::AppendNodePayload
//   :139-179 graph_signature         -> graph_dedup_sig::AppendGraphSignature
//
// ONE source, two runtimes. The CUDA and HIP graph APIs differ only by symbol prefix
// and by two call shapes, so binding them with an alias block keeps this a single path
// instead of the hand-written parallel path AGENTS.md forbids. Include it from a .cu
// for the CUDA leg, or define VT_GRAPH_DEDUP_HIP first and include it from a .hip.
//
// THIS FILE IS ALLOWED TO SEE THE RUNTIME FAIL, AND THAT IS WHY IT NEEDS A LATCH GUARD.
// The cudaGraphExecUpdate probe refusing a fold is the feature working, not an
// exception, and the topology walk has four more escapes that degrade the key rather
// than abort inside a capture. Every one of those latches an error code in the runtime's
// sticky per-thread slot, and until #1184 nothing here consumed it, so the next
// unrelated kernel reported OUR refusal as its own failure. The clear is therefore not
// placed beside each fallible call — a thirteenth fallible call would miss it — but in a
// destructor wrapped around every entry point, by `vt/graph_dedup_latch.h`. Ops() below
// is the file's ONLY export, and it is built solely through MakeLatchGuardedOps, so no
// raw function address in this file ever reaches a GraphDedupOps field.
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

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "vt/graph_dedup.h"
#include "vt/graph_dedup_latch.h"
#include "vt/graph_dedup_signature.h"

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

using vt::graph_dedup_sig::AppendNumber;

// cudaGraphGetEdges gained a fifth `cudaGraphEdgeData*` parameter in CUDA 13; the
// four-argument form is what CUDA 12 and HIP ship. Both shapes are bound here rather
// than at the call sites so the topology walk stays one piece of code. This is
// the one place a local compile against CUDA 12 headers could not have caught, and did
// not: the `cuda-fat-build` job on nvidia/cuda:13.3.0 is what reported it.
inline bool GetEdgesRaw(Graph graph, Node* from, Node* to, std::size_t* num_edges) {
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

// The `Rt` policy vt/graph_dedup_signature.h walks a graph through. Everything
// device-shaped lives here; the ordering, re-indexing and edge emission do not.
struct Runtime {
  using Graph = vt::graph_dedup_rt::Graph;
  using Node = vt::graph_dedup_rt::Node;

  // Consume the runtime's sticky per-thread error. Called from a destructor at every
  // entry point (vt/graph_dedup_latch.h) rather than beside each fallible call. On the
  // HIP leg VTGD_FN resolves this to hipGetLastError, which has the same latch
  // semantics; one line covers both arms because there is one source.
  static void ClearLatchedError() { VTGD_FN(GetLastError)(); }

  static bool GetNodes(Graph graph, std::vector<Node>* out) {
    std::size_t num_nodes = 0;
    if (VTGD_FN(GraphGetNodes)(graph, nullptr, &num_nodes) != VTGD_SUCCESS) return false;
    out->assign(num_nodes, Node{});
    if (num_nodes > 0 &&
        VTGD_FN(GraphGetNodes)(graph, out->data(), &num_nodes) != VTGD_SUCCESS) {
      return false;
    }
    return true;
  }

  static bool GetEdges(Graph graph, std::vector<Node>* from, std::vector<Node>* to) {
    std::size_t num_edges = 0;
    if (!GetEdgesRaw(graph, nullptr, nullptr, &num_edges)) return false;
    from->assign(num_edges, Node{});
    to->assign(num_edges, Node{});
    if (num_edges > 0 && !GetEdgesRaw(graph, from->data(), to->data(), &num_edges)) {
      return false;
    }
    return true;
  }

  // cuda_graph_dedup_mixin.py:117-136, the same five cases in the same order. `*child`
  // is set only for a child-graph node; the depth bound that governs recursing into it
  // belongs to the walk, not to this switch.
  static bool AppendNodePayload(Node node, std::string* out, Graph* child) {
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
        Graph nested = nullptr;
        if (VTGD_FN(GraphChildGraphNodeGetGraph)(node, &nested) != VTGD_SUCCESS) {
          return false;
        }
        *child = nested;
        return true;
      }
      default:
        return true;
    }
  }
};

// The six operations, BEFORE the latch guard. Ops() is what the backends see, and it
// wraps every one of these; nothing outside this namespace takes their addresses.
namespace unguarded {

inline std::string Signature(void* raw_graph) {
  return vt::graph_dedup_sig::Signature<Runtime>(static_cast<Graph>(raw_graph));
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

}  // namespace unguarded

inline const GraphDedupOps& Ops() {
  static const GraphDedupOps ops =
      vt::graph_dedup_latch::MakeLatchGuardedOps<Runtime, &unguarded::Signature,
                                                 &unguarded::Instantiate,
                                                 &unguarded::Update,
                                                 &unguarded::DestroyExec,
                                                 &unguarded::DestroyGraph,
                                                 &unguarded::Launch>();
  return ops;
}

}  // namespace vt::graph_dedup_rt

#endif  // VT_GRAPH_DEDUP_RUNTIME_H_
