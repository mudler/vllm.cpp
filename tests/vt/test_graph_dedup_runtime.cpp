// vllm.cpp original — CPU-tier contract for the two device-free halves of the graph
// dedup runtime binding: the error-latch discipline (src/vt/graph_dedup_latch.h) and the
// structural-signature walk (src/vt/graph_dedup_signature.h).
// Row ENG-CUDAGRAPH-DEDUP, issues #1162 and #1184.
//
// WHY THIS SUITE EXISTS. Until #1184 the whole of src/vt/graph_dedup_runtime.h was
// reached by NO test on any tier: `grep -rn graph_dedup_runtime tests/` returned
// nothing, and `cuda-fat-build` proved only that it compiled. Two things hid inside that
// gap. The first is the defect: twelve runtime calls that this file is DESIGNED to see
// fail — the cudaGraphExecUpdate probe refusing a fold is the feature working, not an
// exception — none of which consumed the runtime's sticky per-thread error, so the next
// unrelated kernel launched with the ordinary `Check(cudaGetLastError())` pattern
// reported OUR refusal as its own failure. On GB10 that presented 6/6 as
// `greedy_argmax launch: invalid device function` from a launch that had succeeded. The
// second is the signature walk itself: Kahn ordering, the topological re-index, the
// sorted edge emission and the depth bound were all unexecuted by anything.
//
// WHAT THIS SUITE CANNOT PROVE, STATED PLAINLY. A CPU test drives a FAKE runtime. It
// cannot observe the CUDA runtime's real latched-error state, so it cannot prove that
// cudaGetLastError is the right call, that hipGetLastError has the same semantics, or
// that #1184 is gone on a device. What it does prove is the STRUCTURE the fix rests on:
// that every entry point clears on every exit path including an unwinding one, that a
// refusal inside the registry leaves nothing latched for the next caller, and that no
// table field holds an address that skips the guard. The device half — a same-binary
// VT_CUDA_GRAPH_DEDUP off/on A/B that survives a real capture — stays owed under #1162
// and is recorded in .agents/specs/eng-cudagraph-dedup.md, not claimed here.
#include <doctest/doctest.h>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vt/graph_dedup.h"
#include "vt/graph_dedup_latch.h"
#include "vt/graph_dedup_signature.h"

namespace {

// ===================================================================================
// Part 1 — the error-latch discipline.
// ===================================================================================

// A stand-in for the CUDA runtime's sticky per-thread error slot. A failing fake call
// SETS it; only a clear consumes it. That is the whole of the mechanism #1184 turned on:
// a return value does not consume the latch, which is why cudaGraphLaunch could return
// success while a stale code was still waiting for the next kernel to read it.
struct FakeLatch {
  bool latched = false;
  int clears = 0;
  // Arming flags: which fake operation is made to fail on its next call.
  bool fail_signature = false;
  bool fail_instantiate = false;
  bool refuse_update = false;
  bool fail_destroy = false;
  bool fail_launch = false;
};

FakeLatch* g_latch = nullptr;

void Latch() { g_latch->latched = true; }

// The policy vt/graph_dedup_latch.h installs at every entry point.
struct CountingRuntime {
  static void ClearLatchedError() {
    ++g_latch->clears;
    g_latch->latched = false;
  }
};

// Stands in for the very next `kernel<<<>>>(...); Check(cudaGetLastError())` the engine
// runs after the capture path returns. This is the caller that #1184 blamed.
void NextUnrelatedKernelLaunch() {
  if (g_latch->latched) {
    throw std::runtime_error("greedy_argmax launch: invalid device function");
  }
}

// --- the fake device, one that latches on every failure -----------------------------
struct LatchGraph {
  std::string sig;
  int id = 0;
};
struct LatchExec {
  int id = 0;
};

std::vector<LatchExec*> g_execs;

std::string RawSignature(void* raw_graph) {
  if (g_latch->fail_signature) Latch();
  return static_cast<LatchGraph*>(raw_graph)->sig;
}

void* RawInstantiate(void* raw_graph) {
  if (g_latch->fail_instantiate) {
    Latch();
    return nullptr;
  }
  auto* exec = new LatchExec{static_cast<LatchGraph*>(raw_graph)->id};
  g_execs.push_back(exec);
  return exec;
}

bool RawUpdate(void* exec, void* raw_graph, std::string* detail) {
  if (g_latch->refuse_update) {
    Latch();
    if (detail != nullptr) *detail = "err=1 result=1";
    return false;
  }
  static_cast<LatchExec*>(exec)->id = static_cast<LatchGraph*>(raw_graph)->id;
  return true;
}

void RawDestroyExec(void* exec) {
  if (g_latch->fail_destroy) Latch();
  delete static_cast<LatchExec*>(exec);
}

void RawDestroyGraph(void* raw_graph) {
  if (g_latch->fail_destroy) Latch();
  (void)raw_graph;
}

void RawLaunch(void* exec, void* stream) {
  (void)stream;
  if (g_latch->fail_launch) {
    Latch();
    // Exactly the shape of the production Launch: the failure is reported by throwing,
    // which means the clear has to survive an unwinding stack.
    VT_CHECK(false, "graph dedup: graph launch failed");
  }
  (void)exec;
}

vt::GraphDedupOps GuardedTable() {
  return vt::graph_dedup_latch::MakeLatchGuardedOps<CountingRuntime, &RawSignature,
                                                    &RawInstantiate, &RawUpdate,
                                                    &RawDestroyExec, &RawDestroyGraph,
                                                    &RawLaunch>();
}

// ===================================================================================
// Part 2 — the structural signature walk.
// ===================================================================================

struct SigGraph;

struct SigNode {
  std::string payload;
  const SigGraph* child = nullptr;
  bool payload_fails = false;
};

// An edge endpoint that is deliberately NOT a member of the graph reporting it, so the
// `edge?` degradation escape has something to fire on.
const SigNode kAlien{"alien", nullptr, false};

struct SigGraph {
  std::vector<const SigNode*> nodes;
  // Index pairs into `nodes`; -1 on either side means "report the alien node instead".
  std::vector<std::pair<int, int>> edges;
  bool nodes_fail = false;
  bool edges_fail = false;
};

struct FakeSigRuntime {
  using Graph = const SigGraph*;
  using Node = const SigNode*;

  static bool GetNodes(Graph graph, std::vector<Node>* out) {
    if (graph->nodes_fail) return false;
    *out = graph->nodes;
    return true;
  }

  static bool GetEdges(Graph graph, std::vector<Node>* from, std::vector<Node>* to) {
    if (graph->edges_fail) return false;
    from->clear();
    to->clear();
    for (const auto& edge : graph->edges) {
      from->push_back(edge.first < 0 ? &kAlien : graph->nodes[static_cast<std::size_t>(edge.first)]);
      to->push_back(edge.second < 0 ? &kAlien : graph->nodes[static_cast<std::size_t>(edge.second)]);
    }
    return true;
  }

  static bool AppendNodePayload(Node node, std::string* out, Graph* child) {
    if (node->payload_fails) return false;
    out->append(node->payload);
    *child = node->child;
    return true;
  }
};

std::string SignatureOf(const SigGraph& graph) {
  return vt::graph_dedup_sig::Signature<FakeSigRuntime>(&graph);
}

}  // namespace

// ===================================================================================

TEST_CASE("every guarded operation consumes the latch its own failure set") {
  FakeLatch latch;
  g_latch = &latch;
  const vt::GraphDedupOps ops = GuardedTable();
  LatchGraph graph{"sig", 1};

  // Each arm: arm the failure, drive the operation through the TABLE (never the raw
  // function), then stand in the shoes of the next unrelated kernel.
  SUBCASE("signature") {
    latch.fail_signature = true;
    ops.signature(&graph);
  }
  SUBCASE("instantiate") {
    latch.fail_instantiate = true;
    CHECK(ops.instantiate(&graph) == nullptr);
  }
  SUBCASE("update") {
    latch.fail_instantiate = false;
    void* exec = ops.instantiate(&graph);
    latch.refuse_update = true;
    std::string detail;
    CHECK_FALSE(ops.update(exec, &graph, &detail));
    // The refusal still reports its reason: clearing the latch must not swallow the
    // driver's own explanation, which is what the caller acts on.
    CHECK(detail == "err=1 result=1");
    ops.destroy_exec(exec);
  }
  SUBCASE("destroy_exec") {
    void* exec = ops.instantiate(&graph);
    latch.fail_destroy = true;
    ops.destroy_exec(exec);
  }
  SUBCASE("destroy_graph") {
    latch.fail_destroy = true;
    ops.destroy_graph(&graph);
  }

  CHECK_FALSE(latch.latched);
  CHECK_NOTHROW(NextUnrelatedKernelLaunch());
  CHECK(latch.clears >= 1);
  g_latch = nullptr;
}

TEST_CASE("the clear survives an operation that throws") {
  // The production Launch reports a failed cudaGraphLaunch by throwing through
  // VT_CHECK. A clear placed after the call in the function body would never run on that
  // path; a clear in a destructor does. This is the case that decides between the two.
  FakeLatch latch;
  g_latch = &latch;
  const vt::GraphDedupOps ops = GuardedTable();
  LatchGraph graph{"sig", 1};
  void* exec = ops.instantiate(&graph);
  const int clears_before = latch.clears;

  latch.fail_launch = true;
  CHECK_THROWS_AS(ops.launch(exec, nullptr), std::runtime_error);

  CHECK(latch.clears == clears_before + 1);
  CHECK_FALSE(latch.latched);
  CHECK_NOTHROW(NextUnrelatedKernelLaunch());

  latch.fail_launch = false;
  ops.destroy_exec(exec);
  g_latch = nullptr;
}

TEST_CASE("a refused fold leaves nothing latched for the next kernel") {
  // #1184 in the shape a CPU tier can hold. The registry's REFUSED probe is normal
  // operation -- it is what "the signature is only a lookup key" means -- so the whole
  // safety argument for this feature depends on a refusal being free of side effects
  // outside the registry. Before the fix the refusal latched, nothing consumed it, and
  // the next unrelated kernel launch reported it as its own failure.
  FakeLatch latch;
  g_latch = &latch;
  vt::GraphDedupRegistry registry(GuardedTable(), nullptr);

  LatchGraph first{"same", 1};
  LatchGraph second{"same", 2};

  void* handle_a = registry.Register(&first);
  CHECK_NOTHROW(NextUnrelatedKernelLaunch());

  latch.refuse_update = true;  // the driver declines to fold the second capture
  void* handle_b = registry.Register(&second);
  CHECK_NOTHROW(NextUnrelatedKernelLaunch());
  // The refusal did what it is supposed to do: two groups, not one.
  CHECK(registry.ExecCount() == 2);
  latch.refuse_update = false;

  registry.Replay(handle_a, nullptr);
  CHECK_NOTHROW(NextUnrelatedKernelLaunch());
  registry.Replay(handle_b, nullptr);
  CHECK_NOTHROW(NextUnrelatedKernelLaunch());

  registry.Close();
  CHECK_NOTHROW(NextUnrelatedKernelLaunch());
  g_latch = nullptr;
}

TEST_CASE("no table field holds an address that skips the guard") {
  // The guard is only structural if the raw addresses cannot reach the table. Assigning
  // `table.signature = &RawSignature` anywhere would compile and would silently restore
  // the defect for that one operation, so the distinctness is asserted rather than
  // trusted to review.
  FakeLatch latch;
  g_latch = &latch;
  const vt::GraphDedupOps ops = GuardedTable();
  CHECK(ops.signature != &RawSignature);
  CHECK(ops.instantiate != &RawInstantiate);
  CHECK(ops.update != &RawUpdate);
  CHECK(ops.destroy_exec != &RawDestroyExec);
  CHECK(ops.destroy_graph != &RawDestroyGraph);
  CHECK(ops.launch != &RawLaunch);
  g_latch = nullptr;
}

TEST_CASE("an operation the guarded builder does not wire cannot reach the registry") {
  // The backstop for the seventh operation. If GraphDedupOps grows a member and
  // MakeLatchGuardedOps is not extended, the field stays null and the registry refuses
  // to construct -- so the bypass is not merely discouraged, it does not run.
  FakeLatch latch;
  g_latch = &latch;
  vt::GraphDedupOps partial = GuardedTable();
  partial.update = nullptr;
  CHECK_THROWS_AS(vt::GraphDedupRegistry(partial, nullptr), std::runtime_error);
  g_latch = nullptr;
}

// --- the signature walk ------------------------------------------------------------

TEST_CASE("the signature is independent of the order the runtime reports nodes in") {
  // THE load-bearing property of the walk. cudaGraphGetNodes promises no order, so two
  // captures of one topology can arrive with their nodes permuted. Without the Kahn
  // re-index they would key differently, fold nothing, and the only observable would be
  // the device-side "captured N graphs, deduped to N execs" line -- exactly the silent
  // mode the spec records.
  const SigNode a{"A,"};
  const SigNode b{"B,"};
  const SigNode c{"C,"};

  SigGraph forward;
  forward.nodes = {&a, &b, &c};
  forward.edges = {{0, 1}, {1, 2}};

  SigGraph reversed;
  reversed.nodes = {&c, &b, &a};
  reversed.edges = {{2, 1}, {1, 0}};  // the same A->B->C chain, reported back to front

  CHECK(SignatureOf(forward) == SignatureOf(reversed));
  CHECK(SignatureOf(forward) == "[A,;B,;C,;]0,1,;1,2,;");
}

TEST_CASE("the signature is independent of the order the runtime reports edges in") {
  const SigNode a{"A,"};
  const SigNode b{"B,"};
  const SigNode c{"C,"};
  const SigNode d{"D,"};

  SigGraph one;
  one.nodes = {&a, &b, &c, &d};
  one.edges = {{0, 1}, {0, 2}, {1, 3}, {2, 3}};  // a diamond

  SigGraph other;
  other.nodes = {&a, &b, &c, &d};
  other.edges = {{2, 3}, {1, 3}, {0, 2}, {0, 1}};  // the same diamond, edges scrambled

  CHECK(SignatureOf(one) == SignatureOf(other));
  CHECK(SignatureOf(one) == "[A,;B,;C,;D,;]0,1,;0,2,;1,3,;2,3,;");
}

TEST_CASE("the signature separates topologies that differ only in their edges") {
  const SigNode a{"A,"};
  const SigNode b{"B,"};
  const SigNode c{"C,"};

  SigGraph chain;
  chain.nodes = {&a, &b, &c};
  chain.edges = {{0, 1}, {1, 2}};

  SigGraph fan;
  fan.nodes = {&a, &b, &c};
  fan.edges = {{0, 1}, {0, 2}};

  CHECK(SignatureOf(chain) != SignatureOf(fan));
}

TEST_CASE("the signature separates topologies that differ only in a node payload") {
  const SigNode a{"A,"};
  const SigNode b{"B,"};
  const SigNode b_other{"B',"};

  SigGraph one;
  one.nodes = {&a, &b};
  one.edges = {{0, 1}};

  SigGraph other;
  other.nodes = {&a, &b_other};
  other.edges = {{0, 1}};

  CHECK(SignatureOf(one) != SignatureOf(other));
}

TEST_CASE("a child graph contributes its own signature, bounded at depth four") {
  // The bound is exact on both sides: level 4's payload is emitted and its child is not
  // walked. A bound of three would drop "L4," and a bound of five would admit "L5,".
  SigGraph level[6];
  SigNode node[6];
  for (int i = 5; i >= 0; --i) {
    node[i].payload = "L" + std::to_string(i) + ",";
    if (i < 5) node[i].child = &level[i + 1];
    level[i].nodes = {&node[i]};
  }

  const std::string signature = SignatureOf(level[0]);
  CHECK(signature.find("L0,") != std::string::npos);
  CHECK(signature.find("L4,") != std::string::npos);
  CHECK(signature.find("L5,") == std::string::npos);
}

TEST_CASE("a child graph is walked, not merely noted") {
  const SigNode leaf{"LEAF,"};
  SigGraph child;
  child.nodes = {&leaf};

  const SigNode parent_node{"P,", &child};
  SigGraph parent;
  parent.nodes = {&parent_node};

  CHECK(SignatureOf(parent) == "[P,[LEAF,;];]");
}

TEST_CASE("each runtime failure degrades the key instead of aborting inside a capture") {
  // Five escapes, five exact strings. Degradation is deliberate: a coarse key costs a
  // wasted probe, and the probe is the authority, so the walk must never throw from
  // inside a stream capture.
  const SigNode a{"A,"};
  const SigNode b{"B,"};

  SUBCASE("the node query fails") {
    SigGraph graph;
    graph.nodes = {&a};
    graph.nodes_fail = true;
    CHECK(SignatureOf(graph) == "nodes?;");
  }
  SUBCASE("the edge query fails") {
    SigGraph graph;
    graph.nodes = {&a, &b};
    graph.edges = {{0, 1}};
    graph.edges_fail = true;
    CHECK(SignatureOf(graph) == "edges?;");
  }
  SUBCASE("an edge names a node the graph did not report") {
    SigGraph graph;
    graph.nodes = {&a, &b};
    graph.edges = {{0, -1}};
    CHECK(SignatureOf(graph) == "edge?;");
  }
  SUBCASE("the reported edges describe a cycle") {
    SigGraph graph;
    graph.nodes = {&a, &b};
    graph.edges = {{0, 1}, {1, 0}};
    CHECK(SignatureOf(graph) == "cycle?;");
  }
  SUBCASE("a node payload query fails") {
    const SigNode broken{"X,", nullptr, true};
    SigGraph graph;
    graph.nodes = {&a, &broken};
    graph.edges = {{0, 1}};
    CHECK(SignatureOf(graph) == "[A,;node?;");
  }
}

TEST_CASE("an empty graph still produces a signature") {
  SigGraph graph;
  CHECK(SignatureOf(graph) == "[]");
}
