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

#include <cstdio>
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

// ===================================================================================
// Part 3 — the key's coarseness (issue #1226).
// ===================================================================================
//
// These helpers call the PRODUCTION field-selection functions. `AppendKernelFields` is
// the one src/vt/graph_dedup_runtime.h reaches after cudaGraphKernelNodeGetParams fills
// its struct, so what is exercised below is the code that decides the real key and not a
// restatement of it in a fake. The device half that stays unreachable from here is the
// runtime query itself, and that limit is the same one the file's header note states.

std::string KernelPayloadFull(long long func, unsigned grid_x, unsigned grid_y,
                              unsigned grid_z, unsigned block_x, unsigned block_y,
                              unsigned block_z, unsigned shared, bool coarse) {
  std::string out;
  vt::graph_dedup_sig::AppendKernelFields(&out, func, grid_x, grid_y, grid_z, block_x,
                                          block_y, block_z, shared, coarse);
  return out;
}

// The one-dimensional shorthand every case below used before the y and z components
// were gated. It delegates rather than repeating the call, so a case that reaches for
// the short form and a case that reaches for the full one drive one code path.
std::string KernelPayload(long long func, unsigned grid, unsigned block, unsigned shared,
                          bool coarse) {
  return KernelPayloadFull(func, grid, 1, 1, block, 1, 1, shared, coarse);
}

std::string MemcpyPayloadFull(long long kind, long long width, long long height,
                              long long depth, bool coarse) {
  std::string out;
  vt::graph_dedup_sig::AppendMemcpyFields(&out, kind, width, height, depth, coarse);
  return out;
}

std::string MemcpyPayload(long long kind, long long width, bool coarse) {
  return MemcpyPayloadFull(kind, width, 1, 1, coarse);
}

std::string MemsetPayload(long long element_size, long long width, long long height,
                          bool coarse) {
  std::string out;
  vt::graph_dedup_sig::AppendMemsetFields(&out, element_size, width, height, coarse);
  return out;
}

// A one-kernel-node graph carrying `payload`, which is what the whole question reduces
// to: two padded decode buckets are the same topology with different parameters.
std::string OneNodeSignature(const std::string& payload) {
  const SigNode node{payload};
  SigGraph graph;
  graph.nodes = {&node};
  return SignatureOf(graph);
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

// --- the exact key, which is the SHIPPED DEFAULT ------------------------------------

TEST_CASE("the exact key discriminates every launch dimension and every copy extent") {
  // COVERAGE OF THE DEFAULT, and it was absent. Every helper in this file passed 1 for
  // grid_y, grid_z, block_y, block_z and the memcpy height and depth, so those six
  // fields were EXECUTED by every case and DISCRIMINATED by none: deleting `grid_z` from
  // the exact key -- a silent coarsening of the key that ships ON, on a path no token
  // gate and no device count can see -- left this suite 22/22 green. Three of the six
  // siblings were caught only incidentally, by `-Werror=unused-parameter`, which is a
  // build guard rather than a test and which a mutation that renames instead of removing
  // walks straight past.
  const std::string base = KernelPayloadFull(0x1000, 8, 1, 1, 128, 1, 1, 0, false);
  CHECK(base != KernelPayloadFull(0x1000, 8, 3, 1, 128, 1, 1, 0, false));  // grid.y
  CHECK(base != KernelPayloadFull(0x1000, 8, 1, 5, 128, 1, 1, 0, false));  // grid.z
  CHECK(base != KernelPayloadFull(0x1000, 8, 1, 1, 128, 7, 1, 0, false));  // block.y
  CHECK(base != KernelPayloadFull(0x1000, 8, 1, 1, 128, 1, 9, 0, false));  // block.z
  // Through the whole signature, not only the payload helper: the walk is what
  // production calls, and a field that separates two payloads must separate two graphs.
  CHECK(OneNodeSignature(base) !=
        OneNodeSignature(KernelPayloadFull(0x1000, 8, 1, 5, 128, 1, 1, 0, false)));

  // A 2-D and a 3-D copy are different copies, and the exact key says so.
  CHECK(OneNodeSignature(MemcpyPayloadFull(2, 64, 1, 1, false)) !=
        OneNodeSignature(MemcpyPayloadFull(2, 64, 4, 1, false)));   // extent.height
  CHECK(OneNodeSignature(MemcpyPayloadFull(2, 64, 1, 1, false)) !=
        OneNodeSignature(MemcpyPayloadFull(2, 64, 1, 4, false)));   // extent.depth
}

TEST_CASE("the coarse key drops the y and z components too, not only x") {
  // The other half of the same gap. The coarse key exists to join two padded decode
  // buckets, and a bucket that grew a second grid dimension would still separate if only
  // the x components were dropped -- so the drop is asserted componentwise rather than
  // inferred from the grid_x case.
  const std::string base = KernelPayloadFull(0x1000, 8, 1, 1, 128, 1, 1, 0, true);
  CHECK(base == KernelPayloadFull(0x1000, 8, 3, 1, 128, 1, 1, 0, true));
  CHECK(base == KernelPayloadFull(0x1000, 8, 1, 5, 128, 1, 1, 0, true));
  CHECK(base == KernelPayloadFull(0x1000, 8, 1, 1, 128, 7, 1, 0, true));
  CHECK(base == KernelPayloadFull(0x1000, 8, 1, 1, 128, 1, 9, 0, true));
  CHECK(OneNodeSignature(MemcpyPayloadFull(2, 64, 1, 1, true)) ==
        OneNodeSignature(MemcpyPayloadFull(2, 64, 4, 7, true)));
}

// --- the coarse key, issue #1226 ----------------------------------------------------

TEST_CASE("the coarse key groups two decode buckets that differ only in launch dimensions") {
  // THE CASE THE ROW WAS FILED FOR, and the one the 2026-08-18 device gate measured never
  // happening. Two padded decode buckets run the same kernel over a different number of
  // sequences, so `gridDim` differs and nothing else does. Under the shipped key those are
  // two signatures, two buckets in `groups_`, no candidate, and cudaGraphExecUpdate is
  // never even attempted -- which is why that gate could report `N == M` with the driver
  // never having been asked.
  const std::string small_exact = KernelPayload(0x1000, 8, 128, 0, false);
  const std::string large_exact = KernelPayload(0x1000, 24, 128, 0, false);
  CHECK(OneNodeSignature(small_exact) != OneNodeSignature(large_exact));

  const std::string small_coarse = KernelPayload(0x1000, 8, 128, 0, true);
  const std::string large_coarse = KernelPayload(0x1000, 24, 128, 0, true);
  CHECK(OneNodeSignature(small_coarse) == OneNodeSignature(large_coarse));

  // The block dimension is dropped for the same reason and by the same branch.
  CHECK(OneNodeSignature(KernelPayload(0x1000, 8, 128, 0, true)) ==
        OneNodeSignature(KernelPayload(0x1000, 8, 256, 0, true)));
  CHECK(OneNodeSignature(KernelPayload(0x1000, 8, 128, 0, false)) !=
        OneNodeSignature(KernelPayload(0x1000, 8, 256, 0, false)));
}

TEST_CASE("the coarse key still separates two different kernel functions") {
  // The discrimination the coarsening must NOT give up. `func` is a host function
  // pointer, strictly stronger than the demangled name SGLang keys, and dropping it would
  // collapse every same-shaped node in the process into one candidate group -- turning a
  // cheap wasted probe into a probe against every capture ever taken.
  CHECK(OneNodeSignature(KernelPayload(0x1000, 8, 128, 0, true)) !=
        OneNodeSignature(KernelPayload(0x2000, 8, 128, 0, true)));
}

TEST_CASE("the coarse key keeps sharedMemBytes") {
  // ARGUED, not swept in. A dynamic shared-memory size is a kernel-node parameter like the
  // dimensions are, so the same reasoning would drop it. It is kept because it is not
  // where the batch dimension lives -- so dropping it buys the hypothesis nothing -- and
  // because a size above the 48 KiB static limit is legal only for a function that opted
  // in through cudaFuncAttributeMaxDynamicSharedMemorySize, an attribute of the FUNCTION
  // and not of the node. It is therefore the field most likely to make the driver
  // re-instantiate rather than re-point, and keeping it holds this experiment to one
  // variable.
  CHECK(OneNodeSignature(KernelPayload(0x1000, 8, 128, 1024, true)) !=
        OneNodeSignature(KernelPayload(0x1000, 8, 128, 2048, true)));
  // And it is kept in a way that does not accidentally re-admit the dimensions.
  CHECK(OneNodeSignature(KernelPayload(0x1000, 8, 128, 1024, true)) ==
        OneNodeSignature(KernelPayload(0x1000, 24, 256, 1024, true)));
}

TEST_CASE("the coarse key drops the memcpy extent and keeps the kind") {
  // The gate named the copy extent alongside the launch dimensions as the second place the
  // padded batch size enters the key. `kind` stays because cudaGraphExecUpdate explicitly
  // refuses a changed memcpy memory type, so a key that dropped it would manufacture
  // refusals rather than folds.
  CHECK(OneNodeSignature(MemcpyPayload(2, 64, false)) !=
        OneNodeSignature(MemcpyPayload(2, 192, false)));
  CHECK(OneNodeSignature(MemcpyPayload(2, 64, true)) ==
        OneNodeSignature(MemcpyPayload(2, 192, true)));
  CHECK(OneNodeSignature(MemcpyPayload(1, 64, true)) !=
        OneNodeSignature(MemcpyPayload(2, 64, true)));
}

TEST_CASE("the coarse key drops the memset width and keeps the element size and height") {
  // `height` stays because the update contract will only change a 1-D memset, so it is a
  // field the driver itself treats as identity; `elementSize` stays for the same reason.
  CHECK(OneNodeSignature(MemsetPayload(4, 64, 1, false)) !=
        OneNodeSignature(MemsetPayload(4, 192, 1, false)));
  CHECK(OneNodeSignature(MemsetPayload(4, 64, 1, true)) ==
        OneNodeSignature(MemsetPayload(4, 192, 1, true)));
  CHECK(OneNodeSignature(MemsetPayload(4, 64, 1, true)) !=
        OneNodeSignature(MemsetPayload(2, 64, 1, true)));
  CHECK(OneNodeSignature(MemsetPayload(4, 64, 1, true)) !=
        OneNodeSignature(MemsetPayload(4, 64, 8, true)));
}

TEST_CASE("the coarse key does not weaken the topology half of the signature") {
  // Coarsening the PAYLOADS must not coarsen the WALK. Two graphs whose nodes are now
  // payload-identical still have to separate on their edges, or the key would offer the
  // driver a fold across genuinely different graphs and pay a probe for every one.
  const std::string payload = KernelPayload(0x1000, 8, 128, 0, true);
  const SigNode a{payload};
  const SigNode b{payload};
  const SigNode c{payload};

  SigGraph chain;
  chain.nodes = {&a, &b, &c};
  chain.edges = {{0, 1}, {1, 2}};

  SigGraph fan;
  fan.nodes = {&a, &b, &c};
  fan.edges = {{0, 1}, {0, 2}};

  CHECK(SignatureOf(chain) != SignatureOf(fan));
  // And a different node COUNT is still a different key.
  SigGraph pair;
  pair.nodes = {&a, &b};
  pair.edges = {{0, 1}};
  CHECK(SignatureOf(pair) != SignatureOf(chain));
}

TEST_CASE("the coarse key is off unless the environment asks for it") {
  // Same polarity function as VT_CUDA_GRAPH_DEDUP, stated once so the two knobs cannot
  // drift, and OFF by default so the shipped key is exactly the one the device gate ran
  // and the two arms are a same-binary A/B.
  CHECK_FALSE(vt::GraphDedupEnabledFor(nullptr));
  CHECK_FALSE(vt::GraphDedupEnabledFor("0"));
  CHECK_FALSE(vt::GraphDedupEnabledFor("true"));
  CHECK(vt::GraphDedupEnabledFor("1"));
  // The suite runs with VT_CUDA_GRAPH_DEDUP_COARSE_KEY unset, so this reads the default.
  CHECK_FALSE(vt::GraphDedupCoarseKeyEnabled());
}

TEST_CASE("two captures the coarse key groups share one executable when the driver accepts") {
  // The end of the chain: a coarser key is worth nothing unless the registry actually
  // folds on it. Same signature, driver accepts, ONE executable for two captures -- which
  // is the `M < N` the device run is being asked for.
  FakeLatch latch;
  g_latch = &latch;
  vt::GraphDedupRegistry registry(GuardedTable(), nullptr);

  const std::string coarse = OneNodeSignature(KernelPayload(0x1000, 8, 128, 0, true));
  const std::string coarse_other = OneNodeSignature(KernelPayload(0x1000, 24, 128, 0, true));
  REQUIRE(coarse == coarse_other);

  LatchGraph small{coarse, 1};
  LatchGraph large{coarse_other, 2};
  registry.Register(&small);
  registry.Register(&large);

  CHECK(registry.CapturedCount() == 2);
  CHECK(registry.ExecCount() == 1);
  CHECK(registry.ProbeCount() == 1);
  CHECK(registry.ProbeRefusalCount() == 0);

  registry.Close();
  g_latch = nullptr;
}

TEST_CASE("the probe counters tell a key that never grouped from a driver that refused") {
  // THE MEASUREMENT INSTRUMENT, and the reason it is in the product rather than in a
  // script. `N == M` has two opposite causes -- the key never grouped, so the driver was
  // never asked; or it grouped and the driver said no -- and the 2026-08-18 gate could
  // only separate them by reading the signature's source afterwards. A device run that
  // cannot separate them measures nothing, so the counters are asserted here on both
  // shapes and printed on the registry's own log line.
  FakeLatch latch;
  g_latch = &latch;

  SUBCASE("distinct signatures: the driver is never asked") {
    vt::GraphDedupRegistry registry(GuardedTable(), nullptr);
    LatchGraph first{"topology-A", 1};
    LatchGraph second{"topology-B", 2};
    registry.Register(&first);
    registry.Register(&second);
    CHECK(registry.ExecCount() == 2);
    CHECK(registry.ProbeCount() == 0);
    CHECK(registry.ProbeRefusalCount() == 0);
    registry.Close();
  }

  SUBCASE("one signature, driver refuses: asked once, refused once, and the reason kept") {
    std::FILE* log = std::tmpfile();
    REQUIRE(log != nullptr);
    {
      vt::GraphDedupRegistry registry(GuardedTable(), log);
      LatchGraph first{"same", 1};
      LatchGraph second{"same", 2};
      registry.Register(&first);
      latch.refuse_update = true;
      registry.Register(&second);
      latch.refuse_update = false;
      CHECK(registry.ExecCount() == 2);
      CHECK(registry.ProbeCount() == 1);
      CHECK(registry.ProbeRefusalCount() == 1);
      registry.Close();
    }
    std::fflush(log);
    std::rewind(log);
    std::string text;
    char buffer[512];
    while (std::fgets(buffer, sizeof(buffer), log) != nullptr) text += buffer;
    std::fclose(log);
    // The driver's own cudaError_t / cudaGraphExecUpdateResult pair, verbatim. This is
    // the evidence the 2026-08-18 record could not produce, because the fold was never
    // attempted and so no refusal existed to quote.
    CHECK(text.find("probe refused a fold (err=1 result=1)") != std::string::npos);
    CHECK(text.find("captured 2 graphs, deduped to 2 execs (probes=1 refused=1)") !=
          std::string::npos);
  }

  CHECK_NOTHROW(NextUnrelatedKernelLaunch());
  g_latch = nullptr;
}
