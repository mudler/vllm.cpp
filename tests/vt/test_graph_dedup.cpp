// vllm.cpp original — CPU-tier contract for the graph-executable dedup registry
// (src/vt/graph_dedup.h), row ENG-CUDAGRAPH-DEDUP, issue #1162.
//
// The production hazard is accelerator-only: today every capture instantiates its own
// graph executable (src/vt/cuda/cuda_backend.cu EndCaptureGraph, and the same shape on
// hipGraph), so a model holds one executable per padded decode bucket and nine drivers
// each hold their own set. The registry folds captures whose topology matches onto ONE
// executable and re-points it with cudaGraphExecUpdate / hipGraphExecUpdate.
//
// A shared executable is shared state, so the guarantee that matters is not "fewer
// executables" — it is that a deduped replay launches EXACTLY what the caller captured.
// This suite pins that on every platform by driving the registry through a fake device.
// The fake's launch() records which graph the executable currently reflects, so "the
// right nodes ran" is an observable sequence rather than an assertion about intent, and
// every case replays MORE THAN ONCE per shape: the whole risk of a shared executable is
// that the SECOND visit to a shape is the one that has to re-point it, so a suite that
// replayed each shape once would pass while measuring nothing.
//
// The device-side proof (a same-binary VT_CUDA_GRAPH_DEDUP off/on A/B that is
// token-identical over a real decode) needs a leased CUDA box and is owed by #1162; it
// is recorded under `## Owed` in .agents/specs/eng-cudagraph-dedup.md, not claimed here.
#include <doctest/doctest.h>

#include <cstdio>
#include <set>
#include <stdexcept>
#include <utility>
#include <string>
#include <vector>

#include "vt/graph_dedup.h"

using vt::GraphDedupOps;
using vt::GraphDedupRegistry;

namespace {

// --- the fake device -------------------------------------------------------------
// A "graph" is a signature plus an identity; an "executable" is a mutable pointer to
// the graph it currently reflects. That is precisely the CUDA model at the level this
// registry works: instantiate binds an executable to a graph, update re-binds it, and
// launch runs whatever it is currently bound to.
struct FakeGraph {
  std::string sig;
  int id = 0;
};
// An executable holds a COPY of what it was last bound to, never a pointer into the
// graph. That is the real contract: cudaGraphInstantiate and cudaGraphExecUpdate copy
// the node parameters out, so an executable stays launchable after its source graph is
// destroyed. A fake that stored the pointer would read freed memory in the
// destroy-a-sibling case and would be modelling a hazard the driver does not have.
struct FakeExec {
  std::string sig;
  int id = 0;
};

struct FakeDevice {
  int instantiated = 0;
  int exec_destroyed = 0;
  int graph_destroyed = 0;
  int updates = 0;
  std::set<int> reject_update_for;  // graph ids the driver refuses to update onto
  // (currently-reflected graph id, target graph id) pairs the driver refuses. This is
  // the shape reject_update_for cannot express and the shape that matters: whether an
  // update is accepted can depend on WHICH graph the executable currently holds, not
  // only on the one it is being pointed at.
  std::set<std::pair<int, int>> reject_update_pair;
  // Graph ids the driver refuses to INSTANTIATE. cudaGraphInstantiate fails for reasons
  // that have nothing to do with the topology -- out of memory is the common one on the
  // unified-memory box this row exists for -- so it is a distinct outcome from a refused
  // update and it has to have its own case.
  std::set<int> fail_instantiate_for;
  // Times the registry handed the driver a NULL executable. cudaGraphExecUpdate has no
  // defined behaviour for one; counting it here rather than dereferencing it keeps the
  // red observable instead of a crash.
  int null_exec_updates = 0;
  std::vector<int> launch_log;      // the id of the graph each launch actually ran
  // With `recycle` set, a destroyed graph's STORAGE is handed back to the test instead
  // of being freed, so a later capture can be made to land on the same address on
  // purpose. That is the only way to test the freed-address hazard deterministically:
  // relying on the allocator to reuse the block would make the case a coin flip.
  bool recycle = false;
  std::vector<void*> recycled;

  int live_execs() const { return instantiated - exec_destroyed; }
};

FakeDevice* g_dev = nullptr;

std::string FakeSignature(void* graph) {
  return static_cast<FakeGraph*>(graph)->sig;
}

void* FakeInstantiate(void* graph) {
  auto* source = static_cast<FakeGraph*>(graph);
  // Mirrors the driver: on failure nothing is allocated and the out-parameter is left
  // null, which is what src/vt/graph_dedup_runtime.h Instantiate() turns into a null
  // return. Counting only the successes keeps live_execs() meaningful.
  if (g_dev->fail_instantiate_for.count(source->id) != 0) return nullptr;
  ++g_dev->instantiated;
  return new FakeExec{source->sig, source->id};
}

// Mirrors cudaGraphExecUpdate's contract: it can only re-point an executable onto a
// graph whose topology matches the one the executable currently holds, and it may
// refuse for a reason the caller cannot see from the topology alone — which is exactly
// what reject_update_for models.
bool FakeUpdate(void* exec_handle, void* graph_handle, std::string* detail) {
  ++g_dev->updates;
  if (exec_handle == nullptr) {
    ++g_dev->null_exec_updates;
    if (detail != nullptr) *detail = "fake driver got a null executable";
    return false;
  }
  auto* exec = static_cast<FakeExec*>(exec_handle);
  auto* graph = static_cast<FakeGraph*>(graph_handle);
  if (g_dev->reject_update_for.count(graph->id) != 0 ||
      g_dev->reject_update_pair.count({exec->id, graph->id}) != 0 ||
      exec->sig != graph->sig) {
    if (detail != nullptr) *detail = "fake driver refused the update";
    return false;
  }
  exec->sig = graph->sig;
  exec->id = graph->id;
  return true;
}

void FakeDestroyExec(void* exec_handle) {
  ++g_dev->exec_destroyed;
  delete static_cast<FakeExec*>(exec_handle);
}

void FakeDestroyGraph(void* graph_handle) {
  ++g_dev->graph_destroyed;
  if (g_dev->recycle) {
    g_dev->recycled.push_back(graph_handle);
    return;
  }
  delete static_cast<FakeGraph*>(graph_handle);
}

void FakeLaunch(void* exec_handle, void* /*stream*/) {
  g_dev->launch_log.push_back(static_cast<FakeExec*>(exec_handle)->id);
}

GraphDedupOps FakeOps() {
  GraphDedupOps ops;
  ops.signature = &FakeSignature;
  ops.instantiate = &FakeInstantiate;
  ops.update = &FakeUpdate;
  ops.destroy_exec = &FakeDestroyExec;
  ops.destroy_graph = &FakeDestroyGraph;
  ops.launch = &FakeLaunch;
  return ops;
}

FakeGraph* MakeGraph(const char* sig, int id) { return new FakeGraph{sig, id}; }

// A registry that never writes to a log stream, so the suite stays quiet.
GraphDedupRegistry MakeRegistry() { return GraphDedupRegistry(FakeOps(), nullptr); }

struct DeviceScope {
  FakeDevice dev;
  DeviceScope() { g_dev = &dev; }
  ~DeviceScope() {
    for (void* graph : dev.recycled) delete static_cast<FakeGraph*>(graph);
    g_dev = nullptr;
  }
};

}  // namespace

TEST_CASE("dedup folds compatible captures onto one executable") {
  DeviceScope scope;
  auto registry = MakeRegistry();

  std::vector<void*> handles;
  for (int i = 1; i <= 5; ++i) handles.push_back(registry.Register(MakeGraph("decode", i)));

  CHECK(registry.CapturedCount() == 5);
  CHECK(registry.ExecCount() == 1);
  // The saving is real only if the extra executables are actually GONE, not merely
  // unreferenced: a transient probe that leaked would leave the count unchanged while
  // ExecCount() reported 1.
  CHECK(scope.dev.live_execs() == 1);

  registry.Close();
  CHECK(scope.dev.live_execs() == 0);
  CHECK(scope.dev.graph_destroyed == 5);
}

TEST_CASE("dedup keeps incompatible captures apart") {
  DeviceScope scope;
  auto registry = MakeRegistry();

  void* a1 = registry.Register(MakeGraph("prefill", 1));
  void* b1 = registry.Register(MakeGraph("decode", 2));
  void* a2 = registry.Register(MakeGraph("prefill", 3));

  CHECK(registry.CapturedCount() == 3);
  CHECK(registry.ExecCount() == 2);
  CHECK(scope.dev.live_execs() == 2);

  // Two shapes, alternating, twice around: neither group may answer for the other.
  for (int round = 0; round < 2; ++round) {
    registry.Replay(a1, nullptr);
    registry.Replay(b1, nullptr);
    registry.Replay(a2, nullptr);
  }
  CHECK(scope.dev.launch_log == std::vector<int>{1, 2, 3, 1, 2, 3});

  registry.Close();
}

TEST_CASE("a deduped replay launches the graph the caller asked for") {
  DeviceScope scope;

  // The control arm: what the SAME sequence launches with no dedup at all, which is
  // trivially the graph each handle was captured from. Comparing against a recomputed
  // expectation rather than a hand-written literal is what makes this an A/B and not a
  // restatement of the implementation.
  const std::vector<int> order = {1, 1, 2, 3, 2, 2, 1, 3, 3, 1, 2};

  auto registry = MakeRegistry();
  std::vector<void*> handles;
  for (int i = 1; i <= 3; ++i) handles.push_back(registry.Register(MakeGraph("decode", i)));
  REQUIRE(registry.ExecCount() == 1);

  for (int id : order) registry.Replay(handles[static_cast<std::size_t>(id - 1)], nullptr);

  CHECK(scope.dev.launch_log == order);
  registry.Close();
}

TEST_CASE("dedup does not re-point the executable for a repeated shape") {
  DeviceScope scope;
  auto registry = MakeRegistry();

  void* a = registry.Register(MakeGraph("decode", 1));
  void* b = registry.Register(MakeGraph("decode", 2));
  const int after_register = scope.dev.updates;

  registry.Replay(a, nullptr);
  const int after_first = scope.dev.updates;
  registry.Replay(a, nullptr);
  registry.Replay(a, nullptr);
  // Three replays of one shape, and only the first may cost an update. This is the
  // whole reason a steady decode workload pays nothing for dedup.
  CHECK(scope.dev.updates == after_first);

  registry.Replay(b, nullptr);
  CHECK(scope.dev.updates == after_first + 1);
  CHECK(after_first >= after_register);
  CHECK(scope.dev.launch_log == std::vector<int>{1, 1, 1, 2});

  registry.Close();
}

TEST_CASE("a capture the driver refuses to fold gets its own executable") {
  DeviceScope scope;
  scope.dev.reject_update_for.insert(2);
  auto registry = MakeRegistry();

  void* g1 = registry.Register(MakeGraph("decode", 1));
  void* g2 = registry.Register(MakeGraph("decode", 2));  // same signature, refused
  void* g3 = registry.Register(MakeGraph("decode", 3));  // same signature, accepted

  // Degrades to today's behaviour for the refused capture only. It must never abort,
  // and it must never leave the refused graph sharing an executable it cannot drive.
  CHECK(registry.CapturedCount() == 3);
  CHECK(registry.ExecCount() == 2);

  for (int round = 0; round < 2; ++round) {
    registry.Replay(g1, nullptr);
    registry.Replay(g2, nullptr);
    registry.Replay(g3, nullptr);
  }
  CHECK(scope.dev.launch_log == std::vector<int>{1, 2, 3, 1, 2, 3});

  registry.Close();
}

TEST_CASE("destroying one capture keeps its siblings replayable") {
  DeviceScope scope;
  auto registry = MakeRegistry();

  void* g1 = registry.Register(MakeGraph("decode", 1));
  void* g2 = registry.Register(MakeGraph("decode", 2));
  void* g3 = registry.Register(MakeGraph("decode", 3));
  REQUIRE(registry.ExecCount() == 1);

  registry.Replay(g2, nullptr);          // the executable now reflects graph 2
  registry.Destroy(g2);                  // and that graph is about to be freed
  CHECK(registry.CapturedCount() == 2);
  CHECK(registry.ExecCount() == 1);
  CHECK(scope.dev.live_execs() == 1);
  CHECK(scope.dev.graph_destroyed == 1);

  // The surviving siblings stay replayable through the executable their destroyed
  // sibling was last pointed at. (The address-reuse half of this is a separate case
  // below; this one cannot see it, because re-pointing onto a DIFFERENT address happens
  // either way.)
  registry.Replay(g1, nullptr);
  registry.Replay(g3, nullptr);
  registry.Replay(g1, nullptr);
  CHECK(scope.dev.launch_log == std::vector<int>{2, 1, 3, 1});

  registry.Destroy(g1);
  CHECK(scope.dev.live_execs() == 1);
  registry.Destroy(g3);
  CHECK(scope.dev.live_execs() == 0);
  CHECK(registry.CapturedCount() == 0);
  CHECK(registry.ExecCount() == 0);
  CHECK(scope.dev.graph_destroyed == 3);

  registry.Close();
}

TEST_CASE("a capture reusing a freed graph address does not inherit its replay state") {
  DeviceScope scope;
  scope.dev.recycle = true;
  auto registry = MakeRegistry();

  registry.Register(MakeGraph("decode", 1));
  void* second = registry.Register(MakeGraph("decode", 2));

  registry.Replay(second, nullptr);  // the shared executable now reflects graph 2
  registry.Destroy(second);          // and graph 2 is freed
  REQUIRE(scope.dev.recycled.size() == 1);

  // A later capture lands on the freed address. The registry remembers which raw graph
  // its executable reflects by ADDRESS, so if it did not forget the destroyed one it
  // would compare equal here, skip the re-point, and replay graph 2's nodes under
  // graph 7's handle. That is a wrong answer, not a lost optimisation.
  auto* reused = static_cast<FakeGraph*>(scope.dev.recycled.back());
  scope.dev.recycled.pop_back();
  *reused = FakeGraph{"decode", 7};

  void* seventh = registry.Register(reused);
  registry.Replay(seventh, nullptr);
  registry.Replay(seventh, nullptr);
  CHECK(scope.dev.launch_log == std::vector<int>{2, 7, 7});

  registry.Close();
}

TEST_CASE("a capture the driver cannot instantiate fails at the capture site") {
  DeviceScope scope;
  scope.dev.fail_instantiate_for.insert(1);
  auto registry = MakeRegistry();

  // The pre-dedup path called Check(cudaGraphInstantiate(...), "cudaGraphInstantiate")
  // inside EndCaptureGraph, so an instantiate failure threw where the capture happened.
  // Accepting a null executable instead would mint a valid-looking handle over nothing,
  // count it live, and defer the failure to the first replay -- a different site, a
  // different message, and the driver's reason already thrown away.
  CHECK_THROWS_AS(registry.Register(MakeGraph("decode", 1)), std::runtime_error);

  // And it must not be counted. A registry that reported a capture it can never replay
  // would make the exec-count ratio this row is measured by a fiction.
  CHECK(registry.CapturedCount() == 0);
  CHECK(registry.ExecCount() == 0);
  CHECK(scope.dev.live_execs() == 0);
  // The capture's ownership had already transferred, so unwinding has to release it.
  CHECK(scope.dev.graph_destroyed == 1);

  // The registry is still usable for a capture the driver will accept.
  void* ok = registry.Register(MakeGraph("decode", 2));
  registry.Replay(ok, nullptr);
  registry.Replay(ok, nullptr);
  CHECK(registry.ExecCount() == 1);
  CHECK(scope.dev.launch_log == std::vector<int>{2, 2});

  registry.Close();
}

TEST_CASE("a probe the driver cannot instantiate degrades instead of driving a null exec") {
  DeviceScope scope;
  auto registry = MakeRegistry();

  void* first = registry.Register(MakeGraph("decode", 1));
  // The probe re-instantiates an EXISTING group member, so its instantiate can fail
  // independently of the capture being registered -- most plausibly under the memory
  // pressure that made this row worth doing at all.
  scope.dev.fail_instantiate_for.insert(1);
  void* second = registry.Register(MakeGraph("decode", 2));

  // cudaGraphExecUpdate has no defined behaviour for a null executable, so a failed
  // probe must answer "cannot fold" rather than ask the driver about nothing.
  CHECK(scope.dev.null_exec_updates == 0);
  CHECK(registry.CapturedCount() == 2);
  CHECK(registry.ExecCount() == 2);

  for (int round = 0; round < 2; ++round) {
    registry.Replay(first, nullptr);
    registry.Replay(second, nullptr);
  }
  CHECK(scope.dev.launch_log == std::vector<int>{1, 2, 1, 2});

  registry.Close();
}

TEST_CASE("a replay update the driver refuses fails loudly rather than launching stale nodes") {
  DeviceScope scope;
  // Register probes (group.raws.front(), candidate) but Replay issues
  // (group.current_raw, target), and from the third group member onwards those pairs
  // differ -- so honouring the probe assumes update compatibility is TRANSITIVE across a
  // group. Nothing asserts that. Refusing exactly the pair Register never asks about
  // reproduces the case: every probe succeeds, the group folds to one executable, and
  // the SECOND replay is where the assumption is tested for real.
  scope.dev.reject_update_pair.insert({2, 3});
  auto registry = MakeRegistry();

  void* g1 = registry.Register(MakeGraph("decode", 1));
  void* g2 = registry.Register(MakeGraph("decode", 2));
  void* g3 = registry.Register(MakeGraph("decode", 3));
  // Every probe was (graph 1, candidate), and the driver accepts those.
  REQUIRE(registry.ExecCount() == 1);

  registry.Replay(g2, nullptr);  // (1 -> 2), accepted; the executable now reflects 2

  // And now the fold nobody probed. What must NOT happen is the silent alternative:
  // leaving the executable pointing at graph 2 and launching it under g3's handle, which
  // is a wrong answer rather than a loud one. This VT_CHECK is the entire reason the
  // transitivity assumption is survivable, so it is gated rather than asserted in prose.
  CHECK_THROWS_AS(registry.Replay(g3, nullptr), std::runtime_error);
  CHECK(scope.dev.launch_log == std::vector<int>{2});

  // g1 is unaffected: its own fold is one the driver still accepts.
  registry.Replay(g1, nullptr);
  CHECK(scope.dev.launch_log == std::vector<int>{2, 1});

  registry.Close();
}

TEST_CASE("the registry does not claim a handle it did not mint") {
  DeviceScope scope;
  auto registry = MakeRegistry();

  void* mine = registry.Register(MakeGraph("decode", 1));
  int not_a_handle = 0;

  // This is what lets one backend serve a deduped and a plain executable through the
  // same void* seam without guessing at a pointer's provenance.
  CHECK(registry.Owns(mine));
  CHECK_FALSE(registry.Owns(&not_a_handle));
  CHECK_FALSE(registry.Owns(nullptr));

  registry.Destroy(mine);
  CHECK_FALSE(registry.Owns(mine));

  registry.Close();
}

TEST_CASE("the dedup log line reports the running capture and executable counts") {
  DeviceScope scope;
  std::FILE* log = std::tmpfile();
  REQUIRE(log != nullptr);

  {
    GraphDedupRegistry registry(FakeOps(), log);
    registry.Register(MakeGraph("decode", 1));
    registry.Register(MakeGraph("decode", 2));
    registry.Register(MakeGraph("prefill", 3));
    registry.Close();
  }

  std::fflush(log);
  std::rewind(log);
  std::string text;
  char buffer[512];
  while (std::fgets(buffer, sizeof(buffer), log) != nullptr) text += buffer;
  std::fclose(log);

  // Mirrors SGLang's "captured %d CUDA graphs, deduped to %d execs"
  // (cuda_graph_dedup_mixin.py:358). The ratio has to be readable off a log, because a
  // dedup that silently folded nothing would otherwise look identical to one that works.
  CHECK(text.find("captured 1 graphs, deduped to 1 execs") != std::string::npos);
  CHECK(text.find("captured 2 graphs, deduped to 1 execs") != std::string::npos);
  CHECK(text.find("captured 3 graphs, deduped to 2 execs") != std::string::npos);
}

TEST_CASE("dedup is off unless the environment asks for it") {
  // The registry only exists when this is true, so its polarity decides whether the
  // production path is byte-identical to the pre-dedup one.
  CHECK_FALSE(vt::GraphDedupEnabledFor(nullptr));
  CHECK_FALSE(vt::GraphDedupEnabledFor(""));
  CHECK_FALSE(vt::GraphDedupEnabledFor("0"));
  CHECK(vt::GraphDedupEnabledFor("1"));

  // Exactly "1", not "starts with 1". Without the terminator check these all enable
  // dedup, and every case above still passes, because none of them ever asks about a
  // value whose FIRST character is the one the polarity accepts. "10" is the one that
  // matters in practice: it is what a hand-edited "1" plus a stray keystroke looks
  // like, and turning a default-off correctness-sensitive path on by accident is the
  // failure this polarity exists to prevent.
  CHECK_FALSE(vt::GraphDedupEnabledFor("10"));
  CHECK_FALSE(vt::GraphDedupEnabledFor("11"));
  CHECK_FALSE(vt::GraphDedupEnabledFor("1 "));
  CHECK_FALSE(vt::GraphDedupEnabledFor("1x"));
  CHECK_FALSE(vt::GraphDedupEnabledFor("01"));
  CHECK_FALSE(vt::GraphDedupEnabledFor("true"));
  CHECK_FALSE(vt::GraphDedupEnabledFor("on"));
}
