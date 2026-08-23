// THE RENDER PHASE INSTRUMENT, held to what its own numbers claim.
//
// Row LTX25-PHASE-INSTRUMENT, issues #1668, #1569, #1571. Spec:
// `.agents/specs/ltx25-phase-instrument.md`.
//
// WHY THIS FILE IS NOT IN `test_ltx2_video.cpp`. The cases below are about the
// INSTRUMENT rather than about LTX-2.5: they build synthetic timelines whose
// leaves contain a `sleep` and nothing else, because the question is where a
// charge LANDS and what the emitter writes, not what a render does. Two of them
// need a table of thousands of records, which no render produces. Keeping them
// beside a 5000-line model suite that renders a fixture would make an instrument
// question cost a model build, and it would put them in the file three other
// issues are actively editing.
//
// WHAT IS STILL GATED IN `test_ltx2_video.cpp`, and has to be. Everything here
// calls `PhaseLog` directly, which proves the class works and never that a
// render reaches it. The reachability half — that `vllm_video_generate` emits a
// table carrying `instrument_seconds` and `gaps` — is asserted on the table the
// ABI writes, in `a render through the ABI emits a phase table that SUMS to
// wall`. Neither file is sufficient alone.
//
// THE ONE NUMBER THIS FILE REFUSES TO ASSERT is a residue measured against the
// instrument's own charge. `.agents/specs/ltx25-phase-residue.md` `## Design` 3
// records three fresh reviews measuring `residue <= 2 * instrument` red 4 times
// in 45 runs at load 88 (max 4.115) and 28 times in 160 at load 125 (max 5.55),
// because the UN-instrumented remainder of a boundary dilates faster than the
// instrumented part when the box slows. Every bound below is either an
// accounting identity, which no scheduler can move, or a one-sided comparison
// whose noise can only push it AWAY from red. Read that section before adding a
// ratio here.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <doctest/doctest.h>
#include <nlohmann/json.hpp>

#include "vllm/multimodal/render_phase_log.h"

namespace {

namespace phase = vllm::multimodal::phase;

std::string ReadAll(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

// A temporary directory and the table written inside it, removed on the way out.
// Every case here writes a real file because `WriteJson` is what is under test:
// the emitted table is the artifact a reader gets, and asserting on an in-memory
// record vector would skip the half of the code these cases exist for.
class TableFile {
 public:
  TableFile() {
    std::snprintf(dir_, sizeof(dir_), "/tmp/vllm_phase_unit_XXXXXX");
    REQUIRE(::mkdtemp(dir_) != nullptr);
    path_ = std::string(dir_) + "/phase-log.json";
  }
  ~TableFile() {
    ::unlink(path_.c_str());
    ::rmdir(dir_);
  }
  TableFile(const TableFile&) = delete;
  TableFile& operator=(const TableFile&) = delete;

  const std::string& path() const { return path_; }

  nlohmann::json Write(const phase::PhaseLog& log) const {
    std::string why;
    REQUIRE_MESSAGE(log.WriteJson(path_, "unit", "cpu", &why), why);
    return nlohmann::json::parse(ReadAll(path_));
  }

 private:
  char dir_[64] = {};
  std::string path_;
};

void SleepMs(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

// The `WALL` total out of a `RenderText` block, at the resolution that block
// prints it. Returns a NEGATIVE sentinel when the block carries no such row or
// carries two, and every caller REQUIREs on that: a capture that caught nothing
// would otherwise read exactly like a comparison that passed, and a second
// occurrence of the label would mean this is reading some other row's number.
double WallFromBlock(const std::string& block) {
  const std::string::size_type at = block.find("WALL");
  if (at == std::string::npos) return -1.0;
  if (block.find("WALL", at + 1) != std::string::npos) return -2.0;
  return std::strtod(block.c_str() + at + 4, nullptr);
}

// FD 2, pointed somewhere else and put back. The last case below needs it for
// two opposite reasons: to READ what the instrument printed, and to keep what
// the instrument prints off the harness's own stderr. `PhaseLog::Close` flushes
// one progress line per scope, and a table of a hundred thousand scopes is
// mostly flushed writes into whatever stderr is attached to -- 3.81 s against
// 0.65 s over 120000 scopes on the box this was written on, with the whole
// difference in that pipe. `ProgressEnabled()` reads its environment variable
// ONCE per process, so a `setenv` from inside a case cannot turn it off: by the
// time any case runs, an earlier one has already latched it.
class StderrTo {
 public:
  explicit StderrTo(const std::string& path) {
    saved_ = ::dup(STDERR_FILENO);
    sink_ = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (saved_ >= 0 && sink_ >= 0) ok_ = ::dup2(sink_, STDERR_FILENO) >= 0;
  }
  ~StderrTo() { Restore(); }
  StderrTo(const StderrTo&) = delete;
  StderrTo& operator=(const StderrTo&) = delete;

  bool ok() const { return ok_; }

  void Restore() {
    if (saved_ < 0) return;
    std::fflush(stderr);
    ::dup2(saved_, STDERR_FILENO);
    ::close(saved_);
    if (sink_ >= 0) ::close(sink_);
    saved_ = -1;
    sink_ = -1;
  }

 private:
  int saved_ = -1;
  int sink_ = -1;
  bool ok_ = false;
};

// One path, gone on the way out.
class RemoveOnExit {
 public:
  explicit RemoveOnExit(const std::string& path) : path_(path) {}
  ~RemoveOnExit() { ::unlink(path_.c_str()); }
  RemoveOnExit(const RemoveOnExit&) = delete;
  RemoveOnExit& operator=(const RemoveOnExit&) = delete;

 private:
  std::string path_;
};

// One environment variable, restored on the way out -- and RAII for the same
// reason `StderrTo` and `RemoveOnExit` are.
//
// The pair this replaces was a bare `::setenv` and a bare `::unsetenv` with two
// `REQUIRE_MESSAGE` standing BETWEEN them, and a failed `REQUIRE` throws. The
// `unsetenv` would then never run, and `VLLM_RENDER_PHASE_LOG_STDERR=1` would
// survive into every case that ran afterwards in the same process -- turning one
// failed precondition into a second, unrelated failure somewhere a reader would
// have to work backwards from. Nothing is wrong today only because that case is
// last in the file and `StderrEnabled()` re-reads `getenv` per call rather than
// latching it the way `ProgressEnabled()` does. Both of those are properties of
// the file as it stands right now: `--order-by=rand` breaks the first, and
// appending a case below breaks it too.
//
// The PREVIOUS value is put back rather than the variable simply removed,
// because a guard that unsets what it found set is a leak in the other
// direction.
class EnvVar {
 public:
  EnvVar(const char* name, const char* value) : name_(name) {
    const char* previous = std::getenv(name);
    had_ = previous != nullptr;
    if (had_) previous_ = previous;
    ::setenv(name_, value, 1);
  }
  ~EnvVar() {
    if (had_) {
      ::setenv(name_, previous_.c_str(), 1);
    } else {
      ::unsetenv(name_);
    }
  }
  EnvVar(const EnvVar&) = delete;
  EnvVar& operator=(const EnvVar&) = delete;

 private:
  const char* name_;
  std::string previous_;
  bool had_ = false;
};

// N sequential leaves under one held span, with the build's own progress lines
// pointed at `sink`. The span is what stops the 100 ms sampler being created
// and joined once per leaf, and closing it leaves nothing live, so no worker is
// running while the clocks in the case are read.
void BuildLeaves(int n, const std::string& sink) {
  const std::string kLeafName = "unit.leaf.with.a.name.past.the.small.string.buffer";
  const StderrTo quiet(sink);
  const phase::Scope holder("unit.holder", /*span=*/true);
  for (int i = 0; i < n; ++i) {
    const phase::Scope leaf(kLeafName);
  }
}

}  // namespace

// ─── the instrument charges its OWN cost to the right place (#1668) ──────────
//
// The rule under test is one sentence from `render_phase_log.cpp`: every
// interval of the instrument's own wall is charged to the innermost live
// NON-SPAN record at the moment it is spent, and to the table when none is live.
// Three consequences, and each one is a different defect if it is wrong:
//
//   * A CHILD'S BOUNDARY IS THE PARENT'S COST. Opening and closing a nested
//     scope costs wall that lies inside the parent and outside the child, which
//     is precisely the uncovered time the coverage gate reads. Charged to the
//     table instead, that gate would have nothing to subtract and the number
//     would say the parent encloses a phase nobody named.
//   * A BOUNDARY WITH NOTHING LIVE IS THE TABLE'S COST. That is the residue the
//     sum gate reads, and it is the whole of `unaccounted_seconds`'s
//     explanation.
//   * A SPAN IS NOT A LEAF. `Sum` skips spans, so time inside a span and outside
//     every leaf IS the residue; charging it to the enclosing span would hide it
//     in a number nothing adds up. This is the case the LTX-2.5 driver actually
//     hits, because `load` and `generate` are spans that stay open across
//     everything beneath them.
//
// NOTHING HERE IS A DURATION COMPARISON. The three assertions are "it moved",
// "it did not move at all" and "it is positive", which is why a loaded box
// cannot change the verdict.
TEST_CASE("ltx2 phase log: the instrument charges its own cost to the innermost LEAF") {
  phase::PhaseLog& log = phase::PhaseLog::Instance();
  log.Reset();
  log.Begin();

  // (1) A SPAN THAT ENCLOSES EVERYTHING, exactly as the driver's `load` does.
  const size_t span = log.Open("unit.span", /*span=*/true);
  // (2) A BOUNDARY WITH NO LEAF LIVE. Only the span is open, so this pair is
  // charged to the TABLE and not to the span.
  const double before_gap = log.Instrument();
  { const phase::Scope gap_probe("unit.gap_probe"); }
  const double after_gap = log.Instrument();
  CHECK_MESSAGE(after_gap > before_gap,
                "opening and closing a leaf under a SPAN charged the table nothing, so either "
                "the instrument is not measuring its own boundaries or it charged them to the "
                "span. A span is not summed, so that time would vanish from the table");

  // (3) A LEAF WITH A NESTED CHILD. The child's boundaries are wall spent inside
  // the parent and outside the child.
  const size_t parent = log.Open("unit.parent", /*span=*/false);
  const double table_before_child = log.Instrument();
  for (int i = 0; i < 8; ++i) {
    const phase::Scope child("unit.child");
  }
  const double table_after_child = log.Instrument();
  // EXACTLY EQUAL, not `Approx`. `doctest::Approx` scales its epsilon by
  // `max(1, |value|)`, so on a quantity of ~1e-4 s it tolerates 1.19e-5 s —
  // 11.9 us, which is about one whole boundary. That is the size of the leak
  // this line exists to detect, so the tolerance would have been the blind spot.
  // Nothing here may charge the table AT ALL while a leaf is live, so the two
  // reads are the same double.
  CHECK_MESSAGE(table_after_child == table_before_child,
                "eight nested boundaries moved the TABLE's charge by "
                    << (table_after_child - table_before_child)
                    << "s while a leaf was live. They belong to the leaf that contains them; "
                       "charging them to the table would report the parent as enclosing a "
                       "phase nobody named");
  log.Close(parent);
  log.Close(span);

  const std::vector<phase::Record> records = log.Records();
  double parent_instrument = -1.0;
  double parent_duration = -1.0;
  double child_total = 0.0;
  int64_t children = 0;
  for (const phase::Record& r : records) {
    if (r.name == "unit.parent") {
      parent_instrument = r.instrument_seconds;
      parent_duration = r.end - r.start;
    }
    if (r.name == "unit.child") {
      child_total += r.end - r.start;
      ++children;
      CHECK_MESSAGE(r.nested, "'unit.child' opened inside a live leaf and is not marked nested");
    }
  }
  REQUIRE(children == 8);
  REQUIRE(parent_duration > 0.0);
  CHECK_MESSAGE(parent_instrument > 0.0,
                "the parent leaf was charged " << parent_instrument
                    << "s although eight children opened and closed inside it. This is the "
                       "quantity a reader of the coverage ratio subtracts");

  // AND WHAT IS **NOT** ASSERTED HERE, because a fresh review of the withdrawn
  // design measured it. This case shipped twice with a bound on
  // `uncovered / parent_instrument`, and the shipped binary reddened 2 of 200
  // consecutive runs at load 85, while a standalone probe of this exact shape
  // reddened 28 of 160 at load 125 and reached 14.1 under ASan. Decomposing the
  // parent's uncovered time explains it: fast, the inter-child gaps are 9-20 us
  // over seven boundaries against a 13-22 us charge; slow, the gaps are
  // 91-105 us against a 52-61 us charge. The UN-instrumented part of a boundary
  // — the `lock_guard` release, the `Close` return, the `Scope` destructor and
  // constructor, the call into `Open` up to its clock read — dilates faster than
  // the instrumented part. Eight bare scopes carry neither a `Tick` nor a
  // `/proc/self/statm` read inside the instrumented region, which makes this the
  // worst-conditioned probe of that ratio anywhere, not the tightest. It is
  // reported so a reader can see it move, and asserted nowhere.
  const double uncovered = parent_duration - child_total;
  MESSAGE("unit.parent = " << parent_duration << "s, children " << child_total
                           << "s, uncovered " << uncovered << "s, charged " << parent_instrument
                           << "s (ratio " << (uncovered / parent_instrument)
                           << ", REPORTED not asserted -- see the note above)");
  log.Reset();
}

// ─── the accounting is CONSERVED (#1668) ─────────────────────────────────────
//
// `instrument_seconds` at the top of the table and `instrument_seconds` on each
// record are ONE quantity split two ways, so a charge that reached neither would
// be an unmeasured cost invisible to every reader of either number. Everything
// asserted here is an inequality between two numbers in the same file:
// non-negative, no record charged more than its own duration, and the table's
// share no larger than the residue it claims to be part of. A box under load
// moves every one of these numbers and moves none of these verdicts.
TEST_CASE("ltx2 phase log: the instrument's own cost is CONSERVED across the table and its records") {
  phase::PhaseLog& log = phase::PhaseLog::Instance();
  log.Reset();
  log.Begin();
  {
    const phase::Scope one("unit.one");
    SleepMs(30);
    { const phase::Scope inner("unit.one.inner"); }
  }
  {
    const phase::Scope two("unit.two");
    SleepMs(30);
  }

  const TableFile file;
  const nlohmann::json table = file.Write(log);

  REQUIRE(table.contains("instrument_seconds"));
  const double table_charge = table["instrument_seconds"].get<double>();
  CHECK(table_charge >= 0.0);
  double record_charge = 0.0;
  for (const nlohmann::json& e : table["phases"]) {
    REQUIRE_MESSAGE(e.contains("instrument_seconds"),
                    "the record for '" << e["name"].get<std::string>()
                                       << "' carries no instrument charge");
    const double c = e["instrument_seconds"].get<double>();
    CHECK_MESSAGE(c >= 0.0, "'" << e["name"].get<std::string>() << "' was charged " << c << "s");
    CHECK_MESSAGE(c <= e["duration_seconds"].get<double>() + 1e-9,
                  "'" << e["name"].get<std::string>() << "' was charged " << c
                      << "s of its own " << e["duration_seconds"].get<double>()
                      << "s duration, which is more instrument than record");
    record_charge += c;
  }
  MESSAGE("instrument: table " << table_charge << "s + records " << record_charge << "s");
  CHECK_MESSAGE(record_charge > 0.0,
                "no record carries any instrument charge, so the per-record half of the "
                "accounting is not reaching the emitted table");

  const double wall = table["wall_seconds"].get<double>();
  const double unaccounted = table["unaccounted_seconds"].get<double>();
  MESSAGE("wall " << wall << "s, unaccounted " << unaccounted << "s, table charge "
                  << table_charge << "s");
  REQUIRE(wall > 0.0);
  CHECK_MESSAGE(table_charge > 0.0,
                "the table's own instrument charge is " << table_charge
                    << "s across a timeline that opened and closed three scopes with nothing "
                       "live between the last two, so `ChargeLocked` never reached the `no live "
                       "leaf` arm. That arm is the whole of `unaccounted_seconds`'s explanation");
  CHECK_MESSAGE(unaccounted >= table_charge - 1e-9,
                "the table reports " << unaccounted << "s of un-named time and claims "
                    << table_charge
                    << "s of it is this instrument's own. A charge larger than the residue it "
                       "is part of means the accounting is charging intervals that are inside a "
                       "leaf to the table, which would make every residue bound too loose");

  log.Reset();
}

// ─── and the TABLE's own share is disjoint too, which is a SECOND arm ────────
//
// `ChargeLocked` has two targets and the clamp landed on both, but only the
// record arm was gated: a second fresh review removed the TABLE arm's
// high-water mark and the whole file stayed green, because every other case
// charges the table from ONE thread, where the intervals are already disjoint.
//
// The shape that breaks it is the hammer above with NOTHING LIVE. `Tick` reads
// its clock before taking the process-wide mutex, so N threads blocked on the
// same acquisition each charge their own full wait — and with no leaf open they
// all land on `instrument_gap`. Overlapping charges to one counter then sum past
// the timeline that contains them.
//
// THE BOUND IS THE TIMELINE ITSELF and it is not a measurement. `Instrument()`
// is a sum of intervals inside `[0, wall]`; if they are disjoint their sum is at
// most `wall`, by arithmetic. A slow box moves both numbers and moves no
// verdict.
TEST_CASE("ltx2 phase log: the TABLE's own charge is disjoint so it cannot exceed the timeline") {
  phase::PhaseLog& log = phase::PhaseLog::Instance();
  log.Reset();
  log.Begin();

  // The same deterministic blocker the record-arm case uses: a device probe that
  // sleeps WHILE HOLDING the mutex makes the pile-up happen every run rather
  // than sometimes, and a detector that fires sometimes is what #1569 is about.
  std::atomic<bool> blocking(true);
  log.SetDeviceProbe([&blocking]() -> int64_t {
    if (blocking.load(std::memory_order_relaxed)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return -1;
  });

  const int kThreads = 24;
  const int kRounds = 12;
  std::atomic<bool> stop_blocker(false);
  std::thread blocker([&stop_blocker]() {
    while (!stop_blocker.load(std::memory_order_relaxed)) phase::SampleNow();
  });
  std::vector<std::thread> hammers;
  hammers.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    hammers.emplace_back([kRounds, t]() {
      for (int i = 0; i < kRounds; ++i) {
        phase::Tick("unit.table.hammer", static_cast<int64_t>(t), "concurrent");
      }
    });
  }
  for (std::thread& h : hammers) h.join();
  stop_blocker.store(true, std::memory_order_relaxed);
  blocker.join();
  blocking.store(false, std::memory_order_relaxed);
  log.SetDeviceProbe(phase::DeviceByteProbe());

  // NOTHING WAS EVER LIVE, so every charge above went to the table.
  const double charge = log.Instrument();
  const double wall = log.Elapsed();
  MESSAGE("table charge " << charge << "s over a timeline of " << wall << "s by " << kThreads
                          << " threads (ratio " << (charge / wall) << ")");
  REQUIRE_MESSAGE(phase::ProgressEnabled(),
                  "VLLM_RENDER_PROGRESS is off in this process, so `phase::Tick` returned "
                  "without taking the mutex and the threads above contended over nothing");
  REQUIRE_MESSAGE(charge > 0.0,
                  "no leaf was ever live and " << kThreads
                      << " threads ticked, and the table was charged nothing. This case is "
                         "measuring an instrument that is switched off");
  CHECK_MESSAGE(charge <= wall + 1e-9,
                "the table charged itself " << charge << "s inside a timeline of " << wall
                    << "s. Charges to the table must be disjoint; overlapping ones are counted "
                       "twice, and every reader who subtracts this number from a residue would "
                       "subtract more than the residue holds");
  log.Reset();
}

// ─── the CONSOLE copy answers the same question as the FILE copy ─────
//
// `RenderText` is what a reader watching a terminal gets, and it printed
// `sum(leaf)`, `unaccounted` and `WALL` while the file alone carried
// `instrument_seconds` — so the residue was on screen with no way to subtract
// the cost of naming the phases from it, which is the whole reason that number
// exists. This case is the only thing holding the two copies together; without
// it the added line is a production statement nothing reaches.
//
// THE TOLERANCE IS THE FORMAT'S OWN RESOLUTION AND NOT A MEASUREMENT.
// `RenderText` prints every total with `%10.3f`, so the printed value differs
// from the number it was given by strictly less than half of the last digit --
// 5e-4 seconds, by the definition of the conversion. Nothing here is timed:
// both sides read the SAME `Instrument()` through two different code paths, so
// no box load can move the verdict.
//
// ── AND THE QUANTITY HAS TO BE ABOVE THAT RESOLUTION, WHICH IS THE WHOLE CASE
//
// The first version of this gate lived inside the conservation case above, on
// its three-scope timeline, and a fresh review measured it VACUOUS. There
// `Instrument()` is 2.80e-4 to 2.95e-4 s — SMALLER than the tolerance — so
// `%10.3f` prints `0.000` for the honest value and the comparison is satisfied
// by anything in `(-5e-4, +5e-4)`, the literal `0.0` included. Mutation
// `NTEXTZERO` replaced `Instrument()` with `0.0` in the production line and the
// whole file stayed GREEN at `5 | 5 passed` and `93 | 93 passed`.
//
// That is #1569's own failure reproduced inside #1569's own repair: a
// discriminator below the instrument's resolution cannot discriminate. The fix
// is the same one #1569 took — make the quantity an EVENT, and REFUSE to assert
// when it is not. The table's own share accrues one boundary at a time in the
// gaps where no leaf is live, so a few thousand sequential scopes put it three
// orders of magnitude above the format's last digit, and the `REQUIRE` below
// says so out loud rather than passing quietly on a table too cheap to print.
TEST_CASE("ltx2 phase log: the CONSOLE copy carries the same instrument charge as the FILE copy") {
  phase::PhaseLog& log = phase::PhaseLog::Instance();
  log.Reset();
  log.Begin();
  // SEQUENTIAL AND NOT NESTED, because it is the gaps BETWEEN leaves that the
  // table's own share is made of: `ChargeLocked` reaches `instrument_gap` only
  // when no leaf is live.
  const int kScopes = 4000;
  for (int i = 0; i < kScopes; ++i) {
    const phase::Scope leaf("unit.console.leaf");
  }

  // `%10.3f` rounds to the nearest millisecond, so half of the last digit is
  // the largest error the conversion can introduce. Both numbers below are the
  // same `double`, so this is the only difference that can exist between them.
  const double kFormatResolution = 5e-4;
  const double charge = log.Instrument();
  REQUIRE_MESSAGE(charge > 20.0 * kFormatResolution,
                  "the table charged itself " << charge << "s over " << kScopes
                      << " scopes, which `RenderText`'s `%10.3f` cannot distinguish from zero at "
                         "its own " << kFormatResolution
                      << "s resolution. A gate over a quantity below the resolution of the "
                         "thing it reads is satisfied by a hardcoded zero -- which is exactly "
                         "how #1569's three-record case stayed green under its own mutation, and "
                         "how the first version of THIS case did too");

  const std::string text = log.RenderText("unit", "cpu");
  const std::string::size_type at = text.find("instrument");
  REQUIRE_MESSAGE(at != std::string::npos,
                  "`RenderText` prints sum(leaf), unaccounted and WALL and no instrument "
                  "charge, so the console copy of this table cannot separate the residue the "
                  "render produced from the residue this instrument produced. The file copy "
                  "carries it and they have to answer the same question:\n"
                      << text.substr(0, 400));
  const double printed = std::strtod(text.c_str() + at + std::string("instrument").size(),
                                     nullptr);
  MESSAGE("console instrument " << printed << "s against Instrument() " << charge << "s over "
                                << kScopes << " scopes");
  CHECK_MESSAGE(std::fabs(printed - charge) < kFormatResolution,
                "`RenderText` prints " << printed << "s of instrument charge and `Instrument()` "
                    << "returns " << charge
                    << "s. The console copy is printing some other number, and "
                    << kFormatResolution
                    << " is this line's own %10.3f resolution rather than a tolerance");
  log.Reset();
}

// ─── WHY THERE IS NO PER-SITE CHARGE CASE HERE (F1, and it was TRIED) ───────
//
// A fresh review found that the attribution rule is gated only in AGGREGATE:
// deleting any ONE charge site -- `Open`'s pre-lock mutex wait, `Open`'s tail,
// `Close`'s tail, the sampler join, `SampleLocked`'s self-charge -- leaves this
// whole file green, and only deleting every site at once reddens it. The worst
// of those is the pre-lock wait, because `## Design` 1 and `Open`'s own comment
// both name that interval as the reason `instrument_seconds` exists.
//
// A case for it was written and run, three times, and it is NOT here because it
// does not measure that site. What each cut found:
//
//   1. Two parents, one contended by a reader thread, compared by charge. The
//      QUIET parent was slower, because it ran first and paid for the record
//      vector's reallocation -- a copy that happens under the mutex inside
//      `Close` and is charged to the parent.
//   2. With a warm-up parent absorbing the growth, the reader thread had not
//      reached its first copy before the 330 us measurement window closed.
//   3. With four readers, a 16 MiB critical section and a start barrier, the
//      case passed in ISOLATION at a separation of 615x -- and failed 5 of 5
//      runs inside this suite, twice because no reader held the lock at all and
//      three times because a contended parent that took 21 ms was charged
//      112 us.
//
// THAT LAST NUMBER IS THE FINDING. Contention through `Records()` -- the only
// public entry point that holds this mutex without charging itself, which is
// what makes it usable as a hold at all -- lands mostly in `PhaseLog::Close`'s
// lock wait. `Close` has no pre-lock clock read, so that wait is charged to
// NOBODY, and it lies inside the CHILD's own duration rather than in the
// parent's charge. So a parent-against-parent comparison does not track the
// site under test, however the contention is arranged.
//
// Both halves are recorded under `## Owed` in
// `.agents/specs/ltx25-phase-instrument.md`: the per-site gate, and the
// uncharged `Close` wait this attempt found. A flaky gate over an instrument
// whose whole subject is flaky gates would be the joke that writes itself.

// ─── the accounting stays CONSERVED when several threads charge one record ───
//
// F3 of the fresh review, and it was a real defect rather than a missing test.
// `instrument_seconds <= duration_seconds` was asserted, documented as an
// invariant, and was NOT one: `Open` and `Tick` read their clock before taking
// the mutex, so their charge to a record spans a window in which another thread
// holding the mutex charges the SAME record. The sum was over OVERLAPPING
// intervals while the duration is one wall interval, and 24 threads calling
// `SampleNow()` inside one live leaf drove the ratio to 1.914, red in 3 runs of
// 5.
//
// `ChargeLocked` now clamps each charge to the end of the last one that reached
// the same target, so the charges are disjoint and their sum is at most their
// union. This case is what holds that, and it is the reviewer's own probe shape.
TEST_CASE("ltx2 phase log: a record charged from MANY THREADS is still charged less than it lasted") {
  phase::PhaseLog& log = phase::PhaseLog::Instance();
  log.Reset();
  log.Begin();

  // THE HAMMER TICKS, AND WHICH ENTRY POINT IT USES IS THE WHOLE REPRODUCTION.
  //
  //   * `PhaseLog::Sample` reads its clock INSIDE the mutex, so concurrent
  //     samples are serialised and their charges are already disjoint. A first
  //     cut of this case used one and measured 0.29 -- and stayed GREEN under
  //     the very mutation it was written for, which is the trap this whole
  //     cluster is about.
  //   * Opening SIBLING scopes does not work either: while several siblings are
  //     live, "the innermost live non-span record" resolves to a sibling rather
  //     than to the leaf under test, so the overlapping charges land somewhere
  //     else.
  //   * `PhaseLog::Tick` reads its clock BEFORE the mutex and pushes nothing, so
  //     the innermost live record stays the leaf. N threads blocked on the same
  //     acquisition each charge their OWN full wait to it, and those waits
  //     overlap by construction. That is the shape that breaks the invariant.
  // AND A BLOCKER MAKES THE PILE-UP DETERMINISTIC. `SetDeviceProbe` installs a
  // callback that `SampleLocked` runs WHILE HOLDING the mutex, so a probe that
  // sleeps holds it for a known time. Without one the ticking threads spread out
  // and mostly do not queue: the unclamped tree measured 1.85 on one run and
  // 0.29 on the next, so the mutation that removes the repair went GREEN. A
  // detector that fires sometimes is exactly what #1569 is about.
  std::atomic<bool> blocking(false);
  log.SetDeviceProbe([&blocking]() -> int64_t {
    if (blocking.load(std::memory_order_relaxed)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return -1;
  });

  const size_t leaf = log.Open("unit.hammered", /*span=*/false);
  const int kThreads = 24;
  const int kRounds = 12;
  std::atomic<int> ready(0);
  std::atomic<bool> stop_blocker(false);
  blocking.store(true, std::memory_order_relaxed);
  std::thread blocker([&stop_blocker]() {
    while (!stop_blocker.load(std::memory_order_relaxed)) phase::SampleNow();
  });
  std::vector<std::thread> hammers;
  hammers.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    hammers.emplace_back([&ready, kRounds, t]() {
      ready.fetch_add(1, std::memory_order_relaxed);
      for (int i = 0; i < kRounds; ++i) {
        phase::Tick("unit.hammer", static_cast<int64_t>(t), "concurrent");
      }
    });
  }
  for (std::thread& h : hammers) h.join();
  stop_blocker.store(true, std::memory_order_relaxed);
  blocker.join();
  blocking.store(false, std::memory_order_relaxed);
  log.SetDeviceProbe(phase::DeviceByteProbe());
  log.Close(leaf);
  REQUIRE(ready.load() == kThreads);
  // THE LIVE LANE HAS TO BE ON FOR THIS CASE TO MEAN ANYTHING. `Tick` returns
  // immediately when `VLLM_RENDER_PROGRESS=0`, so a process that silenced it
  // would run this case as 24 threads doing nothing at all.
  REQUIRE_MESSAGE(phase::ProgressEnabled(),
                  "VLLM_RENDER_PROGRESS is off in this process, so `phase::Tick` returned "
                  "without taking the mutex and the 24 threads above contended over nothing. "
                  "This case measured an instrument that is switched off");

  double charge = -1.0;
  double duration = -1.0;
  for (const phase::Record& r : log.Records()) {
    if (r.name != "unit.hammered") continue;
    charge = r.instrument_seconds;
    duration = r.end - r.start;
  }
  REQUIRE(duration > 0.0);
  // THE RATIO SITS JUST UNDER 1 ON PURPOSE, and a reader should not read that as
  // a bound about to flap. Measured over 45 runs at load 58-85: median 0.9920,
  // maximum 0.9961. This leaf is 24 threads ticking and almost nothing else, so
  // almost all of it IS instrument -- that is what the case constructs. The
  // bound cannot be crossed by a slow box, because the charges to one record are
  // disjoint intervals inside it: the arithmetic, not the margin, is what holds.
  MESSAGE("unit.hammered = " << duration << "s, charged " << charge << "s by " << kThreads
                             << " threads (ratio " << (charge / duration) << ")");
  // THE PRECONDITION. If the hammer threads charged nothing, the bound below is
  // satisfied by an instrument that is not running.
  REQUIRE_MESSAGE(charge > 0.0,
                  "24 threads ticked inside a live leaf and it was charged nothing, so this "
                  "case is measuring an instrument that is switched off rather than an "
                  "invariant");
  // AND NO TOLERANCE, BEYOND THE ONE THE SUM OF DOUBLES NEEDS. This is not a
  // measurement: the charges to one record are disjoint intervals inside it, so
  // their sum cannot exceed it. A box under load moves both numbers and moves
  // neither verdict -- which is what the previous version of this claim said and
  // was wrong about, before the clamp existed.
  CHECK_MESSAGE(charge <= duration + 1e-9,
                "'unit.hammered' lasted " << duration << "s and was charged " << charge
                    << "s, which is more instrument than record. Charges to one record must be "
                       "disjoint; overlapping ones are being counted twice, and every reader "
                       "who subtracts this number from a residue would subtract too much");
  log.Reset();
}

// ─── the residue is DECOMPOSED into the gaps that make it (#1571) ────────────
//
// `unaccounted_seconds` shipped as an aggregate, and four issues — #1439, #1470,
// #1494 and #1536 — argued about whether its 95% floor was the right tolerance
// without anyone splitting it into the gaps between consecutive leaves.
// Splitting it took one pass over the table the render already writes and
// settled the question: **92% of the residue was ONE gap**, the load's prologue
// from the timeline's origin to `Open("load.dit")`, 17.661 ms of 19.178 ms,
// while the sixteen gaps between adjacent named phases held 6.8 us each. That
// pass was a scratch script nobody shipped.
//
// THE IDENTITY IS ARITHMETIC RATHER THAN A TOLERANCE, AND IT HOLDS EXACTLY ONE
// THING. A fresh review did the algebra: the gap sum telescopes to
// `wall - sum(durations)` for ANY record sequence -- ordered or not, overlapping
// or not -- so the identity cannot see a reordering, and the reviewer confirmed
// it by execution, reversing the record order and watching the identity stay
// green while two other assertions fired. The comment here used to claim it
// caught a mis-ordering. It does not.
//
// WHAT IT DOES HOLD is that `GapsBetweenLeaves` and `Sum` select the SAME
// records: both skip `span` and both skip `nested`. A decomposition that walked
// a different population -- one that counted a nested record, or dropped one, or
// stopped short of `wall` -- fails by an amount no box load can supply. That is
// the whole of it, it is worth having, and three of this file's mutations are
// caught by nothing else.
//
// The ORDER, the POSITIONS and the NAMES are held below by (1), (2) and (3),
// which is where those claims belong.
//
// THE ONE DURATION HERE IS A LOWER BOUND ON A SLEEP, which is the only shape of
// wall-clock assertion contention cannot break: `sleep_for` returns no earlier
// than its argument and a loaded box only makes it later.
TEST_CASE("ltx2 phase log: the emitted table DECOMPOSES its residue into the gaps between leaves") {
  phase::PhaseLog& log = phase::PhaseLog::Instance();
  log.Reset();
  log.Begin();

  // THE PROLOGUE, which is the shape of the defect this decomposition was
  // written to find: time before the first named leaf, inside no leaf, invisible
  // in every aggregate. A span is open across it exactly as the driver's `load`
  // span is, so this gap is inside a span and outside every leaf — the position
  // `Sum` cannot see.
  const int kPrologueMs = 60;
  const phase::Scope enclosing("unit.enclosing", /*span=*/true);
  SleepMs(kPrologueMs);
  { const phase::Scope a("unit.a"); SleepMs(5); }
  SleepMs(5);  // an interior gap, between two adjacent named leaves
  {
    const phase::Scope b("unit.b");
    SleepMs(5);
    // A nested child, so the decomposition has to skip a record that is inside a
    // leaf rather than between two. Counting it would produce a negative gap.
    const phase::Scope inner("unit.b.inner");
    SleepMs(1);
  }
  SleepMs(5);  // the tail gap, before the writer reads the clock

  const TableFile file;
  const nlohmann::json table = file.Write(log);

  REQUIRE_MESSAGE(table.contains("gaps"),
                  "the emitter writes `unaccounted_seconds` as an aggregate and nothing else, "
                  "so a reader still cannot see WHICH gap holds it without writing a script -- "
                  "which is how four issues argued about a tolerance instead (#1571)");
  REQUIRE(table["gaps"].is_array());
  REQUIRE(table.contains("gap_rule"));
  CHECK(!table["gap_rule"].get<std::string>().empty());

  // The leaves the decomposition must lie between, in the emitter's own order.
  std::vector<std::string> leaf_names;
  std::vector<double> leaf_starts;
  std::vector<double> leaf_ends;
  for (const nlohmann::json& e : table["phases"]) {
    if (e.value("span", false) || e.value("nested", false)) continue;
    leaf_names.push_back(e["name"].get<std::string>());
    leaf_starts.push_back(e["start_seconds"].get<double>());
    leaf_ends.push_back(e["start_seconds"].get<double>() + e["duration_seconds"].get<double>());
  }
  REQUIRE(leaf_names.size() == 2);

  const nlohmann::json& gaps = table["gaps"];
  REQUIRE(!gaps.empty());

  // (1) NO GAP IS NEGATIVE, EACH ONE'S SECONDS ARE ITS OWN ENDPOINTS, AND THEY
  // ADD TO THE RESIDUE. All non-fatal and all first, so one broken decomposition
  // reports every way it is broken rather than the first one. THE ORDER HERE IS
  // A REPAIR: the count below was a `REQUIRE` above this loop, and a mutation
  // that counted NESTED records as leaves aborted the case on the count and
  // never reached these — so the negative gap that same mutation produces went
  // unobserved, and two of this case's assertions were unproven while the case
  // reddened.
  //
  // THE ENDPOINTS ARE READ, WHICH A FRESH REVIEW FOUND THEY WERE NOT. They were
  // emitted and asserted nowhere: zeroing both left this whole file green. They
  // are the half a reader uses to LOCATE a region, and locating one was the
  // entire content of #1571 — 92% of a residue sat at one position, and a
  // decomposition that says how long a gap is but not where it starts sends the
  // reader back to the script it replaced.
  double gap_total = 0.0;
  for (size_t i = 0; i < gaps.size(); ++i) {
    const nlohmann::json& g = gaps[i];
    REQUIRE(g.contains("after"));
    REQUIRE(g.contains("before"));
    REQUIRE(g.contains("seconds"));
    REQUIRE(g.contains("start_seconds"));
    REQUIRE(g.contains("end_seconds"));
    const double seconds = g["seconds"].get<double>();
    const double from = g["start_seconds"].get<double>();
    const double to = g["end_seconds"].get<double>();
    INFO("gap " << i << " = " << g["after"].get<std::string>() << " -> "
                << g["before"].get<std::string>());
    CHECK_MESSAGE(seconds >= 0.0,
                  "gap " << i << " between '" << g["after"].get<std::string>() << "' and '"
                         << g["before"].get<std::string>() << "' is " << seconds
                         << "s. A negative gap means two records the emitter is treating as "
                            "non-overlapping leaves overlap, which would make every sum in this "
                            "table the residue of double counting");
    CHECK_MESSAGE(std::fabs((to - from) - seconds) < 1e-9,
                  "gap " << i << " runs from " << from << "s to " << to << "s, a span of "
                         << (to - from) << "s, and reports " << seconds
                         << "s. A reader who sorts by `seconds` and then looks up the region at "
                            "those endpoints would be sent somewhere else");
    gap_total += seconds;
  }
  // AND EACH GAP'S ENDPOINTS ARE THE TWO RECORDS IT LIES BETWEEN, read out of
  // the SAME table. This is where the order is held, and getting here took two
  // wrong answers worth recording.
  //
  // THE IDENTITY BELOW HOLDS ONE THING AND IT IS NOT THE ORDER. A fresh review
  // did the algebra: the gap sum telescopes to `wall - sum(durations)` for ANY
  // record sequence, ordered or not, overlapping or not, and confirmed it by
  // execution -- reversing the record order left the identity green while two
  // other assertions fired. What it does hold is that `GapsBetweenLeaves` and
  // `Sum` select the SAME records: both skip `span` and both skip `nested`.
  //
  // AND THE FIRST REPAIR FOR IT WAS TWO TAUTOLOGIES, which a SECOND fresh
  // review measured. `gaps[i].start >= gaps[i-1].end` reduces, inside this
  // emitter, to `records[i-1].end >= records[i-1].start` -- a non-negative leaf
  // duration, which a monotone clock guarantees unconditionally. And
  // `gaps.front().start == 0.0` is constant-true unless the first gap is
  // dropped, because `cursor` is initialised to the literal `0.0`. The reviewer
  // staged `cursor = r.start` instead of `r.end`, so every gap swallows the leaf
  // before it, and BOTH of those passed; only the pre-existing identity fired.
  //
  // Comparing each endpoint against the leaf record it is supposed to touch is
  // what that mutation cannot survive, and it needs no tolerance beyond double
  // rounding because both numbers come out of the same emitted file.
  REQUIRE(gaps.size() == leaf_names.size() + 1);
  for (size_t i = 0; i < gaps.size(); ++i) {
    const double from = gaps[i]["start_seconds"].get<double>();
    const double to = gaps[i]["end_seconds"].get<double>();
    const double expect_from = i == 0 ? 0.0 : leaf_ends[i - 1];
    const double expect_to =
        i == leaf_names.size() ? table["wall_seconds"].get<double>() : leaf_starts[i];
    INFO("gap " << i);
    CHECK_MESSAGE(std::fabs(from - expect_from) < 1e-9,
                  "gap " << i << " starts at " << from << "s and the record before it ends at "
                         << expect_from
                         << "s. A gap that does not begin where the previous leaf ended is "
                            "either overlapping that leaf or skipping part of the timeline, and "
                            "the sum can still reconcile while it does");
    CHECK_MESSAGE(std::fabs(to - expect_to) < 1e-9,
                  "gap " << i << " ends at " << to << "s and the record after it starts at "
                         << expect_to
                         << "s. A reader who looks up the region at those endpoints would be "
                            "sent somewhere else");
  }

  // THE IDENTITY. No tolerance beyond double rounding over a handful of
  // additions: this is the same arithmetic `Sum` does, read from the other side.
  const double unaccounted = table["unaccounted_seconds"].get<double>();
  MESSAGE("gaps sum " << gap_total << "s against an unaccounted " << unaccounted << "s over "
                      << gaps.size() << " gaps");
  CHECK_MESSAGE(std::fabs(gap_total - unaccounted) < 1e-9,
                "the gaps add to " << gap_total << "s and the table reports " << unaccounted
                    << "s of un-named time. A decomposition that does not reconcile with the "
                       "quantity it decomposes sends the next reader after the wrong region");

  // (2) ONE GAP BEFORE EACH LEAF AND ONE AFTER THE LAST. A decomposition with a
  // different count is not a partition of the timeline, whatever its sum says.
  CHECK_MESSAGE(gaps.size() == leaf_names.size() + 1,
                "the table names " << leaf_names.size() << " leaves and reports " << gaps.size()
                    << " gaps. A partition of `[0, wall]` by N non-overlapping leaves has "
                       "exactly N+1 complementary intervals");

  // (3) AND EACH GAP NAMES THE TWO LEAVES IT LIES BETWEEN, which is the half a
  // reader uses. A sum that reconciles while the names are wrong points the next
  // investigation at the wrong region, which is the failure #1571 is about.
  // Guarded on the count, because the pairing below is only defined when the
  // decomposition IS a partition — and the guard is announced rather than
  // silent, since an assertion that turned itself off would look exactly like
  // one that passed.
  if (gaps.size() != leaf_names.size() + 1) {
    MESSAGE("  the gap/leaf pairing is SKIPPED: the counts above already disagree, so there is "
            "no pairing to check. The count assertion is what speaks here.");
  } else {
    for (size_t i = 0; i < gaps.size(); ++i) {
      const std::string after = gaps[i]["after"].get<std::string>();
      const std::string before = gaps[i]["before"].get<std::string>();
      const std::string expect_after = i == 0 ? std::string("<origin>") : leaf_names[i - 1];
      const std::string expect_before =
          i == leaf_names.size() ? std::string("<end>") : leaf_names[i];
      INFO("gap " << i);
      CHECK_MESSAGE(after == expect_after,
                    "gap " << i << " says it follows '" << after
                           << "' and the table's leaf order says '" << expect_after << "'");
      CHECK_MESSAGE(before == expect_before,
                    "gap " << i << " says it precedes '" << before
                           << "' and the table's leaf order says '" << expect_before << "'");
    }
  }

  // (4) AND THE PROLOGUE IS THE ONE A READER NEEDS TO SEE. `sleep_for` returns
  // no earlier than its argument, so this lower bound is one contention can only
  // move away from red. Before this decomposition existed, exactly this region
  // was 92% of a real render's residue and no reader of the file could name it.
  const double prologue = gaps[0]["seconds"].get<double>();
  CHECK_MESSAGE(prologue >= 0.001 * static_cast<double>(kPrologueMs) - 1e-3,
                "the timeline slept " << kPrologueMs
                    << "ms inside a span and outside every leaf, and the decomposition reports "
                    << prologue
                    << "s before the first leaf. The prologue is the region that held 92% of "
                       "the LTX-2.5 load's residue, and a decomposition that cannot see it is "
                       "the aggregate it replaced");
  log.Reset();
}

// ─── the writer's clock stops BEFORE the writer works (#1569) ────────────────
//
// `PhaseLog::WriteJson` reads `Elapsed()` before it copies and sorts the record
// vector, so the writer's own serialization is not charged to `wall_seconds` and
// therefore not to `unaccounted_seconds`. That table measures the RENDER.
//
// **NOTHING ASSERTED IT, AND ITS OWN MUTATION STAYED GREEN 10 OF 10.** A fresh
// review of #1556 restored the late clock read and the case that claimed to pin
// the ordering passed every time, at `wall 0.0608987s, unaccounted 0.000534223s,
// table charge 0.000301655s`, because the copy and the sort of a THREE-record
// table are nanoseconds — far below the slack in any bound that case carried.
// An instrument whose own mutation cannot fail is not an instrument (#1569).
//
// WHAT MAKES IT GATEABLE IS A TABLE BIG ENOUGH FOR THE SORT TO EXIST, and a
// discriminator measured in the same run rather than written down as a constant.
// The case builds `kRecords` leaves, then measures two quantities K times:
//
//   * `head` — the elapsed clock read by this case immediately before the call,
//     against the `wall_seconds` the writer recorded. With the clock read first
//     the writer's clock is one function call and one uncontended mutex behind
//     this case's own, i.e. the instrument's resolution. With it read late the
//     head contains a whole copy and a whole `stable_sort`.
//   * `copy` and `sort` — those two steps, performed by this case through the
//     same public `Records()`, on the same data, on this box, in this run.
//     MEASURED SEPARATELY, and the bound is against the SMALLER of them.
//
// THE TWO ARE SEPARATE BECAUSE A FRESH REVIEW BROKE THE ONE-NUMBER FORM. This
// case first bounded the head against `copy + sort` together, on the argument
// that the mutated head contains one of each and is therefore at least 1.0x
// their sum. That is true only when BOTH move. The reviewer hoisted `Records()`
// above the clock read and left `ByStart` below it -- the natural shape of a
// partial regression, and the exact edit somebody makes while "just moving one
// line" -- and the bound stayed GREEN at a ratio of 0.0588, because the copy is
// about 6% of copy-plus-sort on this data. Against `min(copy, sort)` that same
// mutation is red. The constant did not move; the quantity under it got smaller,
// which is the only direction a repair may take a bound.
//
// AND THE ESTIMATOR IS A MINIMUM, WHICH IS WHY THIS IS NOT THE WITHDRAWN BOUND
// AGAIN. Contention is ONE-SIDED: it can only make a measured interval longer,
// never shorter. The honest head is a floor of ~1e-7 s plus a preemption that
// lands in it sometimes; the mutated head has a HARD floor of one serialization,
// which is present in every single iteration. A minimum over K iterations
// therefore strips the sporadic term from the honest side and cannot strip the
// deterministic term from the defective side. That is the difference between
// this and `residue <= 2 * instrument`: there, both sides were single
// measurements of comparable magnitude and the tail decided the gate; here the
// two sides differ by orders of magnitude and the estimator removes the tail by
// construction.
//
// THE FACTOR IS 0.5 AND IT IS NOT A TOLERANCE. Under the correct ordering the
// head contains ZERO copies and ZERO sorts. Under any wrong ordering it contains
// at least ONE of the two steps in full, so it is at least
// `1.0 x min(copy, sort)` by the definition of the quantities. Any constant
// strictly between 0 and 1 separates them; 0.5 is the midpoint. What the
// measurement decides is not the constant but whether the separation is real,
// and it is: see `## Evidence` in the row's spec.
TEST_CASE("ltx2 phase log: the emitter reads its CLOCK before it serialises the table") {
  phase::PhaseLog& log = phase::PhaseLog::Instance();
  log.Reset();
  log.Begin();

  // ENOUGH RECORDS FOR THE SORT TO BE A MEASURABLE EVENT. Three records made
  // this ungateable; the sort is `n log n` on a vector of records carrying a
  // `std::string`, so the discriminator grows with `kRecords` while the honest
  // head does not depend on it at all.
  // 8000 RATHER THAN 4000, and the reason is the budget below rather than the
  // sort. The bound is against the CHEAPER of the writer's two post-clock steps,
  // which is the copy, and the copy is linear in the record count while the
  // honest head does not depend on it at all. Doubling the table doubles the
  // margin. Three records made this ungateable in the first place (#1569).
  const int kRecords = 8000;
  // AND A NAME LONGER THAN THE SMALL-STRING BUFFER, which is not decoration.
  // The budget below is the CHEAPER of the writer's two post-clock steps, and
  // with a short name the copy is a flat memcpy while the sort is `n log n`, so
  // the copy is the cheap one by a factor of twenty and it sets the whole
  // budget. A name past `std::string`'s inline buffer makes the copy allocate
  // once per record, which is what the LTX-2.5 table's own names
  // (`decode.video.chunk`, `artifacts.frames.ppm`) do anyway. The two steps then
  // sit within one order of magnitude of each other and the budget stops being
  // decided by an implementation detail of `std::string`.
  const std::string kLeafName = "unit.leaf.with.a.name.past.the.small.string.buffer";
  // ONE SPAN HELD OPEN ACROSS THE BUILD, for two reasons. It stops the sampler
  // thread from being created and joined once per leaf, which would dominate the
  // build; and closing it before the measurement leaves NOTHING live, so the
  // 100 ms worker is not running while the clocks below are read.
  {
    const phase::Scope holder("unit.holder", /*span=*/true);
    for (int i = 0; i < kRecords; ++i) {
      const phase::Scope leaf(kLeafName);
    }
  }

  const TableFile file;
  const int kProbes = 5;
  double head = -1.0;
  double copy_cost = -1.0;
  double sort_cost = -1.0;
  int64_t emitted = 0;
  for (int k = 0; k < kProbes; ++k) {
    const double before_call = log.Elapsed();
    const nlohmann::json table = file.Write(log);
    const double writer_clock = table["wall_seconds"].get<double>();
    emitted = static_cast<int64_t>(table["phases"].size());
    const double this_head = writer_clock - before_call;
    if (head < 0.0 || this_head < head) head = this_head;

    // THE DISCRIMINATOR, MEASURED THE SAME WAY THE WRITER DOES IT, AND IN THE
    // SAME TWO STEPS. `Records()` returns a copy taken under the process-wide
    // mutex and `ByStart` stable-sorts that copy; this is the same copy and the
    // same sort through the same public entry point, so these are the costs the
    // writer would pay after its clock read rather than numbers quoted from
    // another box.
    const double before_copy = log.Elapsed();
    std::vector<phase::Record> copy = log.Records();
    const double after_copy = log.Elapsed();
    std::stable_sort(copy.begin(), copy.end(),
                     [](const phase::Record& a, const phase::Record& b) {
                       return a.start < b.start;
                     });
    const double after_sort = log.Elapsed();
    // Kept from being optimised away: the sorted copy has to be observed.
    REQUIRE(!copy.empty());
    const double this_copy = after_copy - before_copy;
    const double this_sort = after_sort - after_copy;
    if (copy_cost < 0.0 || this_copy < copy_cost) copy_cost = this_copy;
    if (sort_cost < 0.0 || this_sort < sort_cost) sort_cost = this_sort;
  }
  // THE SMALLER OF THE TWO STEPS IS THE BUDGET. A wrong ordering puts at least
  // one of them after the clock read, so the head is at least this large.
  const double serialize = copy_cost < sort_cost ? copy_cost : sort_cost;

  REQUIRE_MESSAGE(emitted >= kRecords,
                  "the timeline was built with " << kRecords << " leaves and the table carries "
                      << emitted << " records, so the discriminator below was measured over a "
                                    "table that is not the one this case built");
  // THE INSTRUMENT'S OWN PRECONDITION, and it is what stops this case from being
  // a mute switch. If the copy and the sort cost nothing measurable, then the
  // bound below is `head < 0` and no ordering can satisfy it — but equally, a
  // `serialize` that collapsed toward the clock's resolution would make the
  // comparison meaningless in the other direction. It has to be an event.
  REQUIRE_MESSAGE(serialize > 1e-5,
                  "the CHEAPER of the writer's two post-clock steps over " << emitted
                      << " records measured " << serialize
                      << "s (copy " << copy_cost << "s, sort " << sort_cost
                      << "s), which is at or below this clock's own resolution. A wrong "
                         "ordering is detected by whichever step it moves, so a step this cheap "
                         "cannot be detected at all -- which is exactly why the three-record "
                         "case in #1569 stayed green under its own mutation");
  MESSAGE("writer clock lag " << head << "s against copy " << copy_cost << "s and sort "
                              << sort_cost << "s over " << emitted << " records (min of "
                              << kProbes << " probes, budget " << serialize << "s, ratio "
                              << (head / serialize) << ")");
  CHECK_MESSAGE(head < 0.5 * serialize,
                "`WriteJson` recorded a wall " << head
                    << "s later than the clock this case read immediately before calling it, "
                       "against the cheaper of its two measured post-clock steps at "
                    << serialize << "s (copy " << copy_cost << "s, sort " << sort_cost
                    << "s) over " << emitted
                    << " records. The writer is reading its clock AFTER it serialises the "
                       "table, so its own copy and sort are charged to `wall_seconds` and "
                       "therefore to `unaccounted_seconds`. This table measures the render");
  log.Reset();
}

// ─── the console copy reports the wall this emitter was ENTERED at (#1755) ───
//
// `PhaseLog::RenderText` is `WriteJson`'s SIBLING and it carried #1569's defect
// unrepaired: it read `Elapsed()` AFTER `ByStart(Records())`, so its own copy
// and its own sort were charged to the `WALL` it printed and therefore to the
// `unaccounted` row above it. And the call site made that worse. The console
// block stood at the END of `WriteJson`, after the whole `nlohmann` object was
// assembled, so the console copy also absorbed the JSON build: over five
// `WriteJson` calls on the 8001-record timeline of the case above, `sum(leaf)`
// held at 0.189 s while the console's `unaccounted` climbed 0.065 -> 0.134 ->
// 0.200 -> 0.265 -> 0.329 s. About 66 ms of writer work per call, charged to
// the render, on the copy of the table a reader actually watches.
//
// WHY NOTHING SAW IT. `RenderText` prints every total with `%10.3f`. A copy and
// a sort of a few thousand records are tenths of a millisecond, so the whole
// defect fits inside the last printed digit, and applying #1569's own one-line
// repair to this emitter left the suite at `7 | 7` and `100 | 100`. That is the
// mute switch `### 7` and `## Design` 6 are each about, met a third time.
//
// WHAT THIS CASE IS. Two comparisons against ONE structurally derived constant.
// `kFormatResolution` is half of `%10.3f`'s last digit, read off the emitter's
// own format string: a printed total differs from the number it was given by
// strictly less than that, by the definition of the conversion. The bound is
// ONE FULL step of that format, `2 * kFormatResolution`, which is the smallest
// bound the rounding itself cannot break. The honest side is held under it by
// ARITHMETIC and not by margin -- the two clock reads being compared are one
// function call apart, so the whole difference between them is the rounding --
// and the defective side is pushed over it by a quantity measured in the same
// run. Nothing here is a wall-clock tolerance and nothing here is a ratio:
// `ltx25-phase-residue.md` `## Design` 3 records the withdrawn one and #1668
// forbids re-proposing it.
//
//   (A) THE SHIPPED PATH, on a table small enough to serialise. `WriteJson`
//       with `VLLM_RENDER_PHASE_LOG_STDERR=1`, against the clock read
//       immediately before the call. Under the old call site the block was
//       rendered after the copy, the sort AND the whole JSON build, and that
//       build is what makes this arm's discriminator large. It is measured
//       here, in this run, by assembling the same per-record objects through
//       the same library.
//   (B) THE PUBLIC ENTRY POINT, on a table big enough for the copy alone to
//       cross the last printed digit. `RenderText` against the clock read
//       immediately before it. This is the arm that holds the ORDERING inside
//       the emitter, and it needs the bigger table because a wrong ordering
//       moves only the copy and the sort -- 0.12 ms over 8000 records, a
//       quarter of one step of `%10.3f`, which is precisely why this defect
//       survived the row that repaired its sibling.
//
// BOTH LAGS ARE MINIMA OVER PROBES, and that is what makes them safe on a
// loaded box. Contention is one-sided: it can only make a measured interval
// longer, so a minimum strips a preemption that lands between two adjacent
// statements from the honest side, and cannot strip the deterministic
// serialization from the defective side. Each `REQUIRE` refuses to assert at
// all when its discriminator has collapsed under the format's resolution,
// because a gate that silently loses its discriminator is the defect this case
// exists to close rather than a run of good luck. Two full steps is what each
// one demands, so a defect sitting exactly on that floor still exceeds the
// one-step bound by 1.5x after the worst rounding.
TEST_CASE("ltx2 phase log: the console copy reports the wall this emitter was ENTERED at") {
  phase::PhaseLog& log = phase::PhaseLog::Instance();
  const TableFile file;
  const std::string sink = file.path() + ".stderr";
  // `TableFile` removes its own directory and would fail to while this one
  // still holds a file, so the sink is removed on every exit including the
  // throw a failed `REQUIRE` takes.
  const RemoveOnExit sink_cleanup(sink);

  // Half of `%10.3f`'s last digit. Not a measurement and not a tolerance: it is
  // the largest error the conversion itself can introduce, and the neighbouring
  // console case reads the same number off the same format string.
  const double kFormatResolution = 5e-4;
  const double kStep = 2.0 * kFormatResolution;
  const int kProbes = 3;

  // ── (A) THE SHIPPED PATH ────────────────────────────────────────────────
  log.Reset();
  log.Begin();
  const int kSmall = 16000;
  BuildLeaves(kSmall, sink);

  // THE DISCRIMINATOR FOR THIS ARM, MEASURED THE WAY THE WRITER BUILDS IT.
  // These are the same per-record objects `WriteJson` assembles into `phases`,
  // through the same library, on the same data, on this box, in this run --
  // the work that stood between the writer's clock read and the console block
  // it rendered afterwards.
  double build_cost = -1.0;
  for (int k = 0; k < kProbes; ++k) {
    const std::vector<phase::Record> recs = log.Records();
    const double before = log.Elapsed();
    nlohmann::json phases = nlohmann::json::array();
    for (const phase::Record& r : recs) {
      nlohmann::json e;
      e["name"] = r.name;
      e["render"] = r.render;
      e["start_seconds"] = r.start;
      e["end_seconds"] = r.end;
      e["duration_seconds"] = r.end - r.start;
      e["peak_host_bytes"] = r.peak_host_bytes;
      e["peak_device_bytes"] = r.peak_device_bytes;
      e["span"] = r.span;
      e["nested"] = r.nested;
      e["instrument_seconds"] = r.instrument_seconds;
      phases.push_back(std::move(e));
    }
    const double this_build = log.Elapsed() - before;
    // Kept from being optimised away: the array has to be observed.
    REQUIRE(phases.size() == recs.size());
    if (build_cost < 0.0 || this_build < build_cost) build_cost = this_build;
  }
  MESSAGE("the writer's per-record JSON build over " << kSmall << " records is " << build_cost
                                                     << "s = " << (build_cost / kStep)
                                                     << " steps of the printed format");
  REQUIRE_MESSAGE(build_cost > 2.0 * kStep,
                  "assembling the writer's own `phases` array over " << kSmall
                      << " records measured " << build_cost
                      << "s, which `RenderText`'s `%10.3f` cannot separate from zero. This arm "
                         "detects a console block rendered after that build, so a build this "
                         "cheap cannot be detected at all");

  double console_lag = -1.0;
  double console_wall = -1.0;
  double console_clock = -1.0;
  {
    // Scoped, because the two `REQUIRE_MESSAGE` below throw on a failed
    // precondition and a bare `unsetenv` after them would not run.
    const EnvVar stderr_on("VLLM_RENDER_PHASE_LOG_STDERR", "1");
    for (int k = 0; k < kProbes; ++k) {
      bool wrote = false;
      std::string why;
      double before_write = 0.0;
      {
        const StderrTo capture(sink);
        // The clock is read INSIDE the redirect, because opening and truncating
        // the sink is itself milliseconds once the build has filled it, and this
        // arm's whole subject is a millisecond.
        before_write = log.Elapsed();
        wrote = log.WriteJson(file.path(), "unit", "cpu", &why);
      }
      REQUIRE_MESSAGE(wrote, why);
      const std::string block = ReadAll(sink);
      const double printed = WallFromBlock(block);
      REQUIRE_MESSAGE(printed >= 0.0,
                      "`VLLM_RENDER_PHASE_LOG_STDERR=1` produced no block carrying a unique "
                      "`WALL` row, so the comparison below has no number to make and this arm "
                      "would pass on a capture that caught nothing. Captured "
                          << block.size() << " bytes ending:\n"
                          << block.substr(block.size() > 400 ? block.size() - 400 : 0));
      const double lag = std::fabs(printed - before_write);
      if (console_lag < 0.0 || lag < console_lag) {
        console_lag = lag;
        console_wall = printed;
        console_clock = before_write;
      }
    }
  }
  MESSAGE("console WALL " << console_wall << "s against a clock read " << console_clock
                          << "s immediately before WriteJson, lag " << console_lag
                          << "s (min of " << kProbes << " probes, build " << build_cost
                          << "s, bound " << kStep << "s)");
  CHECK_MESSAGE(console_lag <= kStep,
                "`VLLM_RENDER_PHASE_LOG_STDERR` printed a WALL of " << console_wall
                    << "s and this case read the clock at " << console_clock
                    << "s immediately before calling `WriteJson`, a lag of " << console_lag
                    << "s against ONE step of that line's own `%10.3f`, with the writer's own "
                       "per-record JSON build measured at " << build_cost
                    << "s. The console copy is being rendered AFTER this writer has already "
                       "copied, sorted and serialised the table, so the residue a reader "
                       "watches on a terminal contains the writer that printed it");

  // ── (B) THE PUBLIC ENTRY POINT ──────────────────────────────────────────
  //
  // A SECOND TABLE RATHER THAN A BIGGER FIRST ONE, and the reason is memory.
  // `WriteJson` holds one `nlohmann` object per record while it dumps, about
  // 3.3 KB per record on this box, so a table big enough for this arm costs a
  // gigabyte through arm (A) and 140 MB through this one, which never
  // serialises to JSON at all.
  log.Reset();
  log.Begin();
  const int kBig = 250000;
  BuildLeaves(kBig, sink);

  double copy_cost = -1.0;
  double sort_cost = -1.0;
  for (int k = 0; k < kProbes; ++k) {
    // MEASURED THE WAY THE EMITTER DOES IT AND IN THE SAME TWO STEPS, through
    // the same public entry point. They are measured SEPARATELY and the budget
    // is the SMALLER, because a partial regression moves only one of them --
    // the shape a fresh review of #1569 actually produced against the
    // one-number form of its budget.
    const double before_copy = log.Elapsed();
    std::vector<phase::Record> copy = log.Records();
    const double after_copy = log.Elapsed();
    std::stable_sort(copy.begin(), copy.end(),
                     [](const phase::Record& a, const phase::Record& b) {
                       return a.start < b.start;
                     });
    const double after_sort = log.Elapsed();
    // Kept from being optimised away: the sorted copy has to be observed.
    REQUIRE(!copy.empty());
    const double this_copy = after_copy - before_copy;
    const double this_sort = after_sort - after_copy;
    if (copy_cost < 0.0 || this_copy < copy_cost) copy_cost = this_copy;
    if (sort_cost < 0.0 || this_sort < sort_cost) sort_cost = this_sort;
  }
  const double serialize = copy_cost < sort_cost ? copy_cost : sort_cost;
  MESSAGE("copy " << copy_cost << "s sort " << sort_cost << "s over " << kBig
                  << " records (min of " << kProbes << " probes, budget " << serialize
                  << "s = " << (serialize / kStep) << " steps of the printed format)");
  REQUIRE_MESSAGE(serialize > 2.0 * kStep,
                  "the CHEAPER of this emitter's two post-clock steps over " << kBig
                      << " records measured " << serialize << "s (copy " << copy_cost
                      << "s, sort " << sort_cost
                      << "s), which `RenderText`'s own `%10.3f` cannot separate from zero. A "
                         "wrong ordering is detected by whichever step it moves, so a step "
                         "this cheap cannot be detected at all -- which is exactly how this "
                         "defect survived #1569, and how #1569's own three-record case stayed "
                         "green under its own mutation");

  double render_lag = -1.0;
  double render_wall = -1.0;
  double render_clock = -1.0;
  for (int k = 0; k < kProbes; ++k) {
    double before_render = 0.0;
    std::string text;
    {
      const StderrTo quiet(sink);
      before_render = log.Elapsed();
      text = log.RenderText("unit", "cpu");
    }
    const double printed = WallFromBlock(text);
    REQUIRE_MESSAGE(printed >= 0.0,
                    "`RenderText` returned a block carrying no unique `WALL` row, so the "
                    "comparison below has no number to make -- and a block that carried "
                    "nothing would read exactly like a comparison that passed:\n"
                        << text.substr(0, 400));
    const double lag = std::fabs(printed - before_render);
    if (render_lag < 0.0 || lag < render_lag) {
      render_lag = lag;
      render_wall = printed;
      render_clock = before_render;
    }
  }
  MESSAGE("RenderText printed WALL " << render_wall << "s against a clock read " << render_clock
                                     << "s immediately before the call, lag " << render_lag
                                     << "s (min of " << kProbes << " probes, budget "
                                     << serialize << "s, bound " << kStep << "s)");
  CHECK_MESSAGE(render_lag <= kStep,
                "`RenderText` printed a WALL of " << render_wall
                    << "s and this case read the clock at " << render_clock
                    << "s immediately before calling it, a lag of " << render_lag
                    << "s against ONE step of that line's own `%10.3f`, with the cheaper of "
                       "its two post-clock steps measured at " << serialize
                    << "s. The emitter is reading its clock AFTER it copies and sorts the "
                       "table, so its own serialization is charged to the WALL it prints and "
                       "to the `unaccounted` above it. This table measures the render");
  log.Reset();
}
