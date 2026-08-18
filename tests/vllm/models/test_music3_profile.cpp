// Gate for the MiniMax-Music3 stage profiler (#672).
//
// WHAT IS AT RISK HERE. The instrument's whole value is that its numbers add
// up: `sum(leaf) + unattributed == TOTAL`, spans are printed but never summed,
// and pure counters carry no time at all. A profiler that double-counts does
// not fail — it reports a plausible split with the wrong shares, and a reader
// acts on it. So the accounting rules are asserted here rather than trusted,
// and the OFF-by-default contract is asserted too, because an instrument that
// quietly stayed on would put a `/proc` read and a clock into every synthesis.
//
// This file does NOT gate the placement of the brackets in
// `minimax_music3_speech.cpp` / `minimax_music3_llm.cpp`. Nothing can: where a
// bracket goes is a judgement, and the instrument's answer to a badly placed
// one is the `unattributed` line, which is exactly what the accounting cases
// below make trustworthy.
#include <cstdint>
#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "vllm/model_executor/models/music3_profile.h"  // via -I src

namespace profile = vllm::models::music3::profile;

namespace {

const profile::Bucket* Find(const char* name) {
  for (const profile::Bucket& bucket : profile::Buckets()) {
    if (bucket.name == name) return &bucket;
  }
  return nullptr;
}

// Every case that needs the instrument ON restores the flag, so a later case
// still sees the shipped default.
struct ArmedProfile {
  ArmedProfile() : previous_(profile::EnabledFlag()) { profile::EnabledFlag() = true; }
  ~ArmedProfile() { profile::EnabledFlag() = previous_; }
  bool previous_;
};

}  // namespace

TEST_CASE("music3 profile is OFF unless the environment asks for it") {
  // The shipped default, asserted rather than assumed: the suite runs with no
  // `VLLM_CPP_MUSIC3_PROFILE`, so this is the production polarity.
  CHECK_FALSE(profile::ParseEnabled(nullptr));
  CHECK_FALSE(profile::ParseEnabled(""));
  CHECK_FALSE(profile::ParseEnabled("0"));
  CHECK_FALSE(profile::ParseEnabled("off"));
  // A TYPO must not arm it. This is the case that matters: an operator who
  // writes `=y` and gets a silent no-op has a run with no numbers, which is
  // recoverable; one who gets a surprise arming has a run whose meaning changed.
  CHECK_FALSE(profile::ParseEnabled("y"));
  CHECK_FALSE(profile::ParseEnabled("enabled"));

  CHECK(profile::ParseEnabled("1"));
  CHECK(profile::ParseEnabled("true"));
  CHECK(profile::ParseEnabled("TRUE"));
  CHECK(profile::ParseEnabled("On"));
  CHECK(profile::ParseEnabled("yes"));
}

TEST_CASE("music3 profile records nothing while disabled") {
  const bool previous = profile::EnabledFlag();
  profile::EnabledFlag() = false;
  profile::Begin();
  {
    profile::Timer timer("disabled.leaf");
  }
  profile::Count("disabled.counter", 7);
  profile::Mark("disabled.mark");
  CHECK(profile::Buckets().empty());
  CHECK(profile::Markers().empty());
  profile::EnabledFlag() = previous;
}

TEST_CASE("music3 profile: leaves sum, spans do not, counters carry no time") {
  ArmedProfile armed;
  profile::Begin();

  // Two leaves, one of them entered three times: a bucket accumulates seconds
  // and COUNTS calls, which is how "540 DiT forwards" is reported rather than
  // implied.
  profile::Add("leaf.a", 1.0);
  profile::Add("leaf.b", 2.0);
  profile::Add("leaf.b", 3.0);
  profile::Add("leaf.b", 4.0);
  // A span ENCLOSING both leaves. If it were summed, the table would claim
  // 20 s of work in a 10 s run.
  profile::Add("span.parent", 10.0, /*span=*/true);
  profile::Count("counter.frames", 500);

  const profile::Bucket* a = Find("leaf.a");
  const profile::Bucket* b = Find("leaf.b");
  const profile::Bucket* span = Find("span.parent");
  const profile::Bucket* counter = Find("counter.frames");
  REQUIRE(a != nullptr);
  REQUIRE(b != nullptr);
  REQUIRE(span != nullptr);
  REQUIRE(counter != nullptr);

  CHECK(a->calls == 1);
  CHECK(a->seconds == doctest::Approx(1.0));
  CHECK_FALSE(a->span);

  CHECK(b->calls == 3);
  CHECK(b->seconds == doctest::Approx(9.0));

  CHECK(span->span);
  CHECK(span->seconds == doctest::Approx(10.0));

  // A counter is distinguishable from a zero-second leaf: its sentinel is
  // negative, so `Report` excludes it from the sum and prints "-" for seconds.
  CHECK(counter->calls == 500);
  CHECK(counter->seconds < 0.0);

  // The accounting identity the report prints, computed the same way it does.
  double attributed = 0.0;
  int64_t examined = 0;
  for (const profile::Bucket& bucket : profile::Buckets()) {
    ++examined;
    if (bucket.seconds >= 0.0 && !bucket.span) attributed += bucket.seconds;
  }
  // A gate that cannot say how many things it looked at has not reported.
  CHECK(examined == 4);
  CHECK(attributed == doctest::Approx(10.0));  // 1 + 9, NOT 20
}

TEST_CASE("music3 profile: a second Begin discards the previous run") {
  ArmedProfile armed;
  profile::Begin();
  profile::Add("leaf.a", 1.0);
  profile::Mark("first");
  REQUIRE(profile::Buckets().size() == 1);
  REQUIRE(profile::Markers().size() == 1);

  profile::Begin();
  CHECK(profile::Buckets().empty());
  CHECK(profile::Markers().empty());
}

TEST_CASE("music3 profile: markers carry a timeline position and an RSS reading") {
  ArmedProfile armed;
  profile::Begin();
  profile::Mark("enter");
  profile::Mark("later");
  REQUIRE(profile::Markers().size() == 2);
  CHECK(profile::Markers()[0].label == "enter");
  CHECK(profile::Markers()[1].label == "later");
  CHECK(profile::Markers()[0].at_seconds >= 0.0);
  CHECK(profile::Markers()[1].at_seconds >= profile::Markers()[0].at_seconds);
#if defined(__linux__)
  // The RSS question this instrument exists to answer (a 7.3 -> 14.7 GB climb
  // during one 20 s generation) needs a reading that is actually there. On
  // Linux `/proc/self/statm` always is, so a -1 here is a broken reader rather
  // than an unsupported platform — and a broken reader would present as a flat
  // RSS line that looks like "no growth".
  CHECK(profile::Markers()[0].rss_kb > 0);
#endif
}

TEST_CASE("music3 profile: Timer attributes to the bucket it names") {
  ArmedProfile armed;
  profile::Begin();
  {
    profile::Timer outer("span.outer", /*span=*/true);
    {
      profile::Timer inner("leaf.inner");
    }
    {
      profile::Timer inner("leaf.inner");
    }
  }
  const profile::Bucket* inner = Find("leaf.inner");
  const profile::Bucket* outer = Find("span.outer");
  REQUIRE(inner != nullptr);
  REQUIRE(outer != nullptr);
  CHECK(inner->calls == 2);
  CHECK_FALSE(inner->span);
  CHECK(outer->calls == 1);
  CHECK(outer->span);
  // The enclosing span is at least as long as what it encloses. This is the
  // nesting invariant the leaf/span split depends on.
  CHECK(outer->seconds >= inner->seconds);
}
