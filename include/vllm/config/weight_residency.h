// vllm.cpp ORIGINAL — the host-RAM -> DISK weight-residency tier as a config
// surface. Row `ENG-RESIDENCY-CONFIG`, spec
// .agents/specs/weight-residency-config.md, issue #1110.
//
// WHY THIS IS NOT IN offload.h. `include/vllm/config/offload.h` is a
// transcription of upstream `vllm/config/offload.py` @ 555967922, cited
// line-for-line. Upstream offloads weights device -> host RAM and stops there:
// `OffloadBackend` is `Literal["auto", "uva", "prefetch"]` (offload.py:12),
// `offloader/uva.py:21` is a CPU-blanket UVA offloader and
// `offloader/prefetch.py:557-560` is cpu-only. NOTHING upstream reads a weight
// off a file at inference time, so there is nothing to mirror for this tier and
// adding a field to `OffloadConfig` would break a mirror to describe behaviour
// vLLM does not have. This file is therefore vllm.cpp-original by construction,
// and it names itself so in its JSON: the knobs live under the `vllm_cpp` key of
// the SAME document `--offload-config` already carries, which keeps one
// user-facing flag for one user-facing concept while leaving the mirrored fields
// byte-identical. `parse_offload_config_json` looks its three keys up by name
// and never enumerates the document, so a `vllm_cpp` sibling is invisible to it —
// and, because the same tolerance would swallow a TYPO, this parser enumerates the
// whole document instead of only its own half (see the parser's comment below).
//
// THE SCHEMA:
//
//   {"vllm_cpp": {"mmap":          {"enabled": bool, "prefault": bool},
//                 "expert_stream": {"enabled": bool, "slots": int,
//                                   "slot_bytes": int},
//                 "device_fit":    {"weight_budget_bytes": int}}}
//
// Every field is optional and an absent field means UNCHANGED. That sentence
// binds TWO pieces of code, and the first two shapes of this row broke it in both
// directions (#1133 H1, H2, each measured through `LoadedEngine::FromModelDir`).
// At install time the stored config is MERGED field by field, so a document that
// omits a field leaves the installed value alone — it does not clear it. At refusal
// time a field is scored only when the document SETS it, so omitting a field is not
// a change to it. `nullopt` is the absence of a request, never a request for the
// default.
//
// An `--offload-config` with no `vllm_cpp` key therefore installs nothing and every
// knob resolves exactly as it did before this row existed. One thing about such a
// document DID change: the PARSE now refuses an unknown key at every level, so
// `--offload-config '{"typo":1}'` used to start a server and now aborts at startup.
// That refusal is the point of the row's H1 repair rather than an exception to the
// paragraph above, but it is a behaviour change, so what is byte-identical is the
// ENGINE a legal document produces and not the SET of documents accepted.
//
// PRECEDENCE, and it is deliberate: **environment variable > JSON config >
// built-in default**. The environment keeps winning because several of these
// variables exist so that a benchmark arm is switchable without restarting the
// server with a new config, and a measurement in flight depends on that. The
// config is the DOCUMENTED surface; the environment is the OVERRIDE. An override
// that could not turn a configured knob back OFF would not be one, so `VT_X=0`
// beats a config `true` as well.
//
// WHAT IS DELIBERATELY ABSENT: `VT_MOE_EXPERT_STREAM_STATS_EVERY`. Every knob
// here changes what memory the process reserves or where a weight lives, which
// is a deployment decision and what a config document is for. That one changes
// only how often a diagnostic line reaches stderr: it moves no byte, reserves
// nothing and changes no number. It is the instrument, not the configuration,
// and it is the thing an operator flips while watching a run. Recorded so a
// later reader sees a decision rather than an omission.
//
// THE LATCH, which is the one real hazard here — and it covers THREE of the six
// knobs, not all six. What genuinely freezes:
//
//   * `expert_stream`, because `ResolveExpertStreamRequested` below caches the
//     answer in a function-local static. (`Qwen35ExpertStreamRequested` is the
//     model-side name and, since this row, a pure delegation to it; the static is
//     not there.) It decides both whether an ~18 GiB slot store is built and
//     whether the default-on grouped-MoE path is disabled, and those two must not
//     be able to disagree later in one process.
//   * `expert_stream.slots` and `.slot_bytes`, from the moment the slot STORE is
//     built: the reservation is process-lifetime and cannot be resized. Reading the
//     two sizes freezes nothing; building the store does.
//
// What does NOT freeze: `mmap`, because `GgufLoadPolicy::FromEnv()` is called per
// load and always has been; `prefault`, because this row deliberately removed
// that site's function-local static (see `ResolveGgufPrefault` below); and
// `device_fit.weight_budget_bytes`, which one GGUF load reads once at its fit
// check through no static at all. A second engine in one process may therefore
// still set any of the three.
//
// So the refusal is scoped to what it can actually justify, and it took three
// shapes to get there. `SetWeightResidencyConfig` THROWS only when the incoming
// document SETS a decided field AND the value it sets would make that field's
// resolver return something OTHER than the decision already taken. Everything else
// installs: an empty document, an equal re-install, a document touching only
// `mmap`/`prefault`, a document that OMITS the decided field, and a document that
// asks for exactly what the process decided.
//
// The first shape marked one process-wide flag inside the shared resolvers, so an
// ordinary first load refused a second engine's whole document — a hard failure on
// a legal load, for a reason untrue of two of the five knobs (#1122 M1). The second
// shape narrowed that to the two real decisions but compared `in.<field>` against
// the STORED optional, and `nullopt != engaged` is true, so it still refused a
// second engine whose document merely omitted the latched field, and it refused a
// document asking for what the engine was already running while telling the operator
// the engine was not running it (#1133 H1). The comparison is now against the
// decision, resolved through the same rule the production resolver uses — which also
// makes it right about the environment, since a document `VT_MOE_EXPERT_STREAM`
// overrides changes nothing and is not refused. Install at
// `LoadedEngine::FromModelDir`, in the same block that installs the weight
// offloader, which is already before any weight I/O.
#ifndef VLLM_CONFIG_WEIGHT_RESIDENCY_H_
#define VLLM_CONFIG_WEIGHT_RESIDENCY_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace vllm {

// The `vllm_cpp` extension object of an `--offload-config` document. Each
// optional is "the operator said so"; an empty optional is "unchanged".
struct WeightResidencyConfig {
  // `mmap.enabled` -> VT_GGUF_MMAP. Keep the GGUF file mmap-resident and borrow
  // weight bytes out of the mapping instead of copying them into owned buffers.
  // This is what makes a checkpoint larger than host RAM loadable at all.
  std::optional<bool> mmap;

  // `mmap.prefault` -> VT_GGUF_PREFAULT. Fault the borrowed pages in at load,
  // off the timed prefill. Set it FALSE for a model larger than memory: the
  // prefault would read the whole tower to populate a page cache that cannot
  // hold it.
  std::optional<bool> prefault;

  // `expert_stream.enabled` -> VT_MOE_EXPERT_STREAM. Serve routed expert slices
  // from a bounded host slot cache instead of faulting them out of the mapping
  // in router order.
  std::optional<bool> expert_stream;

  // `expert_stream.slots` -> VT_MOE_EXPERT_STREAM_SLOTS. How many expert slices
  // stay resident. Must be positive.
  std::optional<int64_t> expert_stream_slots;

  // `expert_stream.slot_bytes` -> VT_MOE_EXPERT_STREAM_SLOT_BYTES. Bytes per
  // slot, fixed for the process's life. Must be positive. This IS a user
  // surface, however internal it looks: the engine refuses an oversized slice
  // BY NAME and tells the operator to raise exactly this value, and a dynamic
  // (UD) quant whose `down_proj` outweighs its gate/up pair is precisely the
  // case where the computed default is wrong.
  std::optional<int64_t> expert_stream_slot_bytes;

  // `device_fit.weight_budget_bytes` -> VT_DEVICE_WEIGHT_BUDGET_BYTES. The device
  // memory pool that a GGUF's staged weight bytes are compared against at LOAD
  // time (issue #1123), overriding the platform's own probe. Set it LOWER when
  // something else lives in the pool.
  //
  // ZERO IS LEGAL HERE AND NOWHERE ELSE IN THIS STRUCT, and it is not a
  // degenerate size. `CheckDeviceWeightFit` reads a zero budget as UNKNOWN and
  // decides nothing, so `0` is the documented spelling of "suppress the refusal
  // and get the late failure back" — exactly what
  // `VT_DEVICE_WEIGHT_BUDGET_BYTES=0` already means. `slots` and `slot_bytes` are
  // sizes and refuse a zero, because a slot count that silently became 64 is a
  // cache the operator does not have. A NEGATIVE budget is still refused.
  std::optional<int64_t> device_weight_budget_bytes;

  // True when the operator set nothing, i.e. the byte-identical default path.
  bool empty() const;

  // One line naming every field the operator set, for the install log. Returns
  // an empty string for an empty config.
  std::string Describe() const;

  // The environment variables that WOULD WIN over a field this config sets, as
  // `VT_NAME (field)` pairs; empty when nothing is shadowed.
  //
  // This is the whole mitigation for the env-wins precedence, and it is the one
  // way that precedence hurts: a document silently shadowed by a variable
  // somebody exported weeks ago. "Would win" and not "is set": for a boolean any
  // value wins, so presence is exact, but a COUNT variable that is empty, garbage
  // or non-positive falls THROUGH to the config under the tolerant parse the
  // existing readers use, so `VT_MOE_EXPERT_STREAM_SLOTS=banana` overrides nothing
  // and must not be announced as though it did. It calls no `Resolve*` and marks no
  // latch, which keeps the line ahead of the weight load where it belongs.
  std::string DescribeEnvOverrides() const;

  // Field-by-field equality, INCLUDING which fields are set: two documents that
  // resolve the same way are not equal unless they say the same thing. Used by tests
  // to compare an installed document against the one they built. The install does
  // NOT use it — "equal to the stored document" was the wrong predicate for the
  // refusal, because it makes an omitted field a change (#1133 H1); the install
  // compares each SET field against the decision that was taken.
  bool operator==(const WeightResidencyConfig& other) const;
  bool operator!=(const WeightResidencyConfig& other) const {
    return !(*this == other);
  }
};

// Parse the `vllm_cpp` extension out of the same JSON document
// `--offload-config` carries. An empty/blank document or an absent `vllm_cpp`
// key yields an empty (inert) config.
//
// Throws std::invalid_argument on a malformed document, a non-object document, a
// `vllm_cpp` that is not an object, an UNKNOWN key at ANY level of the document, a
// field of the wrong type, a non-positive `slots` / `slot_bytes`, or a NEGATIVE
// `device_fit.weight_budget_bytes`. "Any level" means the top level, the inside of
// `vllm_cpp`, the inside of `vllm_cpp.mmap`, `vllm_cpp.expert_stream` and
// `vllm_cpp.device_fit`, and the inside of the two MIRRORED sub-objects `uva` and
// `prefetch`. A `weight_budget_bytes` of ZERO is accepted, and it is the only zero
// this parser accepts in a field it OWNS: it is the documented spelling of
// "suppress the device-fit refusal", which `CheckDeviceWeightFit` implements by
// reading a zero budget as UNKNOWN. (Values inside `uva` and `prefetch` are not
// this parser's to accept or refuse; it reads their NAMES only.) The last of those was missing while this comment already claimed it,
// so `{"uva":{"cpu_offload_GB":10}}` started a server with a 0 GiB budget the
// operator believed was set (#1133 H3). TYPE checking inside `uva`/`prefetch` stays
// with the mirrored parser that owns those fields; this one only enumerates names.
//
// The unknown-key refusal is the load-bearing half, and it enumerates the WHOLE
// document, not only the inside of `vllm_cpp`. `parse_offload_config_json` ignores
// a key it does not know, which is what lets this extension exist at all — and it
// is also what made `{"vllm-cpp":{...}}` and `{"vllm_cpp":{"mmapp":{...}}}` parse
// to an empty config and start a server running this tier at its DEFAULTS — mmap
// residency riding the caller's availability predicate, prefault ON and streaming
// OFF — and the last two are exactly what the 370 GiB case exists to change, so the
// operator meets the typo as an out-of-memory kill rather than as an error. The hyphen is the likeliest spelling of
// all, because every flag around it is hyphenated. A typo that quietly disables the
// tier keeping a 370 GiB model in 119 GB is worse than a startup error, so it is an
// error.
//
// `{"vllm_cpp":{"device-fit":...}}` and `{"vllm_cpp":{"device_fit":{"weight_budget_byte":1}}}`
// are refused on the same terms, and for the same consequence one level down: a
// budget the operator believes is set, with the platform probe silently in force.
//
// The four legal top-level keys are `offload_backend`, `uva`, `prefetch` and
// `vllm_cpp`: the three the mirrored parser reads by name, plus this extension.
// Refusing the rest is the MIRROR-FAITHFUL polarity rather than a local invention.
// Upstream has no `--offload-config` flag at the pin — the whole JSON document is
// vllm.cpp's own, so no upstream-legal document exists to break — and vLLM builds
// its config dataclasses with `@config`, whose body sets `ConfigDict(extra="forbid")`
// (vllm/config/utils.py:68-69 @ 555967922). `OffloadConfig` (offload.py:80) and
// `KVTransferConfig` (kv_transfer.py:22-23) both carry it, so `--kv-transfer-config`,
// the JSON-document flag next door, refuses an unknown key. This parser closes the document while `parse_offload_config_json` stays a
// byte-faithful transcription; both read the same string at both entry points.
WeightResidencyConfig parse_weight_residency_extension_json(
    const std::string& json_text);

// Install the process-global config, MERGING it field by field over what is already
// installed. Call BEFORE any weight I/O.
//
// The merge is the schema's "absent means unchanged" applied to the install. An
// earlier shape assigned wholesale, so a second engine's partial document silently
// dropped the first engine's fields — `expert_stream=on` with 8000 slots became OFF
// with 64 — and the slot store reads those values lazily, on the first slice taken,
// which can be long after the second engine loaded (#1133 H2). An empty document is
// therefore a no-op by construction rather than by a special case.
//
// Throws std::logic_error only when the config SETS a field a taken decision has
// already fixed (see THE LATCH above) AND the value it sets would resolve differently
// from that decision: `expert_stream` once the streaming answer has been read, and
// the two sizes once the slot store has been built. Honouring such a call would
// record a configuration the engine is not running, and the message names the
// decision the engine took rather than the document it holds. Everything else is
// accepted — an empty config, an equal re-install, a document that touches only
// `mmap` or `prefault`, a document that OMITS the decided field, and a document
// asking for exactly what was decided.
void SetWeightResidencyConfig(const WeightResidencyConfig& config);

// The installed config, BY VALUE. Empty until something installs one. A reference
// would be read after the lock was released, which is an unsynchronised read behind
// a lock that looks like it covers one; the copy is six optionals.
WeightResidencyConfig ActiveWeightResidencyConfig();

// The decisions that genuinely freeze, one enumerator each. There is no `kMmap` or
// `kPrefault`, and their absence is the contract: those two resolve per load.
enum class ResidencyLatch {
  // `ResolveExpertStreamRequested`'s own function-local static. The flag behind it
  // also carries WHICH answer was cached, because the refusal has to compare an
  // incoming document against the decision rather than against the stored document.
  kExpertStream,
  // The slot store's `slots x slot_bytes` reservation, via
  // `NoteExpertStreamGeometry` when the store is built.
  kExpertStreamGeometry,
};

// True once the named decision has been taken in this process.
bool WeightResidencyLatched(ResidencyLatch knob);

// True once ANY of them has.
bool WeightResidencyLatched();

// Drop the installed config, both latches, and the recorded slot geometry. Tests
// only: a latch is process-wide, so a suite with more than one case needs to be able
// to clear it. It cannot undo the function-local static inside
// `ResolveExpertStreamRequested`, which is why the value that resolver returns is
// observable exactly once per process and has a test binary of its own — a case that
// resets and calls it again gets the SAME cached answer with the latch re-marked,
// which is a state a production process can also be in (streaming decided by the
// environment, nothing installed).
void ResetWeightResidencyConfigForTesting();

// env var (if set) > `configured` (if set) > `builtin_default`.
//
// The env var's value is read with the tree's existing polarity: "", "0",
// "false" and "off" are FALSE, anything else is TRUE. Marks NO latch: a latch
// belongs to the site that caches an answer, not to reading a variable.
bool ResolveResidencyBool(const char* env_name, std::optional<bool> configured,
                          bool builtin_default);

// The integer form, same precedence, and the same absence of a latch. A
// non-positive or unparseable ENV value is ignored and falls through to the config,
// then to `builtin_default` — the tolerant parsing the existing knobs already
// document, kept byte-for-byte so this row changes no behaviour for an
// environment-only run. Such a value is therefore NOT an override, which is why
// `DescribeEnvOverrides` asks the same predicate rather than reporting presence. A
// non-positive CONFIG value cannot reach here: the parser refuses it at startup,
// where the operator can still see the message.
int64_t ResolveResidencyCount(const char* env_name,
                              std::optional<int64_t> configured,
                              int64_t builtin_default);

// ── The six knobs, one named resolver each ────────────────────────────────────
//
// Each one owns its environment NAME and its exact historical POLARITY, and each
// is the SOLE reader of its variable after this row. Two reasons this is six
// functions and not one call at each site with a string literal.
//
// First, the polarities are NOT the same and one of them is deliberately odd.
// `VT_GGUF_MMAP` and `VT_GGUF_PREFAULT` compare the whole value against "", "0",
// "false" and "off" (the tree's `EnvOn`,
// gguf_keep_quant.cpp:61-66). `VT_MOE_EXPERT_STREAM` examines only the FIRST
// CHARACTER — `v[0] != '0' && v[0] != '\0'` — so `VT_MOE_EXPERT_STREAM=false`
// reads as ON, which docs/ENVIRONMENT.md states explicitly. That is documented
// behaviour, not an accident, so it is preserved rather than tidied: a row whose
// subject is "where does this value come from" must not also change what an
// existing value means.
//
// Second, an environment-only run has to resolve BYTE-FOR-BYTE as it did before,
// and the cheapest way to hold that is for each variable to keep having exactly
// one reader with its own transcribed rule.

// `VT_GGUF_MMAP` > `vllm_cpp.mmap.enabled` > `builtin_default`. Not latching:
// `GgufLoadPolicy::FromEnv()` is called per load and always has been.
bool ResolveGgufMmap(bool builtin_default);

// How many borrowed spans have actually been prefaulted in this process, and the
// reset that lets a test A/B it.
//
// THIS EXISTS BECAUSE THE DECISION WAS OTHERWISE UNOBSERVABLE. Measured: with the
// prefault site mutated to never consult its resolver at all, the whole GGUF suite
// stayed green — `test_gguf_keep_quant` 39/39, `test_gguf_qwen36_loader` 6/6,
// `test_gguf_expert_span` 11/11 — because the only prefault case asserts BYTE
// TRANSPARENCY, which holds whether the prefault runs or not. So a config key
// wired to a site nothing watches would have landed looking tested. The counter is
// the cheapest instrument that distinguishes "prefaulted" from "skipped": the
// prefault reads pages and changes no byte, so there is nothing else to see.
uint64_t GgufPrefaultedSpanCount();
void ResetGgufPrefaultedSpanCountForTesting();

// The slot geometry the expert-stream store was actually BUILT with; both zero
// until something builds one. Same reason as the counter above, and the same
// measurement behind it: with the slot-count site mutated to a hardcoded 64, so
// that `VT_MOE_EXPERT_STREAM_SLOTS=8000` would have silently stopped working,
// `test_expert_stream_mixed_slot` and `test_gguf_expert_span` both stayed green.
// No test in this tree had ever exercised a non-default slot count — the gap
// predates this row, since the old inline `getenv` was equally unobserved — and a
// row that turns the knob into a config key is the wrong place to leave it.
struct ExpertStreamGeometry {
  int64_t slots = 0;
  int64_t slot_bytes = 0;
};
ExpertStreamGeometry BuiltExpertStreamGeometry();

// Called by the store's constructor with the values it resolved. Nothing but the
// accessor above reads it.
void NoteExpertStreamGeometry(int64_t slots, int64_t slot_bytes);

// `VT_GGUF_PREFAULT` > `vllm_cpp.mmap.prefault` > ON.
//
// NOT LATCHING, and that is a change: the site used to cache the answer in a
// function-local static. Dropping the cache costs one `getenv` per prefaulted
// span — set against reading megabytes of pages, which is what the function then
// does — and it buys two things. A config installed at load is honoured even if
// some earlier caller had already asked, and the existing A/B case
// (tests/vllm/test_gguf_keep_quant.cpp, "L7 load-time prefault is
// byte-transparent") stops being silently vacuous: under the static its second
// `setenv` could not affect anything, so both of its arms ran the same way.
bool ResolveGgufPrefault();

// Called by the prefault site after it has actually faulted a span in. Feeds
// GgufPrefaultedSpanCount above; nothing else reads it.
void NoteGgufPrefaultedSpan();

// `VT_MOE_EXPERT_STREAM` > `vllm_cpp.expert_stream.enabled` > OFF, with the
// FIRST-CHARACTER environment rule described above.
//
// LATCHING, deliberately: the answer decides whether an ~18 GiB slot store is
// built and whether the default-on grouped-MoE path is disabled, and those two
// must not be able to disagree with each other later in the same process. That
// latch is exactly why `SetWeightResidencyConfig` refuses a late CHANGE to this
// field — not a late install, which it accepts, and not a late document that omits
// the field or agrees with what was decided.
bool ResolveExpertStreamRequested();

// The pure decision behind `ResolveExpertStreamRequested`, with the environment
// value passed in (`nullptr` == unset). Exposed because the wrapper LATCHES and
// can therefore be exercised exactly once per process, which would leave the
// first-character rule and the env-beats-config direction untested — the two
// things most likely to be got wrong here. Takes no lock and touches no global.
bool ExpertStreamRequestedFrom(const char* env_value,
                               std::optional<bool> configured);

// `VT_MOE_EXPERT_STREAM_SLOTS` > `vllm_cpp.expert_stream.slots` > 64.
int64_t ResolveExpertStreamSlots();

// `VT_MOE_EXPERT_STREAM_SLOT_BYTES` > `vllm_cpp.expert_stream.slot_bytes` >
// `computed_default` (the largest gate/up/down slice the caller is about to
// take).
int64_t ResolveExpertStreamSlotBytes(int64_t computed_default);

// `VT_DEVICE_WEIGHT_BUDGET_BYTES` > `vllm_cpp.device_fit.weight_budget_bytes` >
// `probed_total_bytes` (the platform's own `device_memory_total_bytes`).
//
// THE SIXTH KNOB, and the only one of the three integers that does NOT go through
// `ResolveResidencyCount`. That helper ignores a non-positive ENVIRONMENT value,
// which is right for a slot count and wrong here: `0` is this knob's suppression
// spelling and has to survive as a budget of zero, which `CheckDeviceWeightFit`
// then reads as UNKNOWN. (Neither helper filters the CONFIGURED value, because
// the parser already refused what each field considers out of range.) So this function transcribes the environment grammar
// `DeviceWeightBudgetBytes` has used since #1123 — one or more decimal digits and
// nothing else, no sign and no space — and inserts the config UNDER it. A value
// outside that grammar is ignored and falls through to the config, where it used
// to fall through to the probe; a run with no config therefore resolves
// byte-for-byte as before. `strtoull` alone is not enough for the grammar: it
// skips leading whitespace and it ACCEPTS a leading '-' and wraps it to
// ULLONG_MAX, so "-1" would parse as an effectively infinite budget.
//
// NOT LATCHING. It is read once per GGUF load, at the fit check in
// `LoadedEngine::FromModelDir`, through no static, so `ResidencyLatch` gains no
// enumerator and a second engine may still set it.
size_t ResolveDeviceWeightBudgetBytes(size_t probed_total_bytes);

}  // namespace vllm

#endif  // VLLM_CONFIG_WEIGHT_RESIDENCY_H_
