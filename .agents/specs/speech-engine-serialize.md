# SERVE-SPEECH-ENGINE-SERIALIZE — the speech seam states that implementations serialize, and neither implementation does

Issue: [#2836](https://github.com/mudler/vllm.cpp/issues/2836).
Base: `7b8b480b1`.
Filed by [#2712](https://github.com/mudler/vllm.cpp/issues/2712) and carried
under `## Owed` in [`rocm-scratch-sync.md`](rocm-scratch-sync.md), whose
`## First deliverable` names this route as "the one shape that would make it
live".

## Scope

The `vllm::multimodal::SpeechEngine` seam and the two families that implement it.

IN SCOPE: the serialisation contract the base class already states, the two
implementations that do not honour it, and the tests that hold both.

OUT OF SCOPE:

- The `vt::Queue` sharing itself. A lock over `Synthesize` is what makes the
  queue single-threaded again; giving each request its own queue is a different
  change with a different cost (Music3 stages ~28.5 GB per request), and it
  would not fix the other shared state named below.
- `POST /v1/videos` and `VideoEngine`. It spawns a job thread
  (`api_server.cpp:670`) and has the same question to answer, but it is a
  different seam with a different lifecycle. Listed under `## Owed`.
- The `##  Owed` items of `rocm-scratch-sync.md` that are not this one.

## First deliverable: is this live, or latent?

**LIVE, on the default server configuration.** #2712 measured its own site as
latent and named this one as the exception it could not rule out. This row rules
it in, and the chain is read at the base commit rather than assumed:

1. `POST /v1/audio/speech` is registered whenever a synthesizer is attached
   (`src/vllm/entrypoints/openai/api_server.cpp:1246-1255`), which
   `server_main.cpp` does for `--speech-model` (`:1717`) and for the
   speech-only server (`:1014`).
2. The handler calls `synthesizer_(request)` **inline on the cpp-httplib worker
   thread** (`api_server.cpp:636`). It is a `const` method and takes no lock.
   Neither `legacy_engine_mutex` nor `embed_mutex` — the two request-level locks
   this file already has, and the precedent that the shape is understood here —
   is on this path.
3. The pool is **12 threads on a stock server**, not one:
   `HttpWorkerCount` returns `max_concurrent_streams + kControlWorkerHeadroom`
   (`api_server.cpp:39-48`), and the defaults are 8 and 4
   (`api_server.h:70-71`). In the `kDefault` mode the pool is cpp-httplib's own
   `hardware_concurrency() - 1`, which is also greater than one on every box
   this project runs on.
4. Both shipped families contain **zero** `mutex`, `lock_guard`, `unique_lock`
   and `scoped_lock` occurrences: `minimax_music3_speech.cpp` and
   `indextts2.cpp`.
5. **The C ABI is the one caller that got it right, and its doc comment is the
   third written contract.** `include/vllm.h:1370` promises "Serialized per
   engine handle", and `src/capi/vllm_c.cpp:1915` delivers it: the handle owns a
   `std::mutex` (`:1790`) and `vllm_synthesize` holds it across the call. So the
   guarantee exists in a WRAPPER around the seam, for one of the seam's callers.
   The HTTP route has no such wrapper, and there is nothing to tell the next
   entry point that it needs one.

**The contract is written down, and it is written down in the wrong place.**
`include/vllm/multimodal/speech_engine.h:157-158` says of `Synthesize`:
"Implementations serialize internally (staged weights are shared state)". That
sentence is the specification. It is a comment on a pure virtual, so nothing
holds either implementation to it, and neither implementation does it.

**What the overlap actually corrupts** is more than the queue:

- `Music3SpeechEngine::Synthesize` brackets the run with `profile::Begin()`,
  `profile::Mark` and `profile::Count`. `music3_profile.h:25-29` states
  "SINGLE-THREADED BY CONTRACT ... a caller that brackets from a pool worker
  would corrupt the table. There is no such caller and there must not be one."
  `handle_audio_speech` **is** such a caller. That sentence is false at the base
  commit, and this row makes it true rather than editing it.
- One `queue_`, created once (`minimax_music3_speech.cpp:553-560`), so two
  threads submit to one stream. Every per-stream pool in the tree — including
  `vt::GrowOnlyStreamScratch`, which #2712 has just landed — keys on the stream
  handle, so two threads that share a queue share the entry, which is exactly the
  case that repair's own bookkeeping is designed for and not the case its
  reachability argument covered.

The measurement is not this prose. `## Tests` fires four concurrent HTTP POSTs
at a real bound server and observes the overlap.

## Design

**Enforce the contract at the seam, not in each family.** `Synthesize` becomes
non-virtual on `SpeechEngine`: it takes the engine's own mutex and delegates to a
new pure-virtual `SynthesizeLocked`. This is the `## Shared seams` answer rather
than a parallel path — every family present and future is serialised by
construction, and a family cannot forget, because there is no longer a virtual
for it to override without the lock.

**Why the seam and not the route.** A lock in `handle_audio_speech` would close
the one door that is open today and teach nothing. The C ABI shows why that is
not enough: it already holds a per-handle mutex around the same call
(`vllm_c.cpp:1790,1915`), so the tree has TWO callers with opposite dispositions
and no seam saying which is right. Moving the lock into `Synthesize` makes the
guarantee the seam's rather than each caller's, and a future entry point cannot
be added without it because there is no longer a virtual to override.

The C ABI's own mutex becomes redundant. It is LEFT IN PLACE: it is nested
strictly inside the new one, always in that order, so it cannot deadlock, and
removing it is a change to a second file that this row's gate does not cover.
`## Owed` names it.

The lock is a plain `std::mutex` held for the whole synthesis. A Music3 run is
minutes long, so the second request waits minutes. That is the correct
behaviour and not a regression: it is what the caller already experienced,
minus the corruption, and it is what vLLM's own serving does, where every
request enters one engine core loop and one forward runs at a time. Refusing a
concurrent request with 429 or 503 was rejected: it changes the HTTP contract,
and no upstream behaviour asks for it.

`std::mutex` is neither copyable nor movable, so the four `protected` defaulted
copy and move members of `SpeechEngine` become defined-as-deleted. Nothing in
the tree copies an engine — all three subclasses are held through
`unique_ptr`/`shared_ptr` — and a future attempt is now a compile error at the
seam instead of a second engine sharing one lock's worth of nothing.

## Tests

Two tests, and they answer different questions.

`tests/vllm/multimodal/test_speech_engine.cpp` — **the contract gate.** A probe
engine records concurrent entries to `SynthesizeLocked` with an atomic depth and
a deliberately plain `int`. N threads call `Synthesize` on one engine. GREEN is
`max_depth == 1` and, under `VLLM_CPP_SANITIZE=thread`, zero race reports. The
plain `int` is the detector: it stands in for `profile::`'s unguarded table and
is the reason a sanitized arm sees this class at all.

`tests/vllm/entrypoints/openai/test_api_server.cpp` — **the reachability gate.**
A real `ApiServer`, bound to a port, served on a background thread, with a
synthesizer wired through the production mapping
`vllm::openai::SynthesizeSpeechRequest` onto one probe engine. Four concurrent
`httplib::Client` POSTs to `/v1/audio/speech`. This enters through a production
entry point — the registered route on a default configuration — and not through
a hand-constructed object.

**A green non-sanitized run is not coverage for this class.** #2712 measured
that directly: its plain concurrent test passed every race assertion in all 24
rounds and failed only a retire count. The overlap assertion here is
deterministic because the probe sleeps inside the critical section, so the
window is reached rather than hoped for; the sanitized arm is what holds the
part an overlap counter cannot see.

## Gates

- Focused: `ctest -R 'test_speech_engine|test_api_server'`, red before, green
  after, both counts recorded.
- Sanitized: `test_speech_engine` under `-DVLLM_CPP_SANITIZE=thread`, red
  before with race reports at the probe's plain counter, green after with zero.
  `test_api_server` is not built in that lane: it links the engine, the
  scheduler and the runner, and the contract it holds is the reachability one,
  which is not a sanitizer question.
- Reachability: deleting the lock in a scratch copy must red the focused gate on
  both files.

## Stop conditions

- Stop if `SynthesizeLocked` can re-enter `Synthesize` on the same engine, which
  would self-deadlock on a non-recursive mutex. It cannot: neither family calls
  back into the seam, and the parallelism inside a run is
  `host_parallel::ForOutputRows`, which is below the bracket and joins before it
  returns.

## Owed

- `POST /v1/videos` spawns a job thread (`api_server.cpp:670`) against a
  `VideoEngine` whose seam states no serialisation contract at all. The same
  question has not been answered for it, and it is not answered here.
- `music3_profile.h`'s single-threaded contract is now true because one caller
  is serialised, not because the table is safe. A second bracketing caller would
  break it again, and nothing enforces that.
- `vllm_speech_engine::mutex` (`src/capi/vllm_c.cpp:1790,1915`) is now redundant
  with the seam's lock. Harmless and strictly nested, so it is left rather than
  removed in a change whose gate does not reach that file.
