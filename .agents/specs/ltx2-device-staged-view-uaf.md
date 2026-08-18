# The LTX-2 device test read a weight view out of a temporary that had already freed it

Row: `FIX-LTX2-DEVICE-UAF-904`
Issue: [#904](https://github.com/mudler/vllm.cpp/issues/904)
Owning row: `ROAD-V1-LTX25`
Baseline: `origin/main` @ `5a0ffe9e3`

## 1. Scope

Make `sanitize-cpu (address,undefined)` and `sanitize-cpu (thread)` green on
`main` by giving the second staged `Ltx2DitDeviceWeights` in one LTX-2 device
test case a name, so the `vt::Tensor` view copied out of it does not outlive the
storage that view points into.

**Out of scope, deliberately:** any change to `vt::cpu::Threadpool`, to
`ParallelForRows`, or to the ownership contract between a queue and the buffers
an in-flight op reads. §2 explains why: the measured evidence does not implicate
any of them, and #904's title does. Also out of scope: a checker or a type
change that would refuse a borrowed `vt::Tensor` outliving its owner. That is a
real footgun and it is recorded under `## Owed` rather than done here.

## 2. What the defect is, and the correction #904 owes

`.agents/issue-index.md` and the title of #904 both state the cause as:

> `~Ltx2DitDeviceWeights` frees the staged DiT buffers on the main thread
> (`ltx2_device.cpp:1088`) while a `vt::cpu` threadpool worker is still reading
> one inside `AddKernel` (`cpu_layernorm.cpp:33`), so the staged weights'
> lifetime is not joined to the in-flight parallel op that reads them.

**That is the wrong cause.** It would send the next reader to redesign where a
join belongs — the op dispatch, the queue, or the destructor — and none of those
is implicated. The defect is a dangling view into a destroyed temporary, in the
test, on one line.

`tests/vllm/models/test_ltx2_device.cpp`, the case *"ltx2 device: an f32
keyframes embedding under a bf16 stream is REFUSED"*, at `5a0ffe9e3`:

```cpp
723  staged.weights.keyframes_abs_pos_embedding =
724      Ltx2StageDitWeightsToDevice(q, p, set.views, vt::DType::kBF16)
725          .weights.keyframes_abs_pos_embedding;
726  CHECK(RefusalMessage([&] {
727          (void)Ltx2DitForwardDevice(q, p, staged.weights, &m.video, &m.audio, vt::DType::kBF16);
728        }).empty());
```

`Ltx2StageDitWeightsToDevice` returns an `Ltx2DitDeviceWeights` that owns each
staged buffer through a `shared_ptr<void>` whose deleter calls `vt::cpu::Free`
(`src/vllm/model_executor/models/ltx2_device.cpp:1085-1088`). A `vt::Tensor` is
a **borrowed view** — `view.data` is a raw pointer into that storage. Line 723
copies the view out; the temporary the view borrows from is destroyed at the end
of the full-expression, which is line 725; every staged buffer is freed there.
The forward at line 727 then reads through the dangling pointer.

### The evidence that distinguishes the two causes

ASan attributes both the allocation and the free to **line 724**, and the read
to the call at 726-727:

```
freed by thread T0 here:
    #1  FreeAligned64                        src/vt/cpu/cpu_backend.cpp:28
    #3  operator()                           src/vllm/model_executor/models/ltx2_device.cpp:1088
   #14  vllm::Ltx2DitDeviceWeights::~Ltx2DitDeviceWeights()
   #15  DOCTEST_ANON_FUNC_76                 tests/vllm/models/test_ltx2_device.cpp:724

previously allocated by thread T0 here:
    #1  AllocAligned64                       src/vt/cpu/cpu_backend.cpp:20
    #3  vllm::Ltx2StageDitWeightsToDevice    src/vllm/model_executor/models/ltx2_device.cpp:1085
    #4  DOCTEST_ANON_FUNC_76                 tests/vllm/models/test_ltx2_device.cpp:724
```

A teardown-races-an-in-flight-op defect would attribute the free to the case's
closing brace at line 729, where the *named* `staged` dies. It attributes to
724, before the read at 726. The named `staged` is never the freed object.

### There was never a race, and `ParallelForRows` cannot produce one

#904 reads the worker frame in the stack as a race — "a race in principle and
not in practice on this box". It is neither, and the dispatch path says so at
`src/vt/cpu/cpu_threadpool.cpp`. `ParallelForRows` (`:413`) hands the body to
`tp.Run(...)` (`:427`). `Threadpool::Run` (`:354`) runs it on the **caller**, as
worker 0, through `ComputeThread(workers_[0])` (`:383`). Every participating
thread's `ComputeThread` (`:220`) ends in `Barrier()` (`:234`), and that
barrier's exit is a full seq-cst fence (`:208-212`). So `Run` returns only after
every worker has passed the barrier — its own comment at `:352-353` calls the
return "the completion point for every output element". Nothing dispatched can
still be reading once `Run` has returned, so no teardown sequenced after it can
race one. The join #904 asks for is already there.

The thread id in an ASan report is incidental, not the signal, and reading it as
one is how #904 reached the threadpool. #904 recorded the read on worker
`T1`/`T2`; CI run `31885935312` at `04be1390b` recorded the same defect with the
whole chain inline on the main thread (`Threadpool::Run` → `ParallelForRows` →
`AddKernel` in one stack). Same address, same `LoadF32At` frame, same free site,
because the thread was never the variable.

The decisive check is that the **thread** sanitizer was asked and answered no.
The `sanitize-cpu (thread)` arm of that run reports
`ThreadSanitizer: heap-use-after-free`, the read `by main thread`, and
`ThreadSanitizer: reported 2 warnings` — of which **zero** are a data race.
Control for that absence, because a null search otherwise only proves the search
term wrong: the same log yields 5 matches for `ThreadSanitizer`, so the term was
searched in a log that does carry TSan output. The tool whose entire job is
finding data races found none in the run that caught this defect.

### The call-site count

`Ltx2StageDitWeightsToDevice` has **12** other call sites, all in this same test
file. Neither 14 nor 13 is that number, and both are easy to land on:
`tests/vllm/models/test_ltx2_device.cpp` holds 14 occurrences of the identifier,
of which one is the `using` declaration at `:64` and so is not a call, and one
of the 13 remaining call expressions is the offending site itself.

Method, because a count that lands in a commit message cannot be corrected
afterwards: separate occurrences (`grep -o` on the identifier, 14) from call
syntax (`grep -o` on the identifier followed by optional whitespace and `(`,
13), then print the difference rather than assume it — it is the `using` line.
Each of the 13 was then read in context and sits in initializer position, so
none is a redeclaration. The pair is 14/13 at `5a0ffe9e3` and 14/13 again at
this row's head. Control, so that a miscount cannot hide: appending one call
expression and one comment mention to a scratch copy moves the pair to 16/14 —
occurrences up by 2, calls up by 1, which is the only way both classes can be
distinguished — and a deliberately misspelled needle returns 0 rather than a
plausible number.

Outside this test file the identifier appears only as its declaration
(`include/vllm/model_executor/models/ltx2_device.h:107`), its definition
(`src/vllm/model_executor/models/ltx2_device.cpp:1056`), and three prose
mentions. No call expression exists anywhere else, which is what makes "all in
this same test file" true rather than merely unchallenged.

Line 724 at `5a0ffe9e3` is the only one of the 13 that does not bind its result
to a named object, which is also why exactly one case aborts.

## 3. The change

Bind the second staging to a named `const` local that outlives the forward:

```cpp
const Ltx2DitDeviceWeights restaged =
    Ltx2StageDitWeightsToDevice(q, p, set.views, vt::DType::kBF16);
staged.weights.keyframes_abs_pos_embedding =
    restaged.weights.keyframes_abs_pos_embedding;
```

This keeps the case's intent exactly as written: the positive control still runs
a **fresh** staging rather than the object the case already had, so it cannot be
satisfied by a view that was never replaced. Saving the original bf16 view
before line 714 overwrites it would be one line shorter and would weaken the
control that way, which is why it was rejected — see §6.

## 4. Tests and evidence

The test *is* the change, so there is no new case to add; the red is the
existing case under the existing lane. Both runs use the `sanitize-cpu
(address,undefined)` job's own configuration, on this box, with `setarch -R`
(without it the binary SIGSEGVs with no output on this host, which reads exactly
like a crash in the code under test).

```sh
cmake -S . -B build-sanitize -DVLLM_CPP_BUILD_TESTS=ON -DVLLM_CPP_CUDA=OFF \
      -DVLLM_CPP_SANITIZE='address,undefined'
cmake --build build-sanitize --target test_ltx2_device
UBSAN_OPTIONS=print_stacktrace=1 ASAN_OPTIONS=detect_leaks=1:strict_string_checks=1 \
VT_POOL_BYPASS=1 setarch -R ./build-sanitize/tests/test_ltx2_device
```

| | result |
|---|---|
| RED, at `5a0ffe9e3` unmodified | `rc=1`, `AddressSanitizer: heap-use-after-free` at `cpu_layernorm.cpp:33 in LoadF32At`, address `0x50a000038c40` — the same address CI run `31885935312` reports |
| GREEN, with §3 applied | `rc=0`, `18 passed / 0 failed`, `assertions: 546 passed / 0 failed`, `Status: SUCCESS!` |

### The case is not vacuous after the fix

A test that stops aborting can stop asserting. Mutated the guarantee the case
exists for — `ltx2_device.cpp:832`, `keyframes_embedding->dtype == out.x->t()
.dtype` inverted to `!=`, so the production refusal fires on the *matching* bf16
view the positive control restores:

```
compile_rc=0
git diff --stat: src/vllm/model_executor/models/ltx2_device.cpp | 2 +-
run_rc=1
TEST CASE:  ltx2 device: an f32 keyframes embedding under a bf16 stream is REFUSED
[doctest] test cases:  18 | 16 passed | 2 failed | 0 skipped
[doctest] assertions: 528 | 526 passed | 2 failed
[doctest] Status: FAILURE!
```

The mutation compiled, applied to exactly one line, and the named case failed.
The tree was restored byte-for-byte afterwards (`git diff --stat` shows only the
test file).

### The full-suite sweep, and the run of it that had to be thrown away

`ctest --test-dir build-sanitize` over all 484 tests: **483 passed, 2 skipped**
(`test_modelopt_mixed_precision_checkpoint`, `test_voxtral_e2e`), and
`test_ltx2_device` reported `***Failed`.

**That failure was an artefact of this session's own instrument, not a defect.**
Its output is the mutation's signature to the digit — `16 passed | 2 failed`,
`528` assertions, the same two `CHECK`s — because the mutant was still linked
in. The mutated source had been restored with `cp`/`mv`, which gave the restored
file an **mtime older than the object built from the mutant**, so `ninja` judged
it up to date, exited 0 having compiled nothing, and left the mutation in
`libvllm.a`. `git diff` on the source was empty the whole time; only
`stat` disagreed — the object was 24 s newer than the source it came from.

Corrected by `touch`ing the source, rebuilding (399 translation units, including
`ltx2_device.cpp.o`), and rerunning the affected family. `ctest -R 'ltx2|
diffusion'`: **10 of 10 passed, 0 failed**, `test_ltx2_device` among them.

Only `ltx2_device.cpp` carried the mutation, and the sole other translation unit
that reaches it is `src/vllm/multimodal/ltx2_video.cpp`; both are covered by
those 10. The other 479 results stand.

Recorded here because a build that compiles nothing and exits 0 defeats every
ordinary discriminator for a stale binary, and because the sweep is the only
thing in §5 that was mitigating the risk of a second finding hiding behind the
abort.

## 5. Risks

**Low.** One test-local lifetime change; no production file is touched, so no
shipped behaviour can move. The residual risk is that another sanitizer red
hides behind this abort — ASan stops the process at the first report, so the
rest of the binary's cases never ran in the RED. §4 discharges it: the full
`build-sanitize` suite is green apart from the two skips, and the 18 cases of
`test_ltx2_device` itself now run to completion rather than stopping at case 15.

## 6. Rejected alternatives

- **Save the pre-overwrite bf16 view and restore it.** One line shorter. It
  makes the positive control assert over the same view the case already proved
  was accepted at line 700, so a staging path that produced a stale or aliased
  view on a second call would still pass. The fresh restage is the stronger
  control and is what the author wrote.
- **Join the threadpool in `~Ltx2DitDeviceWeights`.** This is what #904's title
  asks for. It would make this case pass while leaving the dangling view in
  place, and it would add a synchronisation cost to every teardown for a defect
  that is not there. §2 gives the evidence.
- **Make `vt::Tensor` own or reference-count its storage.** A borrowed view is
  the deliberate design across the whole of `vt`; changing it for this is a
  project-wide decision that no red requires. Recorded under `## Owed`.

## 7. Cost

One test file, 8 lines added and 2 removed, including the comment that says why
`restaged` has to be named.

## 8. Now

`main` is red on `sanitize-cpu` on **both** arms because of this case, and both
arms are measured rather than one generalised to the pair. Scheduled run
[`31885935312`](https://github.com/mudler/vllm.cpp/actions/runs/31885935312) at
the main-lane SHA `04be1390b`:

| arm | ctest | failing test | report |
|---|---|---|---|
| `sanitize-cpu (address,undefined)` | 1 failed out of 478 | `71 - test_ltx2_device` | `AddressSanitizer: heap-use-after-free` at `cpu_layernorm.cpp:33 in LoadF32At`, address `0x50a000038c40` |
| `sanitize-cpu (thread)` | 1 failed out of 478 | `71 - test_ltx2_device` | `ThreadSanitizer: heap-use-after-free` at the same `cpu_layernorm.cpp:33 in LoadF32At` |

Exactly one test fails on each arm, it is the same test on both, and it is this
row's. So the two arms carry the same single cause, and removing it is what
"green on both arms" rests on. With §3 applied the lane's remaining known reds
are the ones tracked elsewhere; this row claims only its own.

## Owed

- A guard against a borrowed `vt::Tensor` outliving the object that owns its
  storage — filed as [#949](https://github.com/mudler/vllm.cpp/issues/949) and
  indexed in [`.agents/issue-index.md`](../issue-index.md). Nothing in the tree
  refuses it, and it took ASan on a lane that has been cancelled on every `main`
  run for weeks to find this one instance. The review of this row sharpened the
  scope with a measurement rather than an argument: with §3 reverted, a plain
  Release build with no sanitizer runs the case `18` passed of `18`, `546`
  assertions, `rc=0`, because `dtype` lives in the `vt::Tensor` struct and not
  in the freed buffer, so even the refusal path cannot notice garbage. The only
  instrument that sees this defect is therefore a lane that is
  `continue-on-error`, which is how #904 reached `main` at all. #949 weighs
  three remedies — promoting the lane once it has a `main` baseline, a test that
  fails without a sanitizer, and a static check for the pattern — rather than
  presuming one; scoping it stays a `vt` design question, which is why it is an
  issue on its own and not a task inside this row.

#904's stated cause is not owed anything — it is corrected in §2 and in the
pull request body, so the correction carries its evidence and its date in Git
rather than in an edited issue.
