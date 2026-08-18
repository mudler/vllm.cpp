# The `main` baseline runs the MSVC gates it grades

Identity: `ENG-RELEASE-WINDOWS`

Issue: [#503](https://github.com/mudler/vllm.cpp/issues/503)

Parent specification: [windows-binary-release.md](windows-binary-release.md).
Sibling: [windows-msvc-strict-build.md](windows-msvc-strict-build.md), which
repairs what MSVC reports; this one repairs who is listening.

Status: `ACTIVE`. The row does not change lifecycle state here, so this change
owes no `docs/STATUS.md` or `docs/BENCHMARKS.md` edit.

## The defect

`windows-msvc-cpu` and `windows-msvc-vulkan` carried
`if: github.event_name == 'pull_request'`, and the lane
`scripts/main-baseline.py` grades is `schedule` plus `workflow_dispatch`. So the
two MSVC gates were not defined for any event that lane fires on.

That is worse than being skipped, and the distinction is the whole row.
`main-baseline.py` already distinguishes three non-green shapes — a job that ran
and failed, a job still running, and an EXPECTED job the payload never mentioned
— and it added the third precisely because absence could otherwise wear green's
face. It could not see this one, because `EXPECTED_JOBS` did not name the two
jobs. They were absent from `failing`, absent from `pending`, absent from
`not_run`, absent from `missing`, and absent from `covered`. Nothing printed
them, and the verdict read:

```
NEWEST BASELINE: GREEN at bbc482a2de73ae0c433f210c5780b52d6c886f5d
```

while `main` did not compile under MSVC.

Measured, not inferred: `workflow_dispatch` run `32044993401` on `main` reports
`conclusion=success` with both `windows-msvc-*` jobs `skipped`.

### What it cost

Each of these landed green on `main` and then reddened the next contributor's
pull request, presenting to that author as a defect their own diff caused:

| Issue | The defect that reached `main` unseen |
|---|---|
| [#503](https://github.com/mudler/vllm.cpp/issues/503) | `test_cpu_isa_x86.cpp` missing `<ostream>` |
| [#603](https://github.com/mudler/vllm.cpp/issues/603) | POSIX `setenv`/`unsetenv` in `test_backend_cross_device.cpp` |
| [#729](https://github.com/mudler/vllm.cpp/issues/729) | three unguarded POSIX sites in `video_engine.cpp` |
| [#965](https://github.com/mudler/vllm.cpp/issues/965) | `C4456` shadowing in `server_main.cpp:1315` |
| [#968](https://github.com/mudler/vllm.cpp/issues/968) | `C4244` narrowing in `ltx2_video.cpp` |
| [#1068](https://github.com/mudler/vllm.cpp/issues/1068) | `C3493` dropped `kRequired` capture in `qwen3_5_weights.cpp` |

Five of the six were found by an author who had to first prove the red was not
theirs. [`minimax-music3.md`](minimax-music3.md) §10.6 records what that costs
when two causes stack behind one habitually-red job name.

## Scope

In: the `if:` of both `windows-msvc-*` jobs, the pinned literal in
`scripts/check-release-workflow.py::validate_pr_ci` that holds it, the
`baseline-summary` `needs:` list, `EXPECTED_JOBS` in `scripts/main-baseline.py`,
and the two suites that pin all of the above.

Out: [#584](https://github.com/mudler/vllm.cpp/issues/584), the crash that makes
the lane red — diagnosed here and listed under `## Owed`. Out:
[#874](https://github.com/mudler/vllm.cpp/issues/874), the closed-pull-request
start, which stays exactly as it is (see Risks). Out: any C++ change; nothing
under `src/`, `include/` or `tests/*.cpp` is touched.

## Design

```yaml
if: github.event_name == 'pull_request' || github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'
```

`push` is excluded, and that is a decision rather than an omission. The push
lane cannot answer "is `main` green" by construction — every expensive job there
carries a group keyed on `github.ref`, constant for every push, so consecutive
pushes cancel each other (26 cancelled of 40 runs in the window `ci.yml`
records). Adding two `windows-2022` runners to 55 pushes a day buys a bill, not
a baseline.

The condition stays a byte-exact literal in `validate_pr_ci` rather than
becoming a predicate. That function compares each Windows job's WHOLE mapping
against a literal dict, which is how the PR lane proves it holds no release,
upload, write-token or OIDC authority (#117). Broadening the pin into "any
condition that admits a pull request" would trade a property the row does not
own. `always()`, `github.event_name != 'push'` and a bare `true` are each still
rejected, and each is an executed mutation below.

Adding the jobs to `baseline-summary`'s `needs:` and to `EXPECTED_JOBS` is not
optional decoration. Running on the lane and being graded by it are separate
properties, and having only the first rebuilds #274's defect by omission: the
job would fail, and the published verdict would not move.

## Consequence, stated in advance

**The first baseline that can see these jobs is RED**, on
[#584](https://github.com/mudler/vllm.cpp/issues/584). That is the correct first
verdict and it is the same shape `baseline-summary`'s own comment already
records for the six sanitizer failures of #274. The baseline read GREEN before
only because it never ran them. A change that hid the red to keep the badge
green would rebuild the defect it is repairing.

## Tests and red-before evidence

Every mutation below was applied to the head tree, run, and reverted; the tree
was verified byte-for-byte identical afterwards by sha256 on both mutated files.

| Mutation | `check-release-workflow.py` | `test_main_baseline.py` |
|---|---|---|
| revert both `if:` to `pull_request` only | `rc=1` | 4 failures, `..._run_on_the_baseline_lane_and_not_on_push` for both jobs on `schedule` and `workflow_dispatch` |
| drop both from `baseline-summary` `needs:` | `rc=0` | 3 failures, incl. `test_expected_jobs_is_pinned_against_the_workflow_needs_list` |
| widen both `if:` to `always()` | `rc=1` | 8 failures, all four events on both jobs |
| add `\|\| github.event_name == 'push'` | `rc=1` | 2 failures, `event='push'` on both jobs |
| narrow `EXPECTED_JOBS` back to nine | `rc=0` | 4 failures, incl. `..._skipped_back_off_the_lane_is_red_not_green` |

New cases in `tests/scripts/test_main_baseline.py`:

- `test_the_windows_proofs_run_on_the_baseline_lane_and_not_on_push` — resolves
  the condition to a boolean per event through `resolve_boolean`, for the reason
  that helper exists: `always()` and `!= 'push'` both admit the baseline lane
  and both are wrong, and only the `push` half separates them from the intent.
- `test_the_windows_proofs_are_covered_by_the_published_verdict` — `needs:` and
  `EXPECTED_JOBS` together, so neither half can be dropped alone.
- `test_a_red_windows_proof_makes_the_baseline_red` — the state on landing day,
  executed rather than asserted about.
- `test_a_windows_proof_skipped_back_off_the_lane_is_red_not_green` — a revert
  of this row reports the job `skipped`, which is read as absent and not as a
  pass.

New mutations in `tests/scripts/test_release_pipeline.py`: `unconditional
execution`, `push lane added`, `condition inverted to exclude only push`,
`baseline lane dropped again`.

Duplicate-key check, because PyYAML accepts what GitHub rejects and a duplicate
key yields zero jobs: every file under `.github/workflows/` loads under a
duplicate-rejecting loader, and `ci.yml` carries **16 jobs before and 16 after**
with an identical job-name set.

The two mutation anchors in `test_release_pipeline.py` that used to carry the
job header and its three comment lines now anchor on the `if:` line, which
occurs exactly twice in `ci.yml` — once per Windows job, CPU first. Both facts
are asserted in the test rather than assumed. The old anchors coupled a mutation
suite to prose, so editing a comment reddened it.

## Gates

`check-release-workflow.py`, `test_release_pipeline.py`,
`test_main_baseline.py`, `check-agent-record.py`, `check-doc-checkpoint.py`,
`check-public-doc-tables.py`, `check-pr-size.py`, `check-commit-trailers.py`,
`check-commit-style.py`.

## Risks

**The baseline goes red and stays red until #584 is fixed.** Intended; see
above. It does not block any pull request — `baseline-summary` runs only on
`schedule`/`workflow_dispatch` and can never fire on a contributor's branch.

**Runner cost.** Twelve additional `windows-2022` job starts per day at the
current four-hour cadence, against the two per pull request already paid.
Neither job carries a job-level concurrency group (the pinned schema admits no
extra key), so on the baseline lane they are non-cancellable, which is correct —
a cancelled baseline answers nothing — and on the pull-request lane the
workflow-level group still supersedes them exactly as before.

**Wall clock.** `baseline-summary` now waits on two jobs whose timeout is 180
minutes against a suite the workflow measures at 99. A hung Windows job delays
the verdict rather than corrupting it, and the four-hour cadence still admits it.

**#874 is untouched, deliberately.** Both jobs still start on a closed pull
request. Fixing that means giving them a closed-action clause, which means
removing them from `UNGUARDABLE_JOBS` in `test_main_baseline.py`, whose own test
asserts they carry no such clause. That is a second decision with its own
review, and bundling it here would make one change argue two cases.

## Now

`ENG-RELEASE-WINDOWS` stays `ACTIVE`. With this landed, `main` can report a
Windows verdict for the first time, and that verdict is RED on #584. The next
step on the row is #584 itself.

## Owed

- [#584](https://github.com/mudler/vllm.cpp/issues/584) —
  `test_openai_api_server.exe` fast-fails with `0xC0000409` on both Windows
  lanes. **Narrowed here, not fixed.** What the evidence establishes:

  The eight `INFO api: POST /v1/chat/completions` lines in the job log are
  emitted by `LogHttpIngress` at `src/vllm/entrypoints/openai/api_server.cpp:223`,
  which is called for that route and no other, through
  `src/vllm/entrypoints/openai/request_logger.cpp:26` on `std::cerr` — unit
  buffered, so every one of them reaches the pipe immediately and their absence
  after a point is evidence rather than buffering. `tests/vllm/entrypoints/openai/test_api_server.cpp`
  reaches `handle_chat_completions` at exactly eight sites before line 1300, in
  file order `:596 :647 :662 :703 :748 :781 :902 :1292`. The eighth and last
  logged request is `body_bytes=92 stream=0 max_tokens=4 msgs=1 prompt_chars=5`,
  and the body posted at `:1291-1294` is 92 bytes with
  `"max_completion_tokens":4` and one `user` message of `hello`. So the process
  reached `:1292`, and the server answered it — `Finished request chatcmpl-0
  prompt_tokens=1 completion_tokens=4 finish_reason=length` is in the log.

  It then produced no further output for **0.78 s** and fast-failed. That places
  the fault between `test_api_server.cpp:1294` and the `REQUIRE` at `:1325` —
  `REQUIRE(res)` at `:1295`, the `json::parse` at `:1297`, `h.server.stop()` and
  `server_thread.join()` at `:1302-1303`, the scope exit that destroys the
  `httplib::Client` and the `ServerHarness`, and the start of the next socket
  test. The 1.0 s poll loop at `:1323-1324` is excluded by the 0.78 s figure.
  This is the first test case in the file that binds a real socket and runs
  `serve()` on a thread; the ~50 harness-only cases before it all completed.

  **`0xC0000409` is not evidence of a stack buffer overrun.** It is the status
  `__fastfail` raises for EVERY fail-fast code, so `abort()` — and therefore
  `std::terminate()` — and the CRT invalid-parameter handler both surface as it.
  Searching for `sprintf` into a stack buffer is not indicated; the only fixed
  stack buffer in the file, `char buf[512]` at `:1685`, is inside
  `#if defined(__linux__)`. `__fastfail` also bypasses SEH, which is why
  doctest's Windows handler cannot report it and why the whole doctest output is
  the version banner.

  **The mechanism that makes this undiagnosable is in the test, and it is not a
  guess.** Fourteen cases hold a `std::thread server_thread` across throwing
  assertions and only join it at the end — construction and join at `:1243/1303`,
  `:1322/1338`, `:1353/1373`, `:1392/1400`, `:1534/1575`, `:1589/1622`,
  `:1662/1701`, `:2101/2109`, `:2267/2315`, `:2333/2367`, `:2521/2558`,
  `:2574/2590`, `:3065/3074`, `:3159/3222` — and a fifteenth, the
  `#if defined(_WIN32)`-only teardown case at `:2616`, holds two
  (`:2629`, `:2643`, joined at `:2659-2660`) across a `REQUIRE` at `:2658`. Any throw in
  between — a failed `REQUIRE`, or `nlohmann::json::parse` on an unexpected body
  — destroys a joinable `std::thread`, which calls `std::terminate()`, which
  calls `abort()`, which on MSVC is `__fastfail`. An ordinary named assertion
  failure therefore becomes an opaque `0xC0000409` with no reporter output. That
  conversion is platform-independent and provable by inspection; whether it is
  what fired here is not, because no output survives to say so.

  NOT fixed in this change, and the reason is a stop condition rather than a
  preference: it is a C++ edit at fifteen sites in a file this session is not
  permitted to build (the host is at 94 % disk, and a mutation that fails to
  compile reads as a passing test). Landing it unbuilt would risk the Linux
  lanes to repair a Windows instrument. The next step is that RAII change plus a
  rerun, which turns the fast-fail into a named assertion and either resolves
  #584 or hands the next session the exact line.
