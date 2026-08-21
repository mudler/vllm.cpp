# containers: the vulkan arm64 tuple and the cuda lane's build parallelism

Issues: [#1547](https://github.com/mudler/vllm.cpp/issues/1547) and
[#1548](https://github.com/mudler/vllm.cpp/issues/1548)
Row: `ENG-RELEASE-CONTAINERS`. Both defects block that row's `W6` publish step.
They land together on `row/FIX-CI-CONTAINER-PUBLISH` at the developer's
direction. This is a workflow and registry repair, not a roadmap item of its
own, and no lifecycle state moves.

## Now

`IMPLEMENTING`. Both fixes, their tests, and the mutation evidence are on
`row/FIX-CI-CONTAINER-PUBLISH`. The gate evidence below is captured on that
branch.

## Scope

In scope:

- `scripts/release_accelerator_metadata.py`, one `ARTIFACTS` row.
- `scripts/release_manifest.py`, one `_artifact_policy` row.
- `.github/workflows/containers.yml`, the build parallelism in the two
  building jobs and an explicit `timeout-minutes` on each.
- `scripts/check-container-workflow.py`, three new assertions.
- The four matching test files.

Out of scope, and deliberately so:

- `release/release-matrix.json`, `scripts/release_pipeline.py`, `docs/RELEASES.md`
  and `scripts/check-release-workflow.py`. Those describe DOWNLOADABLE release
  archives. See `## Design`.
- Building or gating an arm64 Vulkan leg. No such leg has run here.
- Any claim that the cuda lane's failure is diagnosed. See `## Risks`.

## Why

Run [32447481128](https://github.com/mudler/vllm.cpp/actions/runs/32447481128)
could not publish. Two independent defects failed two publish legs, and
`manifest` declares `needs: [plan, publish]`, so one failed leg skipped the job
that turns pushed digests into usable tags. The `cpu` images built, validated
and pushed by digest on both architectures and sit in the registry untagged.

`#1547` is deterministic. `docker/Dockerfile:145-146` selects
`linux-aarch64-glibc-vulkan` for every non-amd64 build, and no registry knows
that id, so the vulkan arm64 leg fails at 406 seconds with `unsupported Linux
accelerator artifact`. The re-run failed identically.

`#1548` is not deterministic and is not diagnosed. The cuda leg died at object
512 of 787 with exit 143 and `the runner has received a shutdown signal`.

## Design

**#1547 registers the tuple rather than removing the lane.** Two options were
weighed.

Removing vulkan arm64 from the publish matrix is not available at the cost it
appears to have. `scripts/check-container-matrix.py` refuses any lane that does
not declare both `linux/amd64` and `linux/arm64`, so removing the leg means
weakening that gate for every lane. The roadmap `ROAD-V1-CONTAINERS` states
that every lane is a multi-arch manifest on native runners because the project
gate hardware is arm64, and `release/container-matrix.json` already declares the
vulkan lane on both platforms at `channel: preview`. The lane is intended. The
registries never caught up with it.

Registering it is therefore the honest repair, and the honesty is in WHERE.
Three registries carry sibling ids, and only two of them describe this tuple:

| Registry | Describes | Edited |
|---|---|---|
| `scripts/release_accelerator_metadata.py` `ARTIFACTS` | the build tuple a container lane produces | yes |
| `scripts/release_manifest.py` `_artifact_policy` | what a produced manifest may claim | yes |
| `scripts/release_pipeline.py` `PRIMARY_ARTIFACT_FORMATS` | downloadable release archives | no |

`scripts/release_pipeline.py:137` requires `release/release-matrix.json` to equal
`PRIMARY_ARTIFACT_FORMATS` exactly, and `.github/workflows/release.yml` builds no
aarch64 Vulkan tarball. Adding the id there would declare an archive that never
exists, and `scripts/check-release-workflow.py` would refuse it. A test pins the
absence so a later reader does not read it as an oversight.

The channel set is `{"preview"}` and not `{"preview", "stable"}`. No arm64
Vulkan leg has been built or gated here, `.agents/roadmap_v1.md` records both
arm64 container legs as unbuilt, and
`scripts/build-linux-accelerator-release.sh` passes `--channel preview` for
every accelerator artifact. `scripts/release_accelerator_metadata.py` already
writes `correctness`, `runtime` and `performance` evidence as `absent` for a
preview accelerator tuple, so the produced manifest states what it lacks.

**#1548 makes parallelism lane-aware and adds a time budget.** The cpu and
vulkan lanes are not failing and keep `$(nproc)`. The cuda lane takes 2.

The value is measured and not guessed. `.github/workflows/ci.yml:801` already
builds the same ten-SM fat gencode set on a hosted runner at `--parallel 2` and
is green. The observed 512 of 787 objects in about 35 minutes puts a halved
build near two hours, which answers
`.agents/specs/container-images.md:200-204` and its concern that two jobs would
not finish inside a hosted runner's limits.

`timeout-minutes: 300` goes on both building jobs. Its purpose is to LABEL the
next failure, not to police the duration. Under the six-hour default a hang, a
reclaimed runner and an exhausted one all report the same exit 143. With a
budget, a job that dies before it is external and a job that dies at it hung.
The value is loose on purpose because the arm64 legs have no wall-time record.

`scripts/check-container-workflow.py` gates the SHAPE and not the number. Both
building jobs must cap the cuda lane, neither may hand `$(nproc)` to the build,
and both must state a budget. The resolved cap and budget print in the OK line
so a reviewer reads the values without opening the file. A gate on the number
would red on tuning it, which `CLAUDE.md` names as the defect rather than the
discipline.

## Risks

**The cuda fix is a hypothesis-driven mitigation, not a proven root-cause fix.**
Memory exhaustion is the leading hypothesis. GitHub infrastructure reclamation
produces the same runner message and the same exit code, and the available logs
cannot separate them. Lowering the value removes the one cause this repository
controls. If the next run fails the same way, the change remains defensible on
its own terms and the cause is external.

**The cuda lane gets slower.** That is the trade, and only that lane takes it.

**The registered tuple is unproven.** It is `preview`, its manifest records
three evidence axes as `absent`, and it is not a download. A reader who treats a
registered id as a supported artifact would be wrong, which is why the channel
set is narrow and a test refuses `stable`.

**`timeout-minutes: 300` could red a legitimately slow arm64 leg.** No arm64
container leg has ever built, so the projection is from amd64. The value carries
about 2.5 times the projection.

## Tests

1. `tests/scripts/test_release_accelerator_metadata.py`:
   `test_aarch64_vulkan_is_registered_preview_and_claims_no_evidence` drives the
   production path with the id that failed in CI and asserts the arch, the
   backend, the Vulkan boundaries, the `preview` channel and three `absent`
   evidence axes.
2. The same file:
   `test_aarch64_vulkan_is_not_a_downloadable_release_artifact` pins the
   deliberate absence from `release/release-matrix.json`.
3. `tests/scripts/test_release_manifest.py`: the aarch64 Vulkan tuple validates,
   `stable` is refused by name, and an x86_64 host on that id is refused as a
   tuple mismatch.
4. `tests/scripts/test_check_container_workflow.py`,
   `BuildParallelismMutationTests`: four cases. An uncapped `$(nproc)` in either
   building job is rejected, a dropped cuda cap in either is rejected, a dropped
   budget in either is rejected, and retuning the number to 3 is still accepted.
   Every case asserts the mutation applied before asserting the verdict.

## Gates

- `scripts/agent-preflight.sh` and `scripts/agent-preflight.sh --staged`.
- `scripts/check-container-workflow.py`, `scripts/check-container-matrix.py`,
  `scripts/check-release-workflow.py`,
  `scripts/check-release-binary-contract.py`,
  `scripts/check-build-runtime-deps.py`.
- The four test files above.

## Evidence

Recorded in the pull request body and in the commit bodies. The red-before
capture for `#1547` reproduces the CI line byte for byte:

```
accelerator release metadata error: unsupported Linux accelerator artifact 'linux-aarch64-glibc-vulkan'
```

## Owed

- The arm64 Vulkan leg has never been built. `#1547` fixes the registry gap that
  stopped it from starting. Whether it then compiles, links, and passes
  `test_vulkan_backend` and `test_backend_cross_device` on an arm64 runner is
  unmeasured, and the first publish run after this lands is the measurement.
  Owned by `ENG-RELEASE-CONTAINERS`.
- Whether the cuda lane finishes at two jobs inside 300 minutes on either
  architecture is unmeasured. Owned by `ENG-RELEASE-CONTAINERS`.

## Stop conditions

- Stop if the fix requires `release/release-matrix.json` to declare an aarch64
  Vulkan archive. That is a release-download claim and needs its own row.
- Stop if lowering the cuda cap requires weakening
  `scripts/check-container-matrix.py`.
- Stop before writing the cuda change up as a proven fix.
