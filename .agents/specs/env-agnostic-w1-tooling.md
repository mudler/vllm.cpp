# ENV-AGNOSTIC-W1-TOOLING: the tooling wave

Issue: [#1190](https://github.com/mudler/vllm.cpp/issues/1190).
Row: `ENV-AGNOSTIC-W1-TOOLING`, wave 1 of the campaign scoped by
[`env-agnostic.md`](env-agnostic.md). That row landed the rule, the three new
keys, the create-on-first-use route, and one worked example. This row applies
the rule to `scripts/`, `tools/`, and `third_party/README.md`.

## Scope

In scope: every literal in the wave's file set that
[`env-agnostic.md`](env-agnostic.md) `## The classification rule` calls
configuration.

Out of scope: `scripts/dgx-bringup.sh`, which the campaign row already
converted as its worked example; `.agents/completed/`; and every literal the
same rule calls provenance, wherever it sits.

### The file set, re-derived

Re-derived at base `5c8671c50d0abdf29d1dc62e2b6254fe8cdf0df9` with the
campaign's own query, narrowed to this wave's paths:

```sh
git grep -cIE 'dgx\.casa|nas_share|192\.168\.|thor:gpu0' \
  -- scripts tools third_party/README.md ':!scripts/dgx-bringup.sh' \
  | sort -t: -k2 -rn
```

19 files and 22 hits, which is exactly what the campaign's `## Follow-on rows`
table records. `git grep -l` gives the file count and `git grep -ho` the hits.

The set is 19 files **plus one**. `scripts/regen-triton-aot.sh:20-23` names
this row in its own text:

```text
#   VLLM_CPP_CUTLASS_DIR (default: ~/cutlass_probe, which is one developer's
#                        path. dgx-bringup.sh no longer carries that default,
#                        and converting this one to ${CUTLASS_DIR} is owed by
#                        ENV-AGNOSTIC-W1-TOOLING, issue #1190)
```

That file carries no query pattern, so the query cannot see it, and the campaign
row wrote the debt into the file rather than into a count. 20 files.

## The classification, per hit

The rule is applied per hit and never per file, because the campaign measured
that density does not predict the defect. Of the 22 query hits, 13 are
configuration and 9 are provenance. After the conversion the same query returns
9 hits over the same paths, and every one of them is in the provenance column
below.

| File:line | Class | Why |
|---|---|---|
| `scripts/check-gate-commands.py:367` | provenance | "because dgx.casa was unreachable at the SSH layer" is a dated reading of one box |
| `scripts/check-triton-aot-drift.sh:195` | configuration | "Regenerate ... e.g. dgx.casa" is an imperative pointing the reader at a box |
| `scripts/gen-vulkan-spirv.py:11` | provenance | "Neither of our boxes has one ... (measured 2026-07-22)" |
| `scripts/glm4-neartie-gap.py:47` | provenance | "GB10 (dgx.casa) has UNIFIED memory ... REBOOTS the box" is a measured property of one machine |
| `scripts/internlm2-neartie-gap.py:47` | provenance | same sentence |
| `scripts/laguna_longctx_bench.sh:8` | configuration | "RUN ON GB10 (dgx.casa)" is an imperative and a prerequisite list |
| `scripts/mtp-k-gt-1-neartie-gap.py:38` | provenance | the file states that its defaults are the instrument, so that the committed file and the file that executed are byte-identical |
| `scripts/qwen3-neartie-gap.py:47` | provenance | same sentence as glm4 |
| `third_party/README.md:39` | provenance | "The pin matches the loader version on `dgx.casa` (1.4.328)" is a measured quantity |
| `tools/bench/gpu_clock_state.py:4` | provenance | boot ids, sample counts, and ms/step |
| `tools/parity/README.md:4,8` | configuration | an oracle venv and a copyable regeneration command |
| `tools/parity/dump_gdn.py:7` | configuration | copyable command |
| `tools/parity/dump_moe.py:7` | configuration | copyable command |
| `tools/parity/dump_qwen36.py:21` | configuration | "The ACTUAL working command" |
| `tools/parity/dump_qwen3_5_mtp.py:9` | configuration | "Run on ... after acquiring the project GPU lock" |
| `tools/parity/dump_tokenizer.py:4,6` | configuration | "Runs on ... in the oracle venv" plus a copyable command |
| `tools/parity/dump_tokenizer_gpt4o.py:59` | provenance | it records the literal a previous change removed, and why (#1073) |
| `tools/parity/dump_tokenizer_mistral.py:10,14` | configuration | copyable command |
| `tools/parity/verify_tokenizer_gguf.py:6` | configuration | "Diff the two on ...", then the commands |

Three files carry both classes. `scripts/glm4-neartie-gap.py`,
`scripts/internlm2-neartie-gap.py`, and `scripts/qwen3-neartie-gap.py` each hold
a provenance sentence at `:47` and a copyable oracle-venv command at `:22-23`
that the query never matched because it names no host. Both are read, and only
the second is converted.

The worked example converted 16 literals of which 4 were query hits, so the
query names the FILE SET and never the hit set. Every operator-resolved literal
inside the 20 files is read: `~/venvs/vllm-oracle`, `~/work/vllm.cpp`,
`/home/mudler/_git/vllm`, `/home/mudler/work/vllm-pin`, `~/work/apex`,
`~/cutlass_probe`, `/usr/local/cuda`, `$HOME/laguna-xs-nvfp4` and
`$HOME/laguna-n4-build`.

## Design

Prose converts to `${KEY}`. Code resolves from the process environment and
refuses by name when a required value is unset.

`.env.example` gains no key. Every value this wave needs is already declared:
`GATE_HOST`, `GATE_CHECKOUT`, `VLLM_ORACLE`, `VLLM_SOURCE`, `CHECKPOINT_ROOT`,
`CUTLASS_DIR`, and `DEVICE_TOOLKIT_ROOT`.

### Where a value is required, and how it refuses

Five entry points consume a value rather than printing one.

| Entry point | Required | Was |
|---|---|---|
| `tools/parity/dump_gdn.py` | `VLLM_SOURCE` or `--pin` | `--pin` defaulted to `/home/mudler/_git/vllm` |
| `tools/parity/dump_moe.py` | `VLLM_SOURCE` or `--pin` | same default |
| `tools/parity/dump_qwen3_5_mtp.py` | `VLLM_SOURCE` or `--pinned-vllm` | same default |
| `scripts/regen-triton-aot.sh` | `VLLM_ORACLE`, `CUTLASS_DIR` | `~/venvs/vllm-oracle/bin/python`, `~/cutlass_probe` |
| `scripts/laguna_longctx_bench.sh` | both positional arguments | `$HOME/laguna-xs-nvfp4`, `$HOME/laguna-n4-build/...` |

The three dumpers read the process environment only, which is what the already
converted `tools/parity/dump_tokenizer_gpt4o.py:64` does in the same directory,
and what `.env.example` documents when it names `set -a; . ./.env; set +a` as
the loader. No script in this wave sources `.env` itself, so the blank-over-
value defect the worked example found cannot occur here; the shell scripts take
their values from the environment and from arguments and never assign over one.

`scripts/regen-triton-aot.sh` keeps `${DEVICE_TOOLKIT_ROOT}` optional and
prepends it to `PATH` only when set, mirroring `scripts/dgx-bringup.sh:94`. Its
previous `export PATH=/usr/local/cuda/bin:${PATH}` hard-coded one toolkit
layout in a file whose whole job is to run `nvcc`.

`${GPU_LOCK:-$HOME/gpu.lock}` stays exactly as it is in every file that has it.
`.env.example` states that this one key is not blank-means-unavailable and that
every script falls back to the same default on purpose, because a mutex on a
divergent path serialises the holder with nobody (#777).

### What a wrong default costs here

The campaign row measured this on `CUTLASS_DIR`: a configure that does not find
CUTLASS silently drops the sm120a NVFP4 GEMM and FlashAttention-2, and
`.agents/environment.md:388-400` records that turning those off moves the SACRED
`test_qwen27_paged_engine` from 235/235 to 234/235 with the source untouched.
`scripts/regen-triton-aot.sh` carried the same `~/cutlass_probe` default the
worked example removed, and it also regenerates VENDORED artifacts that get
committed, so a wrong toolchain there lands in the tree.

The three parity dumpers are worse in kind rather than in degree. `--pin` names
the checkout the ORACLE MATH executes from. A default that resolves to some
other tree on a second developer's box does not fail; it dumps goldens from the
wrong source and records the wrong provenance in the manifest. That is the
`token gates cannot see it` shape, so an unset value refuses.

## Tests

`tests/scripts/test_env_agnostic_tooling.py`.

- Each of the five entry points refuses by name when its value is unset, names
  the key and `.env`, and refuses BEFORE any expensive step. The dumpers refuse
  before importing torch; `regen-triton-aot.sh` refuses before `cmake`;
  `laguna_longctx_bench.sh` refuses before taking `${GPU_LOCK}`.
- Each entry point resolves the value from the process environment.
- No file in the wave's converted set carries a home-anchored `${KEY:-$HOME/...}`
  fallback or one operator's resolved literal, in either the comment half or the
  executable half. A relative default such as `${BUILD_DIR:-build-triton-regen}`
  is nobody's path and stays, and `${GPU_LOCK:-$HOME/gpu.lock}` is exempt for the
  reason `.env.example` records.
- The provenance hits stay literal. Nine files and twelve sentence fragments are
  pinned by content, so a later blind sweep that rewrites a recorded
  measurement's host goes red.

The last case is the one that protects the second class. Every other assertion
here would pass a `sed` that converted the whole tree.

## Gates

- `python3 tests/scripts/test_env_agnostic_tooling.py`
- `python3 tests/scripts/test_gate_bringup.py`
- `python3 tests/scripts/test_agent_start.py`
- `python3 tests/scripts/test_agent_onboard.py`
- `bash scripts/check-triton-aot-drift.sh`
- `scripts/agent-preflight.sh --fail-on-skip`

No hardware gate applies. This row moves where a value comes from. It changes
no kernel, no dtype, no allocation, and no token, so it claims no CUDA, SACRED,
or throughput gate. The parity dumpers and the bring-up scripts it edits are
maintainer entry points that need a GPU box to RUN; the refusal path is what is
gated here, and it is asserted before any of them reaches hardware.

## Risks

A converted dumper that refuses where it used to run is a behaviour change for
whoever had the old default resolving correctly. That is the intended trade: the
old default was right on exactly one machine and silently wrong everywhere else.
The refusal names the key, so the repair is one `--env-set`.

`tools/parity/dump_tokenizer.py` writes a `regenerate` string into
`tests/parity/goldens/*/encodings.json`. The committed goldens keep the old
string until someone regenerates them, and nothing reads the field, so the two
disagreeing is documentation drift and not a gate. `tests/` belongs to
`ENV-AGNOSTIC-W3-CODE`, so this row does not edit the goldens.

## Stop conditions

The campaign's stop conditions apply unchanged. Report `NEEDS_DECISION` rather
than guessing when one paragraph holds both classes, and never rename a file
whose name carries a host.

## Evidence

Red before green, at base `5c8671c50`, with the suite written and no file
converted:

```text
$ python3 tests/scripts/test_env_agnostic_tooling.py
Ran 11 tests in 1.439s
FAILED (failures=58)
```

Ten of the eleven cases failed. The one that passed is
`ProvenanceTests::test_every_recorded_measurement_keeps_its_host`, which is
correct: that guard has to be green on both sides of this change, because what
it protects is what must NOT move.

Green after, same command, 11 tests `OK`.

The campaign's query over the wave's paths returns 9 hits after the conversion,
down from 22, and each of the 9 is in the provenance column of
`## The classification, per hit`:

```sh
git grep -hoIE 'dgx\.casa|nas_share|192\.168\.|thor:gpu0' \
  -- scripts tools third_party/README.md ':!scripts/dgx-bringup.sh' | wc -l
```

Four mutations, each applied to the committed tree, each restored and confirmed
byte-for-byte identical by `git write-tree` returning `e2d6c00e5` again with
`git status --porcelain` empty. Every mutation compiled, so none of them is a
build failure reading as a passing test:

| Mutation | compile_rc | Failing cases |
|---|---:|---:|
| `sed 's/dgx\.casa/${GATE_HOST}/g'` over the six provenance files | 0 | 6, all `ProvenanceTests` |
| reintroduce `${CUTLASS_DIR:-${HOME}/cutlass_probe}` in `regen-triton-aot.sh` | 0 | 3 |
| restore `--pin` default `/home/mudler/_git/vllm` in `dump_gdn.py` | 0 | 3 |
| delete the `resolve_pinned_source` call site in `dump_moe.py` | 0 | 4 |

The first is the one this row exists to make expensive. A blind sweep of the
tree is the cheapest way to close #1190 and the only way to falsify the records
while doing it, and until now nothing anywhere would have gone red.

The last is the reachability mutation. Removing the call site and resolving the
literal inline keeps `parity_env.py` compiling and its unit behaviour intact,
and the suite still goes red, so the gate measures the entry point rather than
the helper.

`scripts/agent-preflight.sh --staged --fail-on-skip` reported 84 gates `ok`,
zero `FAIL`, and two `SKIP`, both because `origin/main` had moved and the branch
was behind it. The post-merge run is recorded in the pull request.

## Owed

**The derivation query cannot see the defect it was written to find.**
`git grep -cIE '\$\{[A-Z_]+:-[^}]*(HOME|home/mudler|venvs|cutlass|usr/local/cuda)'`
over `scripts/` and `tools/` returns 12 lines in 6 files at this row's base, and
none of them matches `dgx.casa`, `nas_share`, `192.168.` or `thor:gpu0`. Widened
to `/home/mudler`, `~/venvs/vllm-oracle`, `~/work/vllm.cpp`, `cutlass-4.5.0` and
`cutlass_probe`, the same paths hold 73 files and 174 hits against the query's 19
and 22.

This row converts the two of those files it owns by name and leaves the rest,
because the residual is a campaign-level gap rather than a W1 gap: the same blind
spot covers `ENV-AGNOSTIC-W3-CODE` and `ENV-AGNOSTIC-W4-RECORDS`, and the
campaign's claim that its five waves partition 227 files is a claim about the
query's 227 and not about the tree.

Two of the residual files read as provenance on a first pass and must not be
swept blind. `scripts/cpu-x86-llamacpp-floor.sh:33` states that "every recorded
leg used these defaults verbatim", and `scripts/dgx-gdn-packed-bridge-ab.sh:4`
dates its prerequisites. That is the same instrument-identity argument that makes
`scripts/mtp-k-gt-1-neartie-gap.py` provenance here.

Owner: [#1308](https://github.com/mudler/vllm.cpp/issues/1308). No row owns it
yet, so this spec owns it here, which is what `AGENTS.md` requires of an issue
filed and not fixed in the same flow.

## Now

`ENV-AGNOSTIC-W1-TOOLING` converts 13 configuration hits across `scripts/`,
`tools/` and `third_party/README.md`, leaves 9 provenance hits literal, removes
the hard-coded defaults five entry points carried, and gates each refusal.
