# Fetch a model from HuggingFace (`ENG-HF-MODEL-DOWNLOAD`)

Row `ENG-HF-MODEL-DOWNLOAD` in [`engine-matrix.md`](../engine-matrix.md).
Issue [#1280](https://github.com/mudler/vllm.cpp/issues/1280).
The two holes the second review found in this row's tree-listing size rule are
issue [#1339](https://github.com/mudler/vllm.cpp/issues/1339), repaired in the
same flow.
The quickstart page that consumes this row is issue
[#1281](https://github.com/mudler/vllm.cpp/issues/1281).

## Scope

Make `--model` accept a HuggingFace repository identifier and fetch the
checkpoint, so a reader who holds only a container image can run a model.

Two forms land behind one flag. The local path stays the first probe, so a
network call can never shadow a path that exists on disk.

| Form | Result | Reference |
|---|---|---|
| `<dir>` or `<file.gguf>` | Unchanged behavior | n/a |
| `org/repo` | Full snapshot into the HuggingFace cache | vLLM, mirrored |
| `org/repo:Q4_K_M` | One GGUF file | llama.cpp, secondary oracle |

Flags and environment variables, each mirrored rather than invented:

| Name | Meaning |
|---|---|
| `--revision <ref>` | vLLM's own flag. Applies to both forms |
| `--download-dir <path>` | vLLM's flag. Maps to the cache directory |
| `HF_TOKEN` | Bearer token for a gated or private repository |
| `HF_ENDPOINT` | Alternate hub host |
| `HF_HOME` | Cache root. The container already sets `/cache` |
| `HF_HUB_OFFLINE` | Resolve from cache and open no socket |

This row does not add a public application binary interface (ABI) function.
The resolver runs behind the model-path field that `vllm_server_main` already
takes, so `include/vllm.h` and ABI v19 do not move.

Out of scope, and listed again under `## Owed`: ModelScope, `--tokenizer-revision`,
`--code-revision`, LoRA adapter fetch, and the llama.cpp Docker-registry model
path.

## Upstream chain

vLLM is the primary oracle and defines the `org/repo` form. Pin
`5559679229bc961848b121ccdeaa8fa5d79bec98`, read from a local checkout whose
`HEAD` was verified equal to the pin on 18 August 2026.

| Behavior | Upstream anchor |
|---|---|
| Local-or-remote decision | `vllm/model_executor/model_loader/weight_utils.py:345` |
| Two-phase fetch, config JSON first | `weight_utils.py:349-357` |
| Weight format preference | `vllm/model_executor/model_loader/default_loader.py:167-184` |
| First matching pattern wins | `weight_utils.py:493-496` |
| Index-driven file selection | `weight_utils.py:472-490` |
| Offline mode | `weight_utils.py:459` |
| Cross-process download lock | `weight_utils.py:506` |
| `--revision` as its own flag | `vllm/engine/arg_utils.py:839`, `vllm/config/model.py:183` |

llama.cpp is the secondary oracle and covers only the `org/repo:QUANT` form,
which vLLM does not implement. `.agents/oracles/llama-cpp.md` pins stock tag
`b10451` and records `gateable = no`. This row reads that source to port a
protocol. It cites no llama.cpp measurement, so the `gateable = no` value does
not block the work.

W1 read every anchor in the table below on 18 August 2026 at stock tag
`b10451`, fetched from `https://github.com/ggml-org/llama.cpp` into a scratch
directory outside this repository. `b10451` is a lightweight tag, `git cat-file
-t b10451` answers `commit`, and it resolves to commit
`10bf611e533d81f739128304991c5e133c6aebd8`, dated 2026-08-16, subject `llama :
check LoRA tensor data is within file bounds (#27056)`. The checked-out `HEAD`
was verified equal to that object id before any line was read. Every behavior in
the table exists at `b10451`. Four anchors moved against the earlier reading in
the developer fork `237ad9b96`, and the moved rows are marked.

| Behavior | llama.cpp anchor at `b10451` |
|---|---|
| Transport is cpp-httplib, not libcurl | `CMakeLists.txt:195,229`, `common/CMakeLists.txt:144` (moved) |
| TLS selection | `vendor/cpp-httplib/CMakeLists.txt:38-154` (moved) |
| Refusal with no TLS | `common/http.h:107-116` (moved) |
| Reference to commit | `common/hf-cache.cpp:233` |
| Recursive file listing | `common/hf-cache.cpp:311` |
| Byte download address | `common/hf-cache.cpp:347` |
| Cache layout, link then move then copy | `common/hf-cache.cpp:455-496` |
| Repository and tag split | `common/download.h:39-42` (moved) |
| Range resume | `common/download.cpp:222-235` |
| Size, entity tag, and range probe | `common/download.cpp:321-349` |
| Object identifier validation | `common/hf-cache.cpp:161-163` |

The three CMake rows and the `common/http.h` row moved because the fork adds
files and options above them. `CMakeLists.txt:170` at `b10451` sets
`GGML_CUDA_GRAPHS_DEFAULT`, and `common/http.h:75-83` at `b10451` splits a
bracketed IPv6 authority. Neither is the behavior the row names, so the earlier
citations were wrong rather than merely shifted. The `common/hf-cache.cpp` and
`common/download.cpp` line numbers are unchanged between the fork and stock,
because the fork does not touch either file.

## Our baseline

What the tree does today, measured at `32b32bb14`:

1. `examples/server/main.cpp` is 23 lines and calls `vllm_server_main`. All
   flag parsing lives in `src/vllm/entrypoints/openai/server_main.cpp`.
2. `server_main.cpp:433` stores the `--model` value and does not interpret it.
3. `src/vllm/entrypoints/model_loader.cpp:279-303` already converts a repository
   identifier to a local cache snapshot directory. It reads an existing cache
   and never downloads. It serves the DFlash draft path only.
4. No file in `src/` or `third_party/` links a transport layer security (TLS)
   library. `third_party/httplib/httplib.h` is vendored, and no build file
   defines `CPPHTTPLIB_OPENSSL_SUPPORT`.
5. `docker/Dockerfile:188-192` sets `HF_HOME=/cache`, declares
   `VOLUME ["/models", "/cache"]`, and installs `ca-certificates`. The runtime
   stage installs no Python and no `curl`.
6. `scripts/release_metadata.py:123-148` derives the declared dependency rows by
   running `readelf -dW` on the built server. The list is not hand maintained.
7. `scripts/release_manifest.py:345` marks `linux-x86_64-musl-cpu-static` as
   `literal-static`. `scripts/validate-release-archive.py:404` refuses an
   interpreter, a dependency, or a run path on such an artifact.
8. `scripts/build-cpu-release.sh:22` is the only line that sets
   `VLLM_CPP_LITERAL_STATIC=ON`.

## Port map

New files, mirroring vLLM's own `vllm/transformers_utils/` location:

| File | Responsibility |
|---|---|
| `src/vllm/transformers_utils/hf_hub.{h,cpp}` | Hub protocol, authentication, endpoint |
| `src/vllm/transformers_utils/hf_cache.{h,cpp}` | Cache layout and offline resolution |
| `src/vllm/transformers_utils/downloader.{h,cpp}` | Byte transport, resume, progress, lock |
| `src/vllm/transformers_utils/model_resolver.{h,cpp}` | The `--model` grammar and the one entry point |

Internal headers go under `include/vllm/transformers_utils/`. No example and no
server includes them, because both reach this code through `--model`.

Protocol calls, in order:

1. `GET {endpoint}api/models/{repo}/refs` resolves the reference to a commit.
2. Every later call names that commit. A moving `main` branch cannot change what
   a second run loads.
3. `GET {endpoint}api/models/{repo}/tree/{commit}?recursive=true` lists files.
4. `GET {endpoint}{repo}/resolve/{commit}/{path}` returns bytes. Follow the
   redirect to the content delivery network.

Cache layout is HuggingFace's documented local cache:
`{HF_HOME}/hub/models--org--repo/` with `refs`, `blobs`, and
`snapshots/{commit}/{path}`. A host that already holds a Python
`huggingface_hub` cache gets a hit and downloads nothing, and
`model_loader.cpp:279-303` already reads this layout. The snapshot entry is a
symbolic link. If the file system refuses a symbolic link, fall back to a move,
then to a copy, and log the fallback one time. The fallback is required here,
because `/cache` is a declared volume and the shared storage on this fleet is
Common Internet File System (CIFS), which holds no symbolic link.

`model_loader.cpp:279-303` moves into `hf_cache.cpp` and the DFlash path calls
the shared function. The tree gets one implementation, not two.

Integrity, and the one place this row does not port llama.cpp:

- Treat `lfs.oid` as absent unless the request carried a token. This governs
  whether the identifier is USED as a blob name, and nothing else.
- Refuse a listing whose entries share one object identifier and disagree on a
  size the listing REPORTED, whether or not their paths differ. Accept them when
  the reported sizes agree, because that is duplicate content. An entry that
  reports no size is compared against nothing and can never satisfy the rule on
  another entry's behalf.
- Refuse an object identifier whose characters are all the same, for example one
  character repeated 64 times.
- Neither refusal depends on the token.
- Prove a safetensors file complete with
  `8 + header_len + max(data_offsets[1]) == file_size`.
- Prove a GGUF file complete with its magic value, version, tensor count, and
  data end offset.
- Name a blob by the entity tag from the resolve response or by a locally
  computed sha256, and record which one the run used.

Transport details: probe with `HEAD` for size, entity tag, and range support.
Resume with `Range: bytes=N-`. Write to a `.incomplete` file and rename it after
the structural check passes. Take a per-repository lock across processes, as
vLLM does at `weight_utils.py:506`. Print progress to standard error under
`--verbose`. Cancel on `SIGINT`.

Refusal messages name the missing part:

| Condition | Message names |
|---|---|
| HTTP 401 or 403 | The repository and `HF_TOKEN` |
| HTTP 404 | Unknown repository, unknown revision, or unknown tag, and for a tag the tags the listing holds |
| No TLS in the build | The three build options |
| Offline with a cache miss | The cache directory the run searched |

Build and packaging:

| Option | Default | Effect |
|---|---|---|
| `VLLM_CPP_HF_DOWNLOAD` | `ON` | The feature. Resolves `OFF` with no TLS |
| `VLLM_CPP_OPENSSL` | `ON` | `find_package(OpenSSL)` with a version check, then link `OpenSSL::SSL` and `OpenSSL::Crypto` |
| `VLLM_CPP_BUILD_BORINGSSL` | `OFF` | Static BoringSSL through `FetchContent`, for the `literal-static` lane |

`docker/Dockerfile:179-183` adds `libssl3` to the existing package list.
`NOTICE:19` gains a line for OpenSSL and a line for BoringSSL.

## Tests to port

Every test runs against an in-process fake hub over plain hypertext transfer
protocol, reached through `HF_ENDPOINT`.
`tests/vllm/entrypoints/openai/test_api_server.cpp` already starts an
`httplib::Server` in a test, so the fixture follows that file.

| Case | Asserts |
|---|---|
| Local directory passthrough | The fake hub receives no request |
| Snapshot fetch | The decoy file `original/model.safetensors` is never requested |
| Single GGUF fetch | The `org/repo:Q4_K_M` form returns one file |
| Cache hit | The second run issues zero requests, counted on the fake hub |
| Resume | The `Range` header is sent and the final bytes are correct |
| Range ignored | A 200 answer to a range request fails the run |
| Offline, cold cache | The run refuses and names the cache directory |
| Offline, warm cache | The run succeeds with zero requests |
| Gated repository | HTTP 401 names the repository and `HF_TOKEN` |
| Token sent | The fake hub observes the `Authorization` header |
| Fabricated object identifier | A tree where every file carries `"a"` 64 times is refused |
| Size rule not disarmable | An unsized entry first, then two disagreeing sizes, is refused |
| One path, two sizes | One path listed twice at 4096 and 2048 bytes is refused |
| Unreported size | A zero-byte file and an entry with no size read differently |
| Truncated body | A body shorter than `Content-Length` is refused |
| Bad safetensors header | A file failing the data-end check is refused |
| No symbolic link | The snapshot entry is a real file and the model loads |
| No TLS build | An `https` address returns the message naming the build options |

The fabricated-identifier case is the regression test for a measured event. On
17 August 2026 the HuggingFace tree API answered an unauthenticated caller on
the gated repository `Lightricks/LTX-2.5` with an `lfs.oid` of one character
repeated 64 times, identical for all 14 large-file-storage files. That value
passes llama.cpp's `is_valid_oid` at `common/hf-cache.cpp:161`. The case exists
so that a later verbatim port of that function turns the gate red.

These tests prove protocol, cache, resume, integrity, and refusal. They prove
nothing about TLS, because they speak plain hypertext transfer protocol. TLS has
its own instruments, listed under `## Gates`.

## Gates

Reachability, answered as two separate questions, as
[`reachability.md`](../reachability.md) requires.

The production entry point is `vllm-server`, which calls `vllm_server_main`,
which parses `--model` and calls the resolver. It is a registered command-line
path on its default configuration.

`vllm-cli` does NOT reach the resolver, and the earlier sentence claiming it did
was wrong. `examples/cli/main.cpp` includes `vllm.h` and nothing else, because
an example is an application binary interface client only, and this row
deliberately adds no ABI function. So `vllm-cli --model org/repo` and any C ABI
caller of `vllm_model_load` still take a local path. That is listed under
`## Owed`: closing it means adding an ABI entry point, which is a public-surface
decision and its own row.

The smallest failing test calls `vllm_server_main(argc, argv)` with
`--model org/repo` and `HF_ENDPOINT` aimed at the fake hub, then asserts that
the server boots and completes a request. It does not construct
`ModelResolver` directly. A fixture that builds the request itself cannot see a
deleted production call site, which this tree has already shipped once.

Mutations the fresh reviewer runs, each in a scratch copy, each restored
byte for byte:

| Mutation | Must turn red |
|---|---|
| Neutralise the resolver call in `server_main.cpp` with `if (false)` | The end-to-end case |
| Delete the `--revision` branch in `ParseArgs` | The end-to-end case |
| Delete the `--download-dir` branch in `ParseArgs` | The end-to-end case |
| Drop the object-identifier preference in `BlobNameFor` | The blob-name case |
| Range mismatch warns instead of failing | The range-ignored case |
| Accept an all-identical object identifier listing | The fabricated-identifier case |
| Own an identifier with an entry that reports no size | The not-disarmable case |
| Restore the distinct-path guard on the size rule | The one-path-two-sizes case |
| Remove the case fold on the object identifier | The mixed-case-two-sizes case and the blob-name case |
| Remove index-driven selection | The decoy-file case |
| Remove the offline short circuit | The offline cases |

THE RESOLVER MUTATION IS A NEUTRALISATION, NOT A DELETION, and the row above
says `if (false)` for a measured reason. DELETING the call does not compile:
`ResolveModelArgument` is file local, so `server_main.cpp:840` fails with
`error: 'ResolveModelArgument(Args&)' defined but not used
[-Werror=unused-function]` and the exit status is 1. A reviewer who ran the
deletion got a stale binary and a false green, which is exactly the trap this
repository has already recorded. `if (false) ResolveModelArgument(args);` is one
changed line, compiles at exit 0, and reddens the end-to-end case at
`test_serve_hf_model.cpp` `REQUIRE(healthy)` with an empty hub-path list,
because the loader then opens `tiny/llama` as a relative directory.

Each mutation run prints the compiler exit status and `git diff --stat` beside
the test result. A mutation that fails to build and a mutation that never
applied both read as a passing test, and this tree has recorded both.

TLS gates, which the hermetic tests cannot give:

1. `scripts/validate-container-image.py` boots the image with
   `--model does-not-exist/nope` and asserts the failure is an HTTP 404 from the
   hub, not the message that names the build options. A symbol check would pass
   on a build where the option resolved `OFF`.
2. One opt-in online test that fetches a real repository, following the pattern
   in `tests/tools/test_online_gate_server_binary.py`. It does not run in the
   default continuous integration lane.

Gate hygiene for this repository's recorded doctest traps: the focused run
asserts a non-zero case count, reads the `Status:` line rather than grepping
`assertions:`, and uses no `-tc` filter containing a comma.

## Dependencies

| Dependency | State |
|---|---|
| OpenSSL 3.0 or later development files on the build host | Present on the glibc release lanes and the container builders |
| `libssl3` in the runtime image | Added by this row |
| BoringSSL through `FetchContent` | Opt-in, network access at configure time |
| vLLM pin `5559679229` | Present and verified 18 August 2026 |
| llama.cpp stock tag `b10451` | Read by W1 on 18 August 2026 at commit `10bf611e533d81f739128304991c5e133c6aebd8` |
| Issue [#1281](https://github.com/mudler/vllm.cpp/issues/1281) | Depends on this row. Not a dependency of it |

## Work breakdown

| Stage | Content | Exit condition |
|---|---|---|
| W1 | Check out stock `b10451`. Re-verify every llama.cpp anchor in `## Upstream chain` and correct the table | The table cites `b10451` line numbers |
| W2 | `hf_hub` and `hf_cache`, with the fake-hub fixture. Move `model_loader.cpp:279-303` into `hf_cache` | Cache and protocol cases green, DFlash path unchanged |
| W3 | `downloader`: `HEAD`, resume, structural checks, lock, progress | Resume, truncation, and integrity cases green |
| W4 | `model_resolver` and the `server_main.cpp` call site | The end-to-end case green, and red when the call site is deleted |
| W5 | Build and packaging: the three options, `NOTICE`, `libssl3`, the container check | Every lane builds, and the container check distinguishes a working build from a disabled one |
| W6 | `docs/guides/hugging-face-access.md`, `docs/reference/server.md` and `docs/FEATURES.md` | The new flags, environment variables, and workflow are documented |
| W7 | Fresh review, mutation table, repair | A fresh reviewer returns `PASS` |

That plan said W1 through W7 land in one pull request, which is the repository
default when no `## Git integration` preference is recorded. It was overtaken by
what happened. W1 and W2 landed in `31f93787c` through pull request
[#1282](https://github.com/mudler/vllm.cpp/pull/1282), after four fresh reviews
returned findings on the tree-listing size rule, and W3 and W4 land in a SECOND
pull request on `row/ENG-HF-MODEL-DOWNLOAD-W3`. The split is recorded rather
than corrected, because a spec that describes a landing that did not happen is
a false record and this one is read by whoever picks up W5.

| Stage | Landed in |
|---|---|
| W1, W2 | `31f93787c`, pull request #1282 |
| W3, W4 | `row/ENG-HF-MODEL-DOWNLOAD-W3`, the second pull request |
| W5, W6, W7 | not started |

## Risks and decisions

**The one-definition-rule hazard.** `CPPHTTPLIB_OPENSSL_SUPPORT` changes the
vendored httplib header for every consumer, and this repository already uses
that header for the server. A build where some translation units define it and
others do not links clean and misbehaves at run time. Decision: set the define
on the target or on an interface target that every consumer inherits. Never set
it per file. W5 gates this with a build that includes both a server translation
unit and a fetcher translation unit.

**The server gains HTTPS as a side effect.** The same define lets
`httplib::SSLServer` compile. Decision: this row does not enable a TLS listener
and does not document one. W5 asserts the server still binds plain hypertext
transfer protocol by default.

**The `literal-static` lane.** `validate-release-archive.py:404` refuses any
dependency on that artifact. Decision: BoringSSL links fully static there, or
the lane sets `VLLM_CPP_HF_DOWNLOAD=OFF` and the binary refuses a repository
identifier with a message that names the missing feature. W5 records which of
the two the lane took.

**macOS and Windows are undecided, and do not block this row.**
`find_package(OpenSSL)` on macOS needs a root hint, and
`scripts/release_macos_metadata.py` restricts install names. httplib has no
Windows Schannel path. Decision: W5 produces a per-lane table of lane, TLS
source, and feature state. The Linux lanes that the quickstart targets do not
wait for it.

**Two divergences in the cache directory, and one restored fallback.**
`HfHubCacheDir` mirrors llama.cpp `common/hf-cache.cpp:37-63 @ b10451`,
including the passwd-database fallback at `:56-62`, which a container started
with `--user` and no `HOME` needs. Two things differ, and both are deliberate.
The value is resolved on every call rather than cached in a function-local
static, because a process that changes `HF_HOME` must see the change and the
upstream freezes the first reader's environment for the whole process. And a
host with no environment variable and no passwd home gets an empty path rather
than upstream's throw at `:63`, because callers resolve the cache directory
eagerly on paths that may never touch the cache, and every caller reads an empty
path as "this host has no cache". `LLAMA_CACHE` is that project's own name and
is not read.

**The multi-snapshot winner is now defined.** The walk that moved out of
`model_loader.cpp` returned the last entry `std::filesystem::directory_iterator`
happened to yield, and that iterator does not order its entries, so a repository
holding two revisions could resolve differently on two hosts and, after the
directory was re-written, on two runs of one command. The relocation preserved
that, and W7's repair replaced it with the newest entry holding a config.json,
with the greater path breaking a tie. "Newest" is what the comment on the
relocated function always claimed the function did. A repository with one
snapshot, which is every case the DFlash draft path has been run against,
resolves identically either way.

**The any-duplicate refusal was wrong, and review caught it.** This spec first
required refusing any listing in which two distinct files carry one object
identifier, and W2 implemented that. The fresh reviewer refuted the
requirement. `lfs.oid` is the sha256 of the contents and the non-large-file
fallback is the git blob sha1, so two byte-identical files in one repository
share one identifier by construction. A repository that ships one tokenizer
file under two names, or one shard twice, is legitimate, and the rule rejected
it. The rule also sat behind the token, so such a repository loaded anonymously
and began failing the moment `HF_TOKEN` was set, which is the wrong polarity
for an integrity check.

The operator replaced it with two narrower rules. A shared identifier whose
files disagree on SIZE is refused, because no content hash names two sizes, so
that shape is always a broken instrument and never duplicate content. An
identifier that is one character repeated is refused, because no content hash
produces one, and it is exactly what was measured. Neither rule depends on the
token. A shared identifier with an agreed size is accepted.

**The size rule then shipped with two holes, and the second review caught
both.** Issue [#1339](https://github.com/mudler/vllm.cpp/issues/1339) tracks
them, and both are repaired in the same flow.

The first shape was a DISARM. The rule kept the first entry carrying an
identifier as that identifier's owner whatever its size state, and then compared
only against an owner whose size was known, so a first entry that reported no
size silenced the rule for that identifier for the rest of the listing. Every
later entry under it was accepted at any size. `HF_ENDPOINT` is user
configurable, so one omitted `size` field on a mirror bought an unconditional
pass. The owner is now the first entry whose size the listing REPORTED, and an
entry that reports none is never an owner. One owner is enough rather than a
list of them, because size equality is transitive: two entries that disagree
with each other cannot both agree with the first.

The second shape was a distinct-path GUARD, `it->second.path != file.path`,
which exempted one path listed twice. It was pinned by nothing, because the case
named for it gave both entries one size and never reached the rule, and it
contradicted the rule's own sentence. One path that is 4096 bytes and 2048 bytes
in the same listing is self contradictory whichever entry is believed, and it is
the exact shape the rule exists to catch. The guard is removed. A repeated path
at an agreed size stays accepted, and the two cases now measure the boundary
from both sides.

The third shape was LETTER CASE, issue
[#1370](https://github.com/mudler/vllm.cpp/issues/1370), found by the fourth
fresh review. `IsHexString` accepts `A` through `F` as well as `a` through `f`,
which mirrors llama.cpp's `is_valid_oid`, and `oid_owner` was keyed on the raw
identifier with nothing folding case anywhere in the file. Two spellings of one
identifier therefore landed two keys and the rule compared nothing. Measured at
`db0af37b1`: `a.bin` at 4096 bytes under `ab234567...` beside `b.bin` at 2048
bytes under `AB234567...` was ACCEPTED, where the same pair in one case is
refused. That is the first shape's lesson again. `HF_ENDPOINT` is user
configurable, so the listing is input the hub does not have to answer
truthfully, and where one omitted field bought an unconditional pass there, one
changed letter case bought it here.

The identifier is now folded to lower case once, after the hexadecimal form is
validated and before either rule or `HfFile::oid` reads it. FOLDED rather than
REFUSED, because hexadecimal is case-insensitive by definition and both git and
the hub emit lower case, so an upper-case spelling is another spelling of one
value rather than a claim about a second object. Refusing it would reject a
mirror over a difference that means nothing, which is the false positive the
any-duplicate rule was replaced for.

The fold is applied to the identifier ITSELF and not only to the map key,
because the raw-identifier assumption reaches past the map. `HfFile::oid` left
the function in the listing's case, and `HfBlobPath` makes that value a cache
FILE NAME, so one listing would populate a different cache on a case-sensitive
file system than on a case-insensitive one such as this project's CIFS
checkpoint mounts. That half is latent rather than shipped, because `HfBlobPath`
has no production caller until W3, and it is repaired ahead of that caller
rather than left for it. `IsDegenerateOid` needed no repair, because folding a
repeated character leaves it repeated, and an upper-case degenerate identifier
is now pinned so the fold cannot move that verdict silently. The COMMIT is
deliberately not folded: no integrity rule keys on it, so a second spelling
costs at worst a second snapshot directory rather than a rule that stops firing,
and neither reference implementation folds it either.

`HfFile::size` is a `std::optional<uint64_t>` for the same reason. A plain
`uint64_t` spells a zero-byte file and an unreported size both `0`, and W3 sizes
a byte range and a resume offset from this field, so the ambiguity would have
been inherited by the downloader rather than staying inside the listing.

The size the size rule compares is the CONTENT size: the entry's top-level
`size`, falling back to `lfs.size`. `lfs.pointerSize` is the size of the pointer
file, is about 135 bytes for every shard, and would make two different shards
look equal.

This is a corrected requirement rather than a quiet narrowing, which is why it
is recorded here. The regression case for the measured event survives it: the
`Lightricks/LTX-2.5` fixture is refused by the degeneracy rule, and its shards
are given different sizes so that the size rule would catch it independently.

**A range request answered 200 is REFUSED, where llama.cpp warns.** llama.cpp
`common/download.cpp:222-235 @ b10451` logs "server did not respond with 206"
and carries on. W3 throws. A `200` carries the WHOLE body, and appending a whole
body onto a partial file writes the first N bytes twice. On a safetensors the
data-end rule catches the result, and on a format with no structural proof it is
a silently corrupt weight that a token gate cannot see, because the model still
emits tokens. The refusal costs one re-run and names the partial file to delete.
The refusal is taken in the RESPONSE HANDLER rather than after the transfer,
which is a second decision the same case pinned: with the check after the call,
the suite measured a 12 byte partial file grown to 48 bytes under a refusal that
still reported the right reason.

**The end-to-end case leaked a listening socket, and the retry loop that hid it
is gone.** This paragraph previously blamed box contention at load 32 and
scheduler starvation, and said that whether a completion can genuinely hang
under contention was not settled by this work. That was WRONG, the cause was
inside this row's own diff, and it is recorded here because a committed spec is
the binding record and a false cause sends the next reader into the serving
core, which this row does not touch.

`FreePort` in `tests/vllm/entrypoints/openai/test_serve_hf_model.cpp` bound an
ephemeral port with an `httplib::Server` and called `probe.stop()`. That
released nothing. `Server::stop()` at `third_party/httplib/httplib.h:11460` is
guarded by `if (is_running_)`, the function never called `listen_after_bind()`,
so `is_running_` was false and the socket was never shut down, and
`Server::~Server()` is `= default` and does not close it either. httplib sets
`SO_REUSEPORT` at `httplib.h:9455`, so the forked child bound the SAME port
successfully and the kernel load balanced inbound connections between the
orphan, which nobody ever accepted on, and the real server.

Measured with `ss -ltnp` during an unmutated run at `e25d3d344`: two LISTEN rows
on one port, one owned by the parent and one by the child, the parent's carrying
`Recv-Q 1`, which is the hung request sitting in a backlog nobody drains.

| Variant | wall, one binary, one box | verdict |
|---|---|---|
| `e25d3d344`, three runs at load 12 to 15 | 90.4 s, 240.4 s, 180.4 s | one FAILED |
| `FreePort` closing its socket, five runs at load 20 to 25 | 0.12 to 0.15 s | five passed |

Every stall was an exact multiple of a client read timeout, 30 s health and
60 s completion, never an intermediate value. A contended box gives a
distribution, not quantised multiples of the client's own timeout, and the
repaired case is FASTER at load 25 than the committed one was at load 13.

`FreePort` now uses a POSIX `socket`, `bind`, `getsockname`, `close`, because
the close has to be unconditional. The three-attempt retry loop is DELETED: with
the leak gone there is nothing left for it to absorb, and sixty seconds of dead
time read as a green run with no output at all. The serving core was cleared by
inspection in the same pass, and every wait on the path is a predicate-guarded
`condition_variable::wait`, at `httplib.h:10444`, in
`include/vllm/v1/engine/core_proc.h`, and at
`src/vllm/v1/engine/output_processor.cpp:48`.

`--max-num-seqs 4` stays on the command line, and the reason recorded for it was
also wrong. The worker-pool arithmetic is true, the default 32 does start a
36-thread pool for one four-token request, but that is not why the case was
slow, and the comment now says so.

**`--revision` and `--download-dir` were parsed and never reached.** Both
`ParseArgs` branches could be deleted with `test_downloader`, `test_model_resolver`
and `test_serve_hf_model` all still green, because the two cases named for those
flags set `ModelResolveOptions` fields by hand and never touch a flag. That is
`.agents/reachability.md`'s own sentence: a unit test that constructs the type by
hand proves the class works, never that anything reaches it. The end-to-end case
now carries both flags and they are LOAD BEARING. The fake hub publishes the
checkpoint on a non-default branch and lists nothing under the commit `main`
names, so a run without `--revision` fetches nothing and never binds, and the
cache is asserted to land under `--download-dir` and NOT under `HF_HOME`.

**A blob is named by the object identifier, else by the COMMIT and the path.**
The `## Port map` above says "by the entity tag from the resolve response or by
a locally computed sha256". W3 keeps the object identifier as the first choice
and replaces both fallbacks with the commit and the path, for two reasons. An
entity tag is a transport artifact that a mirror may respell without the bytes
changing, and naming a cache file after one costs a `HEAD` request per file on a
WARM cache, where the correct number of requests is zero and this row has a case
asserting it. A locally computed sha256 has to be computed from the whole file,
and the tree's one sha256 (`kv_cache_utils.h:175`, which the tree may not have a
second of) takes a `std::string`, so a 60 GB shard would be hashed in memory. A
commit plus a repository-relative path already names exactly one byte sequence,
which is the property a content hash was wanted for. The choice the run made is
printed under `--verbose`.

**A null grep is not absence.** This spec states that no document pins a runtime
dependency list, based on a grep of `docs/RELEASES.md`, `docs/BUILD.md`, and
`scripts/check-release-binary-contract.py`. That grep proves the search terms
wrong, not the fact. W5 confirms by building and running the archive validator.

**The GGUF form diverges from vLLM by design.** vLLM has no `org/repo:QUANT`
form. This is a tracked exception under the secondary-oracle rule, not a silent
divergence. `docs/guides/hugging-face-access.md` states which upstream defines
which form.

## Owed

- ModelScope resolution, `vllm/transformers_utils/repo_utils.py:239`.
- `--tokenizer-revision` and `--code-revision`, `vllm/config/model.py:186,190`.
- LoRA adapter fetch, `vllm/lora/utils.py:346`.
- The llama.cpp Docker-registry model path, `common/download.cpp:847`.
- **A `.bin`-only repository is fetched and then refused by the loader.**
  `SelectWeightFiles` mirrors vLLM's format preference, `["*.safetensors",
  "*.bin"]` with the first matching pattern winning
  (`default_loader.py:167-184`), so a repository that publishes only
  `pytorch_model.bin` is downloaded in full and then refused by `LoadShards`
  with `no *.safetensors shards found`. vLLM loads that format and this engine
  does not, so mirroring the preference is right and the wasted transfer is
  the cost of it. Refusing after phase one, before the weights, would save the
  bandwidth and is what the two-phase fetch exists for, but it needs its own
  case and its own message. Row `ENG-HF-MODEL-DOWNLOAD`, issue
  [#1280](https://github.com/mudler/vllm.cpp/issues/1280).
- **`vllm-cli` and the C ABI do not resolve a repository identifier.** W4 wires
  the resolver into `server_main.cpp` only. `examples/cli/main.cpp` is an ABI
  client and this row adds no ABI function, so `vllm-cli --model org/repo` and
  `vllm_model_load("org/repo")` still take a local path and refuse anything
  else exactly as they did. Closing it moves the public surface, so it needs its
  own row. Row `ENG-HF-MODEL-DOWNLOAD`, issue
  [#1280](https://github.com/mudler/vllm.cpp/issues/1280).
- The macOS and Windows TLS lanes, resolved as a table in W5.
- The quickstart page, issue
  [#1281](https://github.com/mudler/vllm.cpp/issues/1281).
- ~~**Wiring `hf_hub` to a production entry point.**~~ CLOSED by W4. The
  paragraph below records why the debt existed. `HubResolveCommitCached`,
  `HubListRepoFiles`, `HfBlobPath`, `HfSnapshotPath` and
  `HfFinalizeSnapshotEntry` are now all reached from
  `server_main.cpp`'s `--model` branch through `model_resolver.cpp`, and
  `tests/vllm/entrypoints/openai/test_serve_hf_model.cpp` turns red when that
  call site is deleted. `HfReadRef` and `HfWriteRef` are reached through
  `HubResolveCommitCached` on the same path.

  W2 landed `hf_hub` reached only by its own suite. Of `hf_cache`, only `ResolveCachedSnapshotDir` is
  reached, through the DFlash draft path in `model_loader.cpp`, and
  `tests/vllm/entrypoints/test_dflash_draft_hf_cache.cpp` gates that reach by
  entering the loader with a repository identifier. `HfReadRef`, `HfWriteRef`,
  `HfBlobPath`, `HfSnapshotPath` and `HfFinalizeSnapshotEntry` have no
  production caller at all, and neither does any hub call. W3 gives the
  snapshot-entry and blob surface its first caller, and W4 wires the hub through
  `model_resolver` and `server_main.cpp:433`. Row `ENG-HF-MODEL-DOWNLOAD`,
  issue [#1280](https://github.com/mudler/vllm.cpp/issues/1280).
- **The DFlash draft path does not honor `HF_HOME`.** `ResolveDflashDraftDir`
  reads `$HOME/.cache/huggingface/hub` and passes that root to the shared
  resolver, because W2 is a relocation and must not change what a DFlash run
  resolves. A container that sets `HF_HOME=/cache` therefore still gets the home
  directory for the draft. Migrating it onto `HfHubCacheDir()` is a behavior
  change and needs its own gate. Row `ENG-HF-MODEL-DOWNLOAD`, issue
  [#1280](https://github.com/mudler/vllm.cpp/issues/1280).

## Now

State `READY`, and W1 through W4 have landed. W3 added
`src/vllm/transformers_utils/downloader.{h,cpp}`: a `HEAD` probe for size,
entity tag and range support, a `Range: bytes=N-` resume that REFUSES a `200`
answer rather than appending it, a `.incomplete` temporary file renamed only
after the structural proof passes, the safetensors data-end rule and the GGUF
proof through the tree's own reader, a per-repository `flock` beside the
repository directory, progress under `--verbose`, and `SIGINT` cancellation that
keeps the partial file so the next run resumes.

W4 added `src/vllm/transformers_utils/model_resolver.{h,cpp}` and the call site
in `src/vllm/entrypoints/openai/server_main.cpp`, which is what makes
`--model org/repo` and `--model org/repo:Q4_K_M` fetch and serve. `--revision`
and `--download-dir` are the vLLM flags rather than an invented inline syntax.
The fetch is two phase, configuration JSON before weights, and index driven, so
a published checkpoint's `original/` copy is never requested.

Three suites cover W3 and W4. `tests/vllm/transformers_utils/test_downloader.cpp`
and `tests/vllm/transformers_utils/test_model_resolver.cpp` run against
in-process fake hubs, and `tests/vllm/entrypoints/openai/test_serve_hf_model.cpp`
is the REACHABILITY gate: it enters at `VllmServerMain(argc, argv)` with
`--model tiny/llama --revision serving --download-dir <dir>`, and the server
fetches the committed `llama_embed_e2e` checkpoint from the fake hub, boots, and
completes a `/v1/completions` request. Neutralising the resolver call site in
`server_main.cpp` with `if (false)` leaves the server unable to bind and turns
that case red. DELETING the call instead does not compile, and the mutation
table above records why.

A FIFTH FRESH REVIEW returned FAIL on six findings, and every one is repaired at
this head. The end-to-end case leaked a listening socket and the retry loop hid
it, both recorded under `## Risks and decisions` with the measurement.
`--revision` and `--download-dir` were parsed and never reached, and the
end-to-end case now carries both and reddens when either `ParseArgs` branch is
deleted. The recorded resolver mutation did not compile, and the table now
records the one that does. The two no-TLS cases were wrapped in
`#ifndef CPPHTTPLIB_OPENSSL_SUPPORT` and would have become zero-assertion skips
the moment W5 defines that macro, so the preprocessor now selects WHICH
statement each case makes rather than whether it makes one. Preprocessing both
files with the macro defined yields zero surviving assertions at `e25d3d344` and
the TLS arm at this head. And the object-identifier preference in `BlobNameFor`
was unpinned, so `test_model_resolver.cpp` now gives one listed file an
identifier and asserts the two blob names apart.

#1491 landed while this branch was open and split `docs/USAGE.md` into
`docs/guides/`, `docs/models/` and `docs/reference/`. The merge takes that file
unchanged from `origin/main` and re-applies this row's documentation where the
split put it: the fetch, the cache layout and the limits in
`docs/guides/hugging-face-access.md`, and the `--model`, `--revision` and
`--download-dir` rows in `docs/reference/server.md`.

W5, W6 and W7 have not been done: there is no transport layer security option,
no `docs/FEATURES.md` entry, and no fresh review.

The paragraph below records the state W2 left, and is kept because the size-rule
findings it names are what the current head's integrity rules are built from.

W1 and W2 landed first. W1 corrected the llama.cpp anchor
table onto stock tag `b10451`. W2 landed `hf_hub` and `hf_cache` under
`src/vllm/transformers_utils/`, with `include/vllm/transformers_utils/`
headers, and moved the DFlash draft path's copy of the cache walk onto the
shared `ResolveCachedSnapshotDir`. Three suites cover them:
`tests/vllm/transformers_utils/test_hf_cache.cpp`,
`tests/vllm/transformers_utils/test_hf_hub.cpp` against an in-process fake hub,
and `tests/vllm/entrypoints/test_dflash_draft_hf_cache.cpp`, which enters the
production loader with a repository identifier so that deleting the call site
turns it red.

A second fresh review returned FAIL on the tree-listing size rule, and the two
holes it found are repaired under issue
[#1339](https://github.com/mudler/vllm.cpp/issues/1339): the rule could be
disarmed by an entry that reported no size, and it exempted one path listed
twice.

A fourth fresh review returned FAIL on a third hole in the same rule and on the
header that describes it. Both are repaired at this head. The size rule was
evadable by LETTER CASE, issue
[#1370](https://github.com/mudler/vllm.cpp/issues/1370), and the identifier is
now folded to lower case before either rule or `HfFile::oid` reads it. The
header still stated the superseded distinct-path form of rule 1, and still
carried an orphan fragment left by the edit that removed the any-duplicate rule.
Rule 1 now says what the code does, the fragment is gone, and the fold is
documented on the function and on `HfFile::oid`. `test_hf_hub.cpp` now carries
39 cases and 154 assertions.

The row stays `READY` rather than moving to `ACTIVE`, because an `ACTIVE` row
needs a `CLAIM-*` owner recorded in a claim source and that is the operator's
record to write, not an implementer's.
