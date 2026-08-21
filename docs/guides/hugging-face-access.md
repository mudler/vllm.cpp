# Access Hugging Face checkpoints

`--model` takes four forms, and leaves anything else alone. The two local forms
are probed FIRST, so a network call can never shadow a path that exists on
disk.

| `--model` value | What happens |
|---|---|
| a directory | Opened as it always was. No network |
| a `.gguf` file | Opened as it always was. No network |
| `org/repo` | The checkpoint is fetched into the Hugging Face cache and the snapshot directory is served |
| `org/repo:Q4_K_M` | ONE GGUF file for that quantization is fetched and served |
| anything else | The error it produced before, unchanged |

`org/repo` mirrors vLLM, which is the only upstream that defines it.
`org/repo:QUANT` is llama.cpp's form and vLLM does not implement it, so it is a
tracked divergence rather than a silent one.

Two flags go with them, spelled as vLLM spells them. There is deliberately no
inline `org/repo@revision` syntax, because vLLM does not have one.

| Flag | Meaning |
|---|---|
| `--revision <ref>` | A branch, a tag, or a 40 character commit. Applies to both hub forms |
| `--download-dir <path>` | The directory that holds the `models--org--repo` folders. Overrides the cache root below |

This is a run on a machine that holds no checkpoint.

```sh
vllm-server --model Qwen/Qwen3-0.6B --port 8000
curl localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","prompt":"hello","max_tokens":16}'
```

The name a client puts in `"model"` is the name you typed, not the commit
directory the cache happens to hold.

**`vllm-server` is the only entry point that resolves a repository identifier.**
`vllm-cli` and the C ABI's `vllm_model_load` still take a local path, because an
example is an application binary interface client only and this work adds no ABI
function.

## How a fetch runs

The fetch is TWO PHASE. The configuration JSON, the tokenizer and the shard
index come first, and the weights follow, so a repository that is not a model
fails after a few hundred kilobytes rather than after sixty gigabytes. It is
also INDEX DRIVEN: when the repository ships a `model.safetensors.index.json`,
the exact file names in its `weight_map` are fetched, so a published
checkpoint's duplicate-format `original/` directory is never requested. Where
there is no index, `*.safetensors` is preferred over `*.bin` and only the first
of the two that matches is fetched.

A transfer resumes. A partial file is kept as `{blob}.incomplete` and the next
run asks for `Range: bytes=N-`. **A server that answers that request with `200`
and the whole file is refused, not appended to**: appending a whole body to a
partial file writes the first N bytes twice, and a corrupt weight still emits
tokens, so nothing downstream would notice. Delete the named `.incomplete` file
and run again to start over. `Ctrl-C` cancels and keeps the partial file.

Nothing takes the name of a finished file until it has proven itself from its
own bytes: a safetensors satisfies `8 + header_len + max(data_offsets[1]) ==
file_size`, and a GGUF is opened by this project's own reader, which validates
every tensor span against the real file size. A `Content-Length` that matches is
not accepted as proof on its own.

`--verbose` prints one line per file, and one lock is taken per repository so
two servers started at once against one cache do not write one blob twice.

## Environment

The library reads these Hugging Face environment variables when it resolves a
cache entry:

| Variable | Effect |
|---|---|
| `HF_TOKEN` | Bearer token for a private or gated repository |
| `HF_TOKEN_PATH` | File that contains the token; used when `HF_TOKEN` is unset |
| `HF_ENDPOINT` | Alternate Hub host; the library adds a missing trailing slash |
| `HF_HUB_OFFLINE` | Resolve files from the cache without opening a network connection |
| `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, `HF_HOME`, `XDG_CACHE_HOME`, `HOME` | Cache root, in priority order; `HF_HOME` contributes `$HF_HOME/hub` |

## Cache layout

The resolver reads and writes the standard Hugging Face cache layout:

```text
{hub}/models--org--repo/
├── refs/
├── blobs/
└── snapshots/{commit}/{path}
```

If the repository has multiple cached snapshots, the resolver selects the most
recently written snapshot. A cache another tool already populated is read rather
than re-downloaded.

Where the file system holds no symbolic link, which is the case for a CIFS mount
and can be the case for a `/cache` container volume, a snapshot entry becomes a
real file, and the switch is logged one time for each cache directory it happens
in.

A cached blob is named by the object identifier the listing carried, and by the
commit plus the repository-relative path when it carried none. `--verbose` says
which of the two the run used.

## Working offline, and reusing a cache

A cache another tool already populated is READ, not re-downloaded. A host that
has run Python `huggingface_hub` against the same repository gets a hit and
opens no socket, because the layout above is the one that library writes.

`HF_HUB_OFFLINE=1` resolves from the cache and opens no socket at all. On a
warm cache the run behaves exactly as it would online. On a cold one it refuses,
and it names the directory it searched so you can see which cache root the run
actually chose:

```text
vllm.cpp: HF_HUB_OFFLINE is set and repository 'org/repo' has no recorded 'main'
reference under /home/you/.cache/huggingface/hub/models--org--repo. Fetch it
once with HF_HUB_OFFLINE unset, or point HF_HOME at a cache that already holds
it.
```

## Two refusals you may see

A repository listing is refused, rather than partly used, when it fails either
of two integrity checks. Both run whether or not you set a token.

An object identifier given to two entries that disagree on SIZE is refused,
because no content hash names two sizes:

```text
vllm.cpp: the tree listing for repository 'org/repo' gives object identifier
ab23... to 'a.bin' at 4096 bytes and to 'b.bin' at 2048 bytes. One content hash
cannot name two different sizes. The listing is not usable.
```

An identifier whose characters are all ONE repeated character is refused,
because no content hash produces one:

```text
vllm.cpp: the tree listing for repository 'org/repo' gives 'model.safetensors'
the object identifier aaaa... , whose 64 characters are all 'a'. No content hash
produces that value, so the hub is not answering the truth about this
repository. The listing is not usable.
```

That second one is not hypothetical. On 17 August 2026 the tree API answered an
unauthenticated caller on a gated repository with exactly that value, identical
for all fourteen of its large files.

Two entries that share an identifier AND agree on size are ACCEPTED: that is
duplicate content, which a repository is allowed to have. An entry that reports
no size can never own an identifier, so a mirror named by `HF_ENDPOINT` cannot
switch the check off by omitting one field.

## Transport layer security

`huggingface.co` speaks `https` and nothing else, so the fetch needs a
transport-layer-security library and a certificate trust store. Three build
options decide whether the binary you have carries one.

| Option | Default | What it does |
|---|---|---|
| `VLLM_CPP_HF_DOWNLOAD` | `ON` | The feature. Resolves `OFF`, with a configure-time warning, when no TLS library is available |
| `VLLM_CPP_OPENSSL` | `ON` | Links the system OpenSSL. Needs the development files, version 3.0 or later |
| `VLLM_CPP_BUILD_BORINGSSL` | `OFF` | **UNTESTED.** Fetches and statically links BoringSSL instead. Reaches the network at configure time |

`cmake` prints which one answered:

```text
-- HuggingFace download: HTTPS through OpenSSL 3.0.13 (system, dynamic)
```

A build that carries no transport layer security still builds, still serves a
local checkpoint, and REFUSES an `https` endpoint with a message that names
those three options rather than with something that reads like a network fault:

```text
vllm.cpp: this build cannot speak HTTPS, so it cannot reach
https://huggingface.co/. Rebuild with -DVLLM_CPP_HF_DOWNLOAD=ON and one of
-DVLLM_CPP_OPENSSL=ON (default, needs the OpenSSL development files) or
-DVLLM_CPP_BUILD_BORINGSSL=ON, or set HF_ENDPOINT to an http:// mirror.
```

Where each shipped lane stands. The two that carry no transport layer security
do so deliberately, and both refuse the way shown above:

| Lane | TLS source | `--model org/repo` |
|---|---|---|
| Container images, and every glibc release archive | system OpenSSL, dynamic | works |
| A local build on a host with no OpenSSL development files | none | refused, naming the options |
| `linux-x86_64-musl-cpu-static` | none | refused, naming the options |

`VLLM_CPP_BUILD_BORINGSSL=ON` has NEVER BEEN COMPILED here. The option is
implemented and it is offered, and no build in this repository has ever taken
it, so no lane ships it and nothing measures it. Treat it as untested code and
expect to debug the configure-time fetch yourself. `VLLM_CPP_OPENSSL=ON`, the
default, is the path every release lane and every container image uses.

The static musl archive is the deliberate one. Its contract forbids ANY dynamic
dependency, and a statically fetched BoringSSL would make the archive depend on
a network fetch at configure time, so that lane sets
`VLLM_CPP_HF_DOWNLOAD=OFF`. Use a glibc archive or the container image to fetch
from the hub.

**This adds no `https` LISTENER.** The same build option makes one compilable,
and `vllm-server` still binds plain hypertext transfer protocol on `0.0.0.0:8000`
exactly as before. Put a reverse proxy in front of it if you need TLS on the
serving side.

The container image carries all three parts: `libssl-dev` in the builder stage,
which is what makes the server LINK OpenSSL at all, then `libssl3` for the
library at run time and `ca-certificates` for the trust store. Naming only the
last two was issue [#1517](https://github.com/mudler/vllm.cpp/issues/1517): the
images shipped the runtime half without the build half, so `ldd` on the server
listed no `libssl` and `--model org/repo` refused every repository while every
gate stayed green. `scripts/check-build-runtime-deps.py` now refuses a builder
stage or release lane that compiles the server without the development files.

## Current limitations

- `vllm-cli` and the C ABI do not resolve a repository identifier. Both still
  take a local path.
- The DFlash draft resolver reads `$HOME/.cache/huggingface/hub`. It does not
  honor `HF_HOME`.
- A repository that publishes only `pytorch_model.bin` is fetched in full and
  then refused by the loader, which reads safetensors and GGUF.
- macOS and Windows have no resolved TLS lane. `find_package(OpenSSL)` on macOS
  needs a root hint and the release install-name rules restrict what may be
  linked, and cpp-httplib has no Windows Schannel path, so a Windows build
  needs an OpenSSL to point at. Both are recorded as owed.
- There is no gated fetch of a REAL checkpoint from the live hub in the default
  test lane. The hermetic suites run against an in-process fake hub, and the
  one instrument that exercises a live `https` session is
  `scripts/validate-container-image.py`, which boots the image against an
  unknown repository and requires an answer from the hub.

Every one is recorded under `## Owed` in
[the Hugging Face download specification](../../.agents/specs/hf-model-download.md),
along with the integrity checks and the remaining work.
