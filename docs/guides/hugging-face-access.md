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

This is the SHAPE of a run on a machine that holds no checkpoint. The repository
name is an illustration: no fetch from the live hub has been gated in this tree
yet, because this build speaks no transport layer security and the hermetic
suites run against an in-process fake hub. The online gate is owed by W5 of
[the Hugging Face download specification](../../.agents/specs/hf-model-download.md).

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

A repository listing is refused, rather than partly used, when it fails either
of two integrity checks. An object identifier given to two entries that disagree
on size is refused, because no content hash names two sizes. An identifier whose
characters are all one repeated character is refused, because no content hash
produces one, and an unauthenticated listing for a gated repository is the shape
that produces it. An entry that reports no size can never own an identifier, so
a mirror named by `HF_ENDPOINT` cannot switch the check off by omitting one
field.

## Current limitations

- This build speaks plain hypertext transfer protocol only, so an `https`
  endpoint is refused with a message naming the build options that would add
  transport layer security rather than with a connection error. Point
  `HF_ENDPOINT` at an `http` mirror until that lands.
- `vllm-cli` and the C ABI do not resolve a repository identifier. Both still
  take a local path.
- The DFlash draft resolver reads `$HOME/.cache/huggingface/hub`. It does not
  honor `HF_HOME`.

All three are recorded under `## Owed` in
[the Hugging Face download specification](../../.agents/specs/hf-model-download.md),
along with the integrity checks and the remaining work.
