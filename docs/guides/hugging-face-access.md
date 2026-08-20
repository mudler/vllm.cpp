# Access Hugging Face checkpoints

Pass a local directory or a `.gguf` file to `--model`. The CLI and server do
not download a repository identifier yet.

The library reads these Hugging Face environment variables when it resolves a
cache entry:

| Variable | Effect |
|---|---|
| `HF_TOKEN` | Bearer token for a private or gated repository |
| `HF_TOKEN_PATH` | File that contains the token; used when `HF_TOKEN` is unset |
| `HF_ENDPOINT` | Alternate Hub host; the library adds a missing trailing slash |
| `HF_HUB_OFFLINE` | Resolve files from the cache without opening a network connection |
| `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, `HF_HOME`, `XDG_CACHE_HOME`, `HOME` | Cache root, in priority order; `HF_HOME` contributes `$HF_HOME/hub` |

The resolver reads the standard Hugging Face cache layout:

```text
{hub}/models--org--repo/
├── refs/
├── blobs/
└── snapshots/{commit}/{path}
```

If the repository has multiple cached snapshots, the resolver selects the most
recently written snapshot.

## Current limitations

- The CLI and server do not fetch checkpoints. Setting `HF_TOKEN` does not
  change a server request until repository downloads reach that entry point.
- The DFlash draft resolver reads `$HOME/.cache/huggingface/hub`. It does not
  honor `HF_HOME`.
- Cache-writing support has no public caller yet.

The cache writer supports filesystems without symbolic links, including CIFS
mounts and some container volumes. On such filesystems it writes a regular file
for each snapshot entry.

See the [Hugging Face download specification](../../.agents/specs/hf-model-download.md)
for implementation state, integrity checks, and remaining work.
