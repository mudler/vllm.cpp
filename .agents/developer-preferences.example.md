# Developer preferences (example)

Copy this file to `.agents/developer-preferences.md` and edit it for the current
developer and workspace. The destination is intentionally ignored by Git.
`AGENTS.md` defines project invariants; this file controls operational choices
that legitimately differ between developers. It cannot relax correctness,
testing, evidence, attribution, or lifecycle requirements.

Delete headings that do not apply, but make every allowed remote or destructive
action explicit. An absent answer uses the safe default documented in
`AGENTS.md`.

For protocol placeholders, provide these values explicitly (use `unavailable`
rather than borrowing another developer's path):

- `${VLLM_SOURCE}`: `<absolute checkout path or unavailable>`.
- `${VLLM_ORACLE}`: `<absolute executable/venv path or unavailable>`.
- `${DEPENDENCY_SOURCE}`: `<absolute source/site-packages root or unavailable>`.
- `${GPU_LOCK}`: `<whole-series exclusion command or unavailable>`.

## Git integration

- Commits: allowed.
- Base ref: `upstream/main`.
- Working branch: create or reuse a feature branch; do not work on local
  `main`.
- Fetch: allowed from `<read-only remote>`.
- Push: ask first; if allowed, name the remote and permitted ref namespace.
- Merge to `main`: not allowed unless explicitly requested for the current
  task.
- Force-push or local ref rewrite: ask first.
- Pull requests and CI inspection: ask first.

## Workspace and upstream oracle

- Repository root: `<absolute path>`.
- vLLM source checkout: `${VLLM_SOURCE}` above.
- vLLM oracle executable/venv: `${VLLM_ORACLE}` above.
- Dependency source/site-packages: `${DEPENDENCY_SOURCE}` above.
- Build directories: `<paths or naming rule>`.
- Model/cache roots: `<paths or unavailable>`.
- Evidence root: `<path>`.

## Compute and benchmarks

- Available hosts: `<local only, or explicit SSH targets>`.
- Available backends/accelerators: `<CPU/CUDA/Metal/etc.>`.
- GPU architecture and memory: `<facts>`.
- Gate models runnable here: `<models or none>`.
- Contention policy: `${GPU_LOCK}` above plus when it is required.
- Profilers: `<nsys/ncu/torch profiler/etc.>`.
- External services that must be stopped: `<service and authorized action, or
  none>`.
- Large downloads, package installation, or service management: ask first.

## Collaboration

- Parallel agents: `<allowed, unavailable, or only when explicitly requested>`.
- Claims/worktrees: `<required for parallel roadmap work, or local rule>`.
- Shared build directories: `<policy>`.

## Notes

Record any persistent developer preference that would otherwise require a
session-by-session question. Do not put credentials, tokens, private keys, or
other secrets in this file.
