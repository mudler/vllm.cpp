# Developer preferences (example)

Copy this file to `.agents/developer-preferences.md` and edit it for the current
developer and workspace. The destination is intentionally ignored by Git.
`AGENTS.md` defines project invariants; this file controls operational choices
that legitimately differ between developers. It cannot relax correctness,
testing, evidence, attribution, or lifecycle requirements.

Delete headings that do not apply, but make every allowed remote or destructive
action explicit. An absent answer uses the safe default documented in
`AGENTS.md`.

The machine-readable values behind the protocol placeholders
(`${VLLM_SOURCE}`, `${VLLM_ORACLE}`, `${DEPENDENCY_SOURCE}`, `${GPU_LOCK}`,
gate host, device arch/toolchain) live in the untracked `.env` at the
repository root: copy `.env.example` and fill it in. Leave a variable empty
rather than borrowing another developer's path. This file keeps the policy
choices below; `.env` keeps the paths and hosts.

## Git integration

- Commits: allowed.
- Base ref: `upstream/main`.
- Working branch: create or reuse a feature branch; do not work on local
  `main`.
- Pull request shape: `<one PR for spec and implementation (recommended), or
  separate spec and implementation PRs>`. Record the answer at row claim and do
  not ask again for that row.
- Fetch: allowed from `<read-only remote>`.
- Push: ask first; if allowed, name the remote and permitted ref namespace.
- Merge to `main`: not allowed unless explicitly requested for the current
  task.
- Force-push or local ref rewrite: ask first.
- Pull requests and CI inspection: ask first.

## Workspace and upstream oracle

- Repository root: `<absolute path>`.
- vLLM source checkout: `${VLLM_SOURCE}` from `.env`.
- vLLM oracle executable/venv: `${VLLM_ORACLE}` from `.env`.
- Dependency source/site-packages: `${DEPENDENCY_SOURCE}` from `.env`.
- Build directories: `<paths or naming rule>`.
- Model/cache roots: `<paths or unavailable>`.
- Evidence root: `<path>`.

## Compute and benchmarks

- Available hosts: `<local only, or explicit SSH targets>`.
- Available backends/accelerators: `<CPU/CUDA/Metal/etc.>`.
- GPU architecture and memory: `<facts>`.
- Gate models runnable here: `<models or none>`.
- Contention policy: `${GPU_LOCK}` from `.env` plus when it is required.
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
