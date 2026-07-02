# Upstream sync

- **Reference checkout:** `/home/mudler/_git/vllm`, branch `main`
  (https://github.com/vllm-project/vllm).
- **Current sync point:** `e24d1b24` (2026-07-02).

## Procedure to advance

1. `git fetch origin main && git pull --ff-only` in the reference checkout.
2. Update the sync point above.
3. Scan `git log <old>..<new> --oneline -- vllm/v1/` (and other ported
   subtrees) for changes relevant to already-ported code.
4. Port what applies (each ported file records the upstream commit it matches
   in its header — diff against that).
5. Log the sync + notable ported PRs in [state.md](state.md).
