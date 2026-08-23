# Git hooks

Repo-tracked hooks. They are not active until a clone opts in:

```sh
git config core.hooksPath .githooks
```

`git push --no-verify` bypasses them for one push.

## pre-push

Runs three record and public-doc gates against the **commits being pushed**, not
the working tree:

- `scripts/check-readme-structure.py`
- `scripts/check-now-current.py`
- `scripts/check-prompt-contract.py`

All three enforce shapes that otherwise surface only in CI, and a push that
breaks one turns main red for whoever pushes next.

The hook **refuses a push when it names a checker `scripts/` does not have**
(#1779). It used to skip such a name and exit 0, so it read as six gates while
running three, and a checker that went missing by accident looked exactly like
one removed on purpose. Two of the six were retired by `0f3e44eee` and the
third, `check-public-doc-tables.py`, by #1714.

`scripts/agent-preflight.sh` runs these same three checkers plus the rest of the
record gates and the tool suites, and it stays the thing to run before
committing. The hook is the backstop for the run that gets skipped: it checks
the pushed commit's tree, so a dirty checkout cannot fail a clean push and an
uncommitted fix cannot let a broken commit through.
