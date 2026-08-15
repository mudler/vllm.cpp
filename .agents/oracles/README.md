# The oracle registry — one file per oracle

Every upstream this project compares itself against has exactly one file here,
named `<id>.md`, and every one of them carries a pin. The rule this surface
serves is [`AGENTS.md`](../../AGENTS.md) §"When vLLM has no implementation":
vLLM is the primary reference, a secondary oracle is admissible only where vLLM
implements nothing, and an upstream with no recorded revision is not an oracle at
all.

**Why one file per oracle rather than a table.** A shared table is a lock: two
rows advancing two different pins in the same week conflict on lines neither
owns, and an automatic three-way merge of a keyed record is exactly what the
protocol refuses. Files are read by glob (`.agents/oracles/*.md`), so adding an
oracle touches one new path and nothing else.

**What lives here and what does not.** These files carry *identity*: which
upstream, which revision, measured when, gateable or not, and the evidence.
They do not carry methodology or results. The vLLM parity pin keeps its home in
[`../upstream-sync.md`](../upstream-sync.md) — `vllm.md` points at it rather
than restating it, because a pin transcribed twice is a pin that drifts.

## The record

Each file carries exactly one fenced ` ```oracle-pin ` block with these keys, in
any order, one `key = value` per line:

| Key | Meaning |
|---|---|
| `id` | matches the filename stem, and the `id` column of the AGENTS.md table |
| `role` | `primary` (vLLM, and only vLLM) or `secondary` |
| `upstream` | the canonical repository URL |
| `scope` | what this oracle may answer for — one line |
| `pin` | a 40-hex commit, a version string, or the literal `UNPINNED` |
| `pin_label` | the human name of that revision (`v0.5.15`, `b9892`, `none`) |
| `pinned_on` | ISO date the pin was recorded or measured |
| `gateable` | `yes` only once it demonstrably builds and runs the model |
| `evidence` | a repo path that exists when `gateable = yes`; the issue that owes the measurement (`#N`) when `gateable = no` |

`scripts/check-oracle-pins.py` enforces all of it, including that the AGENTS.md
table and this directory name the same set of ids. Run it with `--self-test` to
sweep its own fixture corpus in both directions.
