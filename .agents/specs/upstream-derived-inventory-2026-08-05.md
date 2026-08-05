# Upstream-derived inventory — what actually needs porting

User-directed 2026-08-05: stop asking which stale claim families are dead and
DERIVE the answer from the reference implementations. This is the first pass.

All three references are present locally and verified at the exact pins the
record claims, so every finding below is grounded, not remembered:

| Reference | Checkout | Pin |
|---|---|---|
| vLLM | `/home/mudler/_git/vllm` | `555967922` (our parity pin) |
| SGLang | `/home/mudler/_git/sglang` | `f63458b5be` (release/v0.5.15) |
| llama.cpp | `/home/mudler/_git/llama.cpp` | `237ad9b96` |
| ds4 / DwarfStar | NOT on this box | ref is `dgx:~/w8run/ds4/` |

## Scope

Answer, from upstream code, three questions the stale record could not:
which of our open rows are OBSOLETE, which are REAL unported work, and what
upstream has that we never inventoried at all.

## Our baseline — finding 1: three arch rows are BELOW vLLM's floor

vLLM's `CMakeLists.txt:120-129` sets `CUDA_SUPPORTED_ARCHS` to
`7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.1;10.3;11.0;12.0;12.1`. **Its floor is 7.5.**

We carry rows for `BACKEND-CUDA-SM060`, `SM061` and `SM070` — all below that
floor, none supported by vLLM at our pin. Under the standing MIRROR-vLLM
directive these are not open work; they are out of scope, and that is now a
code-derived fact rather than a judgement call. Every other arch row we carry
(`SM075` through `SM121`) maps to an arch vLLM does support.

**Disposition:** the three sub-floor rows are the first claim family that can be
closed without asking anyone.

## Finding 2 — the component rows are REAL unported work, not dead

Each was checked for existence in vLLM's `csrc/` at our pin:

| Row | Upstream files | csrc refs | Verdict |
|---|---:|---:|---|
| `BACKEND-CUDA-COMP-MACHETE` | 12 | 13 | real, unported |
| `BACKEND-CUDA-COMP-W4A8` | 13 | 9 | real, unported |
| `BACKEND-CUDA-COMP-MLA` | 39 | 14 | real, unported |
| `BACKEND-CUDA-COMP-ALLSPARK` | 6 | 5 | real, unported |
| `BACKEND-CUDA-COMP-SCALEDMM-C2X` | 6 | 5 | real, unported |
| `BACKEND-CUDA-COMP-MOE-CUTLASS` | 3 | 6 | real, unported |
| `BACKEND-CUDA-COMP-FLASHMLA` | 5 | 0 | real, but a SEPARATE package, not csrc |
| `BACKEND-CUDA-COMP-DEEPGEMM` | 0 | 0 | NOT in vLLM at this pin — external dep |

So the earlier read that these rows were "verified no implementation, therefore
closeable" was RIGHT about our side and WRONG about the conclusion: they have no
implementation HERE precisely because they are unported upstream work. They stay
open. Only `DEEPGEMM` needs re-scoping, since it is not vLLM code at all.

## Finding 3 — all five parallelism modes are real upstream

`pipeline_parallel` (11 files), `expert_parallel` (3), `sequence_parallel` (9),
`tensor_parallel` (16), `data_parallel` (20) all exist in `vllm/distributed` and
`vllm/config`. None of the `BACKEND-DISTRIBUTED-*` rows is obsolete.

## Finding 4 — 43 upstream architectures we never inventoried

vLLM's registry defines **362** architectures. Our model matrix names **319** of
them, leaving **43 never inventoried at all** (the first hand count said 62; the
regex behind it missed `ForMaskedLM`/`ForRetrieval` suffixes, and W1's tooling
corrects it) — a coverage gap invisible to the
existing record because the record only tracks rows we created. The missing set
is dominated by embedding/retrieval and encoder models: `BertForMaskedLM`,
`BertForSequenceClassification`, `BertForTokenClassification`,
`ColModernVBertForRetrieval`, `ColPaliForRetrieval`, `ColQwen3`, `ColQwen3_5`,
`GritLM`, `Cheers`, `Exaone4_5_MTP` and others.

This is the most valuable finding: the backfill was trying to make 79 existing
rows honest while 43 upstream architectures had no row at all.

## Port map

1. close the three sub-floor arch rows as OUT-OF-SCOPE (mirror-vLLM);
2. re-scope `COMP-DEEPGEMM` as an external dependency, not a vLLM component;
3. inventory the 43 missing architectures, classified by family and by whether
   they are generative, embedding or retrieval;
4. leave every other open row alone — upstream proves the work is real.

## Tests to port

None yet; this pass is inventory. Each row that graduates carries its upstream
tests per the standing directive.

## Gates

The enumeration is REPRODUCIBLE, not a one-off reading:
`scripts/upstream-inventory.py` emits this table from the three checkouts,
snapshots it to `.agents/upstream-inventory.json`, and CI fails when the
uninventoried count or vLLM's supported-arch list drifts. `--check` SKIPS
cleanly when the checkouts are absent rather than pretending to have verified
something, because CI has no vLLM checkout and a silent pass there would be a
lie.

## Dependencies

The three reference checkouts at the pins above. ds4 is absent from this box, so
its comparison is deferred rather than guessed.

## Work breakdown

| W | Item |
|---|---|
| W1 | **LANDED** `scripts/upstream-inventory.py` + 10 mutation tests; snapshot at `.agents/upstream-inventory.json`, drift-checked in CI |
| W2 | Close the 3 sub-floor arch rows; re-scope DEEPGEMM |
| W3 | Inventory the 43 missing architectures into `model-matrix.md` |
| W4 | Repeat the enumeration against SGLang and llama.cpp for their distinct surfaces |

## Risks/decisions

- **Risk: the 43 are mostly embedding/retrieval models** we may not want in
  scope. Inventorying is not committing: they land `INVENTORIED`, which claims
  nothing, and the roadmap decides.
- **Risk: counting by grep overstates or understates.** File counts are a
  presence signal, not a port-cost estimate; W1 must cite `file:line` per row
  the way the matrices already require.
- **Decided:** derive from upstream rather than ask which families are dead.
  Findings 1-4 are what that produced on the first pass.
