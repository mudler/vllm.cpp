# The tracked checkpoint paths name `/mnt/nas_share`, which a reboot deletes

Row: `FIX-NAS-PATH-1073`
Issue: [#1073](https://github.com/mudler/vllm.cpp/issues/1073)
Baseline: `origin/main` @ `100026481`

## 1. Scope

Point the live checkpoint defaults at the declared checkpoint root, and record
in [`../environment.md`](../environment.md) where the NAS mounts on `dgx.casa`
and why the old location cannot be restored.

**In scope:** `.agents/environment.md`, the four script and tool defaults that a
fresh run reads, the one product comment that names the source checkpoint, the
one gate whose fallback path is a literal, and the two `docs/USAGE.md` recipes a
reader copies.

**Out of scope:** every record that cites `/mnt/nas_share` as the path a past
measurement used. `.agents/benchmark-record.md`, the LTX-2.5 and Nemotron-H
specs, `.agents/model-matrix.md`, the captured goldens under
`tests/parity/goldens/`, and the generated `.inc` headers state what was read at
the time. That is provenance. AGENTS.md is explicit that rewriting an existing
file to satisfy a rule is out of scope unless a row asks for the rewrite, and
this row does not ask.

## 2. The fact, and why it is not a one-off

The NAS mounts at `/usr/local/nas_share` on `dgx.casa`. `/mnt/nas_share` is
gone. The developer confirmed that `/usr/local` is canonical.

`/mnt` sits on the ephemeral root overlay of the immutable Kairos OS, so a
directory created there does not survive a reboot. `/usr/local` is
`COS_PERSISTENT` and does survive. This is the same property that made an
earlier `/oem` `rootfs`-stage change cost a boot.

Observed 2026-08-16, after the box returned from an 8 h 19 min outage: the mount
itself came back because the `/oem` boot-stage unit worked, and `/mnt/nas_share`
did not. Every checkpoint path built on `/mnt` broke while the untracked `.env`
still declared `CHECKPOINT_ROOT=/mnt/nas_share/checkpoints`. A gate that reads a
path `.env` does not declare is not the gate its spec names, so this state
blocks any checkpoint-loading gate until a person notices.

`.env` is untracked. It was repointed by hand and verified against 18 checkpoint
directories under `/usr/local/nas_share/checkpoints`, including
`nemotron-3.5-lightning-30b-nvfp4` and `nemotron-3.5-lightning-30b-gguf`. The
tracked surfaces are the part this row repairs.

## 3. The change

`.env` declares the location once. Every tracked default now derives from
`CHECKPOINT_ROOT` instead of repeating an absolute path, so the next mount move
costs one untracked line rather than another sweep. The convention already
exists in the tree: `scripts/measure-ltx2-keyframes-meta.py:30`,
`scripts/gen-ltx2-prompt-tokens-goldens.py:28`,
`scripts/measure-ltx2-prompt-adaln.py:27` and `tools/oracle/music3_oracle.py:31`
all write `$CHECKPOINT_ROOT/…` already. The files below were the outliers.

| File | Was | Now |
|---|---|---|
| `.agents/environment.md` | the DGX profile named no NAS location at all | a profile bullet gives `/usr/local/nas_share`, the `COS_PERSISTENT` reason, and the instruction not to restore the old path |
| `scripts/gen-minimax-music3-manifest.py:17` | `--checkpoint /mnt/nas_share/checkpoints/minimax-music3` | `--checkpoint "$CHECKPOINT_ROOT/minimax-music3"` |
| `scripts/gen-ltx2-quant-goldens.py:48` | `--checkpoint-root /mnt/nas_share/checkpoints` | `--checkpoint-root "$CHECKPOINT_ROOT"` |
| `tools/parity/dump_tokenizer_gpt4o.py:36,39` | two literal paths in the by-hand recipe | `$CHECKPOINT_ROOT/…` |
| `tools/parity/dump_tokenizer_gpt4o.py:57` | `DEFAULT_TOKENIZER_JSON`, a literal | read from `CHECKPOINT_ROOT`; `--tokenizer-json` becomes required when the variable is unset |
| `tools/gen_pretok_goldens.py:57` | the regex source path | `$CHECKPOINT_ROOT/muse-glimmer-30b/tokenizer.json` |
| `src/vllm/tokenizer/pretokenizer.cpp:319` | the same source path, in a comment | the same substitution |
| `tests/parity/test_minimax_music3_quant_real.cpp:133,144` | fallback `std::string("/mnt/nas_share/checkpoints")` | no fallback; an unset root skips and names the two variables |
| `docs/USAGE.md:3069,3453` | two literal paths in copyable recipes | `$CHECKPOINT_ROOT` and `CHECKPOINT_ROOT=…`, which is what the rest of the file already writes |

The `.agents/environment.md` bullet carries the reason, not only the path. The
issue asks for that explicitly, because a bare path correction invites the next
reader to restore `/mnt/nas_share` as a convenience symlink, and that symlink
disappears at the next reboot.

The issue reports that `.agents/environment.md` documented the `/mnt` location
and a "canonical symlink". Measured against `100026481`, it documents neither:
the file holds no `/mnt/nas_share` string, and its only "canonical symlink" at
`:87` is `~/venvs/vllm-oracle`, which is the oracle venv and is unrelated. The
defect is therefore an absence rather than a wrong value, and the repair is to
add the fact.

## 4. Two paths that are deliberately left

`tests/vllm/multimodal/test_qwen3_5_moe_vl_hw.cpp:66-67` probes
`/usr/local/nas_share/checkpoints/qwen3.6-35b-a3b-bf16` FIRST and
`/mnt/nas_share/…` second. A fresh run on `dgx.casa` already resolves, so the
second entry costs one `fs::exists` call on an absent path and breaks nothing.
The fact in this row covers `dgx.casa`. It does not cover the cluster nodes, and
deleting a tolerated fallback for a host nobody measured would trade a harmless
probe for a possible refusal.

`tests/vllm/test_pretokenizer.cpp:377` records that the GPT-4o regex was
"transcribed verbatim from `/mnt/nas_share/…/tokenizer.json` into
`tools/gen_pretok_goldens.py`". That sentence is the same shape as the two
comments this row does change, and the issue lists those two and not this one.
It reads as a statement about a past transcription rather than as a pointer a
reader follows, because the pointer it gives is the generator script. Left
unchanged and reported, so the owner decides rather than an implementer guessing.

## 5. Evidence

No behavior changes except in one gate, so the gate is the accuracy of the text
and the reachability of each default.

| Claim | How it was checked at `100026481` |
|---|---|
| every `/mnt/nas_share` hit is classified | `grep -rn '/mnt/nas_share'` returns 41 lines in 26 files; each is named in §3, §4, or the out-of-scope list in §1 |
| `.agents/environment.md` holds no `/mnt` path | `grep -n '/mnt' .agents/environment.md` returns nothing |
| `$CHECKPOINT_ROOT` is an existing convention, not a new one | four sibling scripts already write it; listed in §3 |
| the gate tests do read `CHECKPOINT_ROOT` | `getenv("CHECKPOINT_ROOT")` at `tests/parity/test_minimax_music3_ar_real.cpp:162`, `_e2e_real.cpp:170`, `_llm_real.cpp:137`, `tests/vllm/models/test_ltx2_text_encoder.cpp:2299` |
| the changed gate still skips loudly | `test_minimax_music3_quant_real` builds warning-free and runs. With the root unset, 6 cases, `SKIP music3 q4_k artifact identity: VLLM_CPP_MUSIC3_GGUF and CHECKPOINT_ROOT are both unset`. With `CHECKPOINT_ROOT=/usr/local/nas_share/checkpoints`, the same 6 name the composed path `…/minimax-music3-gguf/rvq_depth_decoder_q4_k.gguf` |
| the tokenizer tool still refuses cleanly | with `CHECKPOINT_ROOT` unset and no arguments, `error: the following arguments are required: --tokenizer-json`; with it set, `--help` prints the `$CHECKPOINT_ROOT/muse-glimmer-30b/tokenizer.json` default |

## 5b. One defect found in flow and fixed here

[#1079](https://github.com/mudler/vllm.cpp/issues/1079). Those six skip
messages printed `SKIP 1` and named no case, because the helper streamed its
`const char*` argument into doctest `MESSAGE` and doctest 2.5.2 stringifies a
`const char*` through its bool overload. It is pre-existing at `100026481` and
is repaired here rather than left, because this row rewrites those exact
messages and would otherwise carry the defect forward under a changed line. The
binary reports `6 passed` with `assertions: 0` when the checkpoint is absent, so
the message text is what separates a skipped run from a gated one. Scope was
measured before the fix: `grep -rn 'MESSAGE("SKIP " << what' tests/` returns 4
hits, all in this file. The fix streams `std::string(what)`, and the output
above is the after.

## 6. Now

The tracked defaults and the DGX profile name the location that survives a
reboot. The records that cite the old location keep it, because they record
where a past measurement read its bytes.

## Owed

[#1077](https://github.com/mudler/vllm.cpp/issues/1077): `.env.example:37`,
`.agents/environment.md:29` and `tests/vllm/multimodal/test_ltx2_video.cpp:2129`
each state that nothing in the tree reads `CHECKPOINT_ROOT`. Six gates read it
today, listed in §5. This row does not repair that claim, because
`test_ltx2_video.cpp:2129` reasons FROM the false premise when it chooses a
separate variable, and reversing that reasoning is a design decision with its own
review rather than a path substitution. Filed and left owned here.
