#!/usr/bin/env python3
"""Compare our DeepSeek-V4 vision token block against the llama.cpp b10766 dump.

Row `MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm` W6, issue #2411.

Every file is `[int32 rows][int32 cols][f32 data]`, which is the format
`tools/mtmd/clip.cpp:5853-5873` writes at the pinned oracle for
`MTMD_DEBUG_EMBEDDINGS=<path>`. Pure standard library, so the worker needs no
numpy.

Usage: dsv4v_w6_compare.py <dir> <tag> <lead_pad> <n_llm_h> <n_llm_w>
  reads <dir>/{ours,oracle}-<tag>-block.f32, and when present the stage files
  -vit.f32, -cells.f32 and -input.f32.

IT ENFORCES A BOUND AND EXITS ON IT. Exit 0 PASS or DIAGNOSTIC, 1 the recorded
bound was exceeded, 2 SHAPE_MISMATCH, 3 the tag falls under no recorded rule and
so nothing was judged, 4 the run could not be judged because the recorded
profile or the data is malformed. Until 2026-09-12 this script returned 0 for
every shape that matched, whatever the magnitude, and the `<= 4.9%` judgement in
the spec was prose arithmetic a reader did against its output; a drifting run
produced a well-formed report, `RC=0` and no signal. The bounds are READ from
`dsv4v_w6_bounds.json` beside this file rather than written here, so no wave can
derive a bound from the run it is judging.

THREE WAYS THIS SCRIPT COULD STILL REPORT SUCCESS OVER A FAILURE, all measured
on synthetic data and all closed here:

  A NON-FINITE STATISTIC IS NOT A PASS. One image row that is zero on BOTH sides
  makes stats() return nan for that row's rel_l2 and cos. The mean over rows is
  then nan, and BOTH bound tests are False -- `nan > 0.049` is False and
  `nan < 0.998` is False -- so a single degenerate row silently disabled both
  magnitude bounds for all 100 rows. Measured: 99 rows 50% off plus one zero row
  printed `VERDICT PASS` and exited 0, while the identical data without the zero
  row exited 1 at 50.0000%. Every judged statistic is now required to be FINITE
  before it is compared, and a judged profile refuses a degenerate row outright:
  an all-zero image row means the tower produced nothing for that aligner cell,
  which is a defect to report and never an average to absorb.

  A MISSING `judged` KEY IS NOT A DIAGNOSTIC. `profile.get("judged", False)`
  meant an incomplete or malformed profile read as DIAGNOSTIC and exited 0 on
  50%-off data. `judged` must now be present and boolean, the profile must exist
  in the file, and anything else is ERROR with a non-zero exit. Only an EXPLICIT
  `judged: false` is a diagnostic leg.

  AN ABSENT STAGE IS NOT A PASSING STAGE. Only `image_rows` was ever judged, so a
  vit or cells stage that was 100x wrong, shape-mismatched, or missing from the
  report entirely still printed `VERDICT PASS`. The spec cites the vit numbers as
  evidence that the error does not jump at a stage, so that sentence rested on
  nothing executable. Stages a profile declares are now REQUIRED to be present
  and are judged against recorded bounds; a stage that is deliberately unbounded
  says so in the profile with its reason, and an absent one always fails.

AND THE SHAPE THOSE THREE REPAIRS EACH LEFT IN PLACE ONE KEY AT A TIME. Each of
them hardened the key it was about, and every OTHER judging key kept the same
fail-open shape: it was read with `profile.get(...)`, and `_check_bound` returns
silently when the limit is `None`, so a judged profile that simply OMITTED a key
was judged without that bound and still exited 0. Measured on data whose every
image row was 50% off, tag `lp0`, profile `shipped_bf16`: dropping
`mean_rel_l2_max` printed `VERDICT PASS` at rc 0, dropping `mean_rel_l2_max` and
`mean_cos_min` together printed `VERDICT PASS` at rc 0, and a profile holding
`judged` and nothing else printed `VERDICT PASS` at rc 0. A mistyped key had the
same effect, because nothing ever read the profile as a whole.

  A KEY THAT IS ABSENT IS NOT A KEY THAT IS UNBOUNDED. The judging keys are
  DECLARED once in `PROFILE_KEYS` and `STAGE_KEYS` below and the resolved profile
  is validated against them BEFORE anything is judged. A missing key, a key of
  the wrong type, and an unknown key are each an ERROR with exit 4, exactly as a
  missing `judged` is. A bound a profile deliberately does not apply is written
  as `null` and names its reason under `unbounded`, so the silence is stated.
  Fixing this per key is what produced this paragraph; it is fixed as a class.

  A PRESENCE-ONLY STAGE IS STILL REQUIRED TO BE PRESENT. `diagnostic_only` used
  to `continue` before the presence check, so for `shipped_bf16` an absent
  `cells` stage exited 0 and a `cells` stage 100x wrong exited 0, against a
  docstring and a bounds file that both say its presence is reported.

  AN ARBITRARY ARGMAX IS NOT A PERMUTATION RESULT. `best_match` asserts nothing
  about the reference rows being separable, and the spec leans on the
  identity-permutation condition as the ORDERING evidence. On rows that are
  near-parallel the argmax is decided by bf16 rounding rather than by content:
  measured, a smooth-ramp fixture separated DIFFERENT rows by 7e-7 in cosine
  while rounding moves a row by about 0.4%, which reads as a false red on clean
  data and would read as a false GREEN on genuinely permuted output. The margin
  between the winner and the runner-up is now reported and bounded.

AND THE TWO SHAPES THAT REPAIR LEFT. Both were found by a fifth review and both
are closed here.

  A STAGE DROPPED FROM `stages` WAS JUDGED BY NOTHING. `PROFILE_KEYS` forced the
  `stages` KEY to exist and said nothing about its MEMBERSHIP, so every judging
  key had to be declared while a whole stage RULE could simply be deleted and
  the run still passed. Measured on a leg whose vit file was 100x wrong:
  deleting the `vit` rule printed `stage 'vit': NOT REQUIRED by this profile`,
  `VERDICT PASS` and exited 0, `"stages": {}` did the same, and the identical
  data with the rule present exited 1 at 9901.8287%. Stage MEMBERSHIP is now
  declared exactly as keys are: a judged profile names every stage in `STAGES`,
  and a stage it does not require is written as `null` with its reason under
  `unbounded` as `stages.<name>`. An omitted stage is ERROR with exit 4.

  A DECLARED CONSTANT IS NOT A MEASUREMENT. `best_match_margin_min` was 0.01,
  and the bounds file said in its own provenance that it was DECLARED rather
  than recorded. Measured with the shipped `best_match()` at realistic width
  (D=1280), the margin falls under 0.01 as soon as per-cell variation is about a
  tenth of what the cells share -- sky, wall, background -- while the identity
  stayed best for 100 of 100 rows in every one of those cases: a shared global
  component with detail 0.10 and 0.05 measured 0.00834 and 0.00210, and a
  20-cell flat region measured 0.00875 and 0.00221. All four would have RED on a
  correct run, and the ordering claim the bound guards was right in all four.
  The margin is now bounded against THIS DATASET's own bf16 rounding scale,
  which is derived per run by `bf16_rounding_scale()`, cannot be accused of
  having been picked by the wave it judges, and still refuses the degenerate
  ramp it was introduced for by three and a half decades.
"""
import fnmatch
import json
import math
import os
import struct
import sys


def load(path):
    with open(path, "rb") as f:
        rows, cols = struct.unpack("<ii", f.read(8))
        data = struct.unpack("<%df" % (rows * cols), f.read(4 * rows * cols))
    return rows, cols, [data[r * cols:(r + 1) * cols] for r in range(rows)]


def bf16(x):
    # round-to-nearest-even to bf16, returned as the widened f32 value
    (u,) = struct.unpack("<I", struct.pack("<f", x))
    if (u & 0x7F800000) == 0x7F800000:
        return x
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", u))[0]


def bf16_ulp(x):
    """The bf16 grid spacing AT x, whether or not x is already on the grid.

    bf16 carries 8 significand bits, so `ulp(x) = 2**(exponent - 7)` and the
    RELATIVE spacing is between 2**-8 and 2**-7. This is deliberately the grid
    spacing rather than the residual `x - bf16(x)`: the residual is exactly ZERO
    for every value already rounded to bf16, and both sides of this comparison
    are bf16 by the time they reach a file, so a scale built from the residual
    would be 0 on real data and bound nothing at all.
    """
    (u,) = struct.unpack("<I", struct.pack("<f", float(x)))
    exponent = (u >> 23) & 0xFF
    if exponent == 0 or exponent == 0xFF:
        return 0.0
    return 2.0 ** (exponent - 127 - 7)


def bf16_rounding_scale(rows):
    """How far bf16 rounding alone can move a row's DIRECTION, worst row.

    This is the quantity the margin has to beat, DERIVED from the dataset being
    judged rather than declared as a constant. Perturbing every element of a row
    by at most half a bf16 ULP moves the row by an angle whose sine is at most
    `||h|| / ||b||`, so the largest cosine change it can produce is
    `1 - sqrt(1 - (||h||/||b||)**2)`, which is what this returns.

    WHY IT IS COMPARED WITH A COSINE MARGIN. The failure being guarded is the
    NEAR-PARALLEL one, where every candidate row points almost the same way. The
    first-order term is then common to the winner and the runner-up and cancels
    out of their difference, and what is left is exactly this second-order
    scale. A margin at or below it was chosen by rounding, not by content.
    """
    worst = 0.0
    for row in rows:
        norm = math.sqrt(sum(v * v for v in row))
        if norm == 0.0:
            continue
        half = math.sqrt(sum((0.5 * bf16_ulp(v)) ** 2 for v in row))
        ratio = min(1.0, half / norm)
        worst = max(worst, 1.0 - math.sqrt(max(0.0, 1.0 - ratio * ratio)))
    return worst


def layout(lead_pad, n_llm_h, n_llm_w):
    # dsv4_get_block_layout + clip.cpp set_input, PROJECTOR_TYPE_DEEPSEEK4V
    rows = n_llm_h + (n_llm_h % 2)
    row_len = n_llm_w + 1
    pad_last = (rows // 2 * row_len) % 2 * 2
    types, cell = ["PAD"] * lead_pad + ["START"], [None] * (lead_pad + 1)
    for t in range(rows * row_len):
        g, rem = divmod(t, 2 * row_len)
        c, r = rem // 2, 2 * g + rem % 2
        if r >= n_llm_h:
            types.append("PAD"); cell.append(None)
        elif c == n_llm_w:
            types.append("NEWLINE"); cell.append(None)
        else:
            types.append("IMAGE"); cell.append(r * n_llm_w + c)
    types += ["PAD"] * pad_last + ["END"]
    cell += [None] * (pad_last + 1)
    return types, cell


def stats(a, b):
    d = [x - y for x, y in zip(a, b)]
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    nd = math.sqrt(sum(x * x for x in d))
    dot = sum(x * y for x, y in zip(a, b))
    return {
        "max_abs": max(abs(x) for x in d),
        "mean_abs": sum(abs(x) for x in d) / len(d),
        "cos": dot / (na * nb) if na and nb else float("nan"),
        "rel_l2": nd / nb if nb else float("nan"),
        "ref_rms": nb / math.sqrt(len(b)),
        # The two norms are reported so a DEGENERATE row is nameable rather than
        # only showing up as a nan that both bound tests then ignore.
        "our_norm": na,
        "ref_norm": nb,
    }


def matrix_summary(name, ours, ref):
    per = [stats(a, b) for a, b in zip(ours, ref)]
    degenerate = [i for i, p in enumerate(per)
                  if p["ref_norm"] == 0.0 or p["our_norm"] == 0.0]
    out = {
        "rows": len(per),
        "max_abs": max(p["max_abs"] for p in per),
        "mean_abs": sum(p["mean_abs"] for p in per) / len(per),
        "min_cos": min(p["cos"] for p in per),
        "mean_cos": sum(p["cos"] for p in per) / len(per),
        "max_rel_l2": max(p["rel_l2"] for p in per),
        "mean_rel_l2": sum(p["rel_l2"] for p in per) / len(per),
        "ref_rms": math.sqrt(sum(p["ref_rms"] ** 2 for p in per) / len(per)),
        # A row with a zero norm on either side has an UNDEFINED cos and rel_l2.
        # Counting them here is what lets judge() refuse rather than average a
        # nan into a bound test that then silently passes.
        "degenerate_rows": len(degenerate),
        "degenerate_row_index": degenerate[:8],
    }
    print("[%s] %s" % (name, json.dumps(out)))
    return out, per


def best_match(ours, ref):
    """For each of our rows: the best reference row, its cosine, and the MARGIN.

    The margin is the winner's cosine minus the runner-up's, and it is what makes
    an argmax mean anything. The identity-permutation condition is the spec's
    ORDERING evidence -- the input stage cannot supply it, because the oracle
    dump writes that file in our own patch-row order -- and until 2026-09-12 it
    rested on an argmax with no separability precondition at all.

    MEASURED, and this is why the margin is reported: a smooth-ramp fixture made
    every reference row near-parallel, cosine 0.9999988 between DIFFERENT rows,
    while rounding a row to bf16 moves each element by about 0.4%. The rounding
    swamped the separation, the argmax became arbitrary, and the identity was
    best for 17 of 100 rows on a CLEAN dataset. The same degeneracy is a false
    GREEN on genuinely permuted output, because any row then matches any row.
    """
    norms = [math.sqrt(sum(x * x for x in r)) or 1.0 for r in ref]
    hits = []
    for a in ours:
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        cos = [sum(x * y for x, y in zip(a, b)) / (na * norms[j])
               for j, b in enumerate(ref)]
        # `sorted` is stable, so a tie keeps the lowest index and the winner is
        # the same row the previous strict-greater-than scan chose.
        order = sorted(range(len(cos)), key=lambda j: cos[j], reverse=True)
        runner_up = cos[order[1]] if len(order) > 1 else -1.0
        hits.append((order[0], cos[order[0]], cos[order[0]] - runner_up))
    return hits


# ── THE BOUND ──────────────────────────────────────────────────────────────
# The numbers judged against are NOT written here. They are recorded
# measurements and they live beside this file, with the rc job that produced
# each one, so that a reader can see what was measured and when.
BOUNDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "dsv4v_w6_bounds.json")
EXIT = {"PASS": 0, "DIAGNOSTIC": 0, "FAIL": 1, "UNJUDGED": 3, "ERROR": 4}
STAGES = ("input", "vit", "cells")


def load_bounds(path=BOUNDS_PATH):
    with open(path) as f:
        return json.load(f)


def profile_for(tag, bounds):
    """The recorded profile this tag falls under, or None when no rule matches."""
    for pattern, name in bounds["tag_rules"]:
        if fnmatch.fnmatchcase(tag, pattern):
            return name
    return None


# ── THE PROFILE SCHEMA ─────────────────────────────────────────────────────
# EVERY JUDGING KEY IS DECLARED HERE, and a judged profile must declare every
# one of them. The value `None` (JSON `null`) means the bound is deliberately
# not applied, and the profile must then name the reason under `unbounded`.
#
# This exists because hardening the keys ONE AT A TIME did not work. `judged`
# was made mandatory on 2026-09-12 and every other judging key kept the same
# shape: read with `profile.get(...)`, silently unbounded when absent. Dropping
# `mean_rel_l2_max` from `shipped_bf16` exited 0 PASS on data 50% off; dropping
# `mean_rel_l2_max` and `mean_cos_min` exited 0 PASS; a profile holding only
# `judged` exited 0 PASS. Validating the resolved profile against this schema
# BEFORE anything is judged is what closes the shape rather than the instances:
# a missing key, a key of the wrong type and an unknown (mistyped) key are each
# an ERROR, because none of them is a bound and all three used to read as one.
NUMBER = "number"
PROFILE_KEYS = {
    "sentinels_bf16_exact": bool,
    "permutation_identity_complete": bool,
    "best_match_margin_above_bf16_rounding": bool,
    "mean_rel_l2_max": NUMBER,
    "mean_cos_min": NUMBER,
    "max_degenerate_rows": int,
    "stages": dict,
}
STAGE_KEYS = {
    "bf16_of_oracle_exact": bool,
    "mean_rel_l2_max": NUMBER,
    "mean_cos_min": NUMBER,
    "max_degenerate_rows": int,
}
# Keys that carry no bound and are therefore not schema violations.
META_KEYS = ("judged", "unbounded", "diagnostic_only")


def _is_typed(value, want):
    """`isinstance(True, int)` is True, so a bool must not satisfy a number."""
    if want is bool:
        return isinstance(value, bool)
    if isinstance(value, bool):
        return False
    if want is NUMBER:
        return isinstance(value, (int, float))
    return isinstance(value, want)


def _validate_keys(where, mapping, schema, bad):
    """Every key of `schema` DECLARED, correctly typed, and nothing invented."""
    unbounded = mapping.get("unbounded", {})
    if not isinstance(unbounded, dict):
        bad.append("%s has a non-object 'unbounded' %r" % (where, unbounded))
        unbounded = {}
    for key in sorted(schema):
        if key not in mapping:
            bad.append(
                "%s does not DECLARE %r. A judged profile must declare every key "
                "it is judged on: an absent key is read by nothing, so the leg "
                "would be judged without that bound and still pass. Write the "
                "bound, or write null and give the reason under 'unbounded'."
                % (where, key))
            continue
        value = mapping[key]
        if value is None:
            reason = unbounded.get(key)
            if not isinstance(reason, str) or not reason.strip():
                bad.append(
                    "%s declares %r as null without naming a reason under "
                    "'unbounded'. A bound that is deliberately not applied must "
                    "say why; silence is how an unbounded key reads as a bound."
                    % (where, key))
            continue
        if not _is_typed(value, schema[key]):
            bad.append("%s declares %r as %r, which is not a %s"
                       % (where, key, value,
                          schema[key] if isinstance(schema[key], str)
                          else schema[key].__name__))
    for key in sorted(mapping):
        if key.startswith("_") or key in META_KEYS or key in schema:
            continue
        bad.append(
            "%s declares unknown key %r. A MISTYPED key is not a bound: nothing "
            "reads it, and the bound it was meant to be would be absent."
            % (where, key))


def validate_profile(name, profile):
    """The whole profile, checked ONCE and up front. Returns [error lines]."""
    bad = []
    # A MISSING `judged` KEY IS AN ERROR. It used to default to False, so an
    # incomplete profile silently downgraded a judged leg to DIAGNOSTIC and
    # exited 0 on data that was 50% off.
    if "judged" not in profile:
        return ["profile %r has no 'judged' key. An incomplete profile is an "
                "ERROR: it must not silently downgrade a judged leg to a "
                "diagnostic one." % name]
    if not isinstance(profile["judged"], bool):
        return ["profile %r has a non-boolean 'judged' value %r"
                % (name, profile["judged"])]
    if not profile["judged"]:
        return bad
    _validate_keys("profile %r" % name, profile, PROFILE_KEYS, bad)
    stages = profile.get("stages")
    if not isinstance(stages, dict):
        return bad
    # STAGE MEMBERSHIP IS DECLARED EXACTLY AS A JUDGING KEY IS. `PROFILE_KEYS`
    # forced this dict to EXIST and constrained nothing about what is IN it, so
    # a whole stage rule could be deleted and that stage was then judged by
    # nothing -- the one bound in this file that needed no declaration to be
    # skipped. Measured on a leg whose vit file was 100x wrong: dropping the
    # `vit` rule exited 0 VERDICT PASS, and so did `"stages": {}`.
    unbounded = profile.get("unbounded")
    if not isinstance(unbounded, dict):
        unbounded = {}
    for stage in STAGES:
        if stage not in stages:
            bad.append(
                "profile %r does not DECLARE stage %r. A judged profile must "
                "name every stage it is judged on: an omitted stage is read by "
                "nothing, so the leg would be judged without it and still pass. "
                "Write the rule, or write null and give the reason under "
                "'unbounded' as %r."
                % (name, stage, "stages.%s" % stage))
    for stage in sorted(stages):
        rule = stages[stage]
        where = "profile %r stage %r" % (name, stage)
        if stage not in STAGES:
            bad.append("%s is not one of %s" % (where, ", ".join(STAGES)))
            continue
        if rule is None:
            reason = unbounded.get("stages.%s" % stage)
            if not isinstance(reason, str) or not reason.strip():
                bad.append(
                    "%s is declared null without naming a reason under "
                    "'unbounded' as %r. A stage a profile deliberately does not "
                    "require must say why; silence is how a dropped stage reads "
                    "as a stage that passed."
                    % (where, "stages.%s" % stage))
            continue
        if not isinstance(rule, dict):
            bad.append("%s is %r, which is not a rule" % (where, rule))
            continue
        flag = rule.get("diagnostic_only", False)
        if not isinstance(flag, bool):
            bad.append("%s has a non-boolean 'diagnostic_only' %r" % (where, flag))
            continue
        if flag:
            # PRESENCE-ONLY, and presence is still REQUIRED. A stage that says
            # nothing about its magnitude must not also say nothing about any
            # bound it forgot to declare, so no judging key may appear here.
            for key in sorted(rule):
                if not key.startswith("_") and key not in META_KEYS:
                    bad.append("%s is diagnostic_only and must carry no bound, "
                               "but it declares %r" % (where, key))
            continue
        _validate_keys(where, rule, STAGE_KEYS, bad)
    return bad


def _check_bound(bad, label, value, limit, kind, notes=None):
    """Compare one statistic, FAIL-CLOSED on a missing or non-finite value.

    `nan > limit` and `nan < limit` are both False, so a non-finite statistic
    used to satisfy every bound at once. It is a failure here instead.

    `limit is None` is reachable ONLY through a profile that declares the key as
    null with a reason: `validate_profile` refuses an absent one.
    """
    if limit is None:
        if notes is not None:
            notes.append("%s: NOT BOUNDED (declared null), value %s"
                         % (label, value))
        return
    if notes is not None:
        notes.append("%s: bound %s %.6g, value %s"
                     % (label, ">=" if kind == "min" else "<=", limit, value))
    if value is None:
        bad.append("%s is MISSING from the report, so its recorded bound could "
                   "not be applied" % label)
        return
    if not math.isfinite(value):
        bad.append("%s is %s, which is NOT FINITE. A bound cannot be applied to "
                   "it and this is a FAILURE, never a pass." % (label, value))
        return
    if kind == "max" and value > limit:
        bad.append("%s %.4f%% EXCEEDS the recorded bound %.4f%%"
                   % (label, 100.0 * value, 100.0 * limit))
    elif kind == "min" and value < limit:
        bad.append("%s %.6f is BELOW the recorded bound %.6f"
                   % (label, value, limit))


def _judge_stage(report, stage, rule, bad, notes):
    """Apply a profile's recorded rule for ONE stage of the report.

    PRESENCE IS CHECKED FIRST AND ALWAYS, `diagnostic_only` included. The
    `diagnostic_only` test used to sit in the caller and `continue` BEFORE this
    function ran, so for `shipped_bf16` an absent `cells` stage exited 0 PASS and
    a `cells` stage 100x wrong exited 0 PASS -- against this module's own
    docstring and against the bounds file, which both say its presence is
    reported. Not judging a magnitude is not the same as not looking.
    """
    got = report.get(stage, "absent")
    if got == "absent":
        bad.append("stage %r is ABSENT from the report and the recorded profile "
                   "REQUIRES it. A stage that never ran is not a stage that "
                   "passed." % stage)
        return
    if isinstance(got, dict) and "shape_mismatch" in got:
        bad.append("stage %r SHAPE MISMATCH %s -- the two sides are not the same "
                   "array and nothing about their agreement was measured"
                   % (stage, got["shape_mismatch"]))
        return
    if not isinstance(got, dict):
        bad.append("stage %r is %r, which is not a summary this bound can be "
                   "applied to" % (stage, got))
        return
    if rule.get("diagnostic_only"):
        notes.append("stage %r: PRESENT and reported, magnitude NOT judged "
                     "(diagnostic_only)" % stage)
        return
    # Every key below is indexed rather than `.get`-ed: validate_profile has
    # already refused a rule that does not declare all four, so an absent key
    # cannot reach this function and read as "no bound".
    if rule["bf16_of_oracle_exact"]:
        notes.append("stage %r bf16_of_oracle_exact: REQUIRED, got %s"
                     % (stage, got.get("bf16_of_oracle_exact")))
        if not got.get("bf16_of_oracle_exact"):
            bad.append("stage %r is not exactly bf16(oracle), which the recorded "
                       "profile requires" % stage)
    else:
        notes.append("stage %r bf16_of_oracle_exact: not required" % stage)
    limit = rule["max_degenerate_rows"]
    if limit is None:
        notes.append("stage %r: degenerate rows NOT BOUNDED (declared null)"
                     % stage)
    else:
        n = got.get("degenerate_rows")
        notes.append("stage %r: at most %d degenerate (zero-norm) rows, got %s"
                     % (stage, limit, n))
        if n is None or n > limit:
            bad.append("stage %r has %s degenerate (zero-norm) rows, above the "
                       "recorded maximum %d" % (stage, n, limit))
    _check_bound(bad, "stage %r mean_rel_l2" % stage, got.get("mean_rel_l2"),
                 rule["mean_rel_l2_max"], "max", notes)
    _check_bound(bad, "stage %r mean_cos" % stage, got.get("mean_cos"),
                 rule["mean_cos_min"], "min", notes)


def judge(report, tag, bounds):
    """Apply the recorded profile. Returns (verdict, [failures], [notes]).

    `notes` is what the run SAYS IT JUDGED. `.agents/verification.md` requires an
    instrument to state what it measured in its own output, and this one printed
    `VERDICT PASS tag=lp0 profile=shipped_bf16` and nothing else: which stages
    were judged, which were presence-only and which bounds were applied were all
    invisible to a reader of the run, so a stage silently going unjudged looked
    exactly like a stage that passed.
    """
    name = profile_for(tag, bounds)
    if name is None:
        return "UNJUDGED", [
            "no rule in %s matches tag %r, so NOTHING was judged. Add a rule for "
            "this leg; do not read this as a pass."
            % (os.path.basename(BOUNDS_PATH), tag)], []
    profile = bounds.get("profiles", {}).get(name)
    if profile is None:
        return "ERROR", [
            "tag %r maps to profile %r, which %s does not define. An unresolvable "
            "profile is an ERROR, never a pass."
            % (tag, name, os.path.basename(BOUNDS_PATH))], []
    # THE WHOLE PROFILE IS VALIDATED BEFORE ANYTHING IS JUDGED. Judging first and
    # checking a key on the way past is the shape that let five separate bounds
    # be absent and unnoticed.
    broken = validate_profile(name, profile)
    if broken:
        return "ERROR", broken, []
    if not profile["judged"]:
        return "DIAGNOSTIC", [], ["profile %r is explicitly judged: false, so "
                                  "NOTHING here is a bound" % name]

    bad, notes = [], ["profile %r, judged" % name]
    if profile["sentinels_bf16_exact"]:
        sentinels = report.get("sentinels")
        if not isinstance(sentinels, dict) or not sentinels:
            bad.append("the profile requires exact sentinels and the report "
                       "carries none")
        else:
            notes.append("sentinels: all %d kinds required exactly bf16(oracle)"
                         % len(sentinels))
            for kind, s in sorted(sentinels.items()):
                if not s["bf16_of_oracle_exact"]:
                    bad.append("sentinel %s is not exactly bf16(oracle), max_abs "
                               "%g" % (kind, s["max_abs"]))
    p = report.get("permutation")
    if profile["permutation_identity_complete"]:
        if not isinstance(p, dict):
            bad.append("the profile requires a complete identity permutation and "
                       "the report carries no permutation")
        else:
            notes.append("permutation: the identity required for all %d image "
                         "rows, got %d" % (p["of"], p["identity_is_best"]))
            if p["identity_is_best"] != p["of"]:
                bad.append("permutation: the identity is best for only %d of %d "
                           "image rows" % (p["identity_is_best"], p["of"]))
    # THE ARGMAX ABOVE NEEDS THE REFERENCE ROWS TO BE SEPARABLE. Without a margin
    # the permutation condition reports an arbitrary winner: on near-parallel
    # rows it is a false red on clean data and a false GREEN on permuted output.
    # The failure is a property of the DATASET, and the message says so rather
    # than claiming a defect in the tower.
    # THE BOUND IS DERIVED FROM THIS RUN'S OWN ROWS, not declared. It was the
    # constant 0.01, and the bounds file said in its own provenance that the
    # number was declared rather than recorded. Measured at realistic width
    # (D=1280), a margin under 0.01 is what an ORDINARY photographic region
    # produces -- a shared global component with 10% or 5% per-cell detail
    # measured 0.00834 and 0.00210, a 20-cell flat region 0.00875 and 0.00221 --
    # and the identity was still best for 100 of 100 rows in every one of them.
    # `min_best_margin` is a MIN over the image rows, so one flat pair decides
    # the run, and the constant would have RED four correct datasets.
    if profile["best_match_margin_above_bf16_rounding"]:
        if not isinstance(p, dict):
            bad.append("the profile requires a best-match margin above the bf16 "
                       "rounding scale and the report carries no permutation")
        else:
            got = p.get("min_best_margin")
            scale = p.get("bf16_rounding_scale")
            notes.append(
                "permutation: best-match margin bound > %s, DERIVED from this "
                "run's own rows as the largest direction change half a bf16 ULP "
                "can cause -- NOT a declared constant -- value %s" % (scale, got))
            if got is None or not math.isfinite(got):
                bad.append("permutation min_best_margin is %s, so the identity "
                           "condition rests on an argmax whose separability was "
                           "never measured" % got)
            elif scale is None or not math.isfinite(scale):
                bad.append("permutation bf16_rounding_scale is %s, so the margin "
                           "was judged against nothing. The scale is derived from "
                           "the rows this run read; its absence is an unjudged "
                           "separability claim, never a pass." % scale)
            elif got <= scale:
                bad.append(
                    "permutation min_best_margin %.6g is NOT ABOVE this dataset's "
                    "own bf16 rounding scale %.6g. The reference rows are not "
                    "separable enough for an argmax to carry the ORDERING claim: "
                    "rounding to bf16 can move the winner past the runner-up, so "
                    "a winner this close is chosen by noise. This is a statement "
                    "about the DATASET, not a defect in the tower." % (got, scale))
    else:
        notes.append("permutation: best-match margin NOT REQUIRED to clear the "
                     "bf16 rounding scale (declared false)")
    stat_name = bounds.get("statistic")
    stat = report.get(stat_name)
    if not isinstance(stat, dict):
        return "ERROR", ["the judged statistic %r is absent from the report"
                         % stat_name], notes
    notes.append("judged statistic: %r over %s rows"
                 % (stat_name, stat.get("rows")))
    # A DEGENERATE ROW IS A DEFECT, not an average to absorb. An all-zero image
    # row means the tower produced nothing for that aligner cell, and it is also
    # the exact shape that used to turn both bounds below into no-ops.
    limit = profile["max_degenerate_rows"]
    if limit is not None:
        n = stat.get("degenerate_rows")
        notes.append("%s: at most %d degenerate (zero-norm) rows, got %s"
                     % (stat_name, limit, n))
        if n is None or n > limit:
            bad.append("%s has %s degenerate (zero-norm) rows at index %s, above "
                       "the recorded maximum %d. A zero row makes rel_l2 and cos "
                       "undefined, which would disable the bounds below."
                       % (stat_name, n, stat.get("degenerate_row_index"), limit))
    else:
        notes.append("%s: degenerate rows NOT BOUNDED (declared null)" % stat_name)
    _check_bound(bad, "%s mean_rel_l2" % stat_name, stat.get("mean_rel_l2"),
                 profile["mean_rel_l2_max"], "max", notes)
    _check_bound(bad, "%s mean_cos" % stat_name, stat.get("mean_cos"),
                 profile["mean_cos_min"], "min", notes)
    for stage in STAGES:
        # INDEXED, not `.get`-ed, for the same reason the stage rule's own keys
        # are: validate_profile has already refused a profile that does not
        # declare every stage, so an OMITTED stage cannot reach this loop and
        # read as "not required".
        rule = profile["stages"][stage]
        if rule is None:
            notes.append("stage %r: NOT REQUIRED (declared null): %s"
                         % (stage, profile.get("unbounded", {})
                            .get("stages.%s" % stage)))
            continue
        _judge_stage(report, stage, rule, bad, notes)
    return ("PASS" if not bad else "FAIL"), bad, notes


def main():
    d, tag = sys.argv[1], sys.argv[2]
    lead_pad, n_llm_h, n_llm_w = map(int, sys.argv[3:6])
    report = {"tag": tag, "lead_pad": lead_pad, "grid": [n_llm_h, n_llm_w]}
    types, cell = layout(lead_pad, n_llm_h, n_llm_w)

    orows, ocols, ours = load(os.path.join(d, "ours-%s-block.f32" % tag))
    rrows, rcols, ref = load(os.path.join(d, "oracle-%s-block.f32" % tag))
    report["tokens"] = {"ours": [orows, ocols], "oracle": [rrows, rcols],
                        "layout": len(types)}
    print("tokens ours=%dx%d oracle=%dx%d layout=%d"
          % (orows, ocols, rrows, rcols, len(types)))
    if (orows, ocols) != (rrows, rcols) or orows != len(types):
        report["verdict"] = "SHAPE_MISMATCH"
        print(json.dumps(report))
        json.dump(report, open(os.path.join(d, "report-%s.json" % tag), "w"), indent=1)
        return 2

    # The four sentinels are COPIED weights. The oracle concatenates them in f32
    # (deepseek4v.cpp); we narrow them to bf16 at the join
    # (deepseek_v4_mm.cpp). So the exact comparison is ours == bf16(oracle), and
    # f32 equality is reported beside it.
    sent = {}
    for kind in ("START", "END", "NEWLINE", "PAD"):
        idx = [i for i, t in enumerate(types) if t == kind]
        f32_eq = all(ours[i] == ref[i] for i in idx)
        bf_eq = all(all(a == bf16(b) for a, b in zip(ours[i], ref[i])) for i in idx)
        st = stats(ours[idx[0]], ref[idx[0]])
        sent[kind] = {"rows": idx, "f32_exact": f32_eq,
                      "bf16_of_oracle_exact": bf_eq, "max_abs": st["max_abs"]}
    report["sentinels"] = sent
    print("sentinels", json.dumps(sent))

    img = [i for i, t in enumerate(types) if t == "IMAGE"]
    summ, per = matrix_summary("block-image-rows", [ours[i] for i in img],
                               [ref[i] for i in img])
    report["image_rows"] = summ
    worst = sorted(range(len(img)), key=lambda k: per[k]["rel_l2"])[-5:]
    report["worst_image_rows"] = [
        {"row": img[k], "cell": cell[img[k]], **per[k]} for k in reversed(worst)]
    for w in report["worst_image_rows"]:
        print("worst", json.dumps(w))
    # STRUCTURE. Precision noise gives a roughly uniform ABSOLUTE error, so the
    # relative error is worst on the smallest rows and nothing tracks position.
    # A positional defect (RoPE axis, unfold order, a padded edge) shows up as
    # error concentrated on an aligner row or column.
    by_r = [[] for _ in range(n_llm_h)]; by_c = [[] for _ in range(n_llm_w)]
    for k, i in enumerate(img):
        r, c = divmod(cell[i], n_llm_w)
        by_r[r].append(per[k]["mean_abs"]); by_c[c].append(per[k]["mean_abs"])
    mean = lambda v: sum(v) / len(v)
    xs = [per[k]["ref_rms"] for k in range(len(img))]
    ya = [per[k]["mean_abs"] for k in range(len(img))]
    yr = [per[k]["rel_l2"] for k in range(len(img))]

    def corr(x, y):
        mx, my = mean(x), mean(y)
        sx = math.sqrt(sum((v - mx) ** 2 for v in x)); sy = math.sqrt(sum((v - my) ** 2 for v in y))
        return sum((u - mx) * (v - my) for u, v in zip(x, y)) / (sx * sy) if sx and sy else float("nan")

    report["structure"] = {
        "mean_abs_by_aligner_row": [round(mean(v), 6) for v in by_r],
        "mean_abs_by_aligner_col": [round(mean(v), 6) for v in by_c],
        "corr_ref_rms_vs_mean_abs": corr(xs, ya),
        "corr_ref_rms_vs_rel_l2": corr(xs, yr),
    }
    print("structure", json.dumps(report["structure"]))
    allrows = [stats(a, b) for a, b in zip(ours, ref)]
    report["all_rows"] = {
        "max_abs": max(p["max_abs"] for p in allrows),
        "mean_abs": sum(p["mean_abs"] for p in allrows) / len(allrows),
        "min_cos": min(p["cos"] for p in allrows)}

    # PERMUTATION: does a re-ordering of our image rows collapse the error? If
    # the identity is already the best match for every row, no permutation can.
    hits = best_match([ours[i] for i in img], [ref[i] for i in img])
    ident = sum(1 for k, (arg, _c, _m) in enumerate(hits) if arg == k)
    report["permutation"] = {"identity_is_best": ident, "of": len(img),
                             "min_best_cos": min(c for _a, c, _m in hits),
                             # The winner's lead over the runner-up. A margin at
                             # the scale of bf16 rounding means the argmax above
                             # was decided by noise; the bound is in the profile.
                             "min_best_margin": min(m for _a, _c, m in hits),
                             # WHAT THAT MARGIN IS JUDGED AGAINST, derived from
                             # the rows this run actually read rather than
                             # declared as a constant. Both sides are measured
                             # because either one's rounding can move the argmax.
                             "bf16_rounding_scale": max(
                                 bf16_rounding_scale([ours[i] for i in img]),
                                 bf16_rounding_scale([ref[i] for i in img]))}
    print("permutation", json.dumps(report["permutation"]))

    for stage in STAGES:
        po = os.path.join(d, "ours-%s-%s.f32" % (tag, stage))
        pr = os.path.join(d, "oracle-%s-%s.f32" % (tag, stage))
        if not (os.path.exists(po) and os.path.exists(pr)):
            report[stage] = "absent"
            continue
        a = load(po); b = load(pr)
        grid = math.isqrt(a[0])
        relaid_out = False
        if (stage == "vit" and a[:2] != b[:2] and grid * grid == a[0]
                and b[1] == grid and b[0] == a[1] * grid):
            # The first W6 run captured llama.cpp's permuted+cont view of
            # vit_out, laid out [hidden][y][x]. Re-index it to [y*gw+x][hidden].
            g = b[1]
            flat = [x for row in b[2] for x in row]
            b = (a[0], a[1], [[flat[(c * g + p // g) * g + p % g] for c in range(a[1])]
                              for p in range(a[0])])
            relaid_out = True
        if a[:2] != b[:2]:
            report[stage] = {"shape_mismatch": [a[:2], b[:2]]}
            print(stage, report[stage])
            continue
        s, _ = matrix_summary(stage, a[2], b[2])
        # RECORD the re-layout. It applies a GUESSED permutation when a numeric
        # coincidence holds, and a reader of report-<tag>.json could not
        # previously tell whether the vit numbers came from the file as written
        # or from this re-indexing.
        s["oracle_relaid_out"] = relaid_out
        if stage == "input":
            s["bf16_of_oracle_exact"] = all(
                x == bf16(y) for ra, rb in zip(a[2], b[2]) for x, y in zip(ra, rb))
        report[stage] = s

    bounds = load_bounds()
    profile = profile_for(tag, bounds)
    verdict, failures, notes = judge(report, tag, bounds)
    report["verdict"] = verdict
    report["bound_profile"] = profile
    report["bound_failures"] = failures
    report["judged"] = notes
    json.dump(report, open(os.path.join(d, "report-%s.json" % tag), "w"), indent=1)
    print("REPORT", os.path.join(d, "report-%s.json" % tag))
    # SAY WHAT WAS JUDGED. A verdict with no account of what it covered cannot be
    # read for what it LEFT OUT, which is the failure every repair on this file
    # has been about.
    for line in notes:
        print("JUDGED", line)
    for line in failures:
        print("BOUND", line)
    print("VERDICT %s tag=%s profile=%s" % (verdict, tag, profile))
    return EXIT[verdict]


if __name__ == "__main__":
    sys.exit(main())
