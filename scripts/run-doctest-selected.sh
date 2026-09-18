#!/usr/bin/env bash
# Run a doctest binary under a filter, and REFUSE when the filter selects no
# test case.
#
# WHY THIS EXISTS. A mutation battery is only evidence if the filtered run
# actually ran something. On 2026-09-13 a W7 battery ran
# `test_qwen4_exp_cuda_reductions '-tc=*W7*'` against case names that did not
# yet carry the token. doctest printed `0 passed | 0 failed | 18 skipped`,
# `Status: SUCCESS!`, and EXITED 0 -- so five kernel mutations each read as
# "green" while nothing had executed. The trap is reproducible on any committed
# doctest binary with a filter that matches nothing, and no test case can catch
# it, because the defect is that no test case was selected.
#
# WHAT IT DOES. It asks doctest for the count first (`--count`, which prints
# "unskipped test cases passing the current filters: N" and runs nothing), and
# stops at exit 3 when N is 0 or unreadable. Otherwise it prints `selected=N`
# and execs the real run, so the binary's own exit status is what the caller
# sees.
#
# USAGE
#   scripts/run-doctest-selected.sh <binary> [doctest args...]
# EXAMPLE
#   scripts/run-doctest-selected.sh build/tests/test_qwen4_exp_cuda_reductions \
#       "-tc=*W7*"
set -uo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <doctest-binary> [doctest args...]" >&2
  exit 2
fi

bin="$1"
shift

if [ ! -x "$bin" ]; then
  echo "REFUSED: not an executable doctest binary: $bin" >&2
  exit 2
fi

count_out="$("$bin" "$@" --count 2>&1)"
count_rc=$?
if [ "$count_rc" -ne 0 ]; then
  printf '%s\n' "$count_out" >&2
  echo "REFUSED: '$bin --count' exited $count_rc, so the selection is unknown" >&2
  exit 3
fi

selected="$(printf '%s\n' "$count_out" \
  | sed -n 's/.*unskipped test cases passing the current filters: *\([0-9][0-9]*\).*/\1/p' \
  | tail -1)"

if [ -z "$selected" ]; then
  printf '%s\n' "$count_out" >&2
  echo "REFUSED: could not read a selection count from '$bin --count'" >&2
  exit 3
fi

echo "selected=$selected"

if [ "$selected" -eq 0 ]; then
  echo "REFUSED: the filter selects NO test case, so a green run would measure nothing" >&2
  echo "         args: $*" >&2
  exit 3
fi

exec "$bin" "$@"
