ID: ISSUE-LOCAL-01M29S9RRXNV4JNPWPXM1HT4BH
Title: DeepSeek-V4 vision W6 parity drivers record tee's exit status, and nothing reads steps.txt back
Row: MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-12
Updated: 2026-09-12
Closed: 2026-09-12

## Problem

tools/parity/dsv4v_w6_parity.sh, dsv4v_w6_floor.sh and dsv4v_w6_f32.sh run 'cmd | tee file; step name $?' under 'set -u' with no pipefail, so the recorded status is TEE's and tee succeeds whenever it can write the file. Demonstrated on this box: 'false | tee /dev/null; echo $?' prints 0. The worst instance is dsv4v_w6_parity.sh, where a failure of dsv4v_w6_image.py -- the generator of the one image the entire gate rests on -- records 'image RC=0' and the sweep proceeds against whatever img392.rgb is left in $OUT, a persistent NAS directory nothing clears. Compounding it, the drivers append every status to steps.txt and NOTHING ever reads it back, so all three jobs ended on their DONE banner at rc 0 no matter which steps failed. Found by fresh review 2026-09-12.

## Resolution

2026-09-12: fixed in this change. All three drivers take 'set -uo pipefail' (none has set -e, so the only behaviour that changes is the value the status readers see), the image step reads ${PIPESTATUS[0]} into IMG_RC and then REFUSES when the generator failed or img392.rgb is empty -- after deleting any stale image first, so a previous run's artefact cannot be swept against -- and the previously unchecked 'sha256sum | tee' sites now record overlay_sha from ${PIPESTATUS[0]}. Each driver now READS steps.txt BACK at the end and exits 1 listing the failing steps, which closes the second half: a failing comparison could not previously reach the job's exit status at all. DEMONSTRATED with the drivers' own step() helper: the old form 'false | tee f; step name $?' records 'image_OLD RC=0', while 'set -o pipefail' records 'image_NEW_pipefail RC=7' and ${PIPESTATUS[0]} records 'image_NEW_pipestatus RC=7'. bash -n passes on all three. tools/parity/dsv4v_w7_cuda.sh was inspected and carries NO site of this shape (its tee pipelines are never followed by a status read), so it is deliberately untouched. The unguarded LP=$((N - 114)) derivation in dsv4v_w6_parity.sh, which could otherwise point the CLI comparison at a previous run's files, is guarded to 0..3 in the same change.
