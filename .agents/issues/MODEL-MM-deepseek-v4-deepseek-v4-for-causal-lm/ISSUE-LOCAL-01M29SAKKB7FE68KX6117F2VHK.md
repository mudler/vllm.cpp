ID: ISSUE-LOCAL-01M29SAKKB7FE68KX6117F2VHK
Title: DeepSeek-V4 vision W6: the input-stage comparison asserts our patch ordering instead of measuring it
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

tools/parity/dsv4v_w6_oracle_dump.cpp rearranges the oracle's normalised pixel buffer into OUR claimed patch-row order [(vy*nvx+vx)][(c*P+dy)*P+dx] before writing oracle-<tag>-input.f32. The -input.f32 comparison in dsv4v_w6_compare.py therefore measures the normalisation ARITHMETIC -- mean, standard deviation and the bf16 narrowing -- and ASSERTS the ordering rather than measuring it: if our patch order were wrong, that file would still compare exact, because both sides were written in the same wrong order. The gate as a whole is NOT blind to ordering, because the block-level permutation check best-matches every image row against the oracle's own block and requires the identity. The input stage line simply says less than its name suggests. RECORDED rather than changed: making the input stage measure ordering needs the oracle's buffer written in the ORACLE's order plus a separately declared mapping, which is a second description of the layout that can drift from the first, and the permutation check already covers ordering downstream. Found by fresh review 2026-09-12.

## Resolution

2026-09-12: recorded, which is the fix chosen here, and the reasoning is stated so a later reader can revisit it. tools/parity/dsv4v_w6_oracle_dump.cpp carries a comment at the rearranging loop saying the input stage measures the normalisation ARITHMETIC and ASSERTS the ordering rather than measuring it, that a wrong patch order would still compare exact because both sides are written in the same order, and that the gate's ordering evidence is the downstream block-level permutation check which requires the identity for every image row. The spec says the same under '## Owed', and the W6 evidence table's 'input pixels' row is annotated so the number cannot be read as ordering evidence. The alternative -- writing the oracle buffer in the ORACLE's order plus a separately declared mapping -- was REJECTED because it is a second description of the layout that can drift from the first, while the permutation check already covers ordering downstream. No claim in the spec rested on the input stage proving ordering, so nothing is withdrawn.
