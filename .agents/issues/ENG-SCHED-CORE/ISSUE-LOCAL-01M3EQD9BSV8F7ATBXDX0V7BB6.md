ID: ISSUE-LOCAL-01M3EQD9BSV8F7ATBXDX0V7BB6
Title: Port: XD-RoPE to M-RoPE unification (upstream 719284fe15)
Row: ENG-SCHED-CORE
State: OPEN
Kind: tech-debt
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-26
Updated: 2026-09-26
Closed: -

## Problem

Upstream commit 719284fe15 unifies XD-RoPE into M-RoPE in transformers_utils/config.py. The C++ port uses mrope_section directly (not xdrope_section), so the deprecation path does not apply. Classification: INVENTORY (C++ already uses M-RoPE). Sync cycle a7c23ac96d.

## Resolution

-
