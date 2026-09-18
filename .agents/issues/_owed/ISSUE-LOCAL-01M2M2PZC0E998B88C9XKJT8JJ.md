ID: ISSUE-LOCAL-01M2M2PZC0E998B88C9XKJT8JJ
Title: Split tenstorrent_ops.cpp into per-module translation units
Row: -
State: OPEN
Kind: refactor
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-16
Updated: 2026-09-16
Closed: -

## Problem

src/vt/tenstorrent/tenstorrent_ops.cpp is 10,445 lines / 501 KB and grows with every keep-quant and GDN wave; it carries at least ten distinct modules (residency/slots, weight staging, embedding shadows, PagedKv shadows, keep-quant decode+grouped+int8-dot ~3.2k lines, PA/RAC metadata, GDN prefill/decode ~2.1k lines, trace capture, host-free decode, W4d alloc trace) separated by banner comments. The size makes every fix require reading past unrelated sections and concentrates merge pressure in one file. Mechanical split along the existing banner seams into per-module TUs plus a shared internal header; staged one module per PR, residency first. Gates: full backend suite unchanged (77/77), 0.8B vehicle unchanged, record anchors repaired per PR. No behavior changes, no checker changes.

## Resolution

-
