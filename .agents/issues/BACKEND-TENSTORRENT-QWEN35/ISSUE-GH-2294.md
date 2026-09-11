ID: ISSUE-GH-2294
Title: CopyDeviceDeviceIfCapture records src's geometry in dst's slot without updating dev_rows/dev_cols — latent until a differing-geometry D2D copy exists
Row: BACKEND-TENSTORRENT-QWEN35
State: CLOSED
Kind: bug
GitHub: 2294
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-29
Updated: 2026-08-30
Closed: 2026-08-30

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> **Row**: `BACKEND-TENSTORRENT-QWEN35` (found by the W7 review, #2282; audited at the repair head).
>
> `CopyDeviceDeviceIfCapture` (`src/vt/tenstorrent/tenstorrent_ops.cpp:5948`) replaces dst's device shadow with a clone of src's device tensor — src's logical shape — **without updating `d->dev_rows/dev_cols`**, while its guard only requires equal slot byte sizes (dtype-blind). If a caller ever presents two tracked slots of equal byte size but different `[rows,cols]` geometry (or different element sizes), a later `EnsureDevice2D(dst, recorded_rows, recorded_cols)` exact-shape hit returns src-shaped bytes to a declared-geometry consumer.
>
> **Static audit at the W7 repair head found no such caller, so this is latent**: every `qwen3_5.cpp` copy site (569, 1185, 1230, 1298, 1308, 1380-1389, 1467) passes a host source and is refused at `FindSlot(src) == nullptr`; the only tracked device→device callers on the tree are GatherRows-style row gathers (`qwen3.cpp:216`, `commandr.cpp:176`, `deepseek_v2.cpp:642`, the gemma/commandr family) whose destination is allocated with the source's dtype and row geometry, so slot byte equality implies equal logical geometry. The stale record is never observed with a differing geometry.
>
> **Fix when it goes live**: the same one-liner W7 applied to `CopyDeviceDeviceIfResident` — set `d->dev_rows/dev_cols` from `src_dev.logical_shape()` in the second lock scope. A bit-identical audit (or a test proving the differing-geometry case) must accompany the fix, since the arm runs under capture where staging semantics are strictest.
>
> Following-Agents-Protocol: true
>

## Resolution

GitHub records closing pull request #2364 (https://github.com/mudler/vllm.cpp/pull/2364) merged on 2026-08-30 as commit `03e0dcd19c9a7539bbaad0e17bebe9a2a892c699`. GitHub closed issue #2294 on 2026-08-30.
