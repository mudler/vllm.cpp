# Spec: NormalizeDevF32Tile geometry mismatch — CaptureSafeReshape in RmsNormGated and AttnQkNormRopeGate

Row: `BACKEND-TENSTORRENT` (the APEX e2e owed item owns this blocker)
Issue: `ISSUE-LOCAL-01M2MKEKWH9J1WMVY0ZQVG4MKS`
State: ACTIVE (2026-09-16)
Git integration: one PR for spec and implementation (developer preference,
recorded for this row).
Base: `origin/main`.

## Scope

Fix the `binary_ng` broadcasting crashes that block the APEX-I-Nano 27B e2e
generation after the L1 overflow fix (#3206) got the probe past the first
forward pass. Two call sites share the same root cause:

1. **RmsNormGatedKernel** (`tenstorrent_ops.cpp:~6739`):
   `ttnn::multiply(ttnn::rms_norm(dev_x, ...), act)` receives operands with
   the same numel but incompatible shapes: `rms_norm` output `[rows, d]`
   (e.g. `[12288, 128]`) vs `act` `[256, 6144]` (the gate shadow's native
   geometry preserved by `NormalizeDevF32Tile`).

2. **AttnQkNormRopeGateKernel** (`tenstorrent_ops.cpp:~4737`):
   `ttnn::multiply(ttnn::multiply(x, inv), dev_w)` inside the `norm_rope`
   lambda receives `dev_k` at `[t, hkv*dh]` (e.g. `[256, 1024]`) instead of
   the intended `[t*hkv, dh]` (e.g. `[1024, 256]`), crashing against
   `dev_w` `[1, dh]` (e.g. `[1, 256]`) with `dim a: 1024, dim b: 256`.

Both crashes stem from `NormalizeDevF32Tile` preserving the shadow's native
rank-2 geometry (the L1 overflow fix from #3206). Out of scope: the L1
overflow itself (already fixed), any keep-quant behavior, any throughput
claim.

## Attribution (measured 2026-09-16, evidence in the issue)

- `VT_TT_SLOT_TRACE=1` e2e probe confirmed:
  `[TT-RNG] rows=12288 d=128 x=12288x128 g=256x6144` — `dev_g` is
  `[256, 6144]` while `dev_x` is `[12288, 128]`.
- `NormalizeDevF32Tile` (`tenstorrent_ops.cpp:7501`) preserves a rank-2
  shadow's native logical shape (the L1 fix: the free reshape to `[1, n]`
  overflows L1, and the member `view_device` on ROW_MAJOR with changed last
  dim corrupts data). So `dev_g` stays at the producer's `[256, 6144]`.
- `rms_norm(dev_x, ...)` reduces along `dev_x`'s last dim (128), producing
  `[12288, 128]`. `binary_ng` (`ttnn::multiply`) sees `[12288, 128]` x
  `[256, 6144]` — same numel (1,572,864), but no subtile broadcast rule
  matches (`binary_ng_device_operation.cpp:226-256`), and it throws at
  line 255.
- The gate and the `rms_norm` output carry the same logical elements
  (256 x 6144 = 12288 x 128 = 1,572,864), just viewed at different 2D
  shapes. The multiply needs them at the same shape.

## Design

The `CaptureSafeReshape` helper (`tenstorrent_ops.cpp:226`) is the
precedent: a metadata-only reshape (member `Tensor::reshape` /
`view_device`) changes the logical shape without a device program. The
L1 fix (#3206) could not use it inside `NormalizeDevF32Tile` because the
tensor was ROW_MAJOR at that point, and `view_device` on ROW_MAJOR with a
changed last dim rebuilds the buffer page mapping and corrupts data
(`tensor_ops.cpp:398` guards this path; lines 412-460 are the corrupting
rebuild).

The key insight: `view_device` line 398 short-circuits for TILE layout:

```cpp
if (input_tensor.layout() != Layout::ROW_MAJOR || !changing_last_dim) {
    // Safe path: reuse same buffer, same page_size, new spec only
```

For TILE layout, `layout != ROW_MAJOR` is true, so the condition is always
satisfied regardless of `changing_last_dim`. The corrupting page_size
rebuild (lines 412-460) is never reached for TILE.

After `NormalizeDevF32Tile` returns, the gate tensor is TILE layout (the
function's final step is `to_layout(TILE)`). After `sigmoid`/`silu`
(elementwise, layout-preserving), `act` is still TILE. At that point,
`CaptureSafeReshape(act, {rows, d})` is safe:

- During capture: member `Tensor::reshape` calls `view_device`, which takes
  the TILE safe path. No device program, no L1 allocation, no page_size
  rebuild.
- During eager: `CaptureSafeReshape` calls the free `ttnn::reshape`, which
  is fine outside capture. The row widths (6144 source, 128 target) are
  well within L1 — the overflow only occurs for `[1, n]` where n = 1.5M.

The fix adds a single reshape between the `sigmoid`/`silu` and the
`multiply` in `RmsNormGatedKernel`:

```cpp
ttnn::Tensor act =
    args.sigmoid_gate ? ttnn::sigmoid(dev_g) : ttnn::silu(dev_g);
// Reshape act to [rows, d] to match rms_norm(dev_x) output. The gate
// shadow's native geometry (e.g. [256, 6144]) is preserved by
// NormalizeDevF32Tile; act is TILE layout here, so CaptureSafeReshape's
// member view_device takes the TILE safe path (tensor_ops.cpp:398:
// layout != ROW_MAJOR short-circuits before the page_size rebuild that
// corrupts ROW_MAJOR reshapes).
const auto target_shape = ttnn::Shape({rows, d});
if (act.logical_shape() != target_shape) {
  act = CaptureSafeReshape(act, target_shape);
}
```

On the host-upload path (eager arm only), `dev_g` is already `[rows, d]`
from `UploadRows`, so the reshape is a no-op (the `if` guard skips it).

### Fix 2: AttnQkNormRopeGateKernel — dev_k reshape after NormalizeDevF32Tile

`AttnQkNormRopeGateKernel` calls `NormalizeDevF32Tile(kf_shadow, t*hkv, dh)`
to produce `dev_k`. The shadow's native geometry is `[t, hkv*dh]` (e.g.
`[256, 1024]` for APEX's T=256, hkv=4, dh=256). The intended target shape
is `[t*hkv, dh]` (e.g. `[1024, 256]`). `NormalizeDevF32Tile` preserves the
native geometry, so `dev_k` stays `[256, 1024]`. The subsequent
`norm_rope` lambda computes `ttnn::multiply(multiply(x, inv), dev_w)`
where `dev_w` is `[1, dh]` = `[1, 256]`. `binary_ng` sees
`dim a: 1024, dim b: 256` and crashes.

The fix adds the same `CaptureSafeReshape` guard after `NormalizeDevF32Tile`:

```cpp
dev_k = NormalizeDevF32Tile(std::move(kf_shadow),
                            static_cast<uint32_t>(t * hkv),
                            static_cast<uint32_t>(dh));
{
  const auto k_target = ttnn::Shape({static_cast<uint32_t>(t * hkv),
                                      static_cast<uint32_t>(dh)});
  if (dev_k.logical_shape() != k_target)
    dev_k = CaptureSafeReshape(dev_k, k_target);
}
```

`dev_k` is TILE layout (NormalizeDevF32Tile's final `to_layout(TILE)`),
so the same `view_device` TILE safe path applies. Same numel, same tile
count — pure metadata view.

Note: `dev_q` does NOT need this fix because it is created via explicit
`ttnn::reshape` + `to_layout(TILE)` + `ttnn::slice`, not via
`NormalizeDevF32Tile`. `dev_cs` also does not need it because its native
geometry `[t, rot]` already matches the target shape.

## Tests

- Red-first (fix 1): the APEX e2e probe with `VT_TT_SLOT_TRACE=1` crashed
  at `binary_ng` "Invalid subtile broadcast type" in `RmsNormGatedKernel`
  (`dim a: 12288, dim b: 128` vs `dim b: 6144`). Green-after: the probe
  advances past the MLP gate multiply.
- Red-first (fix 2): after fix 1, the probe crashed at `binary_ng`
  broadcasting in `AttnQkNormRopeGateKernel` (`dim a: 1024, dim b: 256`).
  Green-after: the probe advances past the attention norm+rope path.
- Bit-exactness guard: the existing gated-norm and attention tests in the
  op suite stay green unchanged (small shapes where the native geometry
  already matches the target hit the same code path; the reshape guard is
  a no-op there).
- Full backend suite green on the P150 (77/77 baseline, 524,495 asserts).

## Risks

- The TILE reshape must preserve the element ordering. For TILE layout,
  the physical storage is tile-major (32x32 tiles in row-major order).
  Reshaping `[256, 6144]` to `[12288, 128]` reinterprets the tile grid:
  256 rows / 32 = 8 tile-rows, 6144 / 32 = 192 tile-cols → 8 * 192 = 1536
  tiles. Reshaped: 12288 / 32 = 384 tile-rows, 128 / 32 = 4 tile-cols →
  384 * 4 = 1536 tiles. Same tile count, same physical bytes. The element
  ordering is preserved because TILE layout stores tiles in row-major
  order and elements within tiles in row-major order — a 2D reshape that
  keeps the same total tile count and tile volume does not move bytes.
  The e2e probe and the op suite pin this.
- `CaptureSafeReshape` passes the old padded shape to `view_device`. For
  `[256, 6144]` (both tile-aligned), the padded shape equals the logical
  shape. `view_device` sees `changing_last_dim = false` (padded shapes
  match), so even without the TILE short-circuit, the safe path at line
  398 would be taken. The fallback path (tile-aligned padded `[12288, 128]`,
  both aligned) would also be safe.

## Gates

1. APEX e2e probe gets past both `binary_ng` crashes (RmsNormGated and
   AttnQkNormRopeGate). Red-first: two distinct crash traces. Green-after:
   the probe advances through all 64 trunk blocks and produces tokens.
2. Full backend suite green on the P150 (77/77, 524,495 asserts).
3. 0.8B vehicle battery unchanged (the gated-norm and attention paths are
   production-reached there; their shapes match the target geometry so the
   reshape guard is a no-op).

## Owed

- The APEX e2e generation gate (this fix unblocks the multiply; later
  blockers get named when reached). Next blocker named: OOM from memory
  fragmentation in `MatmulBTQuant` → `DecodeKeepQuantWordsF32` →
  `to_layout(TILE)` → `ttnn::tilize_with_val_padding`. The 2.5 GB temp is
  the full lm_head weight [248320, 5120] in Q6_K, tilized at once instead of
  being chunked. The chunking policy (`KeepQuantChunkRowsOverride`, 256 MiB
  plane budget at `tenstorrent_ops.cpp:~2876`) does not appear to apply to
  the grouped decode path.
- F1/F2 from #3206 review (served-geometry test, kSigmoidGateBf16/L2Norm
  migration) — unchanged, still owed.

## Stop conditions

- If `CaptureSafeReshape` on TILE throws or corrupts data on the e2e
  probe, stop and record the error. The `view_device` analysis says it
  should not, but the e2e is the authority.
- If the reshape guard fires on a shape where `CaptureSafeReshape` is
  unsafe (e.g. the gate arrives as ROW_MAJOR, not TILE), stop and
  escalate. `NormalizeDevF32Tile`'s final `to_layout(TILE)` guarantees
  TILE, but a future caller that bypasses `NormalizeDevF32Tile` would
  not.
