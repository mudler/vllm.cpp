# Isolate diffusion pool fixtures from earlier cases

Row: `LTX25-VAE-DEVICE-RESIDENCY`.
Issue: [#3068](https://github.com/mudler/vllm.cpp/issues/3068).
Parent spec: [VAE device residency](ltx25-vae-device-residency.md).
Base: `08a34c3a74d78046f83886f242d07110a70ff45e`.

## Gap and scope

The two pool fixtures introduced in `57efbdf3c` share the process-wide pool.
Earlier decodes can leave matching or larger retained blocks. The cold-pool
case then makes no new driver allocation, and the padding case can borrow
different size classes between its first and second decode.

Unchanged main fails two assertions with `--order-by=name` and three with
`--order-by=rand --rand-seed=3066`. The default order passes. Repair only these
two fixtures in `tests/vllm/multimodal/test_diffusion_device_seam.cpp`.
Do not include the separate missing-platform repair from #3066.

## Design and source

Use the existing `DevicePool::Drain(Backend())` boundary operation. It frees
only retained blocks, preserves counters, and is a no-op under pool bypass.
Assert that retained bytes are zero before each fixture's first device decode.
Commit those precondition assertions and execute their red result first.
Then drain before each assertion, keeping the fixtures' counter deltas.

Keep both named cases, every allocation and hit/miss assertion, noise replay,
host comparisons, and both zero-padding comparisons. Never drain between the
first and second decode: the second must still reuse the first one's blocks.
No production allocator, API, memory policy, or oracle behavior changes.
The parent spec records the production pool and CPU zero-fill source anchors.

## Tests and gates

Run the full executable in default, name, and seeded-random orders on the
candidate. Run those same orders with `VT_POOL_BYPASS=1`; bypass retains its
existing positive allocation and correctness assertions, not a skip.
The two pool cases must also pass separately.

In scratch copies, remove each fixture drain and require its cold precondition
to fail after earlier cases. Mutate production pool allocation to bypass the
shared pool and require the allocation fixture to fail. Delete the CPU pad
zero-fill and require the pooled padding fixture's pixel comparisons to fail.
Restore source bytes and rerun focused green after every mutation.

Run the complete host preflight on the immutable head and obtain fresh review.
The unchanged production archive built from the pinned main may be reused to
link this test-only change. Record that provenance. CPU tests do not establish
the separate real-device padding behavior still owed in the parent spec.

## Risks and stop conditions

Draining must not hide reuse within the two-decode fixture. Keep the second
decode's no-new-allocation and unchanged-miss checks. If draining retained
blocks cannot isolate the fixture, stop before changing allocator semantics.
Do not weaken a guarantee or modify the missing-platform case to make this
separate change pass a HIP build.

## Now

ACTIVE: fixture isolation specified before the new cold-pool assertions.
