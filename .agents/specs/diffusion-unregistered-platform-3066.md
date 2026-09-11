# Test the missing diffusion platform independently of registered backends

Row: `LTX25-VAE-DEVICE-RESIDENCY`.
Issue: [#3066](https://github.com/mudler/vllm.cpp/issues/3066).
Parent spec: [VAE device residency](ltx25-vae-device-residency.md).
Base: `08a34c3a74d78046f83886f242d07110a70ff45e`.

## Scope and source

Repair only the missing-platform fixture in
`tests/vllm/multimodal/test_diffusion_device_seam.cpp`.
The case introduced by `57efbdf3c` assumes ROCm has no registered platform.
The HIP static registrar in `src/vllm/platforms/rocm.cpp` can register it.
This is a local test-harness correction, not a change to oracle behavior,
production refusal, device memory policy, or sampling.

## Design and risks

Make the existing refusal case run with ROCm already registered. Register a
process-lifetime fake platform only when no real ROCm platform exists; never
replace a real platform for this regression. This must first reproduce the
old fatal missing-platform precondition on CPU.

Choose a non-CPU device type whose platform is verified absent at execution
time. The registry has no unregister operation, so do not rely on case order.
Require a candidate rather than skipping the test. Register the fake backend
and all four required operation providers for that type, and verify the backend
and absent platform before decoding. Derive the expected device name from the
chosen type. Preserve the nonempty error, device name, platform, and pool checks.
The fixture must not invoke a real accelerator.

### Preserve the registered platform by identity

Review of `6a45286c2f861f8d9a851f6deb4590d3763b14c3` found a coverage gap:
unconditional ROCm registration still passed all 16 focused assertions.
The guarded fixture is correct, but those assertions do not detect replacement.

Put its single registration guard in a test-only helper accepting a fallback
platform reference. Call it with two distinct process-lifetime fallbacks.
The first call fills an absent CPU-test slot or preserves the real HIP platform.
Check the exact expected pointer, then require the second call to retain it.
Keep all existing provider and refusal assertions unchanged.
A separate guarded sentinel setup would move the same blind spot to another
guard, so use one helper for both absent and present contexts.

## Tests and gates

Run the existing case after adding the registered-ROCm regression and capture
its intended red result before changing slot selection. Then run the complete
CPU executable in default, name, and randomized orders. In a scratch copy,
delete the production `RequirePooledDevice` call and require the focused case
to fail its error assertions; restore the source byte-for-byte and rerun.
Run `scripts/agent-preflight.sh` on the immutable candidate.
The operator owns the HIP build and execution under a device lease; CPU results
do not satisfy that device gate. Independent review remains required.

Before the coverage change, reproduce the surviving overwrite mutation in
scratch. After adding the assertions, the same mutation must fail pointer
identity. A no-op helper must fail the CPU case's registration precondition.
Restore scratch bytes and rerun focused green after each mutation. Run all
three full CPU orders with pooling enabled and with `VT_POOL_BYPASS=1`.
No registry API, production code, or pool fixture changes belong to this repair.

## Stop conditions

Stop if the fixture needs a production registry API change, no unregistered
non-CPU slot is available, or the refusal cannot be distinguished from an
unrelated missing provider. Do not weaken or skip the refusal assertions.

## Now

ACTIVE: fixture repair specified before the regression and implementation.
