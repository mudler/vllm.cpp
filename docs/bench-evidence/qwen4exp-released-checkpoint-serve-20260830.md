# The RELEASED `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S artifact through `examples/server`, 30 August 2026

W5n of [`MODEL-MM-QWEN4-EXP`](../../.agents/specs/qwen4-exp-flash-next.md),
[#2031](https://github.com/mudler/vllm.cpp/issues/2031).

**What this file is for.** Every measurement this row has ever taken ran on a
SYNTHETIC fixture whose weights are a deterministic ramp. W5L served a
`POST /v1/completions` through `examples/server` on that fixture. **No byte of a
published `qwen4exp` checkpoint had ever been read by this tree on any host.**
This file records the first attempt to change that, and the
`docs/USAGE.md` weights row AGENTS.md "Say which weights, and from where"
requires beside it.

**Nothing here is a benchmark.** One prompt, greedy, one repetition, no clock
window (the lease has no `CAP_SYS_ADMIN`, so clocks cannot be pinned -- see
`.agents/environment.md`), and a shared network filesystem whose contention this
host can only partly observe. No number in this file may be quoted as a speed,
and no ratio against llama.cpp is computed from it.

## 1. The device, and why it is not `dgx`

Developer instruction, 2026-08-30: CPU work goes to `thor`. Measured capacity,
read from `rc describe`:

| Device | `mem_total_bytes` | GiB | Holds a 67.564 GiB artifact? |
|---|---:|---:|---|
| `dgx:gpu0` | 128,452,960,256 | 119.63 | yes |
| `thor:gpu0` | 131,856,982,016 | 122.80 | yes, and it is the LARGER box |
| `strix:gpu0` | 67,038,691,328 | 62.44 | **no** -- smaller than the artifact |

A dgx job was submitted first and **cancelled** (`646c9f2f-...`,
`killed (cancelled)`, ran 4 s) when the instruction arrived.

**`rc run --as <name>` costs you kill ownership.** That submit passed
`--as w5n@mudler-ubuntu-box`; `rc kill` then refused with
`rc: not_job_owner: only the submitter or an admin may kill this job`, and only
`RC_SUBMITTER=w5n@mudler-ubuntu-box rc kill ...` worked. Later submits pass no
`--as`. A job you cannot kill on a shared fleet is a hazard, not a convenience.

## 2. The lease worker, measured rather than assumed

`thor:gpu0`, pod `rc-worker-n8smh`, uid 0, aarch64, kernel `6.8.12-1021-tegra`,
14 cpus, Mem total 122 GiB / available 118 GiB, swap 30 GiB. `/` is an overlay
with 538 G free. `CapEff = CapBnd = 00000000a80425fb` -- the 14-capability OCI
default, byte-identical to the value `.agents/environment.md` records for dgx,
holding no `CAP_SYS_ADMIN`. Toolchain: cmake 3.28.3, ninja, g++ 13.3.0,
python3 3.12.3, curl.

### Three proofs taken ON THOR, not carried over from dgx

A prior wave established that `checkpoints/` is a SIBLING of `rc/` on the share
and is invisible from a lease, which is why a copy exists under `rc/`. That is a
fact about one device on one day, so it was re-proved here:

1. `findmnt -no SOURCE,FSTYPE,TARGET /workspace` ->
   `//192.168.68.102/Data[/rc]   cifs   /workspace`.
   The `[/rc]` subpath is printed by the mount itself: `/workspace` is the `rc/`
   subfolder, so a sibling directory is unreachable by construction.
2. A token file written by the development box at 2026-08-30T20:26:13Z, naming
   this worktree and base SHA `1ce580453`, was read back from inside the lease.
3. **Artifact identity, not just artifact presence.** Shard 1's sha256 computed
   INSIDE the lease is
   `88a1420825a9304063e882ada29d438263617f51ac8923d438d927496693bafd`, equal to
   the pinned value. The same name on a different mount would not have matched.

## 3. The artifact, censused from its own headers before any lease time was spent

`/mnt/nas_share/rc/q4exp-bench/UD-IQ1_S/`, three shards,
10,946,624 + 49,990,818,368 + 22,544,696,352 = **72,546,461,344 bytes = 67.564 GiB**.

**`split.tensors.count = 1224` is VERIFIED here**, and the spec's `## Owed`
records it as UNVERIFIED on the grounds that shard 1 is the metadata shard and
reports `n_tensors = 0`. That is true of shard 1 and is not the whole file:
shard 2 declares 595 tensors and shard 3 declares 629, and 595 + 629 = 1224.

| ggml id | type | tensors | params | bf16 GiB if expanded |
|---:|---|---:|---:|---:|
| 0 | F32 | 557 | 78,373,760 | 0.15 |
| 8 | Q8_0 | 244 | 747,110,400 | 1.39 |
| 12 | Q4_K | 2 | 1,271,398,400 | 2.37 |
| 13 | Q5_K | 212 | 2,219,704,320 | 4.13 |
| 14 | Q6_K | 40 | 611,450,880 | 1.14 |
| 16 | IQ2_XXS | 28 | 23,488,102,400 | 43.75 |
| 19 | IQ1_S | 68 | 57,042,534,400 | 106.25 |
| 20 | IQ4_NL | 49 | 91,465,564,160 | 170.37 |
| 30 | BF16 | 24 | 19,660,800 | 0.04 |
| | **TOTAL** | **1224** | **176,943,899,520** | **329.58** |

Two facts decided that this was worth a lease at all, both read off the file and
this tree rather than assumed:

- **Every one of the nine encodings has BOTH a `to_float` row decoder and a
  `vec_dot` kernel** in `src/vt/cpu/cpu_quant_dequant.cpp` and
  `src/vt/cpu/cpu_quant_dot.cpp`. `IQ1_S` and `IQ2_XXS`, which carry 80.5 G of
  the 176.9 G parameters, are both present.
- **Every one of the 643 block tensors is block-aligned on `ne0`** (checked
  against 32 for Q4_0/Q5_0/Q8_0/IQ4_NL and 256 for the K- and i-quants): zero
  misaligned. So `RouteGgufTensor`'s `k % vt::BlockElems(dt) == 0` rule forces no
  tensor to expand.

Together these predict a keep-quant residency near the file size rather than the
329.58 GiB bf16 expansion. **That was a prediction before the run, and section 5
says what actually happened.**

The largest tensor is `per_layer_token_embd.weight` `[160, 320001536]`, IQ4_NL,
51,200,245,760 parameters -- the n-gram gather table, 26.822 GiB kept and
95.368 GiB expanded, which is the figure #2083 refuses a non-CPU device over.

## 4. The run

Built in the lease, CPU-only, from base SHA `1ce580453`:

```sh
cmake -S "$SRC" -B "$BLD" -GNinja -DCMAKE_BUILD_TYPE=Release \
  -DVLLM_CPP_CUDA=OFF -DVLLM_CPP_METAL=OFF -DVLLM_CPP_VULKAN=OFF \
  -DVLLM_CPP_HIP=OFF -DVLLM_CPP_TENSTORRENT=OFF \
  -DVLLM_CPP_BUILD_TESTS=OFF -DVLLM_CPP_BUILD_EXAMPLES=ON -DVLLM_CPP_HF_DOWNLOAD=OFF
ninja -C "$BLD" -j 4 server      # -j 4 per AGENTS.md
```

`cmake configure rc=0`, `ninja rc=0` in **535 s**, 530 steps, binary 21,000,672
bytes. `CUDA FA2 compiled-arch manifest: []` and `Triton AOT: default OFF ...
VLLM_CPP_CUDA is OFF` confirm no CUDA entered the build, so thor's `sm_110` is
not exercised by any of this.

The server command, verbatim:

```sh
/tmp/w5n/build/examples/vllm-server \
  --model /workspace/q4exp-bench/UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf \
  --device cpu --host 127.0.0.1 --port 8099 \
  --block-size 16 --num-blocks 128 --max-model-len 256 \
  --served-model-name qwen4exp --verbose
```

No `--offload-config`: the SHIPPED default residency is what a user gets, and
that is what this run is for.

## 5. What happened: it LOADED, it SERVED, it produced NO TOKEN

**The load succeeded.** `load wall time: 4446s  ready=1` -- 74.1 minutes.
`/health` returned 200, `/v1/models` answered, and the engine sized its caches
over all three published groups. The 3-shard split GGUF opened, and **no
encoding, shard, role or device refusal fired at any point in the load.**

**Residency, and the keep-quant prediction of section 3 CONFIRMED:**

| quantity | value |
|---|---|
| `VmHWM` (peak resident) | 72,567,932 kB = **69.206 GiB** |
| artifact on disk | 67.564 GiB |
| overhead above the file | ~1.64 GiB |
| anonymous memory (`free` `used`) peak | ~11 GiB |
| bf16 expansion this AVOIDED | 329.58 GiB |
| `VmPeak` (virtual address space, NOT residency) | 148,393,352 kB = 141.5 GiB |

**The request returned HTTP 500 and no token.** `curl rc=0`, request wall
**1.39 s**. It died in the first prefill:

```text
INFO prefill id=cmpl-0-0 status=begin prompt_tokens=5 already_computed=0 remaining=5 scheduling=5 chunked=0
engine-fatal: EngineCore busy loop threw: vt: qwen4_exp_gated_residual: input_mix_weight_down must be float (f32/bf16 for outputs) at /tmp/w5n/src/src/vt/ops.cpp:2552
```

Response body, verbatim:

```json
{"error":{"code":500,"message":"EngineCore encountered an issue. See stack trace (above) for the root cause. [vt: qwen4_exp_gated_residual: input_mix_weight_down must be float (f32/bf16 for outputs) at /tmp/w5n/src/src/vt/ops.cpp:2552]","param":null,"type":"InternalServerError"}}
```

## 6. The cause, and it is a FIXTURE-SHAPED BLIND SPOT

`src/vt/ops.cpp:2551-2560` applies `check_operand` to `mix_down` and `mix_up`:

```cpp
VT_CHECK(IsFloat(t.dtype) && (!is_out || IsOutFloat(t.dtype)),
         std::string(name) + ": " + what + " must be float (f32/bf16 for outputs)");
```

**The released checkpoint stores every hyper-connection mix weight as Q8_0:**

| tensor | shape | ggml type | count |
|---|---|---|---:|
| `blk.N.hc_attn_down.weight` | `[10240, 320]` | **Q8_0** | 48 |
| `blk.N.hc_attn_up.weight` | `[320, 10240]` | **Q8_0** | 48 |
| `blk.N.hc_ffn_down.weight` | `[10240, 320]` | **Q8_0** | 48 |
| `blk.N.hc_ffn_up.weight` | `[320, 10240]` | **Q8_0** | 48 |
| `output_hc_down.weight` | `[10240, 320]` | **Q8_0** | 1 |
| `output_hc_up.weight` | `[320, 10240]` | **Q8_0** | 1 |

194 tensors, and **the loader is behaving correctly**: Q8_0 has a `vec_dot`
(`src/vt/cpu/cpu_quant_dot.cpp`), `10240 % 32 == 0` and `320 % 32 == 0`, so
`RouteGgufTensor` returns `kKeepQuant` exactly as its policy says it should. The
op then cannot consume what the loader produced. (`hc_*_norm` and
`hc_*_inject` are F32 and are not implicated.)

**Why eleven waves never met this.** `tests/support/qwen4_exp_gguf_fixture.h`
emits **exactly one** Q8_0 tensor -- the n-gram table, line 364
(`Q8_0Bytes(kNgramRows, kPleRow)`) -- and writes `output_hc_down.weight`,
`output_hc_up.weight` and every per-layer `hc_<side>_*` through the default
**F32** path (lines 365-375). So the fixture hands `vt::Qwen4ExpGatedResidual`
float operands and the published artifact hands it Q8_0 blocks. Every earlier
wave, W5L's `examples/server` end-to-end included, gated the float case only.
This is the class of defect a synthetic fixture cannot find: the fixture chose
a residency the real file does not use.

**Nothing was repaired here, deliberately.** Three repairs are conceivable --
dequantize these tensors in the layer loop, give the op a keep-quant arm, or
make the hyper-connection role keep-quant-ineligible in `RouteGgufTensor` -- and
choosing among them is a design decision owed its own spec, test and mutation.
It is recorded under the spec's `## Owed`. **No product code changed in this
wave**, so the forward golden is untouched at `max|diff| = 0.00982457` by
construction rather than by re-measurement.

## 7. The concurrency clamp that printed is NOT the one W5L added

```text
INFO recurrent-state budget: reduced max_num_seqs from 32 to 1. The KV pool
(128 blocks) holds 1 unified pages of 1664 tokens (one page = one 3391504-byte
GDN state), and each sequence owns 1 of them.
```

That is the **recurrent-state budget** clamp. W5L's
`serves_one_sequence_per_step` line did **not** print, and the reason is the
ordering in `LoadedEngine::ResolveMaxNumSeqs` (`model_loader.cpp:1647-1674`):
the budget clamp runs first and already returned 1, so
`serves_one_sequence_per_step && resolved > 1` was false. Both ceilings agree on
1, so the served concurrency is right -- but **W5L's specific line is not
evidenced by this run** and is not reported as verified by it.

## 8. What may NOT be quoted from this file

- **No speed number, and no comparison to llama.cpp.** The 4446 s load ran on a
  shared CIFS share whose contention this host can only partly observe: a
  third-party `sha256sum` on `GLM-5.3-UD-IQ1_S-00003-of-00006.gguf` was reading
  the same share during the first attempt, and other LAN clients and NAS-side
  load are invisible from here. Our own `expand_nk` orientation repacking is ON
  by default and its CPU cost is not separated from the I/O. The llama.cpp arm's
  16m27s was a different device under unknown conditions and is not a control.
- **No token number.** No token was produced.
- **Nothing about any other arm.** Six other published quants are staged and
  none was run.
