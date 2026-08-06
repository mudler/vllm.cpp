# Raspberry Pi 5 CPU vs llama.cpp, 2026-08-06

Binding same-file comparison for the Raspberry Pi 5 Cortex-A76 arm of
`BACKEND-GATE-CPU-LLAMACPP`. Both engines were built locally for AArch64 under
QEMU and copied to the Pi for execution. Nothing was compiled on the Pi.

## Result

The implementations produce the same greedy text, but vllm.cpp does not yet
meet the llama.cpp speed floor on this four-core target. llama.cpp is 2.17x
faster in steady-state prefill and 1.53x faster in decode and combined
prompt-plus-generation time. vllm.cpp uses 24.2% less peak resident memory.

One vllm.cpp request, `--input-len 16 --output-len 64`, tokenized to 17 input
tokens. The binding llama.cpp arm therefore uses `pp17`, `tg64`, and
`pp17+tg64`, not the nominal 16-token target. vllm.cpp medians are three clean
process repetitions. llama.cpp reports three in-process timed samples after
its standard warmups. Every accepted leg was unthrottled.

| Axis | vllm.cpp | llama.cpp | vllm.cpp / llama.cpp |
|---|---:|---:|---:|
| Prefill | 12.81 tok/s (`17 / TTFT`) | **27.77 tok/s** | **0.461x** |
| Decode | 2.55 tok/s (`1000 / TPOT`) | **3.91 tok/s** | **0.653x** |
| Output-equivalent E2E | 2.46 tok/s (`64 / E2E`) | **3.77 tok/s** | **0.653x** |
| E2E latency | 26,018.39 ms | **16,998.49 ms** | **1.531x latency** |
| Peak RSS | **2.841 GiB** | 3.747 GiB | **0.758x, 24.2% less** |

The vllm.cpp median TTFT is 1,327.25 ms and median TPOT is 391.93 ms.
Its three E2E samples are 25,946.07, 26,018.39 and 26,021.14 ms, a 0.29%
range. llama.cpp's p17 result is 27.7681 +/- 0.0320 tok/s, tg64 is
3.9091 +/- 0.0142 tok/s, and combined p17+tg64 takes
16,998.49 +/- 25.13 ms. This is a measured speed gap, not run noise.

## Correctness

The three vllm.cpp performance repetitions emitted the existing 64-token
golden byte-for-byte, SHA-256
`0ec98eabb23e4148d540fcf79a2fe61678fb90fe462cdf28134af7a42fe6a826`.

An additional raw-prompt check ran both CLIs with the same text, greedy
sampling and 64 output tokens. Both tokenizers measured 24 prompt tokens and
both completed 64 tokens. The files differ only in trailing line endings;
after removing trailing whitespace both have SHA-256
`a5a630d7e9774c2300f5dda67a085d43ab1cf9125480c37208ae1c24a2eb25e0`.
Thus the competitor gap is performance-only for the checked stream.

## Provenance

- Pi: Raspberry Pi 5, four Cortex-A76 r4p1 cores, DotProd present, i8mm
  absent, 8 GiB RAM, Debian kernel `6.18.34+rpt-rpi-2712`.
- Model: `Qwen3.5-2B-UD-Q8_K_XL.gguf`, 2,823,978,240 bytes, SHA-256
  `a53988df91157d78acaf3c95e22db179d13f6236061bdb86576494dc99b1bc3b`.
- vllm.cpp: branch head `9044c2a7d`; assembly-default `vllm-bench` SHA-256
  `9eb57cf3760eaade9dcef03dda1648556577c44199369ad38bf42083efbc70a9`.
- llama.cpp: official tag `b9892`, commit
  `ee445f93d8a0a5033a46d1960e901ef5caec9a41`; `llama-bench` SHA-256
  `d9d93d8b38d0d8faa676f7d48f1a8fcbbc235f3a5697fc2ac787422d61783d52`.
  Ubuntu 24.04/GCC 13.3, `GGML_NATIVE=OFF`,
  `GGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16`, OpenMP on, accelerator
  backends off.

The historical project record calls llama.cpp pin
`237ad9b961f009ae19ac29dbce4cd0c1251f94b3` “b9892”. That object is no longer
available from the recorded fork or official remote, so it could not be
reproduced byte-for-byte. The official b9892 tag above is the binding
reconstruction. Its recorded anchors match exactly: portable Q8 dot
`ggml/src/ggml-cpu/quants.c:400`, Arm Q8 dot
`ggml/src/ggml-cpu/arch/arm/quants.c:1076`, Q8 repack
`ggml/src/ggml-cpu/repack.cpp:2725`, and `src/models/qwen35.cpp`. The exact
commit and binary hash are recorded so this substitution is explicit rather
than silently attributed to the unavailable object.

## Commands

vllm.cpp, repeated three times in the interleaved clean series:

```sh
taskset -c 0-3 env VLLM_CPP_CPU_THREADS=4 VT_CPU_Q8_DOT=auto \
  bin/final/vllm-bench \
  --model models/Qwen3.5-2B-UD-Q8_K_XL.gguf \
  --num-prompts 1 --input-len 16 --output-len 64 --concurrency 1 \
  --seed 0 --temperature 0 --output-token-ids evidence/out.json
```

llama.cpp, three timed repetitions in one loaded process:

```sh
taskset -c 0-3 bin/llama-b9892/llama-bench \
  -m models/Qwen3.5-2B-UD-Q8_K_XL.gguf \
  -p 17 -n 64 -pg 17,64 -t 4 -r 3 -o jsonl --progress
```

Peak RSS was measured in separate same-workload passes from Linux `VmHWM`,
sampled once per second with shell built-ins. The sampler does not run in the
timed performance series. An earlier attempt that forked two `awk` processes
every 50 ms is `VOID`: it inflated load and slowed both engines. A clean p16
cross-process series was also non-binding once vllm.cpp reported that the
nominal prompt actually contained 17 tokens; it was retained only as a
corroborating diagnostic.

## Raw evidence hashes

| Evidence | SHA-256 |
|---|---|
| Clean environment log | `ea2955cd99843645ff4c36d293e0fd61e49910b184e81e04f7f29888c4c3cecd` |
| Clean manifest | `3cf4c33c86ad938ff7df1f84f9db38a5ca198a6fb9cf4513e2640d00dc16320c` |
| vllm.cpp outputs, reps 1/2/3 | `07fcd93a0953d258af36e5c5ccd30ee215e399e50ac3d81e6efd8e551812479d` / `3ef8ca5e5497f2f68aeb8a4f74db3db7bfdf18bab4dc333e7437d17a9c03c060` / `1c796eac3e947dc474c7311bae7d5a64a223fe767c9569b4750ae5a99b7f60ce` |
| vllm.cpp RSS record | `426f2ef1c000da16a228f099dd3f1daf896c14a6f6ae1dcc4f6a5843e5457a27` |
| llama.cpp p17 JSONL | `7960035bd312117efa8f9c208c4dc53207d7379af42d9462c5a3c1add5b5e83f` |
| llama.cpp p17 environment | `13f07504d373558ffa74d57f5d2ca8f4453c935b4b3b78268f748ebc89a19fc6` |
| llama.cpp p17 RSS record | `5b00b4bfd1a13a6a0073452095278934cf9be57ea32fa1a70583a285c36ad1b3` |
| Exact-prompt vllm.cpp stdout | `8e0a7064f2b61462926ee554d3d15f3c5b1297c0ff1873cbc0e6391fde498097` |
| Exact-prompt llama.cpp stdout | `490e2379acd27dccf55dbdb0485052f4424a5af6654b6862e675adfaa172b67d` |
| Exact-prompt environment | `ed75ef54044549d379426472e0d935a3710a14049ec4321662492a207fc9bfc3` |

Raw files remain under
`rich@rpi5fan.lan:~/vllm-cpp-assembly/evidence/llama-compare-20260806/`.
