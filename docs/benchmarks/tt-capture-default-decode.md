# Tenstorrent capture-default decode rate

Qwen3-0.6B on the P150 (`thalia`), the #2566 recipe re-taken on the #1625
flip tree (`row/BACKEND-TENSTORRENT-HOST-FREE-FORWARD` @ `d88aed67b`,
2026-09-04) where captured decode is the SHIPPED DEFAULT. Batch 1, 96
completion tokens, `--repeat 5`, leg 1 of each arm discarded, warm medians
over the remaining four runs, order-alternated A/B/A·B/A/B triples, one
`flock $HOME/gpu.lock` and one `tt-smi -r 0` reset for the whole batch.
Zero replays and zero fatals on every leg.

| Arm | Warm median tok/s | Per-triple medians |
|---|---:|---|
| default (capture on, no env) | **27.70** | 27.63 / 27.67 / 27.97 |
| `VT_TT_DECODE_CAPTURE=0` (opt-out) | 12.86 | 12.88 / 13.02 / 12.76 |
| ratio | **2.15x** | |

The arms differ ONLY in capture: `VT_TT_HOST_FREE_DECODE` stays unset in
both, so the ratio isolates the decode-graph capture and not the host-free
residency both arms share. The 2026-09-01 measurement of the same captured
leg as an explicit opt-in (27.47 tok/s, #2566) agrees within 1%, and the
opt-out arm reproduces the eager host-free class measured on 2026-08-30
(12.21 tok/s).

Reproduce (one leg; the batch alternates the two arms and discards leg 1):

```sh
source "$HOME/Sources/tt/env-tt-common.sh"
export TT_METAL_RUNTIME_ROOT="$HOME/Sources/tt/tt-metal"
export LD_LIBRARY_PATH="<build>:$TT_METAL_RUNTIME_ROOT/build_Release/lib:$TT_METAL_RUNTIME_ROOT/build_Release/lib64"
"$HOME/Sources/tt/.venv/bin/tt-smi" -r 0
build/examples/vllm-cli --model <Qwen3-0.6B snapshot dir> \
  --prompt 'Write a short story about a robot learning to paint.' \
  --max-tokens 96 --repeat 5
# opt-out leg: prefix VT_TT_DECODE_CAPTURE=0
```

Limitations: single prompt, single request, 0.6B only — the shape the flip
gates on. Multi-request captured throughput is the #1627 async lane and is
not measured here. The 4B and Mistral captured arms are not gated (#2811,
#2812) and publish no figure.
