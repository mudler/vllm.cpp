# Tenstorrent keep-quant 27B decode (capture-default)

Qwen3.8-27B-Q4_K_M on the P150, with the keep-quant int8-dot decode
path (`VT_TT_KEEPQUANT_INT8DOT=1`). The first end-to-end completion of
the 27B bench on the Tenstorrent backend, after fixing seven layers of
capture-time program-cache misses in the decode graph trace capture
(commits `26b88846a`, `69b0d4f8e`, `ead93289b` on
`row/BACKEND-TT-KEEPQUANT-W3`).

## Results

| Model | Batch | Prompts | In/Out | TTFT (s) | TPOT (s) | Output tok/s | Total tok/s | Duration (s) |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B-Q4_K_M | 2 | 2 | 128/32 | 64.5 | 2.0 | 0.49 | 2.45 | 131 |
| Qwen3.8-27B-Q4_K_M | 2 | 2 | 128/32 | 1389 | 29.7 | 0.03 | 0.14 | 2326 |

Both runs: `--num-prompts 2 --input-len 128 --output-len 32 --concurrency 2
--num-blocks 64 --max-num-batched-tokens 64 --seed 0`, 64 generated tokens,
RC=0, 912/916 JIT cache hits (27B) / 691/695 (0.8B).

## Reproduce

```sh
flock "$HOME/gpu.lock" bash -c '
  ~/Sources/tt/luwen/target/release/reset && sleep 15 &&
  cd /tmp/row-tt-w3 && source ~/Sources/tt/env-tt-common.sh &&
  export VT_TT_KEEPQUANT_INT8DOT=1 &&
  timeout -k 10 3600 ./build/examples/vllm-bench \
    --model /mnt/models/unsloth-qwen3.8-27B-gguf/Qwen3.8-27B-Q4_K_M.gguf \
    --num-prompts 2 --input-len 128 --output-len 32 --concurrency 2 \
    --num-blocks 64 --max-num-batched-tokens 64 --seed 0
'
```

## Limitations

- 2 prompts only (smoke test); multi-request throughput not measured
- No token-exact gate (correctness not verified beyond RC=0)
- Single seed (0); determinism not tested
- TTOT includes device-resident decode-graph replay overhead
- The 27B prefill (TTFT) dominates the run (~23 min for 128 tokens)
