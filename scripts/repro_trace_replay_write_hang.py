# Scratch repro (vllm.cpp TT backend bisect, 2026-08-16) — NOT upstream.
#
# Reproduce the ~38-replay hang from the vllm.cpp decode driver as a standalone
# ttnn script. The driver's toxic pattern: capture a trace of a ttnn op whose
# INT32 inputs (page_table, cur_pos) are persistent device tensors, then replay
# N times with ttnn.copy (copy_to_device) into those inputs between replays,
# trace staying live. Mirrors tests/ttnn/unit_tests/operations/sdpa/sdpa_test_utils.py
# run_test_paged_attention_trace path for the op setup.
#
# Usage:
#   source /home/lu_zero/Sources/tt/env-tt-common.sh
#   python repro_trace_replay_write_hang.py
import os
import torch
import ttnn


def to_paged_cache(cache, batch, num_kv, max_num_blocks_per_seq, block_size, head_dim):
    return (
        cache.reshape(batch, num_kv, max_num_blocks_per_seq, block_size, head_dim)
        .transpose(1, 2)
        .reshape(batch * max_num_blocks_per_seq, num_kv, block_size, head_dim)
    )


def main():
    NUM_ITERS = int(os.environ.get("REPRO_ITERS", "120"))
    DO_WRITE = os.environ.get("REPRO_NO_WRITE") != "1"

    device = ttnn.open_mesh_device(l1_small_size=0, trace_region_size=50 * 1024 * 1024)
    try:
        b, nh, nkv, s, d = 1, 8, 1, 512, 128
        block_size = 32
        max_num_blocks_per_seq = s // block_size
        max_num_blocks = b * s // block_size
        scale = d**-0.5
        dram = ttnn.DRAM_MEMORY_CONFIG

        K = torch.randn(b, nkv, s, d, dtype=torch.bfloat16)
        V = torch.randn(b, nkv, s, d, dtype=torch.bfloat16)
        paged_k = to_paged_cache(K, b, nkv, max_num_blocks_per_seq, block_size, d)
        paged_v = to_paged_cache(V, b, nkv, max_num_blocks_per_seq, block_size, d)
        permutation = torch.randperm(max_num_blocks)
        reverse_permutation = torch.argsort(permutation)
        page_table = reverse_permutation.reshape(b, max_num_blocks_per_seq).to(torch.int32)
        paged_k_shuffled = paged_k[permutation]
        paged_v_shuffled = paged_v[permutation]

        tt_K = ttnn.as_tensor(paged_k_shuffled, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=dram)
        tt_V = ttnn.as_tensor(paged_v_shuffled, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=dram)
        tt_page_table = ttnn.as_tensor(page_table, device=device, dtype=ttnn.int32, memory_config=dram)
        Q = torch.randn(1, b, nh, d, dtype=torch.bfloat16)
        tt_Q = ttnn.as_tensor(Q, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=dram)
        start_indices = torch.tensor([s - 1], dtype=torch.int32)
        tt_cur_pos = ttnn.as_tensor(start_indices, device=device, dtype=ttnn.int32, memory_config=dram)

        NUM_OPS = int(os.environ.get("REPRO_NUM_OPS", "1"))  # sdpa calls chained per trace
        USE_RAC = os.environ.get("REPRO_NO_RAC") != "1"  # include paged_update_cache in the trace

        # paged_update_cache needs a height-sharded K/V input, shard [nkv_pad, d]
        # on one core (nkv padded to TILE_HEIGHT=32). Matches our driver's RAC.
        nkv_pad = 32
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
        shard_spec = ttnn.ShardSpec(shard_grid, [nkv_pad, d], ttnn.ShardOrientation.ROW_MAJOR)
        k_input_memcfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
        k_input = torch.zeros(1, 1, nkv_pad, d, dtype=torch.bfloat16)
        tt_k_input = ttnn.as_tensor(k_input, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=k_input_memcfg)
        v_input = torch.zeros(1, 1, nkv_pad, d, dtype=torch.bfloat16)
        tt_v_input = ttnn.as_tensor(v_input, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=k_input_memcfg)
        update_idxs = [s - 1]
        tt_update_idxs = ttnn.as_tensor(torch.tensor(update_idxs, dtype=torch.int32), device=device, dtype=ttnn.int32, memory_config=dram)

        # Warm both ops (compile + program cache).
        if USE_RAC:
            ttnn.experimental.paged_update_cache(tt_K, tt_k_input, update_idxs=[], update_idxs_tensor=tt_update_idxs, page_table=tt_page_table)
            ttnn.experimental.paged_update_cache(tt_V, tt_v_input, update_idxs=[], update_idxs_tensor=tt_update_idxs, page_table=tt_page_table)
        _ = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            tt_Q, tt_K, tt_V, tt_page_table, is_causal=True, cur_pos_tensor=tt_cur_pos, scale=scale)
        ttnn.synchronize_device(device)

        # Capture: paged_update_cache (RAC) + paged sdpa decode, chained NUM_OPS times.
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        tt_out = tt_Q
        for _ in range(NUM_OPS):
            if USE_RAC:
                ttnn.experimental.paged_update_cache(tt_K, tt_k_input, update_idxs=[], update_idxs_tensor=tt_update_idxs, page_table=tt_page_table)
                ttnn.experimental.paged_update_cache(tt_V, tt_v_input, update_idxs=[], update_idxs_tensor=tt_update_idxs, page_table=tt_page_table)
            tt_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                tt_out, tt_K, tt_V, tt_page_table, is_causal=True, cur_pos_tensor=tt_cur_pos, scale=scale)
        ttnn.end_trace_capture(device, tid, cq_id=0)

        print(f"[repro] captured; now {NUM_ITERS} replays with do_write={DO_WRITE}", flush=True)
        for i in range(NUM_ITERS):
            if DO_WRITE:
                # Refresh the INT32 trace-input buffers in place (the toxic
                # pattern: page_table + cur_pos + update_idxs, matching
                # WarmPaMeta + WarmRacIdx).
                page_table[0, 0] = i % max_num_blocks_per_seq
                tt_pt_fresh = ttnn.as_tensor(page_table, device=device, dtype=ttnn.int32, memory_config=dram)
                ttnn.copy(tt_pt_fresh, tt_page_table)
                cp = torch.tensor([s - 1], dtype=torch.int32)
                tt_cp_fresh = ttnn.as_tensor(cp, device=device, dtype=ttnn.int32, memory_config=dram)
                ttnn.copy(tt_cp_fresh, tt_cur_pos)
                ui = torch.tensor([s - 1], dtype=torch.int32)
                tt_ui_fresh = ttnn.as_tensor(ui, device=device, dtype=ttnn.int32, memory_config=dram)
                ttnn.copy(tt_ui_fresh, tt_update_idxs)
            # Match the driver exactly: blocking=False, NO synchronize_device,
            # the .cpu() readback is the sync point.
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
            if i % 5 == 0:
                print(f"[repro] iter {i}/{NUM_ITERS} ok", flush=True)
            _ = tt_out.cpu()
        print(f"[repro] DONE: {NUM_ITERS} replays, no hang", flush=True)
        ttnn.release_trace(device, tid)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
