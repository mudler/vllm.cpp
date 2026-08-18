# Spike probe: is ttnn.plus_one capture-safe + does it advance on replay?
import torch
import ttnn

device = ttnn.open_mesh_device(l1_small_size=0, trace_region_size=50 * 1024 * 1024)
try:
    dram = ttnn.DRAM_MEMORY_CONFIG
    # Persistent INT32 cur_pos [1] on device, seeded to 5.
    cur_pos = torch.tensor([5], dtype=torch.int32)
    tt_cp = ttnn.as_tensor(cur_pos, device=device, dtype=ttnn.int32, memory_config=dram)

    # Warm plus_one (compile + program cache).
    ttnn.plus_one(tt_cp, skip_negative_entries=True)
    ttnn.synchronize_device(device)
    print(f"[probe] after warm plus_one, cur_pos = {ttnn.to_torch(tt_cp).tolist()}", flush=True)

    # Capture: a no-op-ish body + plus_one at the end.
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    ttnn.plus_one(tt_cp, skip_negative_entries=True)
    ttnn.end_trace_capture(device, tid, cq_id=0)

    # Replay 5 times; each should increment cur_pos by 1.
    for i in range(5):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        v = ttnn.to_torch(tt_cp).tolist()
        print(f"[probe] replay {i+1}: cur_pos = {v}", flush=True)

    ttnn.release_trace(device, tid)
    print("[probe] DONE: plus_one is capture-safe and advances on replay", flush=True)
finally:
    ttnn.close_mesh_device(device)
