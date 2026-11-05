## Description

Follow-up to #6037 (landed as 1e71c2b51) and the duplicate #6102, which bound the per-thread scalar device-buffer cache in `MakeBuffer` by clearing the whole map once it holds 1024 entries.

#6037's own measurements show the cost of a full clear: a workload whose working set of distinct scalars is larger than one cache (a 1500-value cycle) misses on every lookup, 1.19x per op in `DEFER_NEVER` and 1.45x in `DEFER_AND_FUSE`, and in `DEFER_AND_FUSE` a scalar used before and after the clear becomes two graph inputs, which adds a compile (see the review discussion on #6037).

This keeps the cache in two generations instead. Misses go into a young map; after 1024 misses it becomes the old map and the previous old map is dropped. A hit in the old map moves the entry back. So a scalar used within the last 1024 misses always survives, a working set of up to 2048 distinct values keeps hitting after one warm-up pass, and stale values (a learning-rate schedule) are still freed after two generations. The hit path stays a single hash lookup for the common case; the bound becomes 2048 buffers per thread (about 2 MiB of scalars) instead of 1024.

## Test Plan

TPU v6e-1, built on 988b22f12 with 1e71c2b51 cherry-picked (`clear`) and this change on top (`twogen`); the standalone build of current main is blocked by the torch-bundled protobuf headers (separate issue). Script: 4000–6000 ops per scenario, `torch.accelerator.synchronize()` per op in `DEFER_NEVER`, per 50 ops in `DEFER_AND_FUSE`. Dispatch is the host-side time of the Python call.

| scenario | mode | clear | twogen |
|---|---|---|---|
| hit, same scalar | DEFER_NEVER | 53.7 us/op, dispatch p50 9.4 us | 55.0 us/op, p50 12.8 us |
| hit, same scalar | DEFER_AND_FUSE | 27.0 us/op, p50 8.7 us | 27.3 us/op, p50 10.4 us |
| cycle of 1500 distinct scalars | DEFER_NEVER | 60.1 us/op, p50 31.0 us | 57.5 us/op, p50 10.6 us |
| cycle of 1500 distinct scalars | DEFER_AND_FUSE | 48.6 us/op, p50 32.3 us | 33.5 us/op, p50 9.4 us |
| cycle of 512 distinct scalars | DEFER_AND_FUSE | 27.3 us/op, p99 39.2 us | 29.1 us/op, p99 14.3 us |
| fused region re-using 0.9 across the eviction | DEFER_AND_FUSE | 417 us/step, +1 compile | 390 us/step, +0 compiles |
| AdamW + cosine LR, MLP, 1500 steps | DEFER_AND_FUSE | 1.01 ms/step | 1.02 ms/step |

Hits and workloads whose scalars change every step (AdamW with a schedule) are unchanged. The 1500-cycle regime loses the all-miss penalty (dispatch p50 3x lower in both modes, 1.45x per op in `DEFER_AND_FUSE`), and the fused region no longer recompiles after the eviction. Device memory for a stream of distinct scalars tops out at 2 generations (`live +1952` buffers for 4000 distinct values, vs `+928`).

Tests (`tests/accelerator_api_test.py`, run with pytest on the TPU, 5 passed):
- `test_distinct_scalars_do_not_accumulate_device_buffers`: bound updated to two generations.
- `test_queued_op_keeps_scalar_buffer_after_cache_eviction`: turns over both generations before checking the evicted scalar.
- new `test_scalar_cycle_longer_than_one_generation_stays_cached` (both eager modes): two warm-up passes over 1500 scalars, then the third pass creates no scalar buffers.

Assisted-by: Claude Fable 5.1
