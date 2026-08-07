# Q3 GPU overlap validation - GoogLeNet call 13

## Verdict

- Static+Janus selected `x_13+x_17`; their kernel-level overlap was positive in all 5 replays (median 6.943 us).
- TD+Janus selected `x_11+x_13+x_17`; strict three-way kernel overlap was positive in all 5 replays (median 6.560 us; maximum concurrent target operators 3).

This validates actual GPU kernel execution, not merely the scheduler's selected-set log.

## Per-replay measurements (microseconds)

| Variant | Replay | x11&x13 | x11&x17 | x13&x17 | Strict triple | Selected-group overlap | Max concurrent |
|---|---:|---:|---:|---:|---:|---:|---:|
| Static+Janus | 0 | 0.000 | 0.000 | 7.071 | 0.000 | 7.071 | 2 |
| Static+Janus | 1 | 0.000 | 0.000 | 6.880 | 0.000 | 6.880 | 2 |
| Static+Janus | 2 | 0.000 | 0.000 | 6.879 | 0.000 | 6.879 | 2 |
| Static+Janus | 3 | 0.000 | 0.000 | 6.943 | 0.000 | 6.943 | 2 |
| Static+Janus | 4 | 0.000 | 0.000 | 7.007 | 0.000 | 7.007 | 2 |
| TD+Janus | 0 | 7.008 | 6.656 | 6.976 | 6.656 | 6.656 | 3 |
| TD+Janus | 1 | 6.848 | 6.496 | 7.136 | 6.496 | 6.496 | 3 |
| TD+Janus | 2 | 6.880 | 6.528 | 6.880 | 6.528 | 6.528 | 3 |
| TD+Janus | 3 | 6.944 | 6.560 | 6.880 | 6.560 | 6.560 | 3 |
| TD+Janus | 4 | 7.008 | 6.624 | 7.328 | 6.624 | 6.624 | 3 |

## Mapping and reproducibility

- Frozen commit: `3b2880ad5ca4b78d0385c9dd014ac2f4ab420648`.
- GoogLeNet profile SHA-256: `0d29cfcd359efbf8d0630d9ef8171b0f6cd383fbac8ce27d7c6a1b18b3a1ae14`.
- Hardware: NVIDIA RTX A5000; batch size 1; 5 CUDA Graph replays per variant.
- Mapping chain: capture-phase FX NVTX range -> captured `graphNodeId` -> CUDA Graph clone `originalGraphNodeId -> graphNodeId` -> replay kernel `graphNodeId`.
- Overlap is the exact intersection of GPU kernel intervals after merging each operator's kernel intervals. No kernel-name-only or CUDA-stream-pointer heuristic is used.
- The call-13 ready set, enumerated/feasible counts, selected set, and numerical correctness were asserted by the profiling driver before accepting the trace.

## Scope

This is a targeted mechanism check for call 13. It proves whether these selected operators actually overlap on the GPU; it does not by itself establish end-to-end latency superiority or generalize to every scheduler call/model.
