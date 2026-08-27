# Multi-Janus chapter-3 experiment

This directory replaces the earlier mean-only multi-client scripts with an
auditable closed-loop protocol.  It retains raw request timings, includes
queueing delay in response latency, measures aggregate throughput over one
shared wall-clock window, checks eager/graph output correctness, records MPS
and repository provenance, and supports `sequential`, `concurrent`, and
lookup-table admission modes.

The `sm_fraction` argument is retained only for compatibility with the frozen
Janus baseline.  It changes the simulator's effective SM count; it does not
partition the physical GPU and must not be described as hardware isolation.

The smoke test uses two GoogLeNet clients.  Formal lookup-table construction
must use profiling trials that are separate from evaluation trials.

`run_ch3_matrix.py` is the resumable seven-model orchestrator. It separates
offline compatibility profiling from independent evaluation.
`run_ch3_with_mps.sh` owns the GPU lock and MPS lifetime.
