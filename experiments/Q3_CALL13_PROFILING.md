# GoogLeNet call 13 GPU-overlap validation

This archive branch preserves the exact Q3 instrumentation and profiling driver
used to validate whether the scheduler selections at GoogLeNet call 13 become
real GPU kernel overlap.

## Provenance

- Frozen algorithm/configuration base: `3b2880ad5ca4b78d0385c9dd014ac2f4ab420648`.
- Hardware used for the archived evidence: NVIDIA RTX A5000, batch size 1.
- GoogLeNet profile SHA-256:
  `0d29cfcd359efbf8d0630d9ef8171b0f6cd383fbac8ce27d7c6a1b18b3a1ae14`.
- Each variant ran one correctness replay and five measured CUDA Graph replays.

The original traces were captured while the instrumentation diff was applied to
the frozen base in a detached worktree. This branch commits that exact diff so
the source is reproducible. The driver therefore verifies that the frozen base
is an ancestor of the current archive commit and that tracked files are clean.

## Included source

- `Opara/GraphCapturer.py`: capture-only FX NVTX ranges, FX/stream metadata, and
  access to the captured CUDA Graph for stable replay.
- `experiments/q3_profile_call13.py`: variant application, frozen-input/profile
  assertions, call-13 identity checks, correctness check, and replay ranges.
- `experiments/q3_nvtx_smoke.py`: minimal NVTX capture smoke test.
- `experiments/analyze_q3_overlap.py`: SQLite graph-node mapping and exact
  pair/triple interval analysis.
- `evidence/q3_call13/`: compact Markdown/CSV/SVG/PNG result artifacts.

## Capture commands

Run from the repository root with an idle GPU and the `opara` environment:

```bash
/opt/nvidia/nsight-systems/2024.3.1/target-linux-x64/nsys profile \
  --trace=cuda,nvtx,cudnn,cublas \
  --sample=none --cpuctxsw=none --cuda-graph-trace=node \
  --show-output=false --export=sqlite --force-overwrite=true \
  -o report-work/q3_call13/static_full_trace \
  /home/lyx/.conda/envs/opara/bin/python experiments/q3_profile_call13.py \
    --variant Baseline \
    --output-dir report-work/q3_call13/static_full \
    --replays 5
```

Repeat with `--variant TD+Janus`, a different output directory, and output
prefix. Normalize the generated filenames to those expected by the analyzer,
then run:

```bash
python experiments/analyze_q3_overlap.py --root /path/to/q3_profiler
```

## Archived result

- Static+Janus selected `x_13+x_17`; their kernel intervals overlapped in 5/5
  replays, with median overlap `6.943 us`.
- TD+Janus selected `x_11+x_13+x_17`; strict three-way overlap occurred in 5/5
  replays, with median overlap `6.560 us`.

This is a targeted mechanism check. It does not establish end-to-end latency
superiority, same-SM thread-block co-residency, or a result for every call/model.
Other graph nodes may also be active during the target interval; the reported
`max_concurrent=3` is scoped to the three target operators.

## Raw evidence intentionally excluded from Git

The `.nsys-rep` and `.sqlite` files remain in the separately archived evidence
bundle. Their SHA-256 values are:

| File | SHA-256 |
|---|---|
| `static_full_trace.nsys-rep` | `5f6066f8f8e12a36c5ca0588ee3199cd7b6b555769792e8f473716eddb3226a7` |
| `static_full_trace.sqlite` | `0bbac143f34da929718db98ba8493fcc20122700fc2242fff42205b7c71f16c7` |
| `td_janus_full_trace.nsys-rep` | `262850ab0cba40f40119f8902092cdd3f5dcca952aab2fc5d3b889c53712f207` |
| `td_janus_full_trace.sqlite` | `84d6dd3f5e7a78af7a44fd8b2d4fc78cd9e4aa1ed333050286bc99850c1f676b` |
