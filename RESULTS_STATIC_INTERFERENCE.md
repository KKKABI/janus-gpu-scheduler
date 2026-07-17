# Static HP-interference scheduling results

Date: 2026-07-18

## Environment

- GPU: NVIDIA RTX A5000 (64 SM)
- Conda environment: opara
- Batch size: 1
- MPS pipe: /tmp/nvidia-mps-1003
- Baseline: original branch at 73f4da5
- Comparison: time-domain scheduler with alpha=0.9 and max_occupancy
- New policy: dominant resource-time (DRT)
- Final policy: offline schedule portfolio; capture max-occupancy and DRT,
  benchmark their replay latency, and retain one static CUDA Graph

## Latency (ms)

| Model | original | max occupancy | DRT | portfolio | portfolio vs original | DRT vs max occupancy |
|---|---:|---:|---:|---:|---:|---:|
| GoogLeNet | 0.8217 | 0.7462 | 0.7922 | 0.7462 | -9.19% | +6.16% |
| Inception-v3 | 2.0505 | 1.9312 | 1.9741 | 1.9312 | -5.82% | +2.22% |
| NASNet-Large | 8.6873 | 7.9579 | 7.7998 | 7.7998 | -10.22% | -1.99% |
| DeepFM | 0.1526 | 0.1607 | 0.0865 | 0.0865 | -43.32% | -46.17% |
| BERT-base | 1.2937 | 1.3026 | 1.2878 | 1.2878 | -0.46% | -1.14% |
| YOLOv8x backbone | 0.0965 | 0.0965 | 0.0960 | 0.0960 | -0.52% | -0.52% |
| ConvNeXt-Base | 4.9039 | 4.9044 | 4.9227 | 4.9044 | +0.01% | +0.37% |

Portfolio total latency is 16.8519 ms versus 18.0062 ms for original
(-6.41%). Its geometric-mean latency improvement versus original is 11.33%.

## Why DRT differs from cosine similarity

DRT retains demand magnitude and duration. For every candidate HP group it
builds time-weighted demand for registers, shared memory, warps, whole-GPU SM
coverage, and duration-per-block density. The maximum accumulated resource-time
demand predicts round-time inflation. In particular, two HP operators that
each already span all SMs are penalized even when their normalized resource
vectors point in different directions.

## Static portfolio

GraphCapturer.autotune_capturer() captures both schedules, measures each with
CUDA events (median of repeated batches), and returns only the faster runner.
Selection happens before deployment; online inference still replays one fixed
CUDA Graph and performs no dynamic scheduling.

A GoogLeNet smoke test measured 0.754995 ms for max occupancy and 0.799430 ms
for DRT, correctly selecting max occupancy.
