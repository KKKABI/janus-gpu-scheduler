# Janus static scheduling: batch-size matrix

Date: 2026-07-18

GPU: NVIDIA RTX A5000 24 GB, 64 SM

Environment: `opara`, CUDA MPS pipe `/tmp/nvidia-mps-1003`

## Compared policies

- **Original**: clean upstream commit `73f4da5`; the original truncation/default behavior is unchanged.
- **Max occupancy**: selects the feasible HP group with the largest predicted SM occupancy.
- **DRT**: dominant-resource-time scoring implemented by `static_interference`.
- **Guarded autotune**: captures max-occupancy and DRT graphs, measures them in interleaved order, and selects DRT only when its gain clears both a 1% threshold and a robust MAD noise margin.

Each model/batch/policy combination ran in an isolated process. Values are mean ± standard deviation in milliseconds. NASNet batch 16 exceeds the 24 GB device capacity and is excluded from aggregate results.

## Batch size 4

| Model | Original | Max occupancy | DRT | DRT vs max |
|---|---:|---:|---:|---:|
| GoogLeNet | 1.8206 ± 0.0463 | 1.5222 ± 0.0422 | **1.4422 ± 0.0337** | **-5.26%** |
| Inception v3 | 4.1150 ± 0.0587 | 3.9605 ± 0.0399 | **3.8468 ± 0.0159** | **-2.87%** |
| NASNet-A-Large | 33.8045 ± 0.1238 | **32.0641 ± 0.1492** | 32.5858 ± 0.1405 | +1.63% |
| DeepFM | 0.2211 ± 0.0020 | 0.2885 ± 0.0055 | **0.1126 ± 0.0011** | **-60.97%** |
| BERT | 1.8690 ± 0.0159 | 1.8587 ± 0.0274 | **1.8473 ± 0.0127** | -0.62% |
| YOLO backbone | **0.3187 ± 0.0010** | 0.3190 ± 0.0009 | 0.3189 ± 0.0011 | -0.03% |
| ConvNeXt-Base | 14.6760 ± 0.1408 | **14.6225 ± 0.1462** | 14.8451 ± 0.1439 | +1.52% |

The 1%-guarded portfolio selects DRT for GoogLeNet, Inception and DeepFM. Its geometric-mean latency is 13.61% below max occupancy and 13.76% below original.

## Batch size 8

| Model | Original | Max occupancy | DRT | DRT vs max |
|---|---:|---:|---:|---:|
| GoogLeNet | 2.7096 ± 0.0513 | 2.5528 ± 0.0414 | **2.5229 ± 0.0258** | **-1.17%** |
| Inception v3 | 7.4668 ± 0.0210 | 7.2567 ± 0.0085 | **7.2562 ± 0.0267** | -0.01% |
| NASNet-A-Large | 60.2395 ± 0.3613 | **59.1710 ± 0.2753** | 59.6501 ± 0.2905 | +0.81% |
| DeepFM | 0.3242 ± 0.0073 | 0.4431 ± 0.0068 | **0.1487 ± 0.0013** | **-66.43%** |
| BERT | 3.1605 ± 0.0221 | 3.1480 ± 0.0400 | **3.1390 ± 0.0238** | -0.29% |
| YOLO backbone | **0.6113 ± 0.0012** | 0.6119 ± 0.0011 | 0.6114 ± 0.0011 | -0.09% |
| ConvNeXt-Base | **27.0029 ± 0.2688** | 26.9632 ± 0.2889 | 27.3681 ± 0.3267 | +1.50% |

The 1%-guarded portfolio selects DRT for GoogLeNet and DeepFM. Its geometric-mean latency is 14.58% below max occupancy and 12.08% below original.

## Batch size 16

| Model | Original | Max occupancy | DRT | DRT vs max |
|---|---:|---:|---:|---:|
| GoogLeNet | 4.9013 ± 0.0092 | 4.9094 ± 0.0221 | **4.8562 ± 0.0269** | **-1.08%** |
| Inception v3 | 14.3369 ± 0.0480 | 13.9880 ± 0.0482 | **13.9803 ± 0.0548** | -0.06% |
| NASNet-A-Large | OOM | OOM | OOM/unavailable | — |
| DeepFM | 0.5377 ± 0.0100 | 0.7184 ± 0.0184 | **0.2153 ± 0.0011** | **-70.03%** |
| BERT | **5.0426 ± 0.0523** | 5.0510 ± 0.0654 | 5.1268 ± 0.0613 | +1.50% |
| YOLO backbone | 1.1940 ± 0.0022 | **1.1937 ± 0.0022** | 1.1938 ± 0.0021 | +0.01% |
| ConvNeXt-Base | 52.9670 ± 0.3638 | **52.8890 ± 0.3596** | 53.7235 ± 0.4717 | +1.58% |

The 1%-guarded portfolio selects DRT for GoogLeNet and DeepFM. Its geometric-mean latency is 18.34% below max occupancy and 14.63% below original.

## Interpretation

The strongest and most repeatable result is DeepFM: DRT improves over max occupancy by 60.97%, 66.43%, and 70.03% at batch sizes 4, 8, and 16. This is consistent with the intended mechanism: occupancy maximization launches too many similarly dominant HP kernels together, while DRT penalizes the predicted peak resource-time overload.

One global analytic weight is not uniformly optimal. NASNet and ConvNeXt regress slightly under DRT, while YOLO and parts of BERT/Inception are within measurement noise. The guarded hardware-in-the-loop selector therefore keeps the static Janus framework and deploys exactly one CUDA Graph, but makes the final per-model/per-shape decision using measured replay latency.

## End-to-end guarded-autotune validation

DeepFM batch 16 measured max occupancy at 0.694231 ms and DRT at 0.189243 ms during interleaved tuning. The 0.504988 ms gain exceeded the 0.026234 ms robust threshold, so DRT was selected. The subsequent isolated benchmark measured 0.198918 ± 0.000916 ms.

YOLO batch 4 measured max occupancy at 0.316027 ms and DRT at 0.317020 ms. DRT's gain was -0.000993 ms versus a required 0.003160 ms, so the selector correctly retained max occupancy. The subsequent isolated benchmark measured 0.316292 ± 0.001555 ms.
