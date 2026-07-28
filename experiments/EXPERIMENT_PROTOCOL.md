# Janus 单请求实验协议（冻结版）

冻结日期：2026-07-28（Asia/Shanghai）

本文件是后续时域仿真与打分策略实验的唯一口径。历史表格仅作为线索，不与按本协议重新采集的数据混合统计。

## 1. 源码与 Baseline

- 唯一工作目录：`/public_0/LYX/janus_repro`
- 分支：`experiment/td-drt-repro`
- 上游基点：`6d4d4fc061f121cd61b0e669a7fa5184752841e4`
- 当前实验源码快照：`fea25ed`（当前 TD/DRT 源码和已有实验脚本的隔离快照）
- 所有策略必须从同一个实验提交运行，不再从 `janus_original_baseline`、`janus`、`janus_static_interference` 三个 worktree 混取结果。
- **Baseline 的唯一含义**：在当前统一源码中显式调用 `selection_mode="legacy_balance", time_domain=false, alpha=0.9`，复现参考提交 `73f4da51354a554f4abee476ad1a57a6e6a5af42` 的静态资源模型、α=0.9 候选过滤和名称平衡启发式。禁止用 `capturer()` 的默认参数充当 Baseline，因为当前默认值实际是 TD+Cos+α=0.9。
- Baseline 在每个模型、每次独立进程重复中只运行一次，结果字段中的外部 `alpha` 记为 `null`；内部兼容参数单独记录为 `internal_alpha=0.9`。
- 原目录 `/public_0/LYX/janus` 保持为历史开发现场，不作为新一轮数据的运行目录。

## 2. 运行环境

每个实验进程都从非交互 shell 显式激活环境：

```bash
source /usr/local/Anaconda3/etc/profile.d/conda.sh
conda activate opara
```

实际环境以 `experiments/environment/conda-opara-explicit.txt` 和结果目录中的运行时快照为准，而不是旧文档里的版本描述。

- 主机：`ubuntu108`
- GPU：NVIDIA RTX A5000 24 GiB，UUID `GPU-32a478ec-0d66-5b4f-6452-3923afcef998`
- 驱动：555.42.02
- nvidia-smi 报告的 CUDA runtime：12.5
- Python：3.10.4，解释器 `/home/lyx/.conda/envs/opara/bin/python`
- PyTorch：2.4.0+cu124；torchvision：0.19.0+cu124；torch CUDA：12.4；cuDNN：90100
- 系统 nvcc：11.7.64（`/usr/local/cuda-11.7`）
- 本阶段 MPS 关闭；每次只允许一个实验进程占用 GPU。
- 不做需要管理员权限的锁频或 persistence-mode 修改；通过空闲检查、固定随机化顺序和多进程重复控制漂移。

## 3. 模型与输入

主结果使用 batch size 1 的六个模型：GoogLeNet、Inception-v3、BERT、NASNet、YOLOv8x、ConvNeXt。DeepFM 只作为次要验证，不进入六模型主表。

模型构造、权重来源、输入、warm-up 和 iteration 全部以 `/public_0/LYX/PriorityOpara_v0/examples` 中对应示例为唯一标准，并固化到 `experiments/repro_config.json`。参考示例源码通过 `experiments/model_reference_manifest.sha256` 校验，外部权重通过 `experiments/model_asset_manifest.sha256` 校验，profile 通过 `experiments/profile_manifest.sha256` 校验；一轮矩阵执行期间不得重新生成或替换这些文件。

## 4. 实验变量的唯一语义

- `simulator=static`：静态资源可行性/冲突判断，`time_domain=false`。
- `simulator=td`：时域仿真，`time_domain=true`。
- `score=cosine`：Cosine 打分，使用 α 过滤。
- `score=min_resource`：MinRes 打分，使用 α 过滤；只作补充分析。
- `score=drt_no_alpha`：DRT/static_interference 打分，不使用 α 过滤，结果中的 `alpha` 必须为 `null`。
- 如以后实现“带 α 的 DRT”，必须使用新名称 `drt_alpha`，不得复用 `drt_no_alpha`。
- α 网格固定为 `[0.9, 0.8, 0.5, 0.2]`，只用于实际执行 α 过滤的策略。
- 主消融矩阵为 Baseline、Static+Cos、TD+Cos、Static+DRT、TD+DRT；MinRes 和 α 扫描单独呈现。

## 5. 正确性门槛

计时前，在相同模型状态和相同输入上分别执行 eager reference 与待测 runner：

- 模型必须为 `eval()`，测试在 `torch.inference_mode()` 中完成。
- 固定随机种子 `20260728`，同时设置 Python、NumPy、PyTorch CPU/CUDA 随机种子。
- 递归比较 tensor、tuple/list 和 dict 输出。
- 浮点输出使用 `torch.testing.assert_close(rtol=1e-4, atol=1e-5)`；整数/布尔输出要求完全相等。
- 任一输出不一致、出现 NaN/Inf、捕获失败或 profile/权重哈希不匹配时，该配置记为无效并停止计时，禁止只保留“跑得快”的样本。

## 6. 计时方法

每个 `(model, simulator, score, alpha)` 配置在全新的 Python 进程中运行，且独立重复 5 次：

1. 确认无其他 GPU compute 进程，记录实验前温度、功耗、SM/显存时钟和显存占用。
2. 创建 4 MiB CUDA int8 cache buffer；每轮执行前调用 `zero_()`，与参考示例保持一致。
3. 使用 `repro_config.json` 中该模型对应的 warm-up 次数，随后 `torch.cuda.synchronize()`。
4. 每个计时样本在 CUDA Event 开始前执行 `torch.cuda._sleep(1_000_000)`，与参考示例保持一致；sleep 不计入延迟。
5. 用同一 CUDA stream 上的 CUDA Events 采集该模型规定数量的单次延迟样本；每个样本结束后同步。
6. 保存全部原始样本，并记录实验后的 GPU 遥测。
7. 每次进程内统计 median/min/max；跨 5 个进程的主指标是“5 个进程内 median 的 median”，同时报告 mean、样本标准差和 MAD。min/max 仅作诊断，不作为优劣结论。
同一模型内的配置执行顺序按 `seed = 20260728 + repeat_index` 可复现地打乱，以降低温度和动态频率随时间漂移造成的顺序偏差。

## 7. 结果与审计

每次完整运行写入新目录 `experiments/results/<run_id>/`，不得覆盖或续写旧目录。至少保存：

- Git commit、分支和 worktree 路径；
- 完整配置与随机种子；
- conda 环境、Python/PyTorch/CUDA/cuDNN/驱动/GPU 信息；
- profile 与外部权重哈希；
- 正确性检查结果；
- 该模型规定数量的原始延迟样本和聚合统计；
- 前后 GPU 遥测、标准输出与错误日志；
- 失败原因与退出码。

若中途发现竞争 GPU 进程、资产变化或环境变化，整次进程重复作废并重跑，不得人工挑选样本。

## 8. 当前阶段边界

本协议只覆盖单请求/单图性能和候选可行性分析。MPS、多图并发、吞吐量和公平性实验必须使用单独协议，不能与本阶段延迟数据混在同一张主表中。本次“第一步”只冻结源码、环境与口径，不运行性能实验。
