# 七模型三策略正式实验（冻结版）

本目录只比较三条明确的策略线：

1. `Original Janus`：Static 准入、原 Janus `legacy_balance` 排序；
2. `NewTD+DRT`：Static 安全路径并入 NewTD 二元扩展，DRT 排序；
3. `NewTD+NCU-DRT`：与第 2 条使用完全相同的准入，仅把最终排序改为 identity-checked NCU 风险排序。

三条策略都固定 `max-ready=6`。两条 NewTD 策略固定最小预测重叠阈值为 `2.0 us`、launch gap 为 `0.004096 ms`，与 §4.7 得到 99.77% 正例精确率的冻结 NewTD 版本一致。端到端延迟不使用 LP→HP；LP→HP 只属于 Janus §4.7 正例准确率实验。YOLO 一律写作 **YOLOv8x BackboneWrapper**，不代表完整 DetectionModel。

## 运行前提

- 使用干净的正式代码提交和 `/home/lyx/.conda/envs/opara/bin/python`；
- GPU 必须空闲；三个阶段共用 `/tmp/janus_gpu0.lock`，不能与第三章或其他 GPU 任务并发；
- 禁止设置 `JANUS_ALLOW_LEGACY_NCU`；
- 所有输出目录必须是新目录，脚本拒绝覆盖旧结果。
- 三个入口默认根据脚本自身的 `realpath` 向上两级确定仓库，也允许用 `JANUS_FORMAL_REPO` 显式覆盖；两种情况都必须设置交付记录中的 40 位 `JANUS_FORMAL_EXPECTED_COMMIT`，实际 HEAD 不一致时会在使用 GPU 前退出。

服务器部署后先运行一次不使用 GPU 的入口预检，并保存 `bash -n`、脚本 SHA 和提交身份：

```bash
export JANUS_FORMAL_REPO=/public_0/LYX/janus_formal_threeway_20260827
export JANUS_FORMAL_EXPECTED_COMMIT=REPLACE_WITH_REVIEWED_40_CHARACTER_COMMIT
/home/lyx/.conda/envs/opara/bin/python \
  "$JANUS_FORMAL_REPO/experiments/formal_threeway_20260827/verify_stage_entrypoints.py" \
  --repo "$JANUS_FORMAL_REPO" \
  --expected-commit "$JANUS_FORMAL_EXPECTED_COMMIT" \
  --output /public_0/LYX/janus_formal_entrypoint_preflight_20260827.json
```

## A：七模型三次稳健画像

本阶段不复用任何旧 NCU cache。在同一个冻结提交、输入、GPU 和软件环境下，对七个模型各独立采集 3 次，共 21 次；参考之前同规模采集，预计约 37 分钟。只有三次的完整身份、launch 数量、OP 名、kernel 名和 launch geometry 全部一致，才对同一 launch 的 DRAM、L2、Compute、Memory 和 Duration 取中位数，生成正式 cache。YOLO 统一使用 BackboneWrapper，并另外重新采集匹配的 solo OP 时长；旧 DetectionModel 数据不复用。`manifest.json` 会记录 21 次实际采集开销和每项指标的 95 分位相对波动。

```bash
export JANUS_FORMAL_REPO=/public_0/LYX/janus_formal_threeway_20260827
export JANUS_FORMAL_EXPECTED_COMMIT=REPLACE_WITH_REVIEWED_40_CHARACTER_COMMIT
export JANUS_STAGE_A_OUT=/public_0/LYX/janus_formal_stage_a_20260827
export JANUS_FROZEN_SOLO_ROOT=/public_0/LYX/janus_simulator_accuracy_outputs_20260820/solo_operator_all_kernel_v1
bash "$JANUS_FORMAL_REPO/experiments/formal_threeway_20260827/run_stage_a_profiles.sh"
```

完成标志为 `$JANUS_STAGE_A_OUT/COMPLETE`。21 次 raw CSV 和单次 cache 位于 `ncu/raw_repeats`，三次中位数正式 cache 位于 `ncu/ncu_cache`，合并后的七模型 solo 根目录位于 `solo_operator_profiles`。每个正式 cache 还必须通过当前 GraphCapturer 的 fail-closed 加载，并达到 50% duration coverage；任何一项失败都会终止。

## B：七模型三策略正式延迟

每个模型、每条策略启动 10 个独立 Python 进程。每个进程内部仍按冻结配置测量：普通模型 300 次、YOLO BackboneWrapper 100 次、DeepFM 3000 次。每个进程先对内部样本取算术平均，最终再对 10 个进程平均值取算术平均，得到论文表中唯一的最终延迟。原始样本和进程间标准差只作为审计信息保存，不进入主表。策略顺序和模型顺序随 trial 循环轮换。

```bash
export JANUS_FORMAL_REPO=/public_0/LYX/janus_formal_threeway_20260827
export JANUS_FORMAL_EXPECTED_COMMIT=REPLACE_WITH_REVIEWED_40_CHARACTER_COMMIT
export JANUS_STAGE_B_OUT=/public_0/LYX/janus_formal_stage_b_latency_20260827
export JANUS_FORMAL_NCU_CACHE_DIR=$JANUS_STAGE_A_OUT/ncu/ncu_cache
export JANUS_FORMAL_SOLO_ROOT=$JANUS_STAGE_A_OUT/solo_operator_profiles
bash "$JANUS_FORMAL_REPO/experiments/formal_threeway_20260827/run_stage_b_latency.sh"
```

主要结果是 `summary.csv` 和 `summary.json`。每个 NCU 进程的 `result.json` 都保存 `ncu_report.experimental_valid`、mapping coverage、输入字节哈希、输出正确性误差和全部 scheduler calls。三条策略都结束后，runner 会在每个任务目录写入 `cross_policy_audit.json`，并在 `comparisons/` 保存同一 trial 的三策略输入一致性、输出误差以及 NCU 是否改变最终选择。

## C：exact same-ready 的 Janus §4.8 配对干扰

预选只使用 B 阶段调度结果，不看 slowdown。主比较是 `NewTD+DRT` 与 `NewTD+NCU-DRT`，因为两者准入完全相同；次比较是原 Janus 与 `NewTD+NCU-DRT`。仅匹配两条 trace 中唯一出现的、顺序完全相同的 `ready_ops`。

每个候选组在相同微基准中测量：5 个独立 timing 进程，每个进程 100 次；另用 NSYS 标记并重放 10 次。只有一对中的两个组都至少一次出现“组内所有 OP kernel 共同重叠”时，才进入配对 slowdown 主结果。

```bash
export JANUS_FORMAL_REPO=/public_0/LYX/janus_formal_threeway_20260827
export JANUS_FORMAL_EXPECTED_COMMIT=REPLACE_WITH_REVIEWED_40_CHARACTER_COMMIT
export JANUS_STAGE_C_OUT=/public_0/LYX/janus_formal_stage_c_same_ready_20260827
export JANUS_STAGE_B_ROOT=$JANUS_STAGE_B_OUT
bash "$JANUS_FORMAL_REPO/experiments/formal_threeway_20260827/run_stage_c_same_ready.sh"
```

`manifest.json` 保存所有 exact-ready 覆盖、未改变选择、组宽不合格、缺少 OP 映射和结构过滤记录；`analysis/summary.json` 保存逐组与逐对结果。该实验表示“被策略选中的 OP 组合在隔离共同启动时的干扰”，不是完整模型延迟。
