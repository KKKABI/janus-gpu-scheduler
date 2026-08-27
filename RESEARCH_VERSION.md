# 新 TD + NCU-DRT 代码快照

本分支保存第二阶段的最终研究代码，基于 Janus 原始工程继续开发，不包含实验生成的 `.nsys-rep`、SQLite、日志和结果目录。

## 两个核心改动

1. **新 TD 准入**：先保留 Janus Static 能通过的组合；Static 拒绝后，只允许二元组进入更谨慎的 TD 扩展检查。代码位于 `experiments/newtd_accuracy/newtd_pair_admission.py`。
2. **NCU-DRT 排序**：调度器读取离线 NCU 画像中的 DRAM、L2、Compute 和 Duration 信息，并在 DRT 候选中估计资源干扰。核心代码位于 `Opara/Scheduler.py` 和 `Opara/ncu_profiler.py`，七个模型的离线缓存位于 `Opara/ncu_result/`。

## 两种运行口径

- 只验证新 TD 判断正确率时，保持默认 `JANUS_NEW_TD_FINAL_SELECTOR=strategy`。这会隔离准入规则，不让后续 NCU 排序影响正确率统计。
- 运行“新 TD + NCU-DRT”组合版本时，设置：

```bash
export JANUS_NEW_TD_PAIR_EXTENSION=1
export JANUS_NEW_TD_FINAL_SELECTOR=risk_adjusted_interference
export JANUS_NEW_TD_SOLO_ROOT=/path/to/solo_operator_profiles
```

随后使用 `experiments/newtd_accuracy/run_one_newtd.py`，其余模型、输入和阈值参数按实验脚本传入。

## 主要目录

- `experiments/newtd_accuracy/`：新 TD 最终准入、七模型正确率和延迟脚本。
- `experiments/simulator_accuracy/`：TD 设计、独立组合验证及分析脚本。
- `experiments/janus_47_aligned_20260821/`：按 Janus 4.7 口径进行硬件检查的脚本。
- `experiments/test_td_simulator.py`：TD 与 NCU 干扰计算的单元测试。

本分支中的默认参数保留已经完成实验时的口径；新增的 `JANUS_NEW_TD_FINAL_SELECTOR` 仅用于选择已有的后续排序器，不改变新 TD 准入公式。
