# TD + DRT 代码快照

本分支保存第一阶段 TD+DRT 的可复现实验代码。统一 Baseline 为原生 Janus；TD 用时域资源模拟扩展候选判断，DRT 在可行候选中按照预计轮次收益和资源压力排序。

## 主要入口

- `experiments/EXPERIMENT_PROTOCOL.md`：冻结后的实验口径。
- `experiments/run_one.py`：单个模型、单种策略的独立进程运行入口。
- `experiments/run_matrix.py`：多模型策略矩阵入口。
- `Opara/Scheduler.py`：Static、TD、Janus 打分和 DRT 打分实现。
- `experiments/test_td_simulator.py`：TD/DRT 相关单元测试。

主对比配置为原生 Janus、TD+Janus、Static+DRT 和 TD+DRT。实验结果目录、日志以及 `.nsys-rep` 文件不放入本代码分支。
