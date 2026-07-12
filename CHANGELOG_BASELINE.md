# 基线版本变更记录

> 日期：2026-07-12
> 版本：基线 `main` (commit `e84baf9`)

## 变更内容

### 1. Scheduler 组合爆炸修复

**文件：** `Opara/Scheduler.py` L254-259

**问题：** `schedule()` 对就绪算子做 C(n,1)+...+C(n,5) 暴力枚举。ConvNeXt 的 max_width=83 产生约 3000 万种组合，每个组合需要 deepcopy VirtualSM，导致调度阶段无限期挂起。

**修复：** 当就绪节点超过 15 时，按 kernel duration 总和排序取前 15，将组合数从 C(83,5) 降至 C(15,5)=3003。

```python
MAX_READY = 15
if len(ready_ops) > MAX_READY:
    ready_ops = sorted(ready_ops, key=lambda op: sum(
        k.duration for k in op.kernels
    ), reverse=True)[:MAX_READY]
```

**影响：** 绝对值可能略差（截断了候选），但基线/方向B/方向2 统一受此限制，相对比较公平。

### 2. ConvNeXt 示例迭代数调整

**文件：** `examples/convnext_example.py` L88-89

- `warm_ups`: 10 → 3
- `iterations`: 300 → 20

**原因：** 原生代码每轮 iteration 有 1 秒 `torch.cuda._sleep`，300 轮仅 native PyTorch 阶段就耗时 5+ 分钟，不适合快速迭代测试。

## 基线测试结果 (2026-07-12)

| 模型 | Opara Latency (ms) | max_width |
|------|-------------------|-----------|
| ConvNeXt | 4.86 | 83 |
| GoogleNet | 0.80 | 4 |
| InceptionV3 | 2.09 | 4 |
| BERT | 1.34 | 4 |

GPU: NVIDIA RTX A5000, CUDA 12.5, PyTorch 2.0
