# 实验日志 — 2026-07-16

## 分支状态

| 分支 | 最新 commit | 内容 |
|------|-----------|------|
| `original` | `73f4da5` | baseline + 方向4 多客户端脚本 + sm_fraction + 关键路径修复 |
| `方向b测试` | `dd49ef9` | **方向1 主力分支**：时域仿真 + 三选策略 + alpha sweep |
| `方向2_LP分级_Defer` | (旧) | 方向2 LP 分级，已搁置 |

## 环境

- GPU: NVIDIA RTX A5000 (64 SM)
- CUDA: 11.6, PyTorch: 2.0.0
- conda env: `opara`
- 运行前需 `export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$(id -u)`
- 多客户端脚本需先 `rm -rf /tmp/opara_scale_* /tmp/opara_multi_*`

## 关键文件修改

### Opara/Scheduler.py
- `ResourceModel` 新增 `time_domain` 参数
- `can_apply_launch()` 包含原始静态分配 + 时域仿真两种模式
- `Scheduler` 新增 `alpha`、`selection_mode`、`time_domain` 参数
- 三种选择策略：`max_occupancy` / `cosine` / `min_resource`
- 全局 `_CANDIDATE_STATS` 统计候选组合数
- `dump_candidate_stats()` 在调度结束后打印汇总

### Opara/OperatorLauncher.py
- `launch()` / `recompile()` 新增 `alpha`、`selection_mode`、`time_domain` 参数
- 调度结束后调用 `dump_candidate_stats()`

### Opara/GraphCapturer.py
- `capturer()` 新增 `alpha`、`selection_mode`、`time_domain` 参数
- 已补上 `Critical_node.mark_critical_nodes()` 调用

### examples/multi_client_compare.py
- MODEL_REGISTRY 含 GoogLeNet、MobileNetV2、ResNet50、NASNet
- 双客户端 × 三种模式 × sm_fraction 支持
- 总延迟：串行=加和，并发=max

### examples/scale_clients.py
- MODEL_REGISTRY 含上述模型 + ConvNeXt
- N 客户端扩展性测试 × sm_fraction 支持
- Total = max(所有客户端延迟)，单次推理 wall-clock

## 典型命令

```bash
# 单模型基准
python examples/googlenet_example.py

# 双客户端对比
python examples/multi_client_compare.py --iterations 50 --warmups 20

# Scale 测试
python examples/scale_clients.py --model convnext --clients 1,2,3,4,8

# 方向1 三策略对比 (需要 alpha + selection_mode)
python -c "
from Opara import GraphCapturer
runner = GraphCapturer.capturer(x, m, alpha=0.9, selection_mode='max_occupancy', time_domain=True)
"
```

## 方向1 结论 (最终)

1. 修复前 VirtualSM 过于保守（要求 block 同时驻留），大 grid 算子被误杀
2. 时域仿真修复 → Inception-v3 -10.5%, NASNet -9.0%
3. 三策略 (max_occupancy/cosine/min_resource) 在所有 α 下等价
4. 最终方案：α=0.9 + max_occupancy + 时域仿真

## 方向4 结论 (最终)

1. 轻模型 (mob, gln) 多 Graph 完美并发
2. 重模型 (NASNet) 并发互踩 (1.84x 退化)
3. sm_fraction 改 profile 资源 → 无效
4. 统计口径已修正：串行加和、并发 max
