# Opara/Janus 方向讨论总结

> 日期：2026-07-12
> 基于 `janus/` 源码分析，与 `PriorityOpara_v0/` 对照

---

## 项目现状：janus 与 PriorityOpara_v0 关键差异

### 方向 B（HP 间 interference 优化）

| 维度 | PriorityOpara_v0 | janus |
|------|-----------------|-------|
| `Critical_node.mark_critical_nodes()` 调用 | ✅ 有 (GraphCapturer.py L172) | ❌ **缺失**（GraphCapturer.py 未 import，从未调用） |
| 调度组合选择 | 仅 SM 占用率 | 占用率 > 0.9×max → 再按 name-based 分类选 compute/memory 1:1 |
| `is_critical` 保护 | 关键路径节点永不降级 | 无保护（从未调用 mark_critical_nodes） |

**janus 的 `Scheduler.py` 已经实现了 name-based 的计算/内存混搭**（L290-317），但基于算子名硬编码分类，不够精确。

---

## 讨论记录

### 1. 方向 B 的实现方案对比

| 方案 | 分类粒度 | 数据来源 | 效果 |
|------|---------|---------|------|
| name-based（janus现有） | 二分类 | 算子名关键词匹配 | 按名字猜，不准 |
| **方向 A（profile 密度）** | 连续值 | KernelProfile.duration / blocks | 比猜名字好，仍是一维 |
| **方向 B（资源多样度）** | 4 维向量 | reg, shared_mem, warps, duration | 最精确，能发现"类型相同但资源互补"的组合 |

**方向 B 已在 `Scheduler.py` 实现**（commit `2c8744e`，分支 `方向b测试`）：

```
schedule() 流程:
1. 枚举组合 → VirtualSM 仿真 → 算 SM 占用率
2. 保留占用率 ≥ 0.9×最高分的候选
3. 对候选组合算 4 维资源特征向量的余弦相似度
   → 相似度越低 → 资源越互补 → interference 越小
4. 选相似度最低的组合，同分选占用率高的
```

---

### 2. 调度机制深入理解

**核心：`launch()` 的 `while queue` 循环在 CPU 调度阶段一次性执行（在 CUDA Graph 捕获之前）。**

```
Phase 1 (CPU): launch() 的 while queue
  → 分配所有节点的 stream 和 event
  → 重排 FX 图节点顺序（写入 result 列表）

Phase 2 (CPU): 3 次预热 + CUDA Graph 捕获
  → Scheduler (Interpreter) 按重排后的顺序逐个发射节点
  → GPU 异步执行，发射即返回

Phase 3 (GPU, 反复): g.replay()
  → 一次重放整个 CUDAGraph
  → GPU 上无"轮次"概念，只有 event 同步 + SM 资源竞争
```

### 3. "轮次"的本质

CPU 调度阶段的"轮" = 一次 `while queue` 迭代。

**轮次同步机制（OperatorLauncher.py）：** `prestage_ops` 只存 HP 的 name（L208-210）：

```python
prestage_ops.clear()
for op in scheduled_ops:       # scheduled_ops = HP
    prestage_ops.append(op.name)
```

下一轮 HP 只等上一轮所有不同流的 HP（L150-153、L204-207）：

```python
for op in scheduled_ops:       # 下一轮 HP
    node = nodes[op.name]
    for pre_op in prestage_ops: # 上一轮 HP
        pre_node = nodes[pre_op]
        if node.stream != pre_node.stream:
            node.event_to_wait.append(pre_node.event)
```

**LP 从不加入 `prestage_ops`** → 下一轮 HP 不等上一轮 LP → HP 和 LP 可以同时在 SM 上共存。

### 4. 方向二（LP 阻塞问题）的真正本质

**不是"串行"问题，而是"资源竞争"问题。**

```
GPU timeline:
stream0(高优先级): [H1_r1...][H3_r2...]
stream1(高优先级): [H2_r1...][H4_r2...]
stream2(普通):      [L1_r1==================] 占 SM
                                    ↑ H3/r4发射时 L1还在SM上
                                    ↑ 不是串行(不等event)，但L1占SM
```

**CUDA stream priority 的限制：** 只能影响"待发射的 block 的排队顺序"，不能抢占已在 SM 上运行的 block。

---

### 5. 方向 2 新方案：LP 分级 + Slack 感知 Defer

#### 核心思路

不再将所有 LP 一刀切，而是根据执行时长 + DAG 松弛时间分级：

| 级别 | 条件 | 行为 |
|------|------|------|
| **LP-light** | CSS < 0.5 + duration < 阈值 | 保持原逻辑，跟 HP 同轮捡漏 |
| **LP-heavy** | CSS < 0.5 + duration ≥ 阈值 + slack 充足 | Defer 到 Phase 2，集中调度 |
| **HP（原本）** | CSS ≥ 0.5 或 关键路径 或 slack 不足 | 正常走 priority stream |

#### DAG 松弛时间（Slack）的计算

```
slack(node) = latest_finish(node) - earliest_start(node) - duration(node)

其中:
  earliest_start = max(所有前驱的 earliest_start + duration)
  latest_finish  = min(所有后继的 latest_finish) - duration
  critical_path  = max(earliest_start + duration)

slack > 0 → 有松弛时间，可以推迟而不影响总执行时间
slack = 0 → 关键节点，不可推迟
```

#### Phase 2 设计

Phase 2 复用 `schedule()` 引擎（含方向 B 的资源多样度打分），对 deferred LP-heavy 做集中调度：

```
执行流程:
  Phase 1: HP + LP-light 正常调度 (while queue)
           每个迭代检测 LP-heavy → 标记 + defer
           LP-heavy 计入 in_degree 但不分配 stream
  
  Phase 2: 所有 deferred LP-heavy 入队
           复用 schedule() 选组合
           分配 priority stream + 事件同步
```

#### 关键优势

1. **大 LP 不干扰 HP** — 不在同一轮次，不共享 SM
2. **SM 不浪费** — LP-heavy 之间可以互相并行（方向 B 的资源多样度依然适用）
3. **不改 CUDA Graph 结构** — 仍然是一次捕获、多次 replay
4. **改动量小** — 集中在 `OperatorLauncher.py` 一个文件

---

### 6. d1 + d2 - 1 的意义（Critical_node.py）

| 符号 | 含义 |
|------|------|
| d1 | 正向深度：从输入到该节点的最长路径步数 |
| d2 | 反向深度：从该节点到输出的最长路径步数 |
| d1 + d2 - 1 | 经过该节点的最长完整路径长度 |
| critical_path_length | max(d1 + d2 - 1) |

`d1 + d2 - 1 == critical_path_length` → 关键节点，不可推迟

**局限性：** d1/d2 基于拓扑步数，不是实际执行时间。精确的 slack 计算需要 profile 中的 duration。

---

## 隐患与风险

### P1: LP-heavy 后继死锁

**风险最高。** 如果 LP-heavy 在 `queue` 中但从不参与 `schedule()`，它的后继节点的 `in_degree` 永远不会归零，导致死循环。

**解法：** LP-heavy 被 defer 时必须将其计入 `scheduled_node_names`，使它的后继节点的 in_degree 正常递减。

### P2: Phase 2 也需要调度循环

LP-heavy 节点之间也可能存在数据依赖，不能简单全部同时发射。Phase 2 需要复用 `while queue` + `schedule()` 的循环结构。

### P3: Profile duration 基于串行执行

profile 阶段的 duration 是串行执行的，并行后由于资源竞争，实际 duration 会变化。
- slack 计算基于串行 timing → 可能不精确
- 建议加保守系数（如 slack > duration × 0.3 才 defer）

### P4: max_light_dur 阈值需要调参

- 100μs 是经验值，实际最优值取决于 GPU 型号和算子类型
- 建议在 A5000 上做一次扫描测试

### P5: 方向 B 与方向 2 的交互

方向 B 的 `resource_diversity_score()` 在 Phase 1 中对 HP 应用，Phase 2 中对 LP-heavy 应用。需要确认 Phase 2 的 `VirtualSM` 状态正确（时间未推进、SM 全部空闲）。

---

## 后续路线图

```
Step 1: ✅ 方向 B（已完成）
  → Scheduler.py 资源多样度调度
  → 分支: main + 方向b测试

Step 2: 🔄 方向 2（当前实现）
  → OperatorLauncher.py LP分级 + Slack defer
  → 分支: 方向2_LP分级_Defer

Step 3: 📋 A5000 测试
  → 对比 main vs 方向b测试 vs 方向2 三个分支
  → 调参（max_light_dur, slack_ratio）

Step 4: 🔜 (可选) LP 内部再细分
  → LP-heavy 内部分 compute-heavy / memory-heavy
  → Phase 2 里用方向 B 做资源混搭
```
