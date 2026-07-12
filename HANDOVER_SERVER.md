# Janus 优化项目 · 服务器交接文档

> 交接日期：2026-07-12
> 本地已完成方向 B（资源多样度调度）+ 方向 2（LP 分级 Slack Defer），
> 需在 A5000 服务器上验证 + 调参。

---

## 一、GitHub 仓库结构

**地址：** `https://github.com/KKKABI/janus-gpu-scheduler`

| 分支 | 包含内容 | 测试目的 |
|------|---------|---------|
| `original` | 论文原始基线 v0 | 对照组 |
| `main` | 方向 B + 方向 2 合并 | 当前工作主线 |
| `方向b测试` | 仅方向 B（Scheduler.py 资源多样度） | 单独验证方向 B 效果 |
| `方向2_LP分级_Defer` | 仅方向 2（OperatorLauncher.py LP分级+Phase 2） | 单独验证方向 2 效果 |

## 二、在服务器上快速上手

```bash
git clone https://github.com/KKKABI/janus-gpu-scheduler.git
cd janus-gpu-scheduler

# 编译 C++ CUDA 扩展（首次 + CUDA/Python 版本变更时需要）
cd Opara
python setup.py install
cd ..

# 三路对比测试：
git checkout original       && python examples/googlenet_example.py  # 基线
git checkout 方向b测试      && python examples/googlenet_example.py  # 方向 B
git checkout 方向2_LP分级_Defer && python examples/googlenet_example.py  # 方向 2
```

注意：profile 缓存以 `profile_result/<Model>_<shape>.pt.trace.json` 为键，不同 GPU（如 A5000 vs 本机显卡）需要删除缓存重新 profiling：

```bash
rm Opara/profile_result/*.pt.trace.json
```

多模型扫描（推荐测全）：

```bash
for m in googlenet inception_v3 nasnet bert yolov8x convnext resnet50 mobilenetv2 deepfm; do
  python examples/${m}_example.py
done
```

---

## 三、修改了什么

### 方向 B — Scheduler.py（分支 `方向b测试`）

**文件：** `Opara/Scheduler.py` L288-353

**改了什么：** 替换 name-based 计算/内存分类，改资源多样度打分。

```
旧：按算子名猜 (is_mem_access_intensive) → comp/mem 数量接近 1:1
新：4 维资源特征向量(reg, smem, warps, dur) → 余弦相似度越低 → 资源越互补
```

**流程：**
```
schedule():
  1. 枚举 C(n,1)+...+C(n,5) 组合
  2. VirtualSM 仿真 → 算 SM 占用率
  3. 保留 ≥ 0.9×最大占用率的候选
  4. 对保留下来的组合，算 4 维余弦相似度
  5. 选相似度最低的组合（同分选占用率高的）
```

### 方向 2 — OperatorLauncher.py（分支 `方向2_LP分级_Defer`）

#### 改点 1：`compute_node_slack()`（DAG 松弛时间分析）

用 profile 中的 kernel duration 做 forward/backward pass，算每个节点的最早开始时间和最晚结束时间，进而得到 slack：

```
slack = latext_finish - earliest_start - duration
slack > 0 → 可以推迟而不影响总执行时间
```

#### 改点 2：`pop_lowPriorty_from_queue()` 分级

| 级别 | 条件 | 行为 |
|------|------|------|
| **LP-light** | CSS < 0.5 + duration < 100μs | 同原来行为，跟 HP 同轮 |
| **LP-heavy** | CSS < 0.5 + duration ≥ 100μs + slack > duration×0.3 | defer 到 Phase 2 |
| **HP** | 上述不满足 + 关键路径 | 正常 priority stream |

#### 改点 3：`launch()` 双阶段

```
Phase 1 (HP + LP-light):    原 while queue 循环
Phase 2 (LP-heavy 调度):   复用 schedule() + priority stream
```

---

## 四、待办事项（按优先级）

### P0：验证三路对比

```bash
# 在 A5000 上跑
git checkout original       && python examples/googlenet_example.py  > /tmp/result_original.txt
git checkout 方向b测试      && python examples/googlenet_example.py  > /tmp/result_dirb.txt
git checkout 方向2_LP分级_Defer && python examples/googlenet_example.py  > /tmp/result_dir2.txt
```

关注指标：**推理延迟（latency）**，其次是 CUDA Graph 捕获时间。

### P1：调参（如果效果不理想）

两个关键参数在 `pop_lowPriorty_from_queue()`：

```python
def pop_lowPriorty_from_queue(queue, slack=None, tau=0.5, max_light_dur=100.0, slack_ratio=0.3):
```

- **`max_light_dur`** — LP-light 的时长阈值（μs），决定"多大算 LP-heavy"。扫描 50/100/200/500 看效果。
- **`slack_ratio`** — 安全系数。当前 0.3 = 需要 slack > duration×30% 才 defer。扫描 0.1/0.3/0.5 看效果。
- **`tau`** — CSS 阈值（0.5）。如果 LP 太多或太少可以调整。

扫码测试脚本示意：

```python
for max_light_dur in [50, 100, 200, 500]:
    for slack_ratio in [0.1, 0.3, 0.5]:
        # 运行 benchmark，记录延迟
```

### P2：多模型诊断

用 `examples/diagnose_sync.py` 风格脚本统计每个模型的 LP-heavy/LP-light 分布：

```
模型  |  总节点  | LP 数  | LP-light  | LP-heavy  | 被 defer  | Phase 2 平均组合大小
```

用于验证 LP-heavy 分级是否合理。

### P3：方向 B 单独验证

方向 B 的改进点在于：两个"同类型"的算子（都 compute-heavy）会不会因为资源互补而被正确识别"可以并行"。建议在 ConvNeXt（max_width=47，并行度高）上重点观察。

---

## 五、已知隐患

### 1. in_degree 死锁（已处理）
Phase 1 中 LP-heavy 被 defer 时已正确更新其后继的 in_degree，不会死锁。但如果有新的 DAG pattern 导致 LP-heavy 被标记但 in_degree 未减，会出现 queue 为空的假象。

### 2. Profile duration 不精确
slack 计算基于**串行** profile 的 duration。并行后由于 resource contention，实际 duration 会变长，slack 可能高估。保守系数（slack_ratio=0.3）已经加了，但需要实测检验。

### 3. 没有 Barrier event
Phase 1 → Phase 2 之间没有硬同步（没有在所有 HP 流上 record 一个 event 让 LP-heavy 等）。当前靠 FX 图节点顺序做软排序——LP-heavy 在 result 列表末尾，CUDA Graph 捕获时会按顺序发射。如果 GPU 调度器的动态负载均衡打破了软排序，LP-heavy 可能在 Phase 1 的 HP 完成之前就拿到 SM。**这可能不是问题**（充分利用空闲 SM），但如果要严格隔离需要加 barrier event。

### 4. 无 Critical_node 保护
janus 版的 GraphCapturer 没有调用 `Critical_node.mark_critical_nodes()`，所以 `is_critical` 属性在所有节点上都是 False。这导致关键路径上的保护缺失。但 slack 计算本身不依赖它，所以方向 2 不受影响。方向 B 也不依赖它（方向 B 只看资源特征向量，不看是否关键路径）。

---

## 六、关键文件 & 行号速查

| 文件 | 关键函数/类 | 行号 | 作用 |
|------|-----------|------|------|
| `Opara/Scheduler.py` | `Scheduler.schedule()` | L288-318 | 方向 B：资源多样度调度 |
| `Opara/OperatorLauncher.py` | `pop_lowPriorty_from_queue()` | L70-178 | LP 分级决策 |
| `Opara/OperatorLauncher.py` | `compute_node_slack()` | L14-67 | DAG 松弛时间计算 |
| `Opara/OperatorLauncher.py` | `launch()` | L184-384 | 双阶段调度（Phase 1 + Phase 2） |
| `Opara/OperatorLauncher.py` | `recompile()` | L482-507 | 计算 slack 并传给 launch |

---

## 七、建议实验顺序

```yaml
Day 1:
  - 搭环境：clone + 编译 + 删缓存
  - 跑三路对比（original vs 方向b vs 方向2）
  - 记录每个模型的延迟

Day 2:
  - 如果方向 2 效果好：做参数扫描（max_light_dur, slack_ratio）
  - 如果方向 2 效果不好：跑诊断统计，看 LP-heavy 分级是否合理
  - 多模型全跑一遍

Day 3:
  - 写实验结果报告
  - 合并两个方向的最佳参数到 main
  - 推 final 版本
```
