# 方向1 关键问题解答

## 1. min_resource 策略是什么？

**目标**：在 α 阈值筛出的高占用候选组合里，选总资源压力最小、三维最均衡的组合。

**做法**：对候选组合里所有算子的寄存器、共享内存、warp 三种资源，分别累加全局总量，除以全 GPU 容量得到利用率 p_reg、p_smem、p_warp。评分 = 平均利用率 + 最大利用率：

```
score = (p_reg + p_smem + p_warp) / 3 + max(p_reg, p_smem, p_warp)
```

平均部分让总体压力低的胜出，max 部分惩罚某个维度特别高的不均衡组合。选 score 最低的。

**结论**：和 max_occupancy 效果等价，未超越。

---

## 2. 时域仿真是什么？

**背景**：原始 VirtualSM 的 `can_apply_launch()` 要求组合内所有算子**全部 block 同时驻留**在 SM 上。elementwise kernel grid=1568 block，但 64 SM 每条只能塞 12 块 → 总共 768 → 塞不完 → 判不可行。大量实际可行的组合被误杀。

**修复**：塞不下时不立即放弃，而是推进时间到最早完成的 block、释放资源、再继续塞。模拟 GPU 流水式执行——block 跑完释放资源、下一个顶上。循环直到全部 block 分配完或死锁。

**效果**：Inception-v3 -10.5%、NASNet -9.0%。

---

## 3. 余弦相似度策略中，一组 HPOP 超过两个是两两相比吗？

**对，两两计算后取平均。**

对组合内的 N 个算子，每个算子提取资源特征向量：

```
[reg_per_block, smem_per_block, warps_per_block, dur_per_block]
```

每维 min-max 归一化后，对 N 个算子的 C(N,2) 对，计算每对之间的余弦相似度，取所有对的平均值作为组合的相似度分数。值越高 = 算子之间资源越像 = interference 越严重。选择相似度最低（最互补）的组合。

非配对比较，而是全组合内两两平均。单个算子（N=1）返回 0。

---

## 4. ncu 七维余弦与之前四维的差别（除维数外）

**相同点**：
- 都是两两算余弦相似度，取平均
- 都是 min-max 归一化
- 都是选相似度最低的组合

**关键区别**：

| | 四维（PyTorch profiler） | 七维（+ ncu） |
|---|---|---|
| 维度 | reg, smem, warps, dur | + dram_thru, l2_thru, comp_thru |
| 数据来源 | PyTorch profiler (串行一次前向) | Nsight Compute (单 kernel 多次重放采集) |
| 能区分什么 | 计算 vs 内存的粗分类（名字匹配） | 实际 DRAM 带宽占用、L2 命中率、SM 计算占用 |
| 覆盖范围 | 所有模型自动采集（快，~1 秒） | 需手动建缓存（慢，~5 分钟/模型） |
| 能补的短板 | — | 告诉调度器哪些算子抢带宽、哪些算子互补 |
| 实际效果 | GoogLeNet/ConvNeXt 均未超越 max_occupancy | 同左 |

**结论**：七维信息更丰富但未转化实际延迟提升。warp 占用率本身已是最好的 interference 代理指标。
