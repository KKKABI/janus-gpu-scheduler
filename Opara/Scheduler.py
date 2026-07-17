from typing import List
import copy
import os
from itertools import combinations



# -------------------------------
# KernelProfile
# -------------------------------

class KernelProfile:
    def __init__(self, name: str, duration: float, shared_mem: int, registers: int, warps: int, blocks: int):
        self.name = name
        self.duration = duration
        self.shared_mem = shared_mem
        self.registers = registers
        self.warps = warps
        self.blocks = blocks
        self.blocks_remaining = blocks

    def has_pending_blocks(self) -> bool:
        return self.blocks_remaining > 0

    def allocate_block(self):
        if self.blocks_remaining > 0:
            self.blocks_remaining -= 1

# -------------------------------
# OperatorTask
# -------------------------------

class OperatorTask:
    def __init__(self, name: str, kernels: List[KernelProfile]):
        self.name = name
        self.kernels = kernels
        self.kernels_remaining = len(kernels)

    def has_pending_kernels(self) -> bool:
        return self.kernels_remaining > 0
    
    def launch_kernel(self):
        if self.kernels_remaining > 0:
            self.kernels_remaining -= 1



# -------------------------------
# VirtualSM
# -------------------------------

class VirtualSM:
    def __init__(self, shared_mem_total: int, register_total: int, warp_total: int):
        self.shared_mem_total = shared_mem_total
        self.register_total = register_total
        self.warp_total = warp_total
        self.shared_mem_used = 0
        self.registers_used = 0
        self.warps_used = 0
        self.running_blocks = []  # (end_time, kernel_name, block_resource)

    def can_accept(self, block_resource) -> bool:
        return (
            self.shared_mem_used + block_resource["shared_mem"] <= self.shared_mem_total and
            self.registers_used + block_resource["registers"] <= self.register_total and
            self.warps_used + block_resource["warps"] <= self.warp_total
        )
    
    def max_blocks_fit(self, block_resource):
        shared_mem = block_resource['shared_mem']
        registers = block_resource['registers']
        warps = block_resource['warps']

        max_blocks_shared_mem = (self.shared_mem_total - self.shared_mem_used) // shared_mem if shared_mem > 0 else float('inf')
        max_blocks_registers = (self.register_total - self.registers_used) // registers if registers > 0 else float('inf')
        max_blocks_warps = (self.warp_total - self.warps_used) // warps if warps > 0 else float('inf')

        return min(max_blocks_shared_mem, max_blocks_registers, max_blocks_warps)

    def allocate_block(self, kernel_name, block_resource, start_time, duration):
        assert self.can_accept(block_resource)
        self.shared_mem_used += block_resource["shared_mem"]
        self.registers_used += block_resource["registers"]
        self.warps_used += block_resource["warps"]
        end_time = start_time + duration
        self.running_blocks.append((end_time, kernel_name, block_resource))

    def release_finished_blocks(self, current_time):
        still_running = []
        for end_time, kernel_name, block_resource in self.running_blocks:
            if end_time <= current_time:
                self.shared_mem_used -= block_resource["shared_mem"]
                self.registers_used -= block_resource["registers"]
                self.warps_used -= block_resource["warps"]
            else:
                still_running.append((end_time, kernel_name, block_resource))
        self.running_blocks = still_running

    def get_utilization(self) -> float:
        return self.warps_used / self.warp_total if self.warp_total > 0 else 0.0



# -------------------------------
# ResourceModel
# -------------------------------

class ResourceModel:
    def __init__(self, sm_count: int, sm_specs: dict, time_domain=True):
        self.sms = [VirtualSM(**sm_specs) for _ in range(sm_count)]
        self.current_time = 0.0
        self.pending_kernels = []
        self.time_domain = time_domain

    def update_time(self, current_time: float):
        for sm in self.sms:
            sm.release_finished_blocks(current_time)
        self.current_time = current_time

    def can_apply_launch(self, operator: OperatorTask, start_time: float)-> bool:
        virtual_sms = copy.deepcopy(self.sms)
        virtual_operator = copy.deepcopy(operator)
        kernel = virtual_operator.kernels[0]

        if not self.time_domain:
            # 原始静态分配：所有 block 必须同时驻留
            while kernel.has_pending_blocks():
                block_resource = {
                    'shared_mem': kernel.shared_mem,
                    'registers': kernel.registers,
                    'warps': kernel.warps
                }
                sm_capacities = []
                for sm in virtual_sms:
                    max_blocks = sm.max_blocks_fit(block_resource)
                    if max_blocks > 0:
                        sm_capacities.append((max_blocks, sm))
                if not sm_capacities:
                    break
                _, selected_sm = max(sm_capacities, key=lambda x: x[0])
                selected_sm.allocate_block(kernel.name, block_resource, start_time, kernel.duration)
                kernel.allocate_block()
            return not kernel.has_pending_blocks()

        # 时域仿真：推进时间释放已完成 block 再重试
        current_time = start_time
        MAX_ITER = 2000
        for _ in range(MAX_ITER):
            while kernel.has_pending_blocks():
                block_resource = {
                    'shared_mem': kernel.shared_mem,
                    'registers': kernel.registers,
                    'warps': kernel.warps
                }
                sm_capacities = []
                for sm in virtual_sms:
                    max_blocks = sm.max_blocks_fit(block_resource)
                    if max_blocks > 0:
                        sm_capacities.append((max_blocks, sm))
                if not sm_capacities:
                    break
                _, selected_sm = max(sm_capacities, key=lambda x: x[0])
                selected_sm.allocate_block(kernel.name, block_resource, current_time, kernel.duration)
                kernel.allocate_block()

            if not kernel.has_pending_blocks():
                return True

            min_end_time = None
            for sm in virtual_sms:
                for end_time, _, _ in sm.running_blocks:
                    if min_end_time is None or end_time < min_end_time:
                        min_end_time = end_time
            if min_end_time is None or min_end_time <= current_time:
                return False
            current_time = min_end_time
            for sm in virtual_sms:
                sm.release_finished_blocks(current_time)

        return False
            
    def apply_launch(self, operator: OperatorTask, start_time: float):
        kernel = operator.kernels[0]
        # for kernel in operator.kernels:
        while kernel.has_pending_blocks():
            block_resource = {
                'shared_mem': kernel.shared_mem,
                'registers': kernel.registers,
                'warps': kernel.warps
             }

                # 计算每个 SM 当前可容纳的最大线程块数
            sm_capacities = []
            for sm in self.sms:
                max_blocks = sm.max_blocks_fit(block_resource)
                if max_blocks > 0:
                    sm_capacities.append((max_blocks, sm))

            if not sm_capacities:
                break  # 没有 SM 可以容纳该线程块

                # 选择可容纳线程块数最多的 SM
            _, selected_sm = max(sm_capacities, key=lambda x: x[0])

            # 分配一个线程块
            selected_sm.allocate_block(kernel.name, block_resource, start_time, kernel.duration)
            kernel.allocate_block()
        kernel.blocks_remaining = kernel.blocks
                


           

    # def launch_pending_kernels(self, start_time: float):
    #     completed = []
    #     for kernel in self.pending_kernels:
    #         while kernel.has_pending_blocks():
    #             block_resource = {
    #                 'shared_mem': kernel.shared_mem,
    #                 'registers': kernel.registers,
    #                 'warps': kernel.warps
    #             }
    #             allocated = False
    #             for sm in self.sms:
    #                 if sm.can_accept(block_resource):
    #                     sm.allocate_block(kernel.name, block_resource, start_time, kernel.duration)
    #                     kernel.allocate_block()
    #                     allocated = True
    #                     break
    #             if not allocated:
    #                 break
    #         if not kernel.has_pending_blocks():
    #             completed.append(kernel)
    #     for k in completed:
    #         self.pending_kernels.remove(k)


    # def ready_for_next_launch(self) -> bool:
    #     return len(self.pending_kernels) == 0

    def _next_block_end_time(self):
        times = []
        for sm in self.sms:
            for end_time, _, _ in sm.running_blocks:
                times.append(end_time)
        return min(times) if times else self.current_time

    # def run_until_next_launchable(self):
    #     while not self.ready_for_next_launch():
    #         next_time = self._next_block_end_time()
    #         self.update_time(next_time)
    #         self.launch_pending_kernels(next_time)
    #     next_time = self._next_block_end_time()
    #     self.update_time(next_time)
    #     return next_time

    def run_until_next_launchable(self):
        
        next_time = self._next_block_end_time()
        self.update_time(next_time)
           
        
        return next_time
    
    def total_utilization(self) -> float:
        return sum(sm.get_utilization() for sm in self.sms) / len(self.sms)
    


# -------------------------------
# 爆搜
# ---

# 全局统计：每次 schedule() 调用时不同 alpha 下的候选组合数
_CANDIDATE_STATS = []  # list of dict: {total, alpha_0.9, alpha_0.8, alpha_0.5, alpha_0.2, occ_max}

def dump_candidate_stats():
    """打印候选组合统计汇总"""
    if not _CANDIDATE_STATS:
        return
    print("\n" + "=" * 70)
    print("  CANDIDATE COMBO STATS (Direction B scheduler)")
    print("=" * 70)
    print(f"  {'call':>5} {'total':>7} {'a=0.9':>7} {'a=0.8':>7} {'a=0.5':>7} {'a=0.2':>7} {'occ_max':>8}")
    print("  " + "-" * 52)
    for i, s in enumerate(_CANDIDATE_STATS):
        print(f"  {i:>5} {s['total']:>7} {s['a=0.9']:>7} {s['a=0.8']:>7} {s['a=0.5']:>7} {s['a=0.2']:>7} {s['occ_max']:>7.4f}")
    # 汇总
    total_all = sum(s['total'] for s in _CANDIDATE_STATS)
    n_calls = len(_CANDIDATE_STATS)
    for a in ['a=0.9', 'a=0.8', 'a=0.5', 'a=0.2']:
        n_single = sum(1 for s in _CANDIDATE_STATS if s[a] == 1)
        avg = sum(s[a] for s in _CANDIDATE_STATS) / n_calls
        print(f"  {a}: avg={avg:.1f}, only_1_combo={n_single}/{n_calls} ({100*n_single/n_calls:.0f}%)")
    print("=" * 70)
    _CANDIDATE_STATS.clear()


class Scheduler:
    def __init__(self, resource_model, alpha=0.9, selection_mode='cosine', time_domain=True):
        self.resource_model = resource_model
        self.alpha = alpha
        self.selection_mode = os.getenv('JANUS_SELECTION_MODE', selection_mode)
        self.overload_weight = float(os.getenv('JANUS_OVERLOAD_WEIGHT', '1.0'))
        self.tail_weight = float(os.getenv('JANUS_TAIL_WEIGHT', '0.02'))
        self.occupancy_weight = float(os.getenv('JANUS_OCCUPANCY_WEIGHT', '0.005'))
        self.time_domain = time_domain
        self._static_profile_cache = {}

    def _select_static_interference(self, ready_ops, combo_scores):
        """Predict round-time gain from magnitude-aware static pressure."""
        feasible = [(combo, occ) for combo, occ in combo_scores if occ >= 0]
        if not feasible:
            return []

        n_sms = max(1, len(self.resource_model.sms))
        sample_sm = self.resource_model.sms[0]
        reg_cap = float(max(1, sample_sm.register_total))
        smem_cap = float(max(1, sample_sm.shared_mem_total))
        warp_cap = float(max(1, sample_sm.warp_total))

        profiles = {}
        raw_densities = {}
        for op in ready_ops:
            kernels = [k for k in op.kernels if k.blocks > 0]
            cached = self._static_profile_cache.get(op.name)
            if cached is not None:
                profiles[op.name], raw_densities[op.name] = cached
                continue

            duration = sum(max(float(k.duration), 1e-9) for k in kernels)
            if not kernels:
                profiles[op.name] = (1e-9, [0.0, 0.0, 0.0, 0.0])
                raw_densities[op.name] = 0.0
                self._static_profile_cache[op.name] = (profiles[op.name], 0.0)
                continue

            # Resource-time demand includes whole-GPU residency, so two
            # kernels spanning every SM are not treated as complementary
            # merely because their register/shared-memory mix differs.
            pressure = [0.0, 0.0, 0.0, 0.0]
            total_blocks = 0.0
            for k in kernels:
                weight = max(float(k.duration), 1e-9) / duration
                fit_reg = int(reg_cap // k.registers) if k.registers > 0 else 32
                fit_smem = int(smem_cap // k.shared_mem) if k.shared_mem > 0 else 32
                fit_warp = int(warp_cap // k.warps) if k.warps > 0 else 32
                blocks_per_sm = max(1, min(32, fit_reg, fit_smem, fit_warp))
                coverage = min(1.0, float(k.blocks) / (n_sms * blocks_per_sm))
                resident = [
                    min(1.0, blocks_per_sm * k.registers / reg_cap) * coverage,
                    min(1.0, blocks_per_sm * k.shared_mem / smem_cap) * coverage,
                    min(1.0, blocks_per_sm * k.warps / warp_cap) * coverage,
                    coverage,
                ]
                for dim in range(4):
                    pressure[dim] += weight * resident[dim]
                total_blocks += float(k.blocks)

            profiles[op.name] = (duration, pressure)
            raw_densities[op.name] = duration / max(total_blocks, 1.0)
            self._static_profile_cache[op.name] = (
                profiles[op.name], raw_densities[op.name])

        density_scale = max(max(raw_densities.values(), default=0.0), 1e-9)

        def candidate_score(item):
            combo, occupancy = item
            durations = []
            vectors = []
            for op in combo:
                duration, pressure = profiles.get(op.name, (1e-9, [0.0, 0.0, 0.0, 0.0]))
                density = min(1.0, raw_densities.get(op.name, 0.0) / density_scale)
                durations.append(duration)
                vectors.append(pressure + [density])

            sequential = max(sum(durations), 1e-9)
            ideal_round = max(durations)
            # Convert instantaneous pressure into resource-time demand. A
            # short op contends only for its share of the longest op's
            # lifetime; the dominant demand predicts round-time inflation.
            summed = [
                sum(
                    v[d] * durations[i] / max(ideal_round, 1e-9)
                    for i, v in enumerate(vectors)
                )
                for d in range(5)
            ]
            overload = max(0.0, max(summed) - 1.0)
            predicted_round = ideal_round * (1.0 + self.overload_weight * overload)
            gain = (sequential - predicted_round) / sequential

            mean_duration = sequential / len(durations)
            variance = sum((d - mean_duration) ** 2 for d in durations) / len(durations)
            tail = (variance ** 0.5) / max(mean_duration, 1e-9)
            score = (
                gain
                - self.tail_weight * tail
                + self.occupancy_weight * max(0.0, occupancy)
            )
            return (score, -overload, -predicted_round, -len(combo), occupancy)

        return max(feasible, key=candidate_score)[0]

    def schedule(self, ready_ops: List["OperatorTask"], current_time: float) -> List["OperatorTask"]:

        self.resource_model.update_time(current_time)
        # 枚举所有候选组合并计算每个组合的 SM 占用率（occupancy）
        combo_scores = []  # list of (combo_list, occupancy_score)

        max_comb_size = min(5, len(ready_ops))
        # 防止组合爆炸：max_width > 15 时截断，否则 C(83,5) ≈ 3000万无法接受
        MAX_READY = 15
        if max_comb_size == 5 and len(ready_ops) > MAX_READY:
            ready_ops = sorted(ready_ops, key=lambda op: sum(
                k.duration for k in op.kernels
            ), reverse=True)[:MAX_READY]
        for r in range(1, max_comb_size + 1):
            for combo in combinations(ready_ops, r):
                virtual_model = copy.deepcopy(self.resource_model)
                feasible = True
                for op in combo:
                    if not op.kernels:
                        continue
                    if virtual_model.can_apply_launch(op, current_time):
                        virtual_model.apply_launch(op, current_time)
                    else:
                        # 如果单个算子本身无法在虚拟模型中分配完其线程块，则视为不可行
                        feasible = False
                        break
                # 即使不可行，也记录其占用情况（不可行的组合占用为 -inf，后续被丢弃）
                score = virtual_model.total_utilization() if feasible else -1.0
                combo_scores.append((list(combo), score))

        if not combo_scores:
            return []

        # 找到最大占用率
        occ_max = max(score for _, score in combo_scores)

        if self.selection_mode == 'static_interference':
            return self._select_static_interference(ready_ops, combo_scores)

        # 统计不同 alpha 下的候选组合数
        total_feasible = sum(1 for _, s in combo_scores if s >= 0)
        stats = {'total': total_feasible, 'occ_max': occ_max}
        for a_label, a_val in [('a=0.9', 0.9), ('a=0.8', 0.8), ('a=0.5', 0.5), ('a=0.2', 0.2)]:
            cnt = sum(1 for _, s in combo_scores if s >= a_val * occ_max)
            stats[a_label] = cnt
        _CANDIDATE_STATS.append(stats)

        # alpha 控制保留阈值
        alpha = self.alpha
        top_candidates = [combo for combo, score in combo_scores if score >= alpha * occ_max]

        # 如果没有满足阈值的候选，则退回到单纯的最大占用组合
        if not top_candidates:
            # 取占用率最高的组合
            best_combo = max(combo_scores, key=lambda x: x[1])[0]
            return best_combo

        # ===== 选择策略 =====
        if self.selection_mode == 'max_occupancy':
            # 纯最大占用率（基线）
            best_combo = max(top_candidates, key=lambda c: next(s for cc, s in combo_scores if cc is c))
            return best_combo

        elif self.selection_mode == 'min_resource':
            # 资源加和策略：选总资源压力最小的组合
            # 对每个组合计算三大资源的全局占用比例，取最均衡（max 压力最小）的
            N_SM = len(self.resource_model.sms)
            REG_CAP = 65536.0 * N_SM
            SMEM_CAP = 102400.0 * N_SM
            WARP_CAP = 48.0 * N_SM

            def min_resource_score(combo):
                if len(combo) <= 1:
                    return 0.0
                total_reg = 0.0; total_smem = 0.0; total_warps = 0.0
                for op in combo:
                    for k in op.kernels:
                        total_reg += k.registers * k.blocks
                        total_smem += k.shared_mem * k.blocks
                        total_warps += k.warps * k.blocks
                p_reg = total_reg / REG_CAP
                p_smem = total_smem / SMEM_CAP
                p_warp = total_warps / WARP_CAP
                # 返回平均压力 + 最大压力（惩罚不均衡）
                return (p_reg + p_smem + p_warp) / 3.0 + max(p_reg, p_smem, p_warp)

            def combo_sort_key_minres(combo):
                score = min_resource_score(combo)
                occ = next(s for c, s in combo_scores if c is combo)
                return (score, -occ)

            best_combo = min(top_candidates, key=combo_sort_key_minres)
            return best_combo

        else:  # 'cosine' — Direction B 余弦相似度
            def resource_diversity_score(combo):
                if len(combo) <= 1:
                    return 0.0
                profiles = []
                for op in combo:
                    reg = 0.0; smem = 0.0; warps = 0.0; dur = 0.0; blocks = 0
                    for k in op.kernels:
                        reg += k.registers * k.blocks
                        smem += k.shared_mem * k.blocks
                        warps += k.warps * k.blocks
                        dur += k.duration * k.blocks
                        blocks += k.blocks
                    if blocks > 0:
                        profiles.append([reg / blocks, smem / blocks, warps / blocks, dur / blocks])
                    else:
                        profiles.append([0.0, 0.0, 0.0, 0.0])
                n_dims = 4
                for d in range(n_dims):
                    vals = [p[d] for p in profiles]
                    vmin, vmax = min(vals), max(vals)
                    if vmax > vmin:
                        for p in profiles:
                            p[d] = (p[d] - vmin) / (vmax - vmin)
                    else:
                        for p in profiles:
                            p[d] = 0.5
                total_sim = 0.0; pairs = 0
                for i in range(len(profiles)):
                    for j in range(i + 1, len(profiles)):
                        dot = sum(profiles[i][d] * profiles[j][d] for d in range(n_dims))
                        ni = sum(profiles[i][d] ** 2 for d in range(n_dims)) ** 0.5
                        nj = sum(profiles[j][d] ** 2 for d in range(n_dims)) ** 0.5
                        total_sim += dot / (ni * nj) if ni > 0 and nj > 0 else 1.0
                        pairs += 1
                return total_sim / pairs if pairs > 0 else 1.0

            def combo_sort_key(combo):
                sim = resource_diversity_score(combo)
                occ = next(s for c, s in combo_scores if c is combo)
                return (sim, -occ)

            best_combo = min(top_candidates, key=combo_sort_key)
            return best_combo



# # -------------------------------
# # 贪心组合构造
# # ---
# class Scheduler:
#     def __init__(self, resource_model):
#         self.resource_model = resource_model

#     def schedule(self, ready_ops: List["OperatorTask"], current_time: float) -> List["OperatorTask"]:
#         """
#         使用贪心组合构造法，在 ready_ops 中依次选择能调度的算子，直到资源不足。
#         每次选择估计带来最大利用率提升的算子。
#         """
#         self.resource_model.update_time(current_time)

#         best_combination = []
#         virtual_model = copy.deepcopy(self.resource_model)

#         sorted_ops = sorted(ready_ops, key=lambda op: sum(k.blocks for k in op.kernels))  # 可换其他启发策略

       
#         for op in sorted_ops:
#             if not op.kernels:
#                 best_combination.append(op) 
#             else:
#                 before_util = virtual_model.total_utilization()
#                 virtual_model.apply_launch(op, current_time)
#                 after_util = virtual_model.total_utilization()

#                 if after_util > before_util:
#                     best_combination.append(op)
            

#         # 在真实模型中执行
#         for op in best_combination:
#             self.resource_model.apply_launch(op, current_time)

#         return best_combination



