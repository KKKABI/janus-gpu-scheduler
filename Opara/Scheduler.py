from typing import List
from copy import deepcopy
import copy
import heapq
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
    def __init__(self, sm_count: int, sm_specs: dict):
        self.sms = [VirtualSM(**sm_specs) for _ in range(sm_count)]
        self.current_time = 0.0
        self.pending_kernels = []

    def update_time(self, current_time: float):
        for sm in self.sms:
            sm.release_finished_blocks(current_time)
        self.current_time = current_time

    def can_apply_launch(self, operator: OperatorTask, start_time: float)-> bool:
        virtual_sms = copy.deepcopy(self.sms)
        virtual_operator = copy.deepcopy(operator)
        kernel = virtual_operator.kernels[0]
        
        while kernel.has_pending_blocks():
            block_resource = {
                'shared_mem': kernel.shared_mem,
                'registers': kernel.registers,
                'warps': kernel.warps
            }

            # 计算每个 SM 当前可容纳的最大线程块数
            sm_capacities = []
            for sm in virtual_sms:
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

            # 更新 pending_kernels 列表
        if kernel.has_pending_blocks():
            return False
        else:
            return True
            
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
class Scheduler:
    def __init__(self, resource_model):
        self.resource_model = resource_model

    def schedule(self, ready_ops: List["OperatorTask"], current_time: float) -> List["OperatorTask"]:
       
        self.resource_model.update_time(current_time)
        # 枚举所有候选组合并计算每个组合的 SM 占用率（occupancy）
        combo_scores = []  # list of (combo_list, occupancy_score)

        max_comb_size = min(5, len(ready_ops))
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

        # alpha 控制保留阈值
        alpha = 0.9
        top_candidates = [combo for combo, score in combo_scores if score >= alpha * occ_max]

        # 如果没有满足阈值的候选，则退回到单纯的最大占用组合
        if not top_candidates:
            # 取占用率最高的组合
            best_combo = max(combo_scores, key=lambda x: x[1])[0]
            return best_combo

        # 内存密集型判断函数（根据算子名判断）
        memory_intensive_ops = ['add', 'cast', 'ceil', 'clip', 'concat', 'exp', 'floor', 'log',
                                'gelu', 'neg', 'pow', 'reciprocal', 'relu', 'sigmoid', 'slice', 'relu'
                                'sqrt', 'sub', 'tanh', 'transpose', 'unsqueeze', 'view', 'avg_pool',
                                'reshape', 'max_pool', 'adaptive_avg_pool', 'adaptive_max_pool', 'premute',
                                'flatten', 'dropout', 'batch_norm', 'layer_norm', 'instance_norm', 'contiguous', 'ones', 'to']

        def is_mem_access_intensive(op_name: str) -> bool:
            name = op_name.lower()
            for op in memory_intensive_ops:
                if op in name:
                    return True
            return False

        # 在 top_candidates 中选择计算/内存比例最接近 1:1 的组合
        def imbalance_score(combo):
            mem = 0
            comp = 0
            for op in combo:
                if is_mem_access_intensive(op.name):
                    mem += 1
                else:
                    comp += 1
            # 目标是使 comp ~= mem，使用绝对差作为得分（越小越好）
            return abs(comp - mem), -sum(s for c, s in combo_scores if c == combo)

        # 选择最小绝对差；若相同则选择占用率更高的
        best_combo = min(top_candidates, key=lambda c: (imbalance_score(c)))
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



