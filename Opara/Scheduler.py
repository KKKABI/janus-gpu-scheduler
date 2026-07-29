from typing import List
import copy
import json
import math
import os
from itertools import combinations



# -------------------------------
# KernelProfile
# -------------------------------

class KernelProfile:
    def __init__(self, name: str, duration: float, shared_mem: int, registers: int, warps: int, blocks: int,
                 mem_thru=0.0, dram_thru=0.0, l2_thru=0.0, comp_thru=0.0):
        self.name = name
        self.duration = duration
        self.shared_mem = shared_mem
        self.registers = registers
        self.warps = warps
        self.blocks = blocks
        self.blocks_remaining = blocks
        self.mem_thru = mem_thru
        self.dram_thru = dram_thru
        self.l2_thru = l2_thru
        self.comp_thru = comp_thru

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


class _TimelineSM:
    """Batch-oriented SM state used by candidate-level TD simulation."""

    def __init__(self, source: VirtualSM):
        self.shared_mem_total = source.shared_mem_total
        self.register_total = source.register_total
        self.warp_total = source.warp_total
        self.shared_mem_used = source.shared_mem_used
        self.registers_used = source.registers_used
        self.warps_used = source.warps_used
        self.running_batches = []
        for end_time, kernel_name, block_resource in source.running_blocks:
            self.running_batches.append({
                "end_time": float(end_time),
                "owner": None,
                "kernel_index": None,
                "kernel_name": kernel_name,
                "resource": block_resource,
                "count": 1,
            })

    def can_accept(self, block_resource) -> bool:
        return (
            self.shared_mem_used + block_resource["shared_mem"] <= self.shared_mem_total and
            self.registers_used + block_resource["registers"] <= self.register_total and
            self.warps_used + block_resource["warps"] <= self.warp_total
        )

    def max_blocks_fit(self, block_resource):
        limits = []
        for used, total, key in (
                (self.shared_mem_used, self.shared_mem_total, "shared_mem"),
                (self.registers_used, self.register_total, "registers"),
                (self.warps_used, self.warp_total, "warps")):
            amount = block_resource[key]
            if amount > 0:
                limits.append((total - used) // amount)
        return min(limits) if limits else float("inf")

    def allocate_batch(self, owner, kernel_index, kernel, start_time, count):
        if count <= 0:
            return
        block_resource = ResourceModel._block_resource(kernel)
        max_fit = self.max_blocks_fit(block_resource)
        assert max_fit == float("inf") or count <= max_fit
        self.shared_mem_used += block_resource["shared_mem"] * count
        self.registers_used += block_resource["registers"] * count
        self.warps_used += block_resource["warps"] * count
        self.running_batches.append({
            "end_time": float(start_time) + max(float(kernel.duration), 0.0),
            "owner": owner,
            "kernel_index": kernel_index,
            "kernel_name": kernel.name,
            "resource": block_resource,
            "count": int(count),
        })

    def undo_last_batch(self):
        batch = self.running_batches.pop()
        block_resource = batch["resource"]
        count = batch["count"]
        self.shared_mem_used -= block_resource["shared_mem"] * count
        self.registers_used -= block_resource["registers"] * count
        self.warps_used -= block_resource["warps"] * count

    def release_at(self, current_time):
        released = []
        still_running = []
        epsilon = 1e-12
        for batch in self.running_batches:
            if batch["end_time"] <= current_time + epsilon:
                block_resource = batch["resource"]
                count = batch["count"]
                self.shared_mem_used -= block_resource["shared_mem"] * count
                self.registers_used -= block_resource["registers"] * count
                self.warps_used -= block_resource["warps"] * count
                released.append(batch)
            else:
                still_running.append(batch)
        self.running_batches = still_running
        return released

    def get_utilization(self):
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

    @staticmethod
    def _block_resource(kernel: KernelProfile) -> dict:
        return {
            'shared_mem': kernel.shared_mem,
            'registers': kernel.registers,
            'warps': kernel.warps
        }

    @staticmethod
    def _allocate_one_block_on_best_sm(sms, kernel: KernelProfile, start_time: float) -> bool:
        """Allocate one block now without advancing simulated time."""
        block_resource = ResourceModel._block_resource(kernel)
        sm_capacities = []
        for sm_index, sm in enumerate(sms):
            max_blocks = sm.max_blocks_fit(block_resource)
            if max_blocks > 0:
                sm_capacities.append((max_blocks, -sm_index, sm))
        if not sm_capacities:
            return False
        _, _, selected_sm = max(
            sm_capacities, key=lambda item: (item[0], item[1])
        )
        selected_sm.allocate_block(
            kernel.name, block_resource, start_time, kernel.duration
        )
        return True

    @staticmethod
    def _kernel_admission_pressure(kernel: KernelProfile, sample_sm: VirtualSM):
        """Order hard-to-place blocks first during exact initial admission."""
        ratios = (
            float(kernel.shared_mem) / max(float(sample_sm.shared_mem_total), 1.0),
            float(kernel.registers) / max(float(sample_sm.register_total), 1.0),
            float(kernel.warps) / max(float(sample_sm.warp_total), 1.0),
        )
        return max(ratios), sum(ratios)

    @staticmethod
    def _undo_last_block(sm: VirtualSM):
        _, _, block_resource = sm.running_blocks.pop()
        sm.shared_mem_used -= block_resource["shared_mem"]
        sm.registers_used -= block_resource["registers"]
        sm.warps_used -= block_resource["warps"]

    def _admit_one_block_per_kernel(self, sms, kernels, start_time: float) -> bool:
        """Find an exact concurrent placement for one block from every kernel.

        Candidate groups contain at most five operators, so a small backtracking
        search is cheap and avoids feasibility depending on operator order.
        Equivalent SM states are explored only once at each search depth.
        """
        if not kernels:
            return True
        sample_sm = sms[0]
        ordered = sorted(
            kernels,
            key=lambda kernel: self._kernel_admission_pressure(kernel, sample_sm),
            reverse=True,
        )

        def place(kernel_index: int) -> bool:
            if kernel_index == len(ordered):
                return True
            kernel = ordered[kernel_index]
            block_resource = self._block_resource(kernel)
            seen_sm_states = set()
            for sm in sms:
                state = (
                    sm.shared_mem_used,
                    sm.registers_used,
                    sm.warps_used,
                )
                if state in seen_sm_states:
                    continue
                seen_sm_states.add(state)
                if not sm.can_accept(block_resource):
                    continue
                sm.allocate_block(
                    kernel.name, block_resource, start_time, kernel.duration
                )
                if place(kernel_index + 1):
                    return True
                self._undo_last_block(sm)
            return False

        return place(0)

    def try_apply_concurrent_combo(self, operators, start_time: float) -> bool:
        """Atomically apply the TD initial-co-residency rule to a candidate.

        Every resource-using operator must have at least one block resident at
        ``start_time``. Once admission succeeds, the remaining current capacity
        is filled round-robin for occupancy scoring. Blocks that do not fit now
        may execute in later waves; this method never advances time.
        """
        virtual_sms = copy.deepcopy(self.sms)
        entries = []
        for operator in operators:
            if not operator.kernels:
                continue
            kernel = copy.deepcopy(operator.kernels[0])
            pending_blocks = max(0, int(kernel.blocks_remaining))
            if pending_blocks == 0:
                continue
            entries.append({
                "kernel": kernel,
                "remaining": pending_blocks - 1,
            })

        if not self._admit_one_block_per_kernel(
                virtual_sms,
                [entry["kernel"] for entry in entries],
                start_time):
            return False

        made_progress = True
        while made_progress:
            made_progress = False
            for entry in entries:
                if entry["remaining"] <= 0:
                    continue
                if self._allocate_one_block_on_best_sm(
                        virtual_sms, entry["kernel"], start_time):
                    entry["remaining"] -= 1
                    made_progress = True

        self.sms = virtual_sms
        return True

    @staticmethod
    def _advance_timeline_entries(entries, current_time):
        for entry in entries:
            while (
                    entry["kernel_index"] < len(entry["kernels"]) and
                    entry["remaining"][entry["kernel_index"]] == 0 and
                    entry["inflight"] == 0):
                entry["kernel_index"] += 1
            if entry["kernel_index"] >= len(entry["kernels"]):
                if entry["completion_time"] is None:
                    entry["completion_time"] = float(current_time)

    def _make_timeline_entries(
            self, operators, start_time, copy_kernels=True):
        entries = []
        ordered_operators = sorted(
            enumerate(operators),
            key=lambda item: (item[1].name, item[0]),
        )
        for _, operator in ordered_operators:
            kernels = (
                copy.deepcopy(operator.kernels)
                if copy_kernels else list(operator.kernels)
            )
            remaining = [
                max(0, int(kernel.blocks_remaining)) for kernel in kernels
            ]
            entries.append({
                "name": operator.name,
                "kernels": kernels,
                "remaining": remaining,
                "kernel_index": 0,
                "inflight": 0,
                "completion_time": None,
            })
        self._advance_timeline_entries(entries, start_time)
        return entries

    def _timeline_initial_admission(self, sms, entries, start_time):
        active = [
            entry_index for entry_index, entry in enumerate(entries)
            if entry["kernel_index"] < len(entry["kernels"])
        ]
        if not active:
            return True
        if not sms:
            return False
        sample_sm = sms[0]
        ordered = sorted(
            active,
            key=lambda entry_index: (
                self._kernel_admission_pressure(
                    entries[entry_index]["kernels"][
                        entries[entry_index]["kernel_index"]
                    ],
                    sample_sm,
                ),
                entries[entry_index]["name"],
            ),
            reverse=True,
        )

        def place(order_index):
            if order_index == len(ordered):
                return True
            entry_index = ordered[order_index]
            entry = entries[entry_index]
            kernel_index = entry["kernel_index"]
            kernel = entry["kernels"][kernel_index]
            block_resource = self._block_resource(kernel)
            seen_sm_states = set()
            for sm in sms:
                state = (
                    sm.shared_mem_used,
                    sm.registers_used,
                    sm.warps_used,
                )
                if state in seen_sm_states:
                    continue
                seen_sm_states.add(state)
                if not sm.can_accept(block_resource):
                    continue
                sm.allocate_batch(
                    entry_index, kernel_index, kernel, start_time, 1
                )
                if place(order_index + 1):
                    return True
                sm.undo_last_batch()
            return False

        if not place(0):
            return False
        for entry_index in active:
            entry = entries[entry_index]
            kernel_index = entry["kernel_index"]
            entry["remaining"][kernel_index] -= 1
            entry["inflight"] += 1
        return True

    @staticmethod
    def _timeline_full_cycles(sm, entries, entry_indices):
        if not entry_indices:
            return 0
        limits = []
        resource_per_cycle = {
            "shared_mem": 0,
            "registers": 0,
            "warps": 0,
        }
        for entry_index in entry_indices:
            entry = entries[entry_index]
            kernel_index = entry["kernel_index"]
            kernel = entry["kernels"][kernel_index]
            limits.append(entry["remaining"][kernel_index])
            block_resource = ResourceModel._block_resource(kernel)
            for key in resource_per_cycle:
                resource_per_cycle[key] += block_resource[key]
        for used, total, key in (
                (sm.shared_mem_used, sm.shared_mem_total, "shared_mem"),
                (sm.registers_used, sm.register_total, "registers"),
                (sm.warps_used, sm.warp_total, "warps")):
            amount = resource_per_cycle[key]
            if amount > 0:
                limits.append((total - used) // amount)
        return max(0, int(min(limits))) if limits else 0

    def _timeline_fill_available(self, sms, entries, current_time, rotation=0):
        """Launch currently available blocks in fair, batched SM-local rounds."""
        launched = 0
        n_entries = len(entries)
        if n_entries == 0:
            return launched
        for sm_index, sm in enumerate(sms):
            while True:
                ordered_indices = [
                    (rotation + sm_index + offset) % n_entries
                    for offset in range(n_entries)
                ]
                launchable = []
                for entry_index in ordered_indices:
                    entry = entries[entry_index]
                    kernel_index = entry["kernel_index"]
                    if kernel_index >= len(entry["kernels"]):
                        continue
                    if entry["remaining"][kernel_index] <= 0:
                        continue
                    kernel = entry["kernels"][kernel_index]
                    if sm.can_accept(self._block_resource(kernel)):
                        launchable.append(entry_index)
                if not launchable:
                    break

                full_cycles = self._timeline_full_cycles(
                    sm, entries, launchable
                )
                if full_cycles > 0:
                    for entry_index in launchable:
                        entry = entries[entry_index]
                        kernel_index = entry["kernel_index"]
                        kernel = entry["kernels"][kernel_index]
                        sm.allocate_batch(
                            entry_index,
                            kernel_index,
                            kernel,
                            current_time,
                            full_cycles,
                        )
                        entry["remaining"][kernel_index] -= full_cycles
                        entry["inflight"] += full_cycles
                        launched += full_cycles
                    continue

                made_progress = False
                for entry_index in launchable:
                    entry = entries[entry_index]
                    kernel_index = entry["kernel_index"]
                    if entry["remaining"][kernel_index] <= 0:
                        continue
                    kernel = entry["kernels"][kernel_index]
                    if not sm.can_accept(self._block_resource(kernel)):
                        continue
                    sm.allocate_batch(
                        entry_index, kernel_index, kernel, current_time, 1
                    )
                    entry["remaining"][kernel_index] -= 1
                    entry["inflight"] += 1
                    launched += 1
                    made_progress = True
                if not made_progress:
                    break
        return launched

    @staticmethod
    def _timeline_utilization(sms):
        if not sms:
            return 0.0
        return sum(sm.get_utilization() for sm in sms) / len(sms)

    @staticmethod
    def _timeline_active_owners(sms):
        return {
            batch["owner"]
            for sm in sms
            for batch in sm.running_batches
            if batch["owner"] is not None
        }

    def evaluate_initial_combo(self, operators, start_time: float):
        """Fast stage-1 admission and occupancy evaluation for all candidates."""
        sms = [_TimelineSM(sm) for sm in self.sms]
        entries = self._make_timeline_entries(
            operators, start_time, copy_kernels=False
        )
        for entry in entries:
            for kernel_index, kernel in enumerate(entry["kernels"]):
                if entry["remaining"][kernel_index] <= 0:
                    continue
                block_resource = self._block_resource(kernel)
                hardware_fit = any(
                    block_resource["shared_mem"] <= sm.shared_mem_total and
                    block_resource["registers"] <= sm.register_total and
                    block_resource["warps"] <= sm.warp_total
                    for sm in sms
                )
                if not hardware_fit:
                    return {
                        "feasible": False,
                        "failure_reason": "kernel_block_exceeds_sm_capacity",
                        "initial_utilization": -1.0,
                    }
        if not self._timeline_initial_admission(sms, entries, start_time):
            return {
                "feasible": False,
                "failure_reason": "initial_co_residency",
                "initial_utilization": -1.0,
            }
        self._timeline_fill_available(sms, entries, start_time)
        return {
            "feasible": True,
            "failure_reason": None,
            "initial_utilization": self._timeline_utilization(sms),
            "initial_resident_blocks": {
                entry["name"]: sum(
                    batch["count"]
                    for sm in sms
                    for batch in sm.running_batches
                    if batch["owner"] == entry_index
                )
                for entry_index, entry in enumerate(entries)
            },
        }

    def simulate_combo_timeline(self, operators, start_time: float):
        """Simulate a candidate on one shared, non-preemptive block timeline."""
        sms = [_TimelineSM(sm) for sm in self.sms]
        entries = self._make_timeline_entries(operators, start_time)
        if not self._timeline_initial_admission(sms, entries, start_time):
            return {
                "feasible": False,
                "failure_reason": "initial_co_residency",
            }

        self._timeline_fill_available(sms, entries, start_time)
        initial_utilization = self._timeline_utilization(sms)
        initial_resident_blocks = {
            entry["name"]: sum(
                batch["count"]
                for sm in sms
                for batch in sm.running_batches
                if batch["owner"] == entry_index
            )
            for entry_index, entry in enumerate(entries)
        }
        current_time = float(start_time)
        utilization_area = 0.0
        overlap_duration = 0.0
        max_concurrent_operators = len(self._timeline_active_owners(sms))
        event_count = 0
        launch_rounds = 1 if any(initial_resident_blocks.values()) else 0
        peak_utilization = initial_utilization
        max_events = int(os.environ.get("OPARA_TD_MAX_EVENTS", "100000"))

        while any(
                entry["kernel_index"] < len(entry["kernels"])
                for entry in entries):
            running_batches = [
                batch for sm in sms for batch in sm.running_batches
            ]
            if not running_batches:
                launched = self._timeline_fill_available(
                    sms, entries, current_time, rotation=event_count
                )
                if launched == 0:
                    return {
                        "feasible": False,
                        "failure_reason": "pending_kernel_cannot_fit",
                    }
                launch_rounds += 1
                running_batches = [
                    batch for sm in sms for batch in sm.running_batches
                ]

            next_time = min(batch["end_time"] for batch in running_batches)
            if next_time < current_time - 1e-12:
                return {
                    "feasible": False,
                    "failure_reason": "non_monotonic_timeline",
                }
            delta = max(0.0, next_time - current_time)
            current_utilization = self._timeline_utilization(sms)
            active_owners = self._timeline_active_owners(sms)
            utilization_area += current_utilization * delta
            if len(active_owners) >= 2:
                overlap_duration += delta
            peak_utilization = max(peak_utilization, current_utilization)
            max_concurrent_operators = max(
                max_concurrent_operators, len(active_owners)
            )
            current_time = next_time

            for sm in sms:
                for batch in sm.release_at(current_time):
                    owner = batch["owner"]
                    if owner is not None:
                        entries[owner]["inflight"] -= batch["count"]
                        if entries[owner]["inflight"] < 0:
                            raise AssertionError("negative timeline inflight blocks")

            self._advance_timeline_entries(entries, current_time)
            launched = self._timeline_fill_available(
                sms, entries, current_time, rotation=event_count + 1
            )
            if launched > 0:
                launch_rounds += 1
            event_count += 1
            if event_count > max_events:
                return {
                    "feasible": False,
                    "failure_reason": "event_limit",
                    "event_count": event_count,
                }

        makespan = max(0.0, current_time - float(start_time))
        average_utilization = (
            utilization_area / makespan if makespan > 0 else 0.0
        )
        overlap_fraction = (
            overlap_duration / makespan if makespan > 0 else 0.0
        )
        return {
            "feasible": True,
            "failure_reason": None,
            "makespan": makespan,
            "average_utilization": average_utilization,
            "initial_utilization": initial_utilization,
            "peak_utilization": peak_utilization,
            "overlap_duration": overlap_duration,
            "overlap_fraction": overlap_fraction,
            "max_concurrent_operators": max_concurrent_operators,
            "event_count": event_count,
            "launch_rounds": launch_rounds,
            "initial_resident_blocks": initial_resident_blocks,
            "operator_completion_times": {
                entry["name"]: (
                    entry["completion_time"] - float(start_time)
                    if entry["completion_time"] is not None else None
                )
                for entry in entries
            },
        }

    def can_apply_launch(self, operator: OperatorTask, start_time: float)-> bool:
        if not operator.kernels:
            return True
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

        # TD single-operator admission: one block must fit now. Remaining
        # blocks are allowed to execute in later waves.
        if not kernel.has_pending_blocks():
            return True
        return self._allocate_one_block_on_best_sm(
            virtual_sms, kernel, start_time
        )
            
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

# 全局统计：每次 schedule() 调用的 ready 集合、枚举、可行和最终选择。
_CANDIDATE_STATS = []


def clear_candidate_stats():
    _CANDIDATE_STATS.clear()


def get_candidate_stats(clear=False):
    stats = copy.deepcopy(_CANDIDATE_STATS)
    if clear:
        _CANDIDATE_STATS.clear()
    return stats

def dump_candidate_stats():
    """打印逐次 scheduler 统计，并输出一行便于脚本解析的 JSON。"""
    if not _CANDIDATE_STATS:
        return
    print("\n" + "=" * 70)
    print("  CANDIDATE COMBO STATS (per scheduler call)")
    print("=" * 70)
    print(
        f"  {'call':>5} {'ready':>5} {'used':>5} {'enum':>7} "
        f"{'pass':>7} {'score':>7} {'mode':>19} {'sim':>6} selected"
    )
    print("  " + "-" * 110)
    for s in _CANDIDATE_STATS:
        selected = "+".join(s["selected"])
        simulator = "TD" if s["time_domain"] else "Static"
        print(
            f"  {s['call']:>5} {s['ready_count']:>5} {s['ready_used_count']:>5} "
            f"{s['enumerated_count']:>7} {s['feasible_count']:>7} "
            f"{s['scoring_candidate_count']:>7} {s['selection_mode']:>19} "
            f"{simulator:>6} {selected}"
        )

    total_enumerated = sum(s['enumerated_count'] for s in _CANDIDATE_STATS)
    total_feasible = sum(s['feasible_count'] for s in _CANDIDATE_STATS)
    n_calls = len(_CANDIDATE_STATS)
    print(
        f"  calls={n_calls}, enumerated={total_enumerated}, "
        f"feasible={total_feasible}, "
        f"pass_rate={100.0 * total_feasible / max(total_enumerated, 1):.2f}%"
    )
    print("CANDIDATE_STATS_JSON=" + json.dumps(_CANDIDATE_STATS, ensure_ascii=False))
    print("=" * 70)


class Scheduler:
    def __init__(self, resource_model, alpha=0.9, selection_mode='cosine', time_domain=True):
        self.resource_model = resource_model
        self.alpha = alpha
        self.selection_mode = selection_mode  # cosine | min_resource | max_occupancy | static_interference[_alpha] | legacy_balance
        self.overload_weight = float(os.getenv('JANUS_OVERLOAD_WEIGHT', '1.0'))
        self.tail_weight = float(os.getenv('JANUS_TAIL_WEIGHT', '0.02'))
        self.occupancy_weight = float(os.getenv('JANUS_OCCUPANCY_WEIGHT', '0.005'))
        self.time_domain = time_domain
        self._static_profile_cache = {}
        self._td_single_timeline_cache = {}
        self.timeline_shortlist_size = int(os.environ.get(
            "OPARA_TD_TIMELINE_SHORTLIST", "8"
        ))
        if self.timeline_shortlist_size < 1:
            raise ValueError("OPARA_TD_TIMELINE_SHORTLIST must be >= 1")
        self.interference_shortlist_size = int(os.environ.get(
            "OPARA_TD_INTERFERENCE_SHORTLIST", "12"
        ))
        if self.interference_shortlist_size < self.timeline_shortlist_size:
            raise ValueError(
                "OPARA_TD_INTERFERENCE_SHORTLIST must be >= "
                "OPARA_TD_TIMELINE_SHORTLIST"
            )
        self.final_selector = os.environ.get(
            "OPARA_TD_FINAL_SELECTOR", "timeline"
        )
        if self.final_selector not in {
                "strategy", "timeline", "guarded_interference"}:
            raise ValueError(
                "OPARA_TD_FINAL_SELECTOR must be one of "
                "strategy, timeline, guarded_interference"
            )
        self.timeline_speedup_guard = float(os.environ.get(
            "OPARA_TD_SPEEDUP_GUARD", "0.9"
        ))
        if not 0.0 <= self.timeline_speedup_guard <= 1.0:
            raise ValueError("OPARA_TD_SPEEDUP_GUARD must be in [0, 1]")
        self.interference_risk_trigger = float(os.environ.get(
            "OPARA_TD_RISK_TRIGGER", "0.1"
        ))
        if not 0.0 <= self.interference_risk_trigger <= 1.0:
            raise ValueError("OPARA_TD_RISK_TRIGGER must be in [0, 1]")
        self._interference_profile_cache = {}
        self._schedule_call = 0

    @staticmethod
    def _operator_timeline_signature(operator):
        return (
            operator.name,
            tuple(
                (
                    kernel.name,
                    float(kernel.duration),
                    int(kernel.shared_mem),
                    int(kernel.registers),
                    int(kernel.warps),
                    int(kernel.blocks_remaining),
                )
                for kernel in operator.kernels
            ),
        )

    def _resource_timeline_signature(self, current_time):
        return tuple(
            (
                sm.shared_mem_used,
                sm.registers_used,
                sm.warps_used,
                tuple(sorted(
                    (
                        round(float(end_time) - float(current_time), 12),
                        kernel_name,
                        block_resource["shared_mem"],
                        block_resource["registers"],
                        block_resource["warps"],
                    )
                    for end_time, kernel_name, block_resource in sm.running_blocks
                )),
            )
            for sm in self.resource_model.sms
        )

    def _single_timeline_makespan(self, operator, current_time):
        cache_key = (
            self._resource_timeline_signature(current_time),
            self._operator_timeline_signature(operator),
        )
        cached = self._td_single_timeline_cache.get(cache_key)
        if cached is not None:
            return cached
        metrics = self.resource_model.simulate_combo_timeline(
            [operator], current_time
        )
        makespan = (
            float(metrics["makespan"])
            if metrics.get("feasible") else float("inf")
        )
        self._td_single_timeline_cache[cache_key] = makespan
        return makespan

    def _timeline_candidate_score(self, combo, metrics, current_time):
        serial_makespan = sum(
            self._single_timeline_makespan(operator, current_time)
            for operator in combo
        )
        concurrent_makespan = float(metrics["makespan"])
        if not math.isfinite(serial_makespan):
            predicted_speedup = 0.0
        elif serial_makespan <= 0 or concurrent_makespan <= 0:
            predicted_speedup = 1.0
        else:
            predicted_speedup = serial_makespan / concurrent_makespan
        metrics["serial_makespan"] = serial_makespan
        metrics["predicted_speedup"] = predicted_speedup
        metrics["normalized_time_saved"] = (
            max(0.0, 1.0 - concurrent_makespan / serial_makespan)
            if serial_makespan > 0 and math.isfinite(serial_makespan)
            else 0.0
        )
        return predicted_speedup

    def _operator_interference_profile(self, operator):
        """Build a magnitude-aware HP resource-pressure profile."""
        signature = self._operator_timeline_signature(operator)
        cached = self._interference_profile_cache.get(signature)
        if cached is not None:
            return cached

        kernels = [kernel for kernel in operator.kernels if kernel.blocks > 0]
        if not kernels:
            profile = {
                "duration": 0.0,
                "vector": [0.0] * 6,
                "ncu_coverage": 0.0,
            }
            self._interference_profile_cache[signature] = profile
            return profile

        n_sms = max(1, len(self.resource_model.sms))
        sample_sm = self.resource_model.sms[0]
        reg_cap = float(max(1, sample_sm.register_total))
        smem_cap = float(max(1, sample_sm.shared_mem_total))
        warp_cap = float(max(1, sample_sm.warp_total))
        duration = sum(max(float(kernel.duration), 1e-9) for kernel in kernels)
        vector = [0.0] * 6
        ncu_weight = 0.0

        for kernel in kernels:
            kernel_duration = max(float(kernel.duration), 1e-9)
            weight = kernel_duration / duration
            fit_reg = (
                int(reg_cap // kernel.registers)
                if kernel.registers > 0 else 32
            )
            fit_smem = (
                int(smem_cap // kernel.shared_mem)
                if kernel.shared_mem > 0 else 32
            )
            fit_warp = (
                int(warp_cap // kernel.warps)
                if kernel.warps > 0 else 32
            )
            blocks_per_sm = max(1, min(
                32, fit_reg, fit_smem, fit_warp
            ))
            coverage = min(
                1.0, float(kernel.blocks) / (n_sms * blocks_per_sm)
            )
            vector[0] += weight * min(
                1.0, blocks_per_sm * kernel.registers / reg_cap
            ) * coverage
            vector[1] += weight * min(
                1.0, blocks_per_sm * kernel.shared_mem / smem_cap
            ) * coverage
            vector[2] += weight * min(
                1.0, blocks_per_sm * kernel.warps / warp_cap
            ) * coverage

            ncu_values = [
                max(0.0, float(getattr(kernel, "dram_thru", 0.0))) / 100.0,
                max(0.0, float(getattr(kernel, "l2_thru", 0.0))) / 100.0,
                max(0.0, float(getattr(kernel, "comp_thru", 0.0))) / 100.0,
            ]
            if any(value > 0.0 for value in ncu_values):
                ncu_weight += weight
            for index, value in enumerate(ncu_values, start=3):
                vector[index] += weight * min(1.0, value)

        profile = {
            "duration": duration,
            "vector": vector,
            "ncu_coverage": min(1.0, ncu_weight),
        }
        self._interference_profile_cache[signature] = profile
        return profile

    def _combo_interference_metrics(self, combo):
        """Estimate HP-HP conflict without treating low pressure as high risk."""
        profiles = [
            self._operator_interference_profile(operator)
            for operator in combo
        ]
        if len(profiles) <= 1:
            return {
                "risk": 0.0,
                "pair_conflict": 0.0,
                "capacity_overload": 0.0,
                "duration_tail": 0.0,
                "ncu_coverage": (
                    profiles[0]["ncu_coverage"] if profiles else 0.0
                ),
                "pair_count": 0,
            }

        pair_conflicts = []
        for left_index in range(len(profiles)):
            left = profiles[left_index]
            for right_index in range(left_index + 1, len(profiles)):
                right = profiles[right_index]
                products = [
                    left_value * right_value
                    for left_value, right_value in zip(
                        left["vector"], right["vector"]
                    )
                ]
                shared_pressure = (
                    0.5 * max(products)
                    + 0.5 * sum(products) / len(products)
                )
                longer = max(left["duration"], right["duration"], 1e-9)
                temporal_overlap = min(
                    left["duration"], right["duration"]
                ) / longer
                pair_conflicts.append(shared_pressure * temporal_overlap)

        pair_conflict = sum(pair_conflicts) / len(pair_conflicts)
        static_sums = [
            sum(profile["vector"][dimension] for profile in profiles)
            for dimension in range(3)
        ]
        raw_overload = max(0.0, max(static_sums) - 1.0)
        capacity_overload = raw_overload / (1.0 + raw_overload)
        risk = min(1.0, 0.7 * pair_conflict + 0.3 * capacity_overload)

        durations = [profile["duration"] for profile in profiles]
        mean_duration = sum(durations) / len(durations)
        variance = sum(
            (duration - mean_duration) ** 2 for duration in durations
        ) / len(durations)
        duration_tail = (
            variance ** 0.5 / max(mean_duration, 1e-9)
        )
        return {
            "risk": risk,
            "pair_conflict": pair_conflict,
            "capacity_overload": capacity_overload,
            "duration_tail": duration_tail,
            "ncu_coverage": sum(
                profile["ncu_coverage"] for profile in profiles
            ) / len(profiles),
            "pair_count": len(pair_conflicts),
        }

    def _record_candidate_stats(
            self, ready_names, ready_used_names, raw_theoretical_count,
            theoretical_count, combo_scores, top_candidates, selected_combo,
            occ_max, current_time, combo_metrics=None, passthrough_ops=None):
        self._schedule_call += 1
        combo_metrics = combo_metrics or {}
        passthrough_ops = passthrough_ops or []
        feasible_count = sum(1 for _, score in combo_scores if score >= 0)
        if self.selection_mode == 'static_interference':
            scoring_candidate_count = feasible_count
        else:
            scoring_candidate_count = len(top_candidates)
        selected_timeline = combo_metrics.get(id(selected_combo))
        feasible_timeline_metrics = [
            combo_metrics[id(combo)]
            for combo, score in combo_scores
            if score >= 0 and id(combo) in combo_metrics
        ]
        timeline_speedups = [
            float(metrics["predicted_speedup"])
            for metrics in feasible_timeline_metrics
            if "predicted_speedup" in metrics
        ]
        selected_resource_names = [op.name for op in selected_combo]
        passthrough_names = [op.name for op in passthrough_ops]
        combined_ready_used_names = ready_used_names + passthrough_names
        record = {
            'call': self._schedule_call,
            'current_time': float(current_time),
            'time_domain': bool(self.time_domain),
            'candidate_score_kind': 'initial_occupancy',
            'final_score_kind': (
                {
                    'strategy': 'strategy_score',
                    'timeline': 'predicted_speedup',
                    'guarded_interference': 'interference_guarded_speedup',
                }[self.final_selector] if self.time_domain
                else 'strategy_score'
            ),
            'final_selector': (
                self.final_selector if self.time_domain else 'strategy'
            ),
            'timeline_speedup_guard': (
                self.timeline_speedup_guard if self.time_domain else None
            ),
            'interference_risk_trigger': (
                self.interference_risk_trigger if self.time_domain else None
            ),
            'timeline_shortlist_limit': (
                (
                    self.interference_shortlist_size
                    if self.final_selector == 'guarded_interference'
                    else self.timeline_shortlist_size
                ) if self.time_domain else 0
            ),
            'selection_mode': self.selection_mode,
            'alpha': float(self.alpha),
            'ready_count': len(ready_names),
            'ready_used_count': len(combined_ready_used_names),
            'ready_ops': ready_names,
            'ready_used_ops': combined_ready_used_names,
            'resource_ready_count': len(ready_names) - len(passthrough_names),
            'resource_ready_used_count': len(ready_used_names),
            'passthrough_count': len(passthrough_names),
            'passthrough_ops': passthrough_names,
            'raw_theoretical_count': raw_theoretical_count,
            'theoretical_count': theoretical_count,
            'enumerated_count': len(combo_scores),
            'feasible_count': feasible_count,
            'alpha_candidate_count': len(top_candidates),
            'scoring_candidate_count': scoring_candidate_count,
            'occ_max': float(occ_max),
            'candidate_score_max': float(occ_max),
            'selected_resource': selected_resource_names,
            'selected_resource_size': len(selected_combo),
            'selected_passthrough': passthrough_names,
            'selected': selected_resource_names + passthrough_names,
            'selected_size': len(selected_combo) + len(passthrough_ops),
        }
        if timeline_speedups:
            record['timeline_candidate_count'] = len(timeline_speedups)
            record['timeline_speedup_min'] = min(timeline_speedups)
            record['timeline_speedup_max'] = max(timeline_speedups)
            record['timeline_speedup_mean'] = (
                sum(timeline_speedups) / len(timeline_speedups)
            )
        interference_metrics = [
            metrics["interference"]
            for metrics in feasible_timeline_metrics
            if "interference" in metrics
        ]
        if interference_metrics:
            risks = [float(metrics["risk"]) for metrics in interference_metrics]
            record['interference_candidate_count'] = len(risks)
            record['interference_risk_min'] = min(risks)
            record['interference_risk_max'] = max(risks)
            record['interference_risk_mean'] = sum(risks) / len(risks)
        if selected_timeline is not None:
            record['selected_timeline'] = copy.deepcopy(selected_timeline)
        _CANDIDATE_STATS.append(record)

    def _select_static_interference(
            self, ready_ops, combo_scores, return_ranked=False):
        """Predict round-time gain from magnitude-aware static pressure."""
        feasible = [(combo, occ) for combo, occ in combo_scores if occ >= 0]
        if not feasible:
            return []
        n_sms = max(1, len(self.resource_model.sms))
        sample_sm = self.resource_model.sms[0]
        reg_cap = float(max(1, sample_sm.register_total))
        smem_cap = float(max(1, sample_sm.shared_mem_total))
        warp_cap = float(max(1, sample_sm.warp_total))
        profiles = {}; raw_densities = {}
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
            pressure = [0.0, 0.0, 0.0, 0.0]; total_blocks = 0.0
            for k in kernels:
                weight = max(float(k.duration), 1e-9) / duration
                fit_reg = int(reg_cap // k.registers) if k.registers > 0 else 32
                fit_smem = int(smem_cap // k.shared_mem) if k.shared_mem > 0 else 32
                fit_warp = int(warp_cap // k.warps) if k.warps > 0 else 32
                blocks_per_sm = max(1, min(32, fit_reg, fit_smem, fit_warp))
                coverage = min(1.0, float(k.blocks) / (n_sms * blocks_per_sm))
                resident = [min(1.0, blocks_per_sm * k.registers / reg_cap) * coverage,
                            min(1.0, blocks_per_sm * k.shared_mem / smem_cap) * coverage,
                            min(1.0, blocks_per_sm * k.warps / warp_cap) * coverage, coverage]
                for dim in range(4): pressure[dim] += weight * resident[dim]
                total_blocks += float(k.blocks)
            profiles[op.name] = (duration, pressure)
            raw_densities[op.name] = duration / max(total_blocks, 1.0)
            self._static_profile_cache[op.name] = (profiles[op.name], raw_densities[op.name])
        density_scale = max(max(raw_densities.values(), default=0.0), 1e-9)

        def candidate_score(item):
            combo, occupancy = item
            durations = []; vectors = []
            for op in combo:
                duration, pressure = profiles.get(op.name, (1e-9, [0.0, 0.0, 0.0, 0.0]))
                density = min(1.0, raw_densities.get(op.name, 0.0) / density_scale)
                durations.append(duration); vectors.append(pressure + [density])
            sequential = max(sum(durations), 1e-9)
            ideal_round = max(durations)
            summed = [sum(v[d] * durations[i] / max(ideal_round, 1e-9) for i, v in enumerate(vectors)) for d in range(5)]
            overload = max(0.0, max(summed) - 1.0)
            predicted_round = ideal_round * (1.0 + self.overload_weight * overload)
            gain = (sequential - predicted_round) / sequential
            mean_duration = sequential / len(durations)
            variance = sum((d - mean_duration) ** 2 for d in durations) / len(durations)
            tail = (variance ** 0.5) / max(mean_duration, 1e-9)
            score = gain - self.tail_weight * tail + self.occupancy_weight * max(0.0, occupancy)
            return (score, -overload, -predicted_round, -len(combo), occupancy)
        ranked = [
            combo for combo, _ in sorted(
                feasible, key=candidate_score, reverse=True
            )
        ]
        return ranked if return_ranked else ranked[0]

    def schedule(self, ready_ops: List["OperatorTask"], current_time: float) -> List["OperatorTask"]:

        self.resource_model.update_time(current_time)
        ready_names = [op.name for op in ready_ops]
        passthrough_ops = []
        if self.time_domain:
            passthrough_ops = [
                operator for operator in ready_ops
                if not any(
                    kernel.blocks_remaining > 0
                    for kernel in operator.kernels
                )
            ]
            ready_ops = [
                operator for operator in ready_ops
                if any(
                    kernel.blocks_remaining > 0
                    for kernel in operator.kernels
                )
            ]
        raw_max_comb_size = min(5, len(ready_ops))
        raw_theoretical_count = sum(
            math.comb(len(ready_ops), r)
            for r in range(1, raw_max_comb_size + 1)
        )
        # 枚举所有候选组合并计算每个组合的 SM 占用率（occupancy）
        combo_scores = []  # list of (combo_list, simulator_score)
        combo_metrics = {}

        max_comb_size = min(5, len(ready_ops))
        # 防止组合爆炸：默认最多使用15个ready算子；实验可通过环境变量调整。
        max_ready = int(os.environ.get("OPARA_MAX_READY", "15"))
        if max_ready < 5:
            raise ValueError(f"OPARA_MAX_READY must be >= 5, got {max_ready}")
        if max_comb_size == 5 and len(ready_ops) > max_ready:
            ready_ops = sorted(ready_ops, key=lambda op: sum(
                k.duration for k in op.kernels
            ), reverse=True)[:max_ready]
        ready_used_names = [op.name for op in ready_ops]
        max_comb_size = min(5, len(ready_ops))
        theoretical_count = sum(
            math.comb(len(ready_ops), r)
            for r in range(1, max_comb_size + 1)
        )
        for r in range(1, max_comb_size + 1):
            for combo in combinations(ready_ops, r):
                virtual_model = (
                    self.resource_model if self.resource_model.time_domain
                    else copy.deepcopy(self.resource_model)
                )
                if virtual_model.time_domain:
                    initial_metrics = virtual_model.evaluate_initial_combo(
                        combo, current_time
                    )
                    feasible = bool(initial_metrics.get("feasible"))
                else:
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
                combo_list = list(combo)
                if feasible and virtual_model.time_domain:
                    score = float(initial_metrics["initial_utilization"])
                else:
                    score = virtual_model.total_utilization() if feasible else -1.0
                combo_scores.append((combo_list, score))

        if not combo_scores:
            self._record_candidate_stats(
                ready_names, ready_used_names, raw_theoretical_count,
                theoretical_count, combo_scores, [], [], -1.0,
                current_time, combo_metrics, passthrough_ops)
            return list(passthrough_ops)

        # 找到最大占用率
        occ_max = max(score for _, score in combo_scores)

        # alpha 控制保留阈值
        alpha = self.alpha
        top_candidates = [combo for combo, score in combo_scores if score >= alpha * occ_max]

        def finish(selected_combo):
            self._record_candidate_stats(
                ready_names, ready_used_names, raw_theoretical_count,
                theoretical_count, combo_scores, top_candidates,
                selected_combo, occ_max, current_time, combo_metrics,
                passthrough_ops)
            return selected_combo + passthrough_ops

        def finish_ranked(ranked_candidates):
            ranked_candidates = list(ranked_candidates)
            if not ranked_candidates:
                return finish([])
            if not self.time_domain:
                return finish(ranked_candidates[0])

            stage1_ranks = {
                id(combo): rank
                for rank, combo in enumerate(ranked_candidates, start=1)
            }
            if self.final_selector == "strategy":
                finalists = [ranked_candidates[0]]
            else:
                shortlist_limit = (
                    self.interference_shortlist_size
                    if self.final_selector == "guarded_interference"
                    else self.timeline_shortlist_size
                )
                # Preserve one finalist from every available group size before
                # filling the remaining slots by the stage-1 strategy rank.
                finalists = []
                seen_sizes = set()
                for combo in ranked_candidates:
                    if len(combo) in seen_sizes:
                        continue
                    finalists.append(combo)
                    seen_sizes.add(len(combo))
                    if len(finalists) >= shortlist_limit:
                        break
                if (
                    self.final_selector == "guarded_interference"
                    and len(finalists) < shortlist_limit
                ):
                    finalist_ids = {id(combo) for combo in finalists}
                    sizes = sorted({len(combo) for combo in ranked_candidates})
                    for size in sizes:
                        same_size = [
                            combo for combo in ranked_candidates
                            if len(combo) == size
                        ]
                        risk_best = min(
                            same_size,
                            key=lambda combo: (
                                self._combo_interference_metrics(combo)["risk"],
                                stage1_ranks.get(id(combo), float("inf")),
                            ),
                        )
                        if id(risk_best) in finalist_ids:
                            continue
                        finalists.append(risk_best)
                        finalist_ids.add(id(risk_best))
                        if len(finalists) >= shortlist_limit:
                            break
                if len(finalists) < shortlist_limit:
                    finalist_ids = {id(combo) for combo in finalists}
                    for combo in ranked_candidates:
                        if id(combo) in finalist_ids:
                            continue
                        finalists.append(combo)
                        finalist_ids.add(id(combo))
                        if len(finalists) >= shortlist_limit:
                            break
            timeline_ranked = []
            for combo in finalists:
                stage1_rank = stage1_ranks[id(combo)]
                metrics = self.resource_model.simulate_combo_timeline(
                    combo, current_time
                )
                metrics["stage1_rank"] = stage1_rank
                metrics["stage1_shortlist_size"] = len(finalists)
                metrics["interference"] = self._combo_interference_metrics(
                    combo
                )
                combo_metrics[id(combo)] = metrics
                if not metrics.get("feasible"):
                    continue
                timeline_score = self._timeline_candidate_score(
                    combo, metrics, current_time
                )
                timeline_ranked.append((
                    timeline_score,
                    float(metrics["average_utilization"]),
                    -stage1_rank,
                    combo,
                ))

            if not timeline_ranked:
                raise RuntimeError(
                    "no stage-1 finalist completed the shared TD timeline"
                )
            best_item = max(timeline_ranked, key=lambda item: item[:3])
            if self.final_selector == "strategy":
                selected_combo = timeline_ranked[0][3]
            elif self.final_selector == "timeline":
                selected_combo = best_item[3]
            else:
                best_speedup = best_item[0]
                best_risk = combo_metrics[
                    id(best_item[3])
                ]["interference"]["risk"]
                speedup_floor = 1.0 + self.timeline_speedup_guard * max(
                    0.0, best_speedup - 1.0
                )
                guard_activated = best_risk >= self.interference_risk_trigger
                guarded = (
                    [
                        item for item in timeline_ranked
                        if item[0] + 1e-12 >= speedup_floor
                    ]
                    if guard_activated else [best_item]
                )
                if not guarded:
                    guarded = [best_item]
                for _, _, _, combo in timeline_ranked:
                    combo_metrics[id(combo)][
                        "selector_speedup_floor"
                    ] = speedup_floor
                    combo_metrics[id(combo)][
                        "selector_guard_activated"
                    ] = guard_activated
                selected_combo = min(
                    guarded,
                    key=lambda item: (
                        combo_metrics[id(item[3])]["interference"]["risk"],
                        -item[0],
                        -item[1],
                        combo_metrics[id(item[3])]["stage1_rank"],
                    ),
                )[3]
            selected_metrics = combo_metrics[id(selected_combo)]
            selected_metrics["final_selector"] = self.final_selector
            selected_metrics["selector_best_speedup"] = best_item[0]
            selected_metrics["selector_best_speedup_risk"] = (
                combo_metrics[id(best_item[3])]["interference"]["risk"]
            )
            selected_metrics["selector_selected_speedup_loss"] = max(
                0.0, best_item[0] - selected_metrics["predicted_speedup"]
            )
            selected_metrics["selector_eligible_count"] = (
                len(guarded)
                if self.final_selector == "guarded_interference"
                else len(timeline_ranked)
            )
            return finish(selected_combo)

        # 如果没有满足阈值的候选，则退回到单纯的最大占用组合
        if not top_candidates:
            # 取占用率最高的组合
            best_combo = max(combo_scores, key=lambda x: x[1])[0]
            return finish_ranked([best_combo])

        # ===== 检查是否有 ncu memory 数据 =====
        _has_ncu = any(
            hasattr(op, 'kernels') and op.kernels and
            hasattr(op.kernels[0], 'mem_thru') and getattr(op.kernels[0], 'mem_thru', 0) > 0
            for op in ready_ops
        )

        # ===== 选择策略 =====
        if self.selection_mode == 'legacy_balance':
            # Compatibility branch for baseline commit 73f4da5.
            memory_intensive_ops = [
                'add', 'cast', 'ceil', 'clip', 'concat', 'exp', 'floor', 'log',
                'gelu', 'neg', 'pow', 'reciprocal', 'relu', 'sigmoid', 'slice', 'relu'
                'sqrt', 'sub', 'tanh', 'transpose', 'unsqueeze', 'view', 'avg_pool',
                'reshape', 'max_pool', 'adaptive_avg_pool', 'adaptive_max_pool', 'premute',
                'flatten', 'dropout', 'batch_norm', 'layer_norm', 'instance_norm',
                'contiguous', 'ones', 'to'
            ]

            def is_mem_access_intensive(op_name):
                name = op_name.lower()
                return any(token in name for token in memory_intensive_ops)

            def legacy_imbalance_score(combo):
                memory_count = sum(is_mem_access_intensive(op.name) for op in combo)
                compute_count = len(combo) - memory_count
                occupancy = next(score for candidate, score in combo_scores if candidate == combo)
                return abs(compute_count - memory_count), -occupancy

            return finish_ranked(sorted(
                top_candidates, key=legacy_imbalance_score
            ))

        if self.selection_mode in ('static_interference', 'static_interference_alpha'):
            scoring_scores = combo_scores
            if self.selection_mode == 'static_interference_alpha':
                top_candidate_ids = {id(combo) for combo in top_candidates}
                scoring_scores = [
                    item for item in combo_scores if id(item[0]) in top_candidate_ids
                ]
            ranked = self._select_static_interference(
                ready_ops, scoring_scores, return_ranked=True
            )
            return finish_ranked(ranked)

        if self.selection_mode == 'max_occupancy':
            ranked = sorted(
                top_candidates,
                key=lambda c: next(
                    score for candidate, score in combo_scores
                    if candidate is c
                ),
                reverse=True,
            )
            return finish_ranked(ranked)

        elif self.selection_mode == 'memory_aware':
            # Memory-aware 策略：避免两个高 DRAM 算子放一起
            # 评分 = 占用率奖励 - 内存冲突惩罚
            def memory_aware_score(combo):
                if len(combo) <= 1:
                    return 0.0
                occ = next((s for c, s in combo_scores if c is combo), 0)

                # 计算组合内算子的平均 DRAM 吞吐量和最大 DRAM 吞吐量
                dram_vals = []
                for op in combo:
                    for k in op.kernels:
                        d = getattr(k, 'dram_thru', 0)
                        if d > 0:
                            dram_vals.append(d)
                if not dram_vals:
                    return -occ  # 无 ncu 数据 → 退化为纯占用率

                avg_dram = sum(dram_vals) / len(dram_vals)
                max_dram = max(dram_vals)

                # 冲突惩罚：两个高 DRAM 算子在一起 → 惩罚
                penalty = 0.0
                high_dram_count = sum(1 for d in dram_vals if d > 30)
                if high_dram_count >= 2:
                    penalty = max_dram * 0.01  # 越高的 DRAM 冲突越严重

                return occ - penalty

            ranked = sorted(
                top_candidates, key=memory_aware_score, reverse=True
            )
            return finish_ranked(ranked)

        elif self.selection_mode == 'min_resource':
            # 资源加和策略：选总资源压力最小的组合
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

            return finish_ranked(sorted(
                top_candidates, key=combo_sort_key_minres
            ))

        else:  # 'cosine' — 余弦相似度（自动扩展维度）
            def resource_diversity_score(combo):
                if len(combo) <= 1:
                    return 0.0
                profiles = []
                for op in combo:
                    reg = 0.0; smem = 0.0; warps = 0.0; dur = 0.0; blocks = 0
                    mem = 0.0; dram = 0.0; l2 = 0.0; comp = 0.0
                    for k in op.kernels:
                        reg += k.registers * k.blocks
                        smem += k.shared_mem * k.blocks
                        warps += k.warps * k.blocks
                        dur += k.duration * k.blocks
                        blocks += k.blocks
                        if _has_ncu:
                            mem += getattr(k, 'mem_thru', 0.0) * k.blocks
                            dram += getattr(k, 'dram_thru', 0.0) * k.blocks
                            l2 += getattr(k, 'l2_thru', 0.0) * k.blocks
                            comp += getattr(k, 'comp_thru', 0.0) * k.blocks
                    if blocks > 0:
                        vec = [reg/blocks, smem/blocks, warps/blocks, dur/blocks]
                        if _has_ncu:
                            vec += [dram/blocks, l2/blocks, comp/blocks]
                        profiles.append(vec)
                    else:
                        n = 7 if _has_ncu else 4
                        profiles.append([0.0] * n)
                n_dims = len(profiles[0])
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

            return finish_ranked(sorted(
                top_candidates, key=combo_sort_key
            ))



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
