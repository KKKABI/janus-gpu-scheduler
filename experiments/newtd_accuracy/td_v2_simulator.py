#!/usr/bin/env python3
"""A non-reserving, launch-ordered time-domain feasibility simulator.

Unlike the current TD admission rule, this simulator does not reserve one
block for every future operator.  Kernels become eligible in launch order and
consume the capacity actually left by earlier kernels.  Pending blocks execute
in later waves.  A group is positive only if all operators have an active block
for a non-zero common interval.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Any, Sequence


def number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Resource:
    shared: float
    registers: float
    warps: float


@dataclass
class SMState:
    shared_total: float
    register_total: float
    warp_total: float
    block_total: int = 32
    shared_used: float = 0.0
    register_used: float = 0.0
    warp_used: float = 0.0
    block_used: int = 0

    def fit_count(self, resource: Resource) -> int:
        limits = [max(0, self.block_total - self.block_used)]
        for total, used, amount in (
            (self.shared_total, self.shared_used, resource.shared),
            (self.register_total, self.register_used, resource.registers),
            (self.warp_total, self.warp_used, resource.warps),
        ):
            if amount > 0:
                limits.append(max(0, int((total - used) // amount)))
        return min(limits)

    def empty_fit_count(self, resource: Resource) -> int:
        """Architectural block capacity on this SM without transient use."""
        limits = [self.block_total]
        for total, amount in (
            (self.shared_total, resource.shared),
            (self.register_total, resource.registers),
            (self.warp_total, resource.warps),
        ):
            if amount > 0:
                limits.append(max(0, int(total // amount)))
        return min(limits)

    def allocate(self, resource: Resource, count: int) -> None:
        if count <= 0 or self.fit_count(resource) < count:
            raise RuntimeError("invalid block allocation")
        self.shared_used += resource.shared * count
        self.register_used += resource.registers * count
        self.warp_used += resource.warps * count
        self.block_used += count

    def release(self, resource: Resource, count: int) -> None:
        self.shared_used = max(0.0, self.shared_used - resource.shared * count)
        self.register_used = max(
            0.0, self.register_used - resource.registers * count
        )
        self.warp_used = max(0.0, self.warp_used - resource.warps * count)
        self.block_used -= count
        if self.block_used < 0:
            raise RuntimeError("negative resident block count")


@dataclass
class OperatorState:
    operator: Any
    kernel_index: int = 0
    remaining_blocks: int = 0
    active_blocks: int = 0
    launched: bool = False
    launch_time: float = 0.0
    completed: bool = False

    @property
    def name(self) -> str:
        return str(self.operator.name)

    @property
    def kernels(self) -> list[Any]:
        return list(getattr(self.operator, "kernels", []) or [])

    @property
    def kernel(self) -> Any | None:
        return (
            self.kernels[self.kernel_index]
            if self.kernel_index < len(self.kernels)
            else None
        )


def kernel_resource(kernel: Any) -> Resource:
    return Resource(
        shared=max(0.0, number(getattr(kernel, "shared_mem", 0.0))),
        registers=max(0.0, number(getattr(kernel, "registers", 0.0))),
        warps=max(0.0, number(getattr(kernel, "warps", 0.0))),
    )


def sm_states(resource_model: Any, block_limit: int) -> list[SMState]:
    output = []
    for sm in list(getattr(resource_model, "sms", []) or []):
        output.append(
            SMState(
                shared_total=max(
                    0.0, number(getattr(sm, "shared_mem_total", 0.0))
                ),
                register_total=max(
                    0.0, number(getattr(sm, "register_total", 0.0))
                ),
                warp_total=max(0.0, number(getattr(sm, "warp_total", 0.0))),
                block_total=block_limit,
                shared_used=max(
                    0.0, number(getattr(sm, "shared_mem_used", 0.0))
                ),
                register_used=max(
                    0.0, number(getattr(sm, "registers_used", 0.0))
                ),
                warp_used=max(
                    0.0, number(getattr(sm, "warps_used", 0.0))
                ),
                block_used=len(list(getattr(sm, "running_blocks", []) or [])),
            )
        )
    if not output:
        raise ValueError("resource model contains no SMs")
    return output


def wave_duration(kernel: Any, resource: Resource, sms: Sequence[SMState]) -> float:
    """Convert profiled whole-grid kernel duration to one simulated wave.

    PyTorch/NSYS profiles report the elapsed time of the complete kernel grid,
    not the service time of every individual thread block.  Treating that
    value as a per-block duration multiplies long grids by their wave count and
    creates artificial overlap with later kernels.
    """
    blocks = max(1, integer(getattr(kernel, "blocks", 0), 1))
    solo_capacity = sum(sm.empty_fit_count(resource) for sm in sms)
    if solo_capacity <= 0:
        return max(number(getattr(kernel, "duration", 0.0)), 1e-12)
    waves = max(1, math.ceil(blocks / solo_capacity))
    total_duration = max(number(getattr(kernel, "duration", 0.0)), 1e-12)
    return total_duration / waves


def simulate_strict_overlap(
    operators: Sequence[Any],
    resource_model: Any,
    *,
    launch_gap: float,
    kernel_gap: float = 0.0,
    block_limit_per_sm: int = 32,
    max_events: int = 200_000,
) -> dict[str, Any]:
    if len(operators) < 2:
        return {
            "feasible": False,
            "strict_parallel": False,
            "failure_reason": "group_size_lt_2",
        }
    if launch_gap < 0 or kernel_gap < 0:
        raise ValueError("launch gaps must be non-negative")
    sms = sm_states(resource_model, block_limit_per_sm)
    states = [
        OperatorState(operator=operator, launch_time=index * launch_gap)
        for index, operator in enumerate(operators)
    ]
    for state in states:
        if not state.kernels:
            return {
                "feasible": False,
                "strict_parallel": False,
                "failure_reason": "missing_kernel_profile",
                "failure_operator": state.name,
            }
        state.remaining_blocks = integer(getattr(state.kernel, "blocks", 0))
        if state.remaining_blocks <= 0:
            return {
                "feasible": False,
                "strict_parallel": False,
                "failure_reason": "nonpositive_grid",
                "failure_operator": state.name,
            }
        resource = kernel_resource(state.kernel)
        if max(sm.fit_count(resource) for sm in sms) <= 0:
            return {
                "feasible": False,
                "strict_parallel": False,
                "failure_reason": "single_block_illegal",
                "failure_operator": state.name,
            }

    # (end_time, sequence, sm_index, op_index, kernel_index, resource, count)
    events: list[tuple] = []
    sequence = 0
    current_time = 0.0
    strict_overlap = 0.0
    any_overlap = 0.0
    max_concurrent = 0
    event_steps = 0
    first_active_time: dict[str, float | None] = {
        state.name: None for state in states
    }
    last_active_end: dict[str, float | None] = {
        state.name: None for state in states
    }
    final_initial_launch_time = (len(states) - 1) * launch_gap

    def activate_launches() -> None:
        for state in states:
            if (
                not state.completed
                and not state.launched
                and state.launch_time <= current_time + 1e-15
            ):
                state.launched = True

    def allocate_pending() -> None:
        nonlocal sequence
        # Earlier kernel arrivals consume capacity first.  No capacity is
        # reserved for an operator whose kernel has not arrived.
        order = sorted(
            range(len(states)),
            key=lambda index: (states[index].launch_time, index),
        )
        for op_index in order:
            state = states[op_index]
            if (
                state.completed
                or not state.launched
                or state.remaining_blocks <= 0
            ):
                continue
            kernel = state.kernel
            resource = kernel_resource(kernel)
            duration = wave_duration(kernel, resource, sms)
            fits = [sm.fit_count(resource) for sm in sms]
            # A batch fills an SM to its current limiting capacity.  Therefore
            # every SM needs to be visited at most once for this kernel at the
            # current event time; repeatedly rescanning all SMs is equivalent
            # but quadratic in the SM count.
            for sm_index in sorted(
                range(len(sms)), key=lambda index: (-fits[index], index)
            ):
                if state.remaining_blocks <= 0:
                    break
                capacity = fits[sm_index]
                if capacity <= 0:
                    continue
                count = min(capacity, state.remaining_blocks)
                sms[sm_index].allocate(resource, count)
                state.remaining_blocks -= count
                state.active_blocks += count
                sequence += 1
                heapq.heappush(
                    events,
                    (
                        current_time + duration,
                        sequence,
                        sm_index,
                        op_index,
                        state.kernel_index,
                        resource,
                        count,
                    ),
                )
                if first_active_time[state.name] is None:
                    first_active_time[state.name] = current_time

    def current_utilization() -> float:
        if not sms:
            return 0.0
        return sum(
            sm.warp_used / sm.warp_total if sm.warp_total > 0 else 0.0
            for sm in sms
        ) / len(sms)

    def active_block_counts() -> dict[str, int]:
        return {state.name: int(state.active_blocks) for state in states}

    while not all(state.completed for state in states):
        activate_launches()
        allocate_pending()
        active_owners = sum(state.active_blocks > 0 for state in states)
        max_concurrent = max(max_concurrent, active_owners)

        # TD-v2 predicts initial all-operator concurrency.  It deliberately
        # does not call a group positive merely because a missing operator may
        # enter after several waves of earlier work have completed.  At the
        # last initial launch, every group member must already own a resident
        # block; otherwise the group is rejected.
        if (
            current_time >= final_initial_launch_time - 1e-15
            and active_owners < len(states)
        ):
            return {
                "feasible": True,
                "strict_parallel": False,
                "failure_reason": "no_full_group_overlap_at_initial_launch_window",
                "makespan_until_decision": current_time,
                "strict_overlap_duration": 0.0,
                "any_overlap_duration": any_overlap,
                "max_concurrent_operators": max_concurrent,
                "initial_utilization": current_utilization(),
                "initial_resident_blocks": active_block_counts(),
                "first_active_time": first_active_time,
                "last_active_end": last_active_end,
                "event_steps": event_steps,
                "launch_gap": launch_gap,
                "kernel_gap": kernel_gap,
                "block_limit_per_sm": block_limit_per_sm,
                "duration_model": "profiled_whole_grid_duration_divided_by_solo_waves",
                "early_decision": "initial_launch_window_closed",
            }

        future_launches = [
            state.launch_time
            for state in states
            if not state.completed and not state.launched
        ]
        next_launch = min(future_launches) if future_launches else float("inf")
        next_finish = events[0][0] if events else float("inf")
        next_time = min(next_launch, next_finish)
        if next_time == float("inf"):
            return {
                "feasible": False,
                "strict_parallel": False,
                "failure_reason": "deadlock_without_event",
                "max_concurrent_operators": max_concurrent,
                "initial_utilization": current_utilization(),
                "initial_resident_blocks": active_block_counts(),
            }
        if next_time < current_time - 1e-15:
            return {
                "feasible": False,
                "strict_parallel": False,
                "failure_reason": "non_monotonic_time",
            }
        delta = max(0.0, next_time - current_time)
        # The classifier only asks whether a positive common interval exists.
        # Once it exists, later waves cannot change the yes/no result.
        if active_owners == len(states) and delta > 0.0:
            strict_overlap += delta
            any_overlap += delta
            return {
                "feasible": True,
                "strict_parallel": True,
                "failure_reason": None,
                "makespan_until_decision": next_time,
                "strict_overlap_duration": strict_overlap,
                "any_overlap_duration": any_overlap,
                "max_concurrent_operators": max_concurrent,
                "initial_utilization": current_utilization(),
                "initial_resident_blocks": active_block_counts(),
                "first_active_time": first_active_time,
                "last_active_end": last_active_end,
                "event_steps": event_steps,
                "launch_gap": launch_gap,
                "kernel_gap": kernel_gap,
                "block_limit_per_sm": block_limit_per_sm,
                "duration_model": "profiled_whole_grid_duration_divided_by_solo_waves",
                "early_decision": "strict_overlap_observed",
            }
        if active_owners >= 2:
            any_overlap += delta
        if active_owners == len(states):
            strict_overlap += delta
        current_time = next_time

        completed_batches = []
        while events and events[0][0] <= current_time + 1e-15:
            completed_batches.append(heapq.heappop(events))
        for _, _, sm_index, op_index, kernel_index, resource, count in completed_batches:
            sms[sm_index].release(resource, count)
            state = states[op_index]
            if kernel_index != state.kernel_index:
                raise RuntimeError("operator advanced before its blocks completed")
            state.active_blocks -= count
            last_active_end[state.name] = current_time

        for state in states:
            if (
                state.completed
                or not state.launched
                or state.remaining_blocks > 0
                or state.active_blocks > 0
            ):
                continue
            state.kernel_index += 1
            if state.kernel_index >= len(state.kernels):
                state.completed = True
                continue
            state.remaining_blocks = integer(
                getattr(state.kernel, "blocks", 0)
            )
            if state.remaining_blocks <= 0:
                return {
                    "feasible": False,
                    "strict_parallel": False,
                    "failure_reason": "nonpositive_grid",
                    "failure_operator": state.name,
                }
            state.launched = False
            state.launch_time = current_time + kernel_gap

        # If any operator has fully completed before a strict common interval
        # appeared, that operator can never overlap with all remaining ones.
        if strict_overlap <= 0.0 and any(state.completed for state in states):
            return {
                "feasible": True,
                "strict_parallel": False,
                "failure_reason": "operator_completed_before_strict_overlap",
                "makespan_until_decision": current_time,
                "strict_overlap_duration": 0.0,
                "any_overlap_duration": any_overlap,
                "max_concurrent_operators": max_concurrent,
                "first_active_time": first_active_time,
                "last_active_end": last_active_end,
                "event_steps": event_steps,
                "launch_gap": launch_gap,
                "kernel_gap": kernel_gap,
                "block_limit_per_sm": block_limit_per_sm,
                "duration_model": "profiled_whole_grid_duration_divided_by_solo_waves",
                "early_decision": "operator_completed",
            }

        event_steps += 1
        if event_steps > max_events:
            return {
                "feasible": False,
                "strict_parallel": False,
                "failure_reason": "event_limit",
                "event_steps": event_steps,
            }

    return {
        "feasible": True,
        "strict_parallel": strict_overlap > 0.0,
        "failure_reason": None if strict_overlap > 0.0 else "no_strict_overlap",
        "makespan": current_time,
        "strict_overlap_duration": strict_overlap,
        "any_overlap_duration": any_overlap,
        "strict_overlap_fraction": (
            strict_overlap / current_time if current_time > 0 else 0.0
        ),
        "duration_model": "profiled_whole_grid_duration_divided_by_solo_waves",
        "max_concurrent_operators": max_concurrent,
        "first_active_time": first_active_time,
        "last_active_end": last_active_end,
        "event_steps": event_steps,
        "launch_gap": launch_gap,
        "kernel_gap": kernel_gap,
        "block_limit_per_sm": block_limit_per_sm,
    }
