#!/usr/bin/env python3
"""Select bounded, exact-ready §4.8 pairs without looking at timing outcomes."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
REPO = EXPERIMENTS.parent
sys.path[:0] = [str(HERE), str(EXPERIMENTS), str(REPO)]

from common import (
    DISPLAY_NAMES,
    MODEL_CLASSES,
    MODELS,
    MODEL_SLUGS,
    sha256_file,
    write_json_atomic,
)


COMPARISONS = (
    ("newtd_drt_vs_newtd_ncu", "newtd_drt", "newtd_ncu_drt", "primary"),
    ("janus_vs_newtd_ncu", "janus", "newtd_ncu_drt", "secondary"),
)
COMPUTE_FAMILIES = {"Conv", "GEMM"}
MEMORY_FAMILIES = {
    "GEMV",
    "Pool",
    "LayerNorm",
    "BatchNorm",
    "Reduce",
    "Concat",
    "LayoutTransform",
    "CopyGather",
    "Elementwise",
}
CLASSIFIED_RESOURCE_CLASSES = (
    "pure_compute",
    "pure_memory",
    "mixed_resource",
)
# Janus §4.8 uses 3/7/12 compute/memory/mixed representative groups.  Freeze
# the same per-comparison upper bounds before any timing is observed.
SAME_CLASS_QUOTAS = {
    "pure_compute": 3,
    "pure_memory": 7,
    "mixed_resource": 12,
}
HETEROGENEOUS_QUOTA = 6
MAX_PAIRS_PER_COMPARISON = sum(SAME_CLASS_QUOTAS.values()) + HETEROGENEOUS_QUOTA
MAX_TOTAL_PAIRS = MAX_PAIRS_PER_COMPARISON * len(COMPARISONS)
CNN_MODELS = {"GoogLeNet", "Inception-v3", "NASNet", "ConvNeXt", "YOLOv8x"}


def kernel_family(name: str) -> str:
    text = name.lower()
    if any(token in text for token in ("batch_norm", "batchnorm", "bn_fw", "bn_bw")):
        return "BatchNorm"
    if any(token in text for token in ("layer_norm", "layernorm")):
        return "LayerNorm"
    if "gemv" in text:
        return "GEMV"
    if any(token in text for token in ("convol", "conv2d", "conv_fwd", "conv_dgrad", "conv_wgrad")):
        return "Conv"
    if "cask" in text and "computeoffsets" not in text:
        return "Conv"
    if any(token in text for token in ("gemm", "sgemm", "hgemm", "cutlass", "matmul")):
        return "GEMM"
    if "pool" in text:
        return "Pool"
    if any(token in text for token in ("softmax", "reduce", "reduction", "welford")):
        return "Reduce"
    if any(token in text for token in ("concat", "catarray", "cat_")):
        return "Concat"
    if any(token in text for token in ("transpose", "permute", "contiguous", "layout", "nchw", "nhwc")):
        return "LayoutTransform"
    if any(token in text for token in ("copy", "indexselect", "index_select", "gather", "scatter", "embedding", "masked_select")):
        return "CopyGather"
    if any(token in text for token in ("elementwise", "pointwise", "clamp", "relu", "gelu", "sigmoid", "tanh", "add", "mul", "sub", "div")):
        return "Elementwise"
    # Unknown kernels are never silently treated as memory-bound.
    return "Unclassified"


def group_class(families: list[str]) -> str:
    if not families or any(value == "Unclassified" for value in families):
        return "unclassified"
    if all(value in COMPUTE_FAMILIES for value in families):
        return "pure_compute"
    if all(value in MEMORY_FAMILIES for value in families):
        return "pure_memory"
    return "mixed_resource"


def load_profiles(cache_dir: Path):
    profiles = {}
    sources = []
    for model in MODELS:
        cache = cache_dir / f"{MODEL_CLASSES[model]}.ncu.v2.json"
        payload = json.loads(cache.read_text(encoding="utf-8"))
        identity = payload.get("identity") or {}
        aggregation = payload.get("aggregation") or {}
        if payload.get("schema_version") != 2:
            raise RuntimeError(f"{model}: Stage C cache is not schema v2")
        if identity.get("model_class") != MODEL_CLASSES[model]:
            raise RuntimeError(f"{model}: Stage C cache model class differs")
        if (
            aggregation.get("method") != "identity-checked per-launch median"
            or aggregation.get("repeat_count") != 3
        ):
            raise RuntimeError(f"{model}: Stage C cache is not the frozen median")

        by_op = defaultdict(list)
        all_launches = payload.get("kernels", [])
        for launch in all_launches:
            by_op[str(launch.get("op_name", ""))].append(launch)
        unclassified_names = Counter()
        unclassified_name_duration = Counter()
        family_launch_counts = Counter()
        family_duration_ns = Counter()
        classified_launches = 0
        classified_duration = 0.0
        total_duration = 0.0
        classified_ops = 0
        unclassified_ops = 0
        for op, launches in by_op.items():
            family_duration = Counter()
            duration = 0.0
            work = 0.0
            has_unclassified = False
            for launch in launches:
                metrics = launch.get("metrics", {})
                value = float(metrics.get("dur_ns", 0.0) or 0.0)
                family = kernel_family(str(launch.get("name", "")))
                family_duration[family] += value
                family_launch_counts[family] += 1
                family_duration_ns[family] += value
                duration += value
                total_duration += value
                work += float(launch.get("grid_size", 0) or 0) * float(
                    launch.get("block_size", 0) or 0
                )
                if family == "Unclassified":
                    has_unclassified = True
                    unclassified_names[str(launch.get("name", ""))] += 1
                    unclassified_name_duration[
                        str(launch.get("name", ""))
                    ] += value
                else:
                    classified_launches += 1
                    classified_duration += value
            # Conservative: an OP with any unknown launch is excluded from the
            # resource-class main table instead of being guessed.
            if has_unclassified:
                dominant = "Unclassified"
                unclassified_ops += 1
            else:
                dominant = sorted(
                    family_duration.items(), key=lambda item: (-item[1], item[0])
                )[0][0]
                classified_ops += 1
                if (
                    model in CNN_MODELS
                    and dominant == "GEMM"
                    and not re.search(r"(?:linear|classifier|\bfc\b)", op.lower())
                ):
                    dominant = "Conv"
            profiles[(model, op)] = {
                "family": dominant,
                "duration_ns": duration,
                "work_items_proxy": work,
                "classified_launches": sum(
                    kernel_family(str(row.get("name", ""))) != "Unclassified"
                    for row in launches
                ),
                "total_launches": len(launches),
            }
        total_launches = len(all_launches)
        sources.append(
            {
                "model": model,
                "model_class": MODEL_CLASSES[model],
                "path": str(cache.resolve()),
                "sha256": sha256_file(cache),
                "fx_code_sha256": identity.get("fx_code_sha256"),
                "profile_sha256": identity.get("profile_sha256"),
                "aggregation_method": aggregation.get("method"),
                "aggregation_repeat_count": aggregation.get("repeat_count"),
                "classification": {
                    "kernel_launches": total_launches,
                    "classified_kernel_launches": classified_launches,
                    "unclassified_kernel_launches": total_launches - classified_launches,
                    "total_duration_ns": total_duration,
                    "classified_duration_ns": classified_duration,
                    "unclassified_duration_ns": total_duration - classified_duration,
                    "kernel_family_launch_counts": dict(
                        sorted(family_launch_counts.items())
                    ),
                    "kernel_family_duration_ns": dict(
                        sorted(family_duration_ns.items())
                    ),
                    "kernel_classification_coverage": (
                        classified_launches / total_launches if total_launches else 0.0
                    ),
                    "duration_classification_coverage": (
                        classified_duration / total_duration if total_duration else 0.0
                    ),
                    "operators": len(by_op),
                    "classified_operators": classified_ops,
                    "unclassified_operators": unclassified_ops,
                    "unknown_kernel_names_sample": [
                        {
                            "name": name,
                            "launches": count,
                            "duration_ns": unclassified_name_duration[name],
                        }
                        for name, count in unclassified_names.most_common(20)
                    ],
                },
            }
        )
    return profiles, sources


def validate_stage_b_provenance(
    summary: dict,
    cache_sources: list[dict],
    current_head: str,
    latency_root: Path | None = None,
) -> dict:
    if summary.get("git_head") != current_head:
        raise RuntimeError(
            "Stage B git HEAD differs from the current frozen checkout: "
            f"{summary.get('git_head')} != {current_head}"
        )
    expected_rows = summary.get("ncu_cache_identity")
    if not isinstance(expected_rows, list) or len(expected_rows) != len(MODELS):
        raise RuntimeError("Stage B summary lacks seven-model NCU cache identity")
    expected = {row.get("model"): row for row in expected_rows}
    observed = {row.get("model"): row for row in cache_sources}
    if set(expected) != set(MODELS) or set(observed) != set(MODELS):
        raise RuntimeError("Stage B/Stage C NCU cache model sets differ")
    fields = (
        "model_class",
        "ncu_cache_sha256",
        "profile_sha256",
        "ncu_fx_code_sha256",
        "aggregation_method",
        "aggregation_repeat_count",
    )
    for model in MODELS:
        actual = {
            "model_class": observed[model]["model_class"],
            "ncu_cache_sha256": observed[model]["sha256"],
            "profile_sha256": observed[model]["profile_sha256"],
            "ncu_fx_code_sha256": observed[model]["fx_code_sha256"],
            "aggregation_method": observed[model]["aggregation_method"],
            "aggregation_repeat_count": observed[model]["aggregation_repeat_count"],
        }
        for field in fields:
            if expected[model].get(field) != actual[field]:
                raise RuntimeError(
                    f"{model}: Stage B/Stage C NCU identity differs at {field}: "
                    f"{expected[model].get(field)!r} != {actual[field]!r}"
                )
    if latency_root is not None:
        if not (latency_root / "COMPLETE").is_file():
            raise RuntimeError("Stage B COMPLETE marker is missing")
        plan = json.loads((latency_root / "plan.json").read_text(encoding="utf-8"))
        run_status = json.loads(
            (latency_root / "run_status.json").read_text(encoding="utf-8")
        )
        expected_task_count = len(MODELS) * 3 * 10
        if (
            plan.get("git_head") != current_head
            or plan.get("repeats") != 10
            or plan.get("task_count") != expected_task_count
            or len(plan.get("tasks", [])) != expected_task_count
        ):
            raise RuntimeError("Stage B plan identity/count is incomplete")
        if (
            run_status.get("status") != "completed"
            or run_status.get("completed") != expected_task_count
            or run_status.get("total") != expected_task_count
        ):
            raise RuntimeError("Stage B run_status is not a complete 7x3x10 run")
        task_records = summary.get("task_records")
        if not isinstance(task_records, list) or len(task_records) != expected_task_count:
            raise RuntimeError("Stage B summary does not contain 210 task records")
        expected_keys = {
            (model, policy, trial)
            for model in MODELS
            for policy in ("janus", "newtd_drt", "newtd_ncu_drt")
            for trial in range(10)
        }
        plan_keys = {
            (row.get("model"), row.get("policy"), row.get("trial"))
            for row in plan.get("tasks", [])
        }
        if plan_keys != expected_keys or len(plan_keys) != len(plan.get("tasks", [])):
            raise RuntimeError("Stage B plan task identities are missing or duplicated")
        record_keys = {
            (row.get("model"), row.get("policy"), row.get("trial"))
            for row in task_records
        }
        if record_keys != expected_keys or len(record_keys) != len(task_records):
            raise RuntimeError("Stage B task identities are missing or duplicated")
        for row in task_records:
            relative = row.get("result_relative_path")
            if not relative:
                raise RuntimeError("Stage B task record lacks a relative result path")
            expected_relative = (
                Path("tasks")
                / f"trial_{int(row['trial']):02d}"
                / MODEL_SLUGS[row["model"]]
                / row["policy"]
                / "result.json"
            )
            if Path(relative) != expected_relative:
                raise RuntimeError(
                    "Stage B task record points to an unexpected result: "
                    f"{relative} != {expected_relative}"
                )
            result_path = (latency_root / relative).resolve()
            if latency_root not in result_path.parents or not result_path.is_file():
                raise RuntimeError(f"Stage B task result path is invalid: {relative}")
            if sha256_file(result_path) != row.get("result_sha256"):
                raise RuntimeError(f"Stage B task result SHA differs: {relative}")

    verification_relative = summary.get("asset_verification_relative_path")
    verification_path = (
        (latency_root / verification_relative).resolve()
        if latency_root is not None and verification_relative
        else Path(summary.get("asset_verification", ""))
    )
    verification_sha = summary.get("asset_verification_sha256")
    if not verification_path.is_file() or not verification_sha:
        raise RuntimeError("Stage B asset verification file/hash is unavailable")
    if sha256_file(verification_path) != verification_sha:
        raise RuntimeError("Stage B asset verification file SHA differs")
    audits = summary.get("model_reference_identity_audit")
    audit_models = (
        [row.get("model") for row in audits]
        if isinstance(audits, list)
        else []
    )
    if (
        not isinstance(audits, list)
        or len(audits) != len(MODELS)
        or set(audit_models) != set(MODELS)
        or len(set(audit_models)) != len(audit_models)
        or any(
            row.get("all_reference_output_identities_identical") is not True
            or row.get("process_policy_results") != 30
            for row in audits
        )
    ):
        raise RuntimeError("Stage B reference-output identity audit is incomplete")
    return {
        "git_head": current_head,
        "stage_b_ncu_identities_match": True,
        "asset_verification_sha256": verification_sha,
        "reference_output_identities_match": True,
        "stage_b_task_results_verified": 210 if latency_root is not None else None,
    }


def schedule_signature(result: dict) -> str:
    rows = [
        {"ready": call.get("ready_ops", []), "selected": call.get("selected_resource", [])}
        for call in result["scheduler"]["calls"]
    ]
    raw = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def unique_ready_map(result: dict):
    observed = defaultdict(list)
    for call in result["scheduler"]["calls"]:
        observed[tuple(call.get("ready_ops", []))].append(call)
    return {key: rows[0] for key, rows in observed.items() if len(rows) == 1}


def group_metadata(model: str, names: list[str], profiles):
    rows = [profiles.get((model, name)) for name in names]
    if any(row is None for row in rows):
        return None
    families = [row["family"] for row in rows]
    classified_launches = sum(row["classified_launches"] for row in rows)
    total_launches = sum(row["total_launches"] for row in rows)
    return {
        "operator_families": families,
        "composition": "+".join(sorted(families)),
        "resource_class": group_class(families),
        "profiled_duration_us": sum(row["duration_ns"] for row in rows) / 1000.0,
        "work_items_proxy": sum(row["work_items_proxy"] for row in rows),
        "kernel_classification_coverage": (
            classified_launches / total_launches if total_launches else 0.0
        ),
        "classified_kernel_launches": classified_launches,
        "total_kernel_launches": total_launches,
    }


def paired_subset_filter(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    kept = []
    removed = []
    for row in rows:
        left = set(row["left_group"])
        right = set(row["right_group"])
        supersets = []
        for other in rows:
            if row is other or row["model"] != other["model"] or row["comparison"] != other["comparison"]:
                continue
            other_left = set(other["left_group"])
            other_right = set(other["right_group"])
            if left <= other_left and right <= other_right and (left < other_left or right < other_right):
                supersets.append(other["raw_pair_id"])
        if supersets:
            removed.append({**row, "removed_by_paired_supersets": supersets})
        else:
            kept.append(row)
    return kept, removed


def candidate_rank(row: dict) -> tuple:
    """Frozen ranking uses only offline profile size, never measured slowdown."""
    left = row["left_group_metadata"]
    right = row["right_group_metadata"]
    return (
        -max(left["profiled_duration_us"], right["profiled_duration_us"]),
        -(left["profiled_duration_us"] + right["profiled_duration_us"]),
        -max(left["work_items_proxy"], right["work_items_proxy"]),
        row["model"],
        row["raw_pair_id"],
    )


def select_with_frozen_quotas(rows: list[dict]):
    selected = []
    rejected = []
    audits = []
    for comparison, _, _, _ in COMPARISONS:
        comparison_rows = [row for row in rows if row["comparison"] == comparison]
        comparison_selected = []
        for resource_class, quota in SAME_CLASS_QUOTAS.items():
            bucket = sorted(
                (
                    row for row in comparison_rows
                    if row["same_resource_class"] and row["paired_resource_class"] == resource_class
                ),
                key=candidate_rank,
            )
            chosen = bucket[:quota]
            comparison_selected.extend(chosen)
            rejected.extend(
                {**row, "selection_rejection_reason": "same_class_quota_exceeded"}
                for row in bucket[quota:]
            )
            audits.append(
                {
                    "comparison": comparison,
                    "bucket": resource_class,
                    "quota": quota,
                    "available": len(bucket),
                    "selected": len(chosen),
                    "shortfall": max(0, quota - len(chosen)),
                    "rejected_by_quota": max(0, len(bucket) - quota),
                }
            )
        heterogeneous = sorted(
            (row for row in comparison_rows if not row["same_resource_class"]),
            key=candidate_rank,
        )
        chosen_heterogeneous = heterogeneous[:HETEROGENEOUS_QUOTA]
        comparison_selected.extend(chosen_heterogeneous)
        rejected.extend(
            {**row, "selection_rejection_reason": "heterogeneous_quota_exceeded"}
            for row in heterogeneous[HETEROGENEOUS_QUOTA:]
        )
        audits.append(
            {
                "comparison": comparison,
                "bucket": "heterogeneous_exploratory",
                "quota": HETEROGENEOUS_QUOTA,
                "available": len(heterogeneous),
                "selected": len(chosen_heterogeneous),
                "shortfall": max(0, HETEROGENEOUS_QUOTA - len(chosen_heterogeneous)),
                "rejected_by_quota": max(0, len(heterogeneous) - HETEROGENEOUS_QUOTA),
            }
        )
        if len(comparison_selected) > MAX_PAIRS_PER_COMPARISON:
            raise AssertionError("per-comparison frozen pair cap was exceeded")
        selected.extend(comparison_selected)
    if len(selected) > MAX_TOTAL_PAIRS:
        raise AssertionError("formal total pair cap was exceeded")
    return sorted(selected, key=lambda row: (row["comparison"], candidate_rank(row))), rejected, audits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latency-root", type=Path, required=True)
    parser.add_argument("--ncu-cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    latency = args.latency_root.resolve()
    summary_path = latency / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "completed" or summary.get("protocol") != "seven_model_three_policy_ten_process_mean_latency_v1":
        raise RuntimeError("latency root is not a completed formal stage B run")
    profiles, cache_sources = load_profiles(args.ncu_cache_dir.resolve())
    current_head = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    provenance_audit = validate_stage_b_provenance(
        summary, cache_sources, current_head, latency
    )

    results = {}
    source_results = []
    for model in MODELS:
        for policy in ("janus", "newtd_drt", "newtd_ncu_drt"):
            trial_payloads = []
            signatures = set()
            for trial in range(10):
                path = latency / "tasks" / f"trial_{trial:02d}" / MODEL_SLUGS[model] / policy / "result.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                signatures.add(schedule_signature(payload))
                trial_payloads.append(payload)
                source_results.append(
                    {"model": model, "policy": policy, "trial": trial, "path": str(path), "sha256": sha256_file(path)}
                )
            if len(signatures) != 1:
                raise RuntimeError(f"{model}/{policy}: schedule changed across ten processes")
            results[(model, policy)] = trial_payloads[0]

    raw_pairs = []
    classification_rejected = []
    coverage = []
    for model in MODELS:
        for comparison, left_policy, right_policy, role in COMPARISONS:
            left_map = unique_ready_map(results[(model, left_policy)])
            right_map = unique_ready_map(results[(model, right_policy)])
            exact_keys = sorted(set(left_map) & set(right_map))
            unchanged = 0
            ineligible_width = 0
            unmapped = 0
            unclassified = 0
            eligible = []
            for ready in exact_keys:
                left_call = left_map[ready]
                right_call = right_map[ready]
                left_group = list(left_call.get("selected_resource", []))
                right_group = list(right_call.get("selected_resource", []))
                if left_group == right_group:
                    unchanged += 1
                    continue
                if not (2 <= len(left_group) <= 5 and 2 <= len(right_group) <= 5):
                    ineligible_width += 1
                    continue
                left_metadata = group_metadata(model, left_group, profiles)
                right_metadata = group_metadata(model, right_group, profiles)
                if left_metadata is None or right_metadata is None:
                    unmapped += 1
                    continue
                ready_text = json.dumps(list(ready), ensure_ascii=False, separators=(",", ":"))
                ready_sha = hashlib.sha256(ready_text.encode("utf-8")).hexdigest()
                row = {
                    "raw_pair_id": f"{comparison}:{model}:L{left_call['call']}:R{right_call['call']}",
                    "comparison": comparison,
                    "comparison_role": role,
                    "model": model,
                    "ready": list(ready),
                    "ready_signature_sha256": ready_sha,
                    "left_policy": left_policy,
                    "right_policy": right_policy,
                    "left_call": int(left_call["call"]),
                    "right_call": int(right_call["call"]),
                    "left_group": left_group,
                    "right_group": right_group,
                    "left_group_metadata": left_metadata,
                    "right_group_metadata": right_metadata,
                }
                left_class = left_metadata["resource_class"]
                right_class = right_metadata["resource_class"]
                if "unclassified" in (left_class, right_class):
                    unclassified += 1
                    classification_rejected.append({**row, "selection_rejection_reason": "unclassified_kernel_family"})
                    continue
                row["same_resource_class"] = left_class == right_class
                row["paired_resource_class"] = left_class if left_class == right_class else None
                row["resource_class_transition"] = f"{left_class}->{right_class}"
                row["analysis_bucket"] = "same_class_formal" if left_class == right_class else "heterogeneous_exploratory"
                row["source_comparison_role"] = role
                if left_class != right_class:
                    row["comparison_role"] = "exploratory"
                eligible.append(row)
            deduplicated = []
            seen = set()
            for row in eligible:
                key = (tuple(sorted(row["left_group"])), tuple(sorted(row["right_group"])))
                if key not in seen:
                    seen.add(key)
                    deduplicated.append(row)
            raw_pairs.extend(deduplicated)
            coverage.append(
                {
                    "model": model,
                    "comparison": comparison,
                    "exact_unique_ready_states": len(exact_keys),
                    "unchanged_selection_states": unchanged,
                    "divergent_before_filters": len(eligible) + unclassified,
                    "ineligible_group_width": ineligible_width,
                    "missing_identity_bound_op_profile": unmapped,
                    "unclassified_kernel_family": unclassified,
                    "after_exact_group_dedup": len(deduplicated),
                }
            )

    filtered, subset_removed = paired_subset_filter(raw_pairs)
    selected, quota_rejected, quota_audit = select_with_frozen_quotas(filtered)
    pairs = []
    cases = []
    per_comparison_index = Counter()
    cache_by_model = {row["model"]: row for row in cache_sources}
    for row in selected:
        comparison = row["comparison"]
        per_comparison_index[comparison] += 1
        pair_id = f"{comparison}_{per_comparison_index[comparison]:03d}"
        case_ids = []
        for side in ("left", "right"):
            policy = row[f"{side}_policy"]
            group = row[f"{side}_group"]
            metadata = row[f"{side}_group_metadata"]
            case_id = f"{pair_id}_{side}"
            case_ids.append(case_id)
            cases.append(
                {
                    "case_id": case_id,
                    "pair_id": pair_id,
                    "comparison": comparison,
                    "comparison_role": row["comparison_role"],
                    "analysis_bucket": row["analysis_bucket"],
                    "side": side,
                    "model": row["model"],
                    "display_name": DISPLAY_NAMES[row["model"]],
                    "policy": policy,
                    "call": row[f"{side}_call"],
                    "ready_signature_sha256": row["ready_signature_sha256"],
                    "ready": row["ready"],
                    "group": group,
                    "width": len(group),
                    **metadata,
                    "profile_sha256": cache_by_model[row["model"]]["profile_sha256"],
                    "fx_code_sha256": cache_by_model[row["model"]]["fx_code_sha256"],
                    "ncu_cache_sha256": cache_by_model[row["model"]]["sha256"],
                }
            )
        pairs.append({"pair_id": pair_id, **row, "case_ids": case_ids})

    for row in coverage:
        row["after_paired_subset_filter"] = sum(
            pair["model"] == row["model"] and pair["comparison"] == row["comparison"]
            for pair in filtered
        )
        row["selected_for_timing"] = sum(
            pair["model"] == row["model"] and pair["comparison"] == row["comparison"]
            for pair in pairs
        )
    primary_pairs = [
        pair for pair in pairs
        if pair["comparison_role"] == "primary" and pair["same_resource_class"]
    ]
    payload = {
        "schema_version": 2,
        "status": "ready_for_isolated_measurement" if primary_pairs else "inconclusive_no_primary_pairs",
        "protocol": "janus_4_8_exact_same_ready_paired_bounded_v2",
        "primary_comparison": "NewTD+DRT vs NewTD+NCU-DRT; admission is identical and only final scoring differs",
        "secondary_comparison": "Original Janus vs NewTD+NCU-DRT baseline view",
        "alignment_rule": "exact ordered ready_ops list, unique within both traces",
        "selection_rule": [
            "match exact-ready divergent selections of width 2 through 5",
            "require identity-bound NCU-v2 OP profiles and reject unclassified kernels",
            "remove exact duplicate selected group pairs and paired strict subsets",
            "rank only by frozen offline duration/work proxies, never measured slowdown",
            "apply per-comparison 3/7/12 same-class quotas and a six-pair heterogeneous appendix cap",
        ],
        "frozen_quotas": {
            "same_class_per_comparison": SAME_CLASS_QUOTAS,
            "heterogeneous_per_comparison": HETEROGENEOUS_QUOTA,
            "max_pairs_per_comparison": MAX_PAIRS_PER_COMPARISON,
            "max_total_pairs": MAX_TOTAL_PAIRS,
        },
        "important_boundary": "isolated group replay measures interference of the selected OP group, not end-to-end model latency",
        "latency_root": str(latency),
        "latency_summary_sha256": sha256_file(summary_path),
        "provenance_audit": provenance_audit,
        "ncu_cache_sources": cache_sources,
        "source_results": source_results,
        "classification_coverage": {
            "per_model": [{"model": row["model"], **row["classification"]} for row in cache_sources],
            "rejected_pairs": len(classification_rejected),
            "rejection_reason": "unclassified_kernel_family",
        },
        "coverage": coverage,
        "raw_deduplicated_pairs": raw_pairs,
        "classification_rejected_pairs": classification_rejected,
        "paired_subset_removed": subset_removed,
        "quota_rejected_pairs": quota_rejected,
        "quota_audit": quota_audit,
        "primary_pair_count": len(primary_pairs),
        "heterogeneous_pair_count": sum(not pair["same_resource_class"] for pair in pairs),
        "pair_count": len(pairs),
        "case_count": len(cases),
        "pairs": pairs,
        "cases": cases,
    }
    write_json_atomic(args.output.resolve(), payload)
    print(json.dumps({"status": payload["status"], "pairs": len(pairs), "primary_pairs": len(primary_pairs), "cases": len(cases), "classification_rejections": len(classification_rejected), "quota_audit": quota_audit}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
