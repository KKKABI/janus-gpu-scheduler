#!/usr/bin/env python3
"""Select §4.8 groups from exact ordered ready states without outcome peeking."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
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
    "OtherMemory",
}
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
    return "OtherMemory"


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
        by_op = defaultdict(list)
        for launch in payload.get("kernels", []):
            by_op[str(launch.get("op_name", ""))].append(launch)
        for op, launches in by_op.items():
            family_duration = Counter()
            duration = 0.0
            work = 0.0
            for launch in launches:
                metrics = launch.get("metrics", {})
                value = float(metrics.get("dur_ns", 0.0) or 0.0)
                family_duration[kernel_family(str(launch.get("name", "")))] += value
                duration += value
                work += float(launch.get("grid_size", 0) or 0) * float(
                    launch.get("block_size", 0) or 0
                )
            dominant = sorted(
                family_duration.items(), key=lambda item: (-item[1], item[0])
            )[0][0]
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
            }
        sources.append(
            {
                "model": model,
                "path": str(cache.resolve()),
                "sha256": sha256_file(cache),
                "fx_code_sha256": payload.get("identity", {}).get(
                    "fx_code_sha256"
                ),
                "profile_sha256": payload.get("identity", {}).get(
                    "profile_sha256"
                ),
            }
        )
    return profiles, sources


def schedule_signature(result: dict) -> str:
    rows = [
        {
            "ready": call.get("ready_ops", []),
            "selected": call.get("selected_resource", []),
        }
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
    return {
        "operator_families": families,
        "composition": "+".join(sorted(families)),
        "resource_class": group_class(families),
        "profiled_duration_us": sum(row["duration_ns"] for row in rows) / 1000.0,
        "work_items_proxy": sum(row["work_items_proxy"] for row in rows),
    }


def paired_subset_filter(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    kept = []
    removed = []
    for row in rows:
        left = set(row["left_group"])
        right = set(row["right_group"])
        supersets = []
        for other in rows:
            if row is other or row["model"] != other["model"] or row[
                "comparison"
            ] != other["comparison"]:
                continue
            other_left = set(other["left_group"])
            other_right = set(other["right_group"])
            if left <= other_left and right <= other_right and (
                left < other_left or right < other_right
            ):
                supersets.append(other["raw_pair_id"])
        if supersets:
            removed.append({**row, "removed_by_paired_supersets": supersets})
        else:
            kept.append(row)
    return kept, removed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latency-root", type=Path, required=True)
    parser.add_argument("--ncu-cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    latency = args.latency_root.resolve()
    summary = json.loads((latency / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "completed" or summary.get("protocol") != "seven_model_three_policy_ten_process_mean_latency_v1":
        raise RuntimeError("latency root is not a completed formal stage B run")
    profiles, cache_sources = load_profiles(args.ncu_cache_dir.resolve())

    results = {}
    source_results = []
    for model in MODELS:
        for policy in ("janus", "newtd_drt", "newtd_ncu_drt"):
            trial_payloads = []
            signatures = set()
            for trial in range(10):
                path = (
                    latency
                    / "tasks"
                    / f"trial_{trial:02d}"
                    / MODEL_SLUGS[model]
                    / policy
                    / "result.json"
                )
                payload = json.loads(path.read_text(encoding="utf-8"))
                signatures.add(schedule_signature(payload))
                trial_payloads.append(payload)
                source_results.append(
                    {
                        "model": model,
                        "policy": policy,
                        "trial": trial,
                        "path": str(path),
                        "sha256": sha256_file(path),
                    }
                )
            if len(signatures) != 1:
                raise RuntimeError(
                    f"{model}/{policy}: schedule changed across ten processes"
                )
            results[(model, policy)] = trial_payloads[0]

    raw_pairs = []
    coverage = []
    for model in MODELS:
        for comparison, left_policy, right_policy, role in COMPARISONS:
            left_map = unique_ready_map(results[(model, left_policy)])
            right_map = unique_ready_map(results[(model, right_policy)])
            exact_keys = sorted(set(left_map) & set(right_map))
            unchanged = 0
            ineligible_width = 0
            unmapped = 0
            eligible = []
            for ready in exact_keys:
                left_call = left_map[ready]
                right_call = right_map[ready]
                left_group = list(left_call.get("selected_resource", []))
                right_group = list(right_call.get("selected_resource", []))
                if left_group == right_group:
                    unchanged += 1
                    continue
                if not (
                    2 <= len(left_group) <= 5 and 2 <= len(right_group) <= 5
                ):
                    ineligible_width += 1
                    continue
                left_metadata = group_metadata(model, left_group, profiles)
                right_metadata = group_metadata(model, right_group, profiles)
                if left_metadata is None or right_metadata is None:
                    unmapped += 1
                    continue
                ready_text = json.dumps(
                    list(ready), ensure_ascii=False, separators=(",", ":")
                )
                ready_sha = hashlib.sha256(
                    ready_text.encode("utf-8")
                ).hexdigest()
                eligible.append(
                    {
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
                )
            # Remove exact duplicate selected-pair identities before the paired
            # subset filter.  No timing or slowdown data is consulted.
            deduplicated = []
            seen = set()
            for row in eligible:
                key = (
                    tuple(sorted(row["left_group"])),
                    tuple(sorted(row["right_group"])),
                )
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
                    "divergent_before_filters": len(eligible),
                    "ineligible_group_width": ineligible_width,
                    "missing_identity_bound_op_profile": unmapped,
                    "after_exact_group_dedup": len(deduplicated),
                }
            )

    filtered, subset_removed = paired_subset_filter(raw_pairs)
    pairs = []
    cases = []
    per_comparison_index = Counter()
    cache_by_model = {row["model"]: row for row in cache_sources}
    for row in filtered:
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
                    "side": side,
                    "model": row["model"],
                    "display_name": DISPLAY_NAMES[row["model"]],
                    "policy": policy,
                    "call": row[f"{side}_call"],
                    "ready_signature_sha256": row[
                        "ready_signature_sha256"
                    ],
                    "ready": row["ready"],
                    "group": group,
                    "width": len(group),
                    **metadata,
                    "profile_sha256": cache_by_model[row["model"]][
                        "profile_sha256"
                    ],
                    "fx_code_sha256": cache_by_model[row["model"]][
                        "fx_code_sha256"
                    ],
                    "ncu_cache_sha256": cache_by_model[row["model"]][
                        "sha256"
                    ],
                }
            )
        pairs.append(
            {
                "pair_id": pair_id,
                **row,
                "case_ids": case_ids,
            }
        )

    for row in coverage:
        row["after_paired_subset_filter"] = sum(
            pair["model"] == row["model"]
            and pair["comparison"] == row["comparison"]
            for pair in pairs
        )
    payload = {
        "schema_version": 1,
        "status": "ready_for_isolated_measurement",
        "protocol": "janus_4_8_exact_same_ready_paired_v1",
        "primary_comparison": "NewTD+DRT vs NewTD+NCU-DRT; admission is identical and only final scoring differs",
        "secondary_comparison": "Original Janus vs NewTD+NCU-DRT baseline view",
        "alignment_rule": "exact ordered ready_ops list, unique within both traces",
        "selection_rule": [
            "use all exact-ready divergent selections with group width 2 through 5",
            "require every selected OP in the identity-bound NCU-v2 cache",
            "remove exact duplicate selected group pairs",
            "remove paired strict subsets without consulting measured slowdown",
            "do not force the Janus paper's 3/7/12 quotas when the eligible set is smaller",
        ],
        "important_boundary": "isolated group replay measures interference of the selected OP group, not end-to-end model latency",
        "latency_root": str(latency),
        "latency_summary_sha256": sha256_file(latency / "summary.json"),
        "ncu_cache_sources": cache_sources,
        "source_results": source_results,
        "coverage": coverage,
        "raw_deduplicated_pairs": raw_pairs,
        "paired_subset_removed": subset_removed,
        "pair_count": len(pairs),
        "case_count": len(cases),
        "pairs": pairs,
        "cases": cases,
    }
    write_json_atomic(args.output.resolve(), payload)
    print(
        json.dumps(
            {"pairs": len(pairs), "cases": len(cases), "coverage": coverage},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
