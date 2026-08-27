#!/usr/bin/env python3
"""Combine the original 38-case holdout and the new 462-case TD validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


MODEL_ORDER = [
    "GoogLeNet",
    "Inception-v3",
    "NASNet",
    "BERT",
    "ConvNeXt",
    "DeepFM",
    "YOLOv8x",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-root", required=True, type=Path)
    ap.add_argument("--new-root", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args()

    sources = []
    all_cases = []
    seen_groups = set()
    for phase, root in (("original_holdout", args.old_root), ("expanded_unseen", args.new_root)):
        files = sorted(root.glob("*/result_v2.json"))
        if not files:
            raise SystemExit(f"no result_v2.json under {root}")
        for path in files:
            data = json.loads(path.read_text(encoding="utf-8"))
            sources.append({
                "phase": phase,
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "model": data["model"],
                "case_count": data["case_count"],
                "evidence": data["evidence"],
            })
            for case in data["cases"]:
                key = (case["model"], int(case["call"]), tuple(case["group"]))
                if key in seen_groups:
                    raise SystemExit(f"duplicate operator group: {key}")
                seen_groups.add(key)
                row = dict(case)
                row["phase"] = phase
                row["combined_case_id"] = f"{phase}:{case['case_id']}"
                row["source_result"] = str(path.resolve())
                all_cases.append(row)

    if len(all_cases) != 500:
        raise SystemExit(f"expected exactly 500 cases, got {len(all_cases)}")

    grouped = defaultdict(list)
    for case in all_cases:
        grouped[case["model"]].append(case)

    per_model = []
    for model in MODEL_ORDER:
        cases = grouped.get(model, [])
        auditable = sum(bool(c.get("auditable")) for c in cases)
        strict = sum(bool(c.get("isolated_strict_parallel")) for c in cases)
        widths = defaultdict(lambda: {"cases": 0, "strict": 0})
        for c in cases:
            w = str(c["size"])
            widths[w]["cases"] += 1
            widths[w]["strict"] += int(bool(c.get("isolated_strict_parallel")))
        per_model.append({
            "model": model,
            "cases": len(cases),
            "auditable": auditable,
            "strict_parallel": strict,
            "positive_precision": (strict / auditable) if auditable else None,
            "by_width": {
                w: {**v, "positive_precision": v["strict"] / v["cases"]}
                for w, v in sorted(widths.items(), key=lambda x: int(x[0]))
            },
        })

    total_auditable = sum(x["auditable"] for x in per_model)
    total_strict = sum(x["strict_parallel"] for x in per_model)
    aggregate = {
        "schema_version": 2,
        "protocol": "janus_4_7_td_final_positive_precision_500_case_combined",
        "truth_definition": "every target OP maps to replay kernels and the intersection of all merged OP kernel intervals has positive duration",
        "selection_rule": "unseen TD-final positive predictions; no 5/5 stability requirement; one isolated replay per group",
        "case_count": len(all_cases),
        "auditable_cases": total_auditable,
        "strict_parallel_cases": total_strict,
        "positive_precision": total_strict / total_auditable,
        "per_model": per_model,
        "sources": sources,
    }

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    (out / "aggregate.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "cases.json").write_text(json.dumps(all_cases, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "cases.csv").open("w", newline="", encoding="utf-8-sig") as f:
        cols = ["phase", "combined_case_id", "model", "case_id", "call", "size", "group", "auditable", "strict_overlap_ns", "isolated_strict_parallel", "missing_ops", "streams_by_op", "common_intervals", "source_result"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for c in all_cases:
            w.writerow({k: json.dumps(c.get(k), ensure_ascii=False) if isinstance(c.get(k), (list, dict)) else c.get(k) for k in cols})
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
