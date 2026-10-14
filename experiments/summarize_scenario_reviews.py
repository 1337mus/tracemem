"""Account for blinded reviews of locally generated scenario claim origins."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.prepare_scenario_reviews import prepare_reviews
from experiments.review_annotations import _preflight, _read_json, _write_many
from tracemem.annotation_report import build_review_report
from tracemem.annotations import _canonical_json, validate_responses
from tracemem.challenge import _reject_constant, _unique_object
from tracemem.scenarios import load_scenarios


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def summarize_reviews(
    receipt_bytes_sequence: Sequence[bytes],
    response_documents: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    """Regenerate review provenance and count reviewer judgments over claim origins."""
    bundle, answer_key, provenance = prepare_reviews(receipt_bytes_sequence, seed=seed)
    build_review_report(bundle, answer_key, response_documents)

    reviewers: list[dict[str, Any]] = []
    for document in response_documents:
        rows = validate_responses(bundle, document)
        if not rows:
            raise ValueError("response document has no tasks to identify its reviewer")
        reviewers.append(
            {
                "reviewer_id": rows[0]["reviewer_id"],
                "reviewer_kind": rows[0]["reviewer_kind"],
                "labels": {row["task_id"]: row["label"] for row in rows},
                "canonical_sha256": _canonical_hash(document),
            }
        )

    cases, _, _ = load_scenarios()
    split_by_case = {case["case_id"]: case["split"] for case in cases}
    split_by_task = {row["task_id"]: row["split"] for row in answer_key["task_keys"]}
    receipt_rows = provenance["receipts"]
    model_keys = [row["model_key"] for row in receipt_rows]
    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for model_key in model_keys:
        by_model[model_key] = {}
        for split in ("dev", "test"):
            by_model[model_key][split] = {
                "raw_output_count": 0,
                "parse_excluded_output_count": 0,
                "schema_excluded_output_count": 0,
                "anchoring_excluded_output_count": 0,
                "schema_valid_empty_output_count": 0,
                "schema_valid_candidate_count": 0,
                "unique_review_task_count": 0,
                "reviewers": [],
                "_task_ids": set(),
            }

    reviewer_cells: dict[tuple[str, str, str], dict[str, Any]] = {}
    for model_key, split_rows in by_model.items():
        for split, cell in split_rows.items():
            for reviewer in reviewers:
                row = {
                    "reviewer_id": reviewer["reviewer_id"],
                    "reviewer_kind": reviewer["reviewer_kind"],
                    "supported_origin_count": 0,
                    "unsupported_origin_count": 0,
                    "uncertain_origin_count": 0,
                    "all_reviewed_origin_count": 0,
                    "decisive_origin_count": 0,
                }
                cell["reviewers"].append(row)
                reviewer_cells[(model_key, split, reviewer["reviewer_id"])] = row

    for output in provenance["outputs"]:
        model_key, split = output["model_key"], split_by_case[output["case_id"]]
        cell = by_model[model_key][split]
        cell["raw_output_count"] += 1
        if output["raw_status"] == "parse_invalid":
            cell["parse_excluded_output_count"] += 1
        elif output["raw_status"] == "schema_invalid":
            cell["schema_excluded_output_count"] += 1
        elif output["anchored_status"] == "excluded":
            cell["anchoring_excluded_output_count"] += 1
        else:
            candidate_count = output["candidate_count"]
            cell["schema_valid_candidate_count"] += candidate_count
            if candidate_count == 0:
                cell["schema_valid_empty_output_count"] += 1

    for task in provenance["task_provenance"]:
        task_id = task["task_id"]
        split = split_by_task[task_id]
        for origin in task["origins"]:
            model_key = origin["model_key"]
            cell = by_model[model_key][split]
            cell["_task_ids"].add(task_id)
            for reviewer in reviewers:
                reviewer_id = reviewer["reviewer_id"]
                label = reviewer["labels"][task_id]
                row = reviewer_cells[(model_key, split, reviewer_id)]
                row[f"{label}_origin_count"] += 1
                row["all_reviewed_origin_count"] += 1
                if label != "uncertain":
                    row["decisive_origin_count"] += 1

    for split_rows in by_model.values():
        for cell in split_rows.values():
            cell["unique_review_task_count"] = len(cell.pop("_task_ids"))
            for reviewer in cell["reviewers"]:
                all_reviewed = reviewer["all_reviewed_origin_count"]
                decisive = reviewer["decisive_origin_count"]
                supported = reviewer["supported_origin_count"]
                reviewer["supported_rate_all_reviewed_origins"] = (
                    supported / all_reviewed if all_reviewed else None
                )
                reviewer["supported_rate_decisive_origins"] = (
                    supported / decisive if decisive else None
                )

    canonical_receipts: list[dict[str, Any]] = []
    for index, receipt_bytes in enumerate(receipt_bytes_sequence):
        receipt_document = json.loads(
            receipt_bytes.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        canonical_receipts.append(
            {
                "model_key": receipt_rows[index]["model_key"],
                "receipt_sha256": receipt_rows[index]["receipt_sha256"],
                "canonical_sha256": _canonical_hash(receipt_document),
            }
        )

    return {
        "version": 1,
        "dataset_sha256": bundle["dataset_sha256"],
        "bundle_sha256": bundle["bundle_sha256"],
        "bundle_canonical_sha256": _canonical_hash(bundle),
        "seed": seed,
        "receipts": canonical_receipts,
        "responses": [
            {
                "reviewer_id": reviewer["reviewer_id"],
                "reviewer_kind": reviewer["reviewer_kind"],
                "canonical_sha256": reviewer["canonical_sha256"],
            }
            for reviewer in reviewers
        ],
        "provenance_sha256": provenance["provenance_sha256"],
        "by_model": by_model,
        "interpretation": (
            "These fractions are not recall, independent human truth, precision calibration, "
            "semantic acceptance, or Gate A. Citation anchoring, semantic judgment, and raw "
            "citation integrity are separate. Reviewer kind is self-attested."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", required=True, nargs="+", type=Path)
    parser.add_argument("--responses", nargs="*", type=Path, default=[])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    inputs = (*args.receipts, *args.responses)
    try:
        outputs = _preflight((args.output,), inputs)
        receipts = [path.read_bytes() for path in args.receipts]
        response_documents = [_read_json(path) for path in args.responses]
        report = summarize_reviews(receipts, response_documents, seed=args.seed)
        _write_many(outputs, (report,))
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"summarize_scenario_reviews: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
