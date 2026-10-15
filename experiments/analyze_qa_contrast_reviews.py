"""Offline descriptive accounting for QA contrast reviews."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from experiments import prepare_qa_contrast_reviews as qa_reviews

_SOURCE_ANSWER_LABELS = ("correct", "incorrect", "uncertain")
_PROMPT_SUPPORT_LABELS = ("supported", "unsupported", "uncertain")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _label_counts(
    rows: list[dict[str, Any]], field: str, labels: tuple[str, ...]
) -> dict[str, int]:
    counts = Counter(row[field] for row in rows)
    return {label: counts[label] for label in labels}


def _joint_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        source: {
            support: sum(
                row["source_answer"] == source and row["prompt_support"] == support for row in rows
            )
            for support in _PROMPT_SUPPORT_LABELS
        }
        for source in _SOURCE_ANSWER_LABELS
    }


def _denominators(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "total": len(rows),
        "answerable": sum(row["is_answerable"] for row in rows),
        "unknown": sum(not row["is_answerable"] for row in rows),
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "denominator": len(rows),
        "source_answer_counts": _label_counts(rows, "source_answer", _SOURCE_ANSWER_LABELS),
        "prompt_support_counts": _label_counts(rows, "prompt_support", _PROMPT_SUPPORT_LABELS),
        "joint_counts": _joint_counts(rows),
    }


def _strata(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        "all": _summary(rows),
        "answerable": _summary([row for row in rows if row["is_answerable"]]),
        "unknown": _summary([row for row in rows if not row["is_answerable"]]),
    }


def analyze_reviews(
    receipt: dict[str, Any], response_documents: list[dict[str, Any]], *, seed: int
) -> dict[str, Any]:
    """Analyze complete response documents separately by reviewer."""
    if type(response_documents) is not list or not response_documents:
        raise ValueError("response_documents must be a nonempty list")
    _bundle, key, _template = qa_reviews.prepare_reviews(receipt, seed=seed)
    reviewers = []
    response_provenance = []
    for document in response_documents:
        rows = qa_reviews.validate_responses(receipt, document, seed=seed)
        first = rows[0]
        response_provenance.append(
            {
                "reviewer_id": first["reviewer_id"],
                "sha256": _sha256(_canonical_bytes(document)),
            }
        )
        response_by_task = {row["task_id"]: row for row in rows}
        key_tasks = key["tasks"]
        key_by_task = {row["task_id"]: row for row in key_tasks}
        cells: list[dict[str, Any]] = []
        for task in key_tasks:
            response = response_by_task[task["task_id"]]
            for mapping in task["cells"]:
                cells.append(
                    {
                        "task_id": task["task_id"],
                        "prompt_id": task["prompt_id"],
                        "question_id": task["question_id"],
                        "cell_id": mapping["cell_id"],
                        "mode": mapping["mode"],
                        "ceiling": mapping["ceiling"],
                        "is_answerable": task["is_answerable"],
                        "source_answer": response["source_answer"],
                        "prompt_support": response["prompt_support"],
                    }
                )
        cells.sort(
            key=lambda row: (
                row["mode"],
                row["ceiling"],
                row["question_id"],
                row["prompt_id"],
                row["task_id"],
            )
        )
        group_keys = sorted({(row["mode"], row["ceiling"]) for row in cells})
        groups = []
        for mode, ceiling in group_keys:
            group_rows = [row for row in cells if row["mode"] == mode and row["ceiling"] == ceiling]
            groups.append(
                {
                    "mode": mode,
                    "ceiling": ceiling,
                    "unique_prompt_count": len({row["prompt_id"] for row in group_rows}),
                    "denominators": _denominators(group_rows),
                    "source_answer_counts": _label_counts(
                        group_rows, "source_answer", _SOURCE_ANSWER_LABELS
                    ),
                    "prompt_support_counts": _label_counts(
                        group_rows, "prompt_support", _PROMPT_SUPPORT_LABELS
                    ),
                    "joint_counts": _joint_counts(group_rows),
                    "strata": _strata(group_rows),
                }
            )
        unique_task_rows = [
            {**response_by_task[task_id], "is_answerable": key_by_task[task_id]["is_answerable"]}
            for task_id in sorted(key_by_task)
        ]
        reviewers.append(
            {
                "reviewer_id": first["reviewer_id"],
                "reviewer_kind_self_attested": first["reviewer_kind"],
                "unique_task_summary": {
                    "denominator": len(rows),
                    "denominators": _denominators(unique_task_rows),
                    "source_answer_counts": _label_counts(
                        unique_task_rows, "source_answer", _SOURCE_ANSWER_LABELS
                    ),
                    "prompt_support_counts": _label_counts(
                        unique_task_rows, "prompt_support", _PROMPT_SUPPORT_LABELS
                    ),
                    "joint_counts": _joint_counts(unique_task_rows),
                    "strata": _strata(unique_task_rows),
                },
                "expanded_cells": {
                    "denominator": len(cells),
                    "denominators": _denominators(cells),
                    "unique_prompt_count": len({row["prompt_id"] for row in cells}),
                    "source_answer_counts": _label_counts(
                        cells, "source_answer", _SOURCE_ANSWER_LABELS
                    ),
                    "prompt_support_counts": _label_counts(
                        cells, "prompt_support", _PROMPT_SUPPORT_LABELS
                    ),
                    "joint_counts": _joint_counts(cells),
                    "strata": _strata(cells),
                    "groups": groups,
                    "cells": cells,
                },
            }
        )
    reviewer_ids = [row["reviewer_id"] for row in reviewers]
    if len(set(reviewer_ids)) != len(reviewer_ids):
        raise ValueError("duplicate reviewer ID across response documents")
    reviewers.sort(key=lambda row: row["reviewer_id"])
    response_provenance.sort(key=lambda row: row["reviewer_id"])
    return {
        "schema_version": 1,
        "label": "provisional descriptive QA review accounting",
        "dev_only": True,
        "provisional": True,
        "test": [],
        "authored_reference_human_label_count": key["human_label_count"],
        "independent_trials": False,
        "independent_human_reviewers_verified": 0,
        "reviewer_count": len(reviewers),
        "question_count": len({row["question_id"] for row in key["tasks"]}),
        "unique_prompt_count": len({row["prompt_id"] for row in key["tasks"]}),
        "expanded_cell_count": sum(len(row["cells"]) for row in key["tasks"]),
        "reused_cell_count": sum(len(row["cells"]) for row in key["tasks"])
        - len({row["prompt_id"] for row in key["tasks"]}),
        "reviewers": reviewers,
        "provenance": {
            "seed": seed,
            "bundle_sha256": key["bundle_sha256"],
            "receipt_sha256": key["receipt_sha256"],
            "literal_analysis_report_sha256": key["literal_analysis_report_sha256"],
            "accounting_tool_source_sha256": _sha256(Path(__file__).read_bytes()),
            "review_preparer_source_sha256": key["review_preparer_source_sha256"],
            "response_documents": response_provenance,
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Run the offline review accounting command-line interface."""
    parser = argparse.ArgumentParser(prog="analyze-qa-contrast-reviews")
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--responses", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError(f"refusing to overwrite existing output path: {args.output}")
        resolved_output = args.output.resolve()
        inputs = [args.receipt, *args.responses]
        if resolved_output in {path.resolve() for path in inputs}:
            raise ValueError("output path aliases an input")
        receipt = qa_reviews._read_json(args.receipt)
        responses = [qa_reviews._read_json(path) for path in args.responses]
        report = analyze_reviews(receipt, responses, seed=args.seed)
        data = (
            json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(args.output, flags, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            os.fchmod(output.fileno(), 0o600)
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"analyze_qa_contrast_reviews: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
