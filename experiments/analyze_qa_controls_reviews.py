"""Offline response validation and descriptive accounting for QA controls reviews."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from experiments import analyze_qa_interpretation_answers as _answer_analysis
from experiments import prepare_qa_controls_reviews as _reviews
from experiments import qa_interpretation_answer_receipt as _receipt
from tracemem.annotations import (
    _require_nonblank_text,
    _require_sha256,
    _validate_utc_timestamp,
)

_PREPARER_SHA256 = "e4300d44839ff816721215b6b8aa0f02a38082a1b2f23e6205d361a35bc04924"
_SOURCE_ANSWER_LABELS = ("correct", "incorrect", "uncertain")
_PROMPT_SUPPORT_LABELS = ("supported", "unsupported", "uncertain")
_RESPONSE_FIELDS = {
    "task_id",
    "source_answer",
    "prompt_support",
    "rationale",
    "reviewer_id",
    "reviewer_kind",
    "timestamp",
}


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _packet(receipt: Any, seed: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_hash = _sha256(Path(_reviews.__file__).read_bytes())
    if source_hash != _PREPARER_SHA256:
        raise ValueError("pinned QA controls review preparer source hash differs")
    return _reviews.prepare_reviews(receipt, seed=seed)


def validate_responses(receipt: Any, response_document: Any, *, seed: int) -> list[dict[str, Any]]:
    bundle, _key, template = _packet(receipt, seed)
    return _validate_response_document(bundle, template, response_document)


def _validate_response_document(
    bundle: dict[str, Any], template: dict[str, Any], document: Any
) -> list[dict[str, Any]]:
    if type(document) is not dict or set(document) != set(template):
        raise ValueError("response document has missing or unexpected fields")
    if (
        type(document["version"]) is not int
        or document["version"] != 1
        or document["kind"] != template["kind"]
    ):
        raise ValueError("unsupported response document kind or version")
    bundle_sha256 = _require_sha256(document["bundle_sha256"], "bundle_sha256")
    if bundle_sha256 != bundle["bundle_sha256"]:
        raise ValueError("response bundle hash does not match regenerated QA controls bundle")
    if document["allowed_choices"] != template["allowed_choices"]:
        raise ValueError("allowed_choices differs from response template")
    raw_rows = document["responses"]
    if type(raw_rows) is not list:
        raise TypeError("responses must be a list")
    expected_ids = [task["task_id"] for task in bundle["tasks"]]
    expected_set = set(expected_ids)
    rows: dict[str, dict[str, Any]] = {}
    reviewer_ids: set[str] = set()
    reviewer_kinds: set[str] = set()
    for row in raw_rows:
        if type(row) is not dict or set(row) != _RESPONSE_FIELDS:
            raise ValueError("response row has missing or unexpected fields")
        task_id = row["task_id"]
        if type(task_id) is not str or task_id not in expected_set:
            raise ValueError("unknown response task ID")
        if task_id in rows:
            raise ValueError("duplicate response task ID")
        if (
            type(row["source_answer"]) is not str
            or row["source_answer"] not in _SOURCE_ANSWER_LABELS
        ):
            raise ValueError("source_answer label is invalid")
        if (
            type(row["prompt_support"]) is not str
            or row["prompt_support"] not in _PROMPT_SUPPORT_LABELS
        ):
            raise ValueError("prompt_support label is invalid")
        _require_nonblank_text(row["rationale"], "rationale", 4_000)
        reviewer_id = _require_nonblank_text(row["reviewer_id"], "reviewer_id", 256)
        reviewer_kind = row["reviewer_kind"]
        if type(reviewer_kind) is not str or reviewer_kind not in {"human", "agent"}:
            raise ValueError("reviewer_kind must be human or agent")
        _validate_utc_timestamp(row["timestamp"])
        reviewer_ids.add(reviewer_id)
        reviewer_kinds.add(reviewer_kind)
        rows[task_id] = dict(row)
    if len(reviewer_ids) > 1 or len(reviewer_kinds) > 1:
        raise ValueError("one response document must have a single reviewer ID and kind")
    if set(rows) != expected_set:
        missing = sorted(expected_set.difference(rows))
        raise ValueError(f"missing response task ID(s): {', '.join(missing)}")
    return [rows[task_id] for task_id in expected_ids]


def _canonical_sha256(value: Any) -> str:
    return _sha256(_receipt._canonical_bytes(value))


def _label_counts(
    rows: list[dict[str, Any]], field: str, labels: tuple[str, ...]
) -> dict[str, int]:
    return {label: sum(row[field] == label for row in rows) for label in labels}


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


def _counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "denominator": len(rows),
        "source_answer_counts": _label_counts(rows, "source_answer", _SOURCE_ANSWER_LABELS),
        "prompt_support_counts": _label_counts(rows, "prompt_support", _PROMPT_SUPPORT_LABELS),
        "joint_counts": _joint_counts(rows),
    }


def _cell_row(task_key: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    return {
        field: task_key[field]
        for field in (
            "task_id",
            "cell_id",
            "scenario_id",
            "question_id",
            "variant_id",
            "prompt_id",
        )
    } | {
        "source_answer": response["source_answer"],
        "prompt_support": response["prompt_support"],
    }


def analyze_reviews(
    receipt: Any, response_documents: list[dict[str, Any]], *, seed: int
) -> dict[str, Any]:
    if type(response_documents) is not list or not response_documents:
        raise ValueError("response_documents must be a nonempty list")
    bundle, key, template = _packet(receipt, seed)
    task_keys = key["task_keys"]
    if len(task_keys) != 36 or len(bundle["tasks"]) != 36:
        raise ValueError("pinned QA controls review scope must contain 36 tasks")
    reviewers = []
    response_provenance = []
    reviewer_ids: set[str] = set()
    variants: dict[str, list[dict[str, Any]]] = {}
    for task_key in task_keys:
        variants.setdefault(task_key["variant_id"], []).append(task_key)
    if len(variants) != 3 or any(len(rows) != 12 for rows in variants.values()):
        raise ValueError("pinned QA controls review variants must contain 12 tasks each")

    for document in response_documents:
        rows = _validate_response_document(bundle, template, document)
        reviewer_id = rows[0]["reviewer_id"]
        if reviewer_id in reviewer_ids:
            raise ValueError("duplicate reviewer ID across response documents")
        reviewer_ids.add(reviewer_id)
        response_by_task = {row["task_id"]: row for row in rows}
        unique_rows = [_cell_row(task, response_by_task[task["task_id"]]) for task in task_keys]
        variant_reports = []
        for variant_id in sorted(variants):
            variant_rows = [
                _cell_row(task, response_by_task[task["task_id"]])
                for task in sorted(variants[variant_id], key=lambda row: row["cell_id"])
            ]
            variant_reports.append(
                {"variant_id": variant_id, **_counts(variant_rows), "cells": variant_rows}
            )
        reviewers.append(
            {
                "reviewer_id": reviewer_id,
                "reviewer_kind_self_attested": rows[0]["reviewer_kind"],
                "unique_task_summary": _counts(unique_rows),
                "variants": variant_reports,
            }
        )
        response_provenance.append(
            {
                "reviewer_id": reviewer_id,
                "canonical_sha256": _canonical_sha256(document),
            }
        )
    reviewers.sort(key=lambda row: row["reviewer_id"])
    response_provenance.sort(key=lambda row: row["reviewer_id"])
    pins = key["source_pins"]
    return {
        "schema_version": 1,
        "label": "provisional descriptive QA controls review accounting",
        "dev_only": True,
        "provisional": True,
        "test": [],
        "independent_trials": False,
        "authored_reference_human_label_count": 0,
        "independent_human_reviewers_verified": 0,
        "reviewer_count": len(reviewers),
        "scope": {
            "correlated_groups": 6,
            "questions": 12,
            "tasks": 36,
            "cells": 36,
            "unique_prompts": 36,
        },
        "reviewers": reviewers,
        "provenance": {
            "seed": seed,
            "bundle_sha256": key["bundle_sha256"],
            "receipt_sha256": pins["receipt_sha256"],
            "result_sha256": pins["result_sha256"],
            "report_sha256": pins["report_sha256"],
            "source_pins": dict(pins),
            "review_preparer_source_sha256": pins["review_preparer_source_sha256"],
            "accounting_tool_source_sha256": _sha256(Path(__file__).read_bytes()),
            "response_documents": response_provenance,
        },
    }


def _preflight_output(output: Path, inputs: list[Path]) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output path: {output}")
    if output.resolve() in {path.resolve() for path in inputs}:
        raise ValueError("output path aliases an input")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="analyze-qa-controls-reviews")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--responses", type=Path, action="append", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _preflight_output(args.output, [args.receipt, *args.responses])
        receipt = _answer_analysis.read_receipt(args.receipt)
        response_documents = [_answer_analysis.read_receipt(path) for path in args.responses]
        report = analyze_reviews(receipt, response_documents, seed=args.seed)
        _answer_analysis.write_analysis(args.output, report)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "reviewers": report["reviewer_count"],
                    "remote_started": False,
                },
                sort_keys=True,
            )
        )
        return 0
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"analyze_qa_controls_reviews: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
