"""Descriptive local comparison of reviews to authored expectations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .annotations import (
    _BUNDLE_VERSION,
    _require_nonblank_text,
    _validate_blind_bundle,
    _validate_case_metadata,
    validate_responses,
)


def build_review_report(
    blind_bundle: Mapping[str, Any],
    answer_key: Mapping[str, Any],
    response_documents: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare complete reviewer documents without changing or promoting labels."""
    tasks = _validate_blind_bundle(blind_bundle)
    references = _validate_answer_key(answer_key, blind_bundle, tasks)
    if not isinstance(response_documents, Sequence) or isinstance(response_documents, (str, bytes)):
        raise TypeError("response_documents must be a sequence")
    stats: dict[str, dict[str, int | float | None]] = {
        kind: {
            "reviewers": 0,
            "responses": 0,
            "agreement": 0,
            "disagreement": 0,
            "uncertain": 0,
            "response_uncertain": 0,
            "response_supported": 0,
            "response_unsupported": 0,
            "reference_supported": 0,
            "reference_unsupported": 0,
            "reference_labeled_responses": 0,
            "unlabeled_responses": 0,
        }
        for kind in ("human", "agent")
    }
    reviewer_ids: set[str] = set()
    for document in response_documents:
        rows = validate_responses(blind_bundle, document)
        if not rows:
            raise ValueError("response document has no tasks to identify its reviewer")
        reviewer_id, reviewer_kind = rows[0]["reviewer_id"], rows[0]["reviewer_kind"]
        if reviewer_id in reviewer_ids:
            raise ValueError(f"duplicate reviewer document: {reviewer_id}")
        reviewer_ids.add(reviewer_id)
        group = stats[reviewer_kind]
        _increment(group, "reviewers")
        for row in rows:
            reference, label = references[row["task_id"]], row["label"]
            _increment(group, "responses")
            if reference is None:
                _increment(group, "unlabeled_responses")
            else:
                _increment(group, "reference_labeled_responses")
                _increment(group, f"reference_{reference}")
            if label == "uncertain":
                _increment(group, "uncertain")
                _increment(group, "response_uncertain")
            else:
                _increment(group, f"response_{label}")
                if reference is not None:
                    _increment(group, "agreement" if label == reference else "disagreement")
    for group in stats.values():
        expected = int(group["reviewers"] or 0) * len(tasks)
        group["expected_responses"] = expected
        group["coverage_rate"] = _ratio(int(group["responses"] or 0), expected)
        decisive = int(group["agreement"] or 0) + int(group["disagreement"] or 0)
        group["decisive_responses"] = decisive
        group["agreement_denominator"] = decisive
        group["agreement_rate_on_decisive"] = _ratio(int(group["agreement"] or 0), decisive)
    return {
        "version": _BUNDLE_VERSION,
        "dataset_sha256": blind_bundle["dataset_sha256"],
        "bundle_sha256": blind_bundle["bundle_sha256"],
        "task_count": len(tasks),
        "reviewer_document_count": len(response_documents),
        "by_reviewer_kind": stats,
        "human_identity_policy": (
            "reviewer_kind is self-attested; reviewer identity is not verified"
        ),
        "reference_label_policy": (
            "authored expectations are descriptive comparisons; this report never modifies or "
            "promotes them"
        ),
        "agreement_scope": (
            "Agreement uses decisive responses to gold and foil tasks only; unlabeled candidates "
            "are excluded from its denominator"
        ),
    }


def _validate_answer_key(
    key: Mapping[str, Any], bundle: Mapping[str, Any], tasks: list[dict[str, Any]]
) -> dict[str, str | None]:
    expected_fields = {
        "version",
        "dataset_sha256",
        "bundle_sha256",
        "case_ids",
        "task_keys",
        "case_metadata",
    }
    if type(key) is not dict or set(key) != expected_fields:
        raise ValueError("answer key has missing or unexpected fields")
    if type(key["version"]) is not int or key["version"] != _BUNDLE_VERSION:
        raise ValueError("unsupported answer key version")
    if key["dataset_sha256"] != bundle["dataset_sha256"]:
        raise ValueError("answer key dataset hash does not match blind bundle")
    if key["bundle_sha256"] != bundle["bundle_sha256"]:
        raise ValueError("answer key bundle hash does not match blind bundle")
    raw_case_ids = key["case_ids"]
    if type(raw_case_ids) is not list:
        raise TypeError("answer key case_ids must be a list")
    case_ids: set[str] = set()
    for case_id in raw_case_ids:
        value = _require_nonblank_text(case_id, "case_id", 256)
        if value in case_ids:
            raise ValueError(f"duplicate answer key case_id: {value}")
        case_ids.add(value)
    rows = key["task_keys"]
    if type(rows) is not list:
        raise TypeError("answer key task_keys must be a list")
    expected_ids = {task["task_id"] for task in tasks}
    references: dict[str, str | None] = {}
    row_fields = {"task_id", "case_id", "split", "membership", "reference_label"}
    for row in rows:
        if type(row) is not dict or set(row) != row_fields:
            raise ValueError("answer key task row has missing or unexpected fields")
        task_id = row["task_id"]
        if type(task_id) is not str or task_id in references:
            raise ValueError("answer key contains an invalid or duplicate task ID")
        case_id = _require_nonblank_text(row["case_id"], "case_id", 256)
        if case_id not in case_ids:
            raise ValueError(f"answer key task references unknown case_id: {case_id}")
        _require_nonblank_text(row["split"], "split", 128)
        membership, label = row["membership"], row["reference_label"]
        if type(membership) is not str or (
            (membership == "gold" and label != "supported")
            or (membership == "foil" and label != "unsupported")
            or (membership == "unlabeled" and label is not None)
            or membership not in {"gold", "foil", "unlabeled"}
        ):
            raise ValueError("answer key has invalid membership or authored label")
        references[task_id] = label if type(label) is str else None
    if set(references) != expected_ids:
        raise ValueError("answer key task coverage does not match blind bundle")
    metadata = key["case_metadata"]
    if type(metadata) is not dict:
        raise TypeError("answer key case_metadata must be an object")
    if any(type(case_id) is not str or case_id not in case_ids for case_id in metadata):
        raise ValueError("answer key metadata references an unknown case_id")
    for row in metadata.values():
        _validate_case_metadata(row)
    return references


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _increment(group: dict[str, int | float | None], field: str) -> None:
    value = group[field]
    if not isinstance(value, int):
        raise ValueError(f"report counter {field} is not an integer")
    group[field] = value + 1
