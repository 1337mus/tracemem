"""Summarize validated synthetic review outcomes by scenario source tag."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.evaluate_integrity_selectivity import evaluate_integrity_selectivity
from experiments.prepare_scenario_reviews import prepare_reviews
from experiments.review_annotations import _preflight, _read_json
from tracemem.annotations import _canonical_json, validate_responses
from tracemem.scenarios import load_scenarios

_LIMITATIONS = [
    "Tags describe whole-scenario membership; they do not identify "
    "which phenomenon caused a label.",
    "Tags overlap, so tag-cell case and origin counts must not be summed as corpus totals.",
    "Small per-tag samples are descriptive and do not establish stable rates or model superiority.",
    "Expected claim counts are not denominators; these results do not imply recall or precision.",
    "Reviewer identity and kind are self-attested; no independent human truth is established.",
    "This retrospective evidence is not semantic error attribution or an acceptance rule.",
]


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _validated_rows(
    bundle: Mapping[str, Any],
    answer_key: Mapping[str, Any],
    provenance: Mapping[str, Any],
    integrity: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    model_keys: Sequence[str],
) -> dict[tuple[str, str], dict[str, Any]]:
    task_ids = [row["task_id"] for row in bundle["tasks"]]
    answer_rows = answer_key["task_keys"]
    task_cases = {row["task_id"]: row["case_id"] for row in answer_rows}
    if len(task_ids) != len(set(task_ids)) or set(task_ids) != set(task_cases):
        raise ValueError("bundle and answer-key task mappings do not match")
    if len(task_cases) != len(answer_rows):
        raise ValueError("answer key contains duplicate task mappings")

    expected: dict[tuple[str, str, int], str] = {}
    mapped_tasks: set[str] = set()
    for task in provenance["task_provenance"]:
        task_id, case_id = task["task_id"], task["case_id"]
        if task_id not in task_cases or task_id in mapped_tasks or task_cases[task_id] != case_id:
            raise ValueError("task provenance mapping is missing, extra, or contradictory")
        mapped_tasks.add(task_id)
        if not task["origins"]:
            raise ValueError("task provenance has no origin mapping")
        for origin in task["origins"]:
            key = (origin["model_key"], case_id, origin["claim_index"])
            if key in expected:
                raise ValueError("task provenance contains a duplicate origin mapping")
            expected[key] = task_id
    if mapped_tasks != set(task_ids):
        raise ValueError("task provenance does not cover every review task")

    audit: dict[tuple[str, str, int], str] = {}
    for origin in integrity["origin_audit"]:
        key = (origin["model_key"], origin["case_id"], origin["claim_index"])
        if key in audit:
            raise ValueError("integrity report contains a duplicate origin mapping")
        audit[key] = origin["task_id"]
    if audit != expected:
        raise ValueError("integrity and provenance origin mappings do not match")

    case_ids = {case["case_id"] for case in cases}
    output_rows = provenance["outputs"]
    outputs = {(row["model_key"], row["case_id"]): row for row in output_rows}
    expected_outputs = {(model, case_id) for model in model_keys for case_id in case_ids}
    if len(outputs) != len(output_rows) or set(outputs) != expected_outputs:
        raise ValueError("receipt output provenance mapping is incomplete or contradictory")
    return outputs


def summarize_phenomena(
    receipt_bytes_sequence: Sequence[bytes],
    response_documents: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    """Return a retrospective source-tag breakdown of validated origin judgments."""
    bundle, answer_key, provenance = prepare_reviews(receipt_bytes_sequence, seed=seed)
    integrity = evaluate_integrity_selectivity(
        receipt_bytes_sequence, response_documents, seed=seed
    )
    cases, _, dataset_sha256 = load_scenarios()
    model_keys = sorted(integrity["by_model"])
    if integrity["dataset_sha256"] != dataset_sha256 or bundle["dataset_sha256"] != dataset_sha256:
        raise ValueError("dataset hash differs between validated scenario reports")
    if integrity["bundle_sha256"] != bundle["bundle_sha256"] or integrity[
        "bundle_canonical_sha256"
    ] != _canonical_hash(bundle):
        raise ValueError("bundle hashes differ between validated scenario reports")
    if (
        integrity["provenance_sha256"] != provenance["provenance_sha256"]
        or integrity["task_mapping_sha256"] != provenance["task_mapping_sha256"]
    ):
        raise ValueError("provenance mapping hashes differ between validated scenario reports")

    outputs = _validated_rows(bundle, answer_key, provenance, integrity, cases, model_keys)
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
            }
        )
    reviewers.sort(key=lambda row: (row["reviewer_id"], row["reviewer_kind"]))

    cases_by_tag: dict[tuple[str, str], set[str]] = defaultdict(set)
    for case in cases:
        for tag in set(case["tags"]):
            cases_by_tag[(case["split"], tag)].add(case["case_id"])

    by_model: dict[str, Any] = {}
    for model_key in model_keys:
        by_model[model_key] = {}
        for split in ("dev", "test"):
            by_tag: dict[str, Any] = {}
            tags = sorted(tag for tag_split, tag in cases_by_tag if tag_split == split)
            for tag in tags:
                tagged_cases = cases_by_tag[(split, tag)]
                cell: dict[str, Any] = {
                    "source_case_count": len(tagged_cases),
                    "parse_excluded_output_count": 0,
                    "schema_excluded_output_count": 0,
                    "anchoring_excluded_output_count": 0,
                    "schema_valid_empty_output_count": 0,
                    "schema_valid_nonempty_output_count": 0,
                    "schema_valid_origin_count": 0,
                    "unique_review_task_count": 0,
                    "reviewers": [],
                }
                for case_id in tagged_cases:
                    row = outputs[(model_key, case_id)]
                    if row["raw_status"] == "parse_invalid":
                        cell["parse_excluded_output_count"] += 1
                    elif row["raw_status"] == "schema_invalid":
                        cell["schema_excluded_output_count"] += 1
                    elif row["anchored_status"] == "excluded":
                        cell["anchoring_excluded_output_count"] += 1
                    elif row["candidate_count"] == 0:
                        cell["schema_valid_empty_output_count"] += 1
                    else:
                        cell["schema_valid_nonempty_output_count"] += 1

                origins = [
                    row
                    for row in integrity["origin_audit"]
                    if row["model_key"] == model_key and row["case_id"] in tagged_cases
                ]
                cell["schema_valid_origin_count"] = len(origins)
                cell["unique_review_task_count"] = len({row["task_id"] for row in origins})
                for reviewer in reviewers:
                    counts: Counter[str] = Counter()
                    for origin in origins:
                        label = reviewer["labels"].get(origin["task_id"])
                        if label is None:
                            raise ValueError(
                                "review response mapping omits a validated origin task"
                            )
                        counts[label] += 1
                    decisive = counts["supported"] + counts["unsupported"]
                    reviewed = decisive + counts["uncertain"]
                    cell["reviewers"].append(
                        {
                            "reviewer_id": reviewer["reviewer_id"],
                            "reviewer_kind": reviewer["reviewer_kind"],
                            "supported_origin_count": counts["supported"],
                            "unsupported_origin_count": counts["unsupported"],
                            "uncertain_origin_count": counts["uncertain"],
                            "decisive_origin_count": decisive,
                            "supported_rate_all_reviewed_origins": _ratio(
                                counts["supported"], reviewed
                            ),
                            "supported_rate_decisive_origins": _ratio(
                                counts["supported"], decisive
                            ),
                        }
                    )
                by_tag[tag] = cell
            by_model[model_key][split] = {"by_tag": by_tag}

    return {
        "schema_version": 1,
        "seed": seed,
        "dataset_sha256": dataset_sha256,
        "inputs": integrity["inputs"],
        "bundle_sha256": bundle["bundle_sha256"],
        "bundle_canonical_sha256": integrity["bundle_canonical_sha256"],
        "provenance_sha256": provenance["provenance_sha256"],
        "task_mapping_sha256": provenance["task_mapping_sha256"],
        "tag_definition": (
            "A case belongs to each distinct agent-authored source tag on that scenario."
        ),
        "limitations": _LIMITATIONS,
        "by_model": by_model,
    }


def _write_json(path: Path, value: Any) -> None:
    """Create a private JSON file exclusively and clean up only our partial file."""
    _preflight((path,), ())
    descriptor = -1
    created: tuple[int, int] | None = None
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.fchmod(descriptor, 0o600)
        info = os.fstat(descriptor)
        created = (info.st_dev, info.st_ino)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except OSError:
        if descriptor >= 0:
            os.close(descriptor)
        if created is not None:
            try:
                info = path.lstat()
                if (info.st_dev, info.st_ino) == created:
                    path.unlink()
            except FileNotFoundError:
                pass
        raise ValueError("cannot create report output") from None


def main(argv: Sequence[str] | None = None) -> int:
    """Write an offline tag report from existing receipt and response JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", required=True, nargs="+", type=Path)
    parser.add_argument("--responses", nargs="*", type=Path, default=[])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        output = _preflight((args.output,), (*args.receipts, *args.responses))[0]
        receipts = [path.read_bytes() for path in args.receipts]
        responses = [_read_json(path) for path in args.responses]
        _write_json(output, summarize_phenomena(receipts, responses, seed=args.seed))
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"summarize_scenario_phenomena: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
