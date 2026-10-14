"""Compare retrospective evidence-integrity filters with blinded support labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.evaluate_scenarios import evaluate_receipt
from experiments.prepare_scenario_reviews import prepare_reviews
from experiments.review_annotations import _preflight, _read_json, _write_many
from experiments.summarize_scenario_reviews import summarize_reviews
from tracemem.anchoring import anchor_unique_quotes
from tracemem.annotations import _canonical_json, validate_responses
from tracemem.challenge import _case_records, _reject_constant, _unique_object
from tracemem.claims import Claim, parse_candidates, verify_candidate
from tracemem.scenarios import load_scenarios

_FILTER_NAMES = ("all_schema_valid", "raw_integrity", "anchored_integrity")
_FILTER_DEFINITIONS = {
    "all_schema_valid": (
        "Every retrospectively selected schema-valid generated claim origin, regardless of "
        "evidence-integrity result."
    ),
    "raw_integrity": (
        "A schema-valid raw claim passes when verify_candidate returns REVIEW for its original "
        "evidence offsets."
    ),
    "anchored_integrity": (
        "A schema-valid claim passes when unique exact-quote anchoring is applied and "
        "verify_candidate returns REVIEW."
    ),
}
_LIMITATIONS = (
    "Filters are applied retrospectively only to schema-valid origins selected for review.",
    "REVIEW means evidence-integrity checks passed; it is not semantic acceptance or truth.",
    "Review labels are self-attested; reviewer identity and independence are unverified.",
    "Candidate-denominator coverage is not whole-output success, recall, or precision.",
    "No calibration or precision guarantee, Gate A result, prospective test, runtime acceptance "
    "policy, or model superiority follows from this report.",
    "Unique quote anchoring changes offsets only and does not establish entailment.",
)


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _parse_claims(output: object) -> tuple[Claim, ...]:
    if type(output) is not str:
        raise ValueError("validated receipt output must be text")
    try:
        payload = json.loads(
            output,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        return parse_candidates(payload)
    except (json.JSONDecodeError, RecursionError, TypeError, ValueError):
        raise ValueError("validated receipt candidate output changed during analysis") from None


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _error_counts(origins: Sequence[dict[str, Any]], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for origin in origins:
        counts.update(origin[key])
    return dict(sorted(counts.items()))


def _filter_cell(
    origins: Sequence[dict[str, Any]],
    reviewer_labels: Sequence[dict[str, Any]],
    *,
    filter_name: str,
) -> dict[str, Any]:
    def passes(origin: dict[str, Any]) -> bool:
        if filter_name == "all_schema_valid":
            return True
        if filter_name == "raw_integrity":
            return origin["raw_integrity_passes"] is True
        return origin["anchored_integrity_passes"] is True

    passed = [origin for origin in origins if passes(origin)]
    reviewers: list[dict[str, Any]] = []
    for reviewer in reviewer_labels:
        label_counts = {
            state: {"passed": 0, "rejected": 0}
            for state in ("supported", "unsupported", "uncertain")
        }
        for origin in origins:
            label = reviewer["labels"][origin["task_id"]]
            disposition = "passed" if passes(origin) else "rejected"
            label_counts[label][disposition] += 1

        passed_supported = label_counts["supported"]["passed"]
        passed_unsupported = label_counts["unsupported"]["passed"]
        passed_uncertain = label_counts["uncertain"]["passed"]
        all_reviewed_supported = sum(label_counts["supported"].values())
        reviewers.append(
            {
                "reviewer_id": reviewer["reviewer_id"],
                "reviewer_kind": reviewer["reviewer_kind"],
                "all_reviewed_supported_origin_count": all_reviewed_supported,
                "all_reviewed_unsupported_origin_count": sum(label_counts["unsupported"].values()),
                "all_reviewed_uncertain_origin_count": sum(label_counts["uncertain"].values()),
                "passed_supported_origin_count": passed_supported,
                "passed_unsupported_origin_count": passed_unsupported,
                "passed_uncertain_origin_count": passed_uncertain,
                "rejected_supported_origin_count": label_counts["supported"]["rejected"],
                "rejected_unsupported_origin_count": label_counts["unsupported"]["rejected"],
                "rejected_uncertain_origin_count": label_counts["uncertain"]["rejected"],
                "supported_rate_all_passed_origins": _ratio(passed_supported, len(passed)),
                "supported_rate_decisive_passed_origins": _ratio(
                    passed_supported, passed_supported + passed_unsupported
                ),
                "supported_retention_rate": _ratio(passed_supported, all_reviewed_supported),
            }
        )
    return {
        "passed_origin_count": len(passed),
        "rejected_origin_count": len(origins) - len(passed),
        "coverage": _ratio(len(passed), len(origins)),
        "reviewers": reviewers,
    }


def evaluate_integrity_selectivity(
    receipt_bytes_sequence: Sequence[bytes],
    response_documents: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    """Compare three integrity filters over reviewed schema-valid claim origins."""
    if type(seed) is not int:
        raise TypeError("seed must be an integer")

    bundle, answer_key, provenance = prepare_reviews(receipt_bytes_sequence, seed=seed)
    accounting = summarize_reviews(receipt_bytes_sequence, response_documents, seed=seed)
    accounting["exclusions"] = provenance["exclusions"]

    cases, _, dataset_sha256 = load_scenarios()
    split_by_case = {case["case_id"]: case["split"] for case in cases}
    receipt_rows = provenance["receipts"]
    output_rows = {(row["receipt_index"], row["case_id"]): row for row in provenance["outputs"]}
    if (
        len(receipt_rows) != len(receipt_bytes_sequence)
        or any(row["receipt_index"] != index for index, row in enumerate(receipt_rows))
        or len(output_rows) != len(provenance["outputs"])
        or len(output_rows) != len(receipt_rows) * len(cases)
    ):
        raise ValueError("receipt and output provenance mappings are incomplete")
    origins_by_key: dict[tuple[int, str, str, int], dict[str, Any]] = {}
    for receipt_index, receipt_bytes in enumerate(receipt_bytes_sequence):
        evaluation = evaluate_receipt(receipt_bytes)
        model_key = evaluation["model"]["key"]
        if model_key != receipt_rows[receipt_index]["model_key"]:
            raise ValueError("receipt model mapping changed during analysis")
        for case in cases:
            case_id = case["case_id"]
            output = output_rows.get((receipt_index, case_id))
            if output is None or output["model_key"] != model_key:
                raise ValueError("receipt output mapping is incomplete")
            diagnostic = evaluation["output_diagnostics"][case_id]
            raw_status = output["raw_status"]
            anchored_status = output["anchored_status"]
            if raw_status != "valid_schema":
                expected_status = {
                    "parse_invalid": "parse_invalid",
                    "schema_invalid": "schema_invalid",
                }.get(raw_status)
                if expected_status is None or diagnostic["status"] != expected_status:
                    raise ValueError("receipt exclusion status changed during analysis")
                if output.get("candidate_count") != 0:
                    raise ValueError("excluded output unexpectedly contains candidates")
                continue

            if anchored_status == "excluded":
                if diagnostic["status"] not in {"valid_schema", "repaired"}:
                    raise ValueError("anchoring exclusion does not match the validated receipt")
                if output.get("candidate_count") != 0:
                    raise ValueError("anchoring-excluded output unexpectedly contains candidates")
                continue

            expected_anchored_status = {
                "unchanged": "valid_schema",
                "repaired": "repaired",
            }.get(anchored_status)
            if expected_anchored_status is None or diagnostic["status"] != expected_anchored_status:
                raise ValueError("anchored output status changed during analysis")
            raw_claims = _parse_claims(evaluation["raw_outputs"][case_id])
            anchored_claims = _parse_claims(evaluation["anchored_outputs"][case_id])
            candidate_count = output.get("candidate_count")
            if (
                type(candidate_count) is not int
                or candidate_count != len(raw_claims)
                or candidate_count != len(anchored_claims)
            ):
                raise ValueError("raw and anchored candidate counts do not match provenance")
            records = _case_records(case)
            for claim_index, (raw_claim, anchored_claim) in enumerate(
                zip(raw_claims, anchored_claims, strict=True)
            ):
                expected_anchored_claim = anchor_unique_quotes(raw_claim, records).claim
                if expected_anchored_claim != anchored_claim:
                    raise ValueError("anchoring changed claim content or candidate order")
                raw_result = verify_candidate(raw_claim, records)
                anchored_result = verify_candidate(anchored_claim, records)
                origin_key = (receipt_index, model_key, case_id, claim_index)
                if origin_key in origins_by_key:
                    raise ValueError("duplicate schema-valid origin")
                origins_by_key[origin_key] = {
                    "receipt_index": receipt_index,
                    "model_key": model_key,
                    "case_id": case_id,
                    "claim_index": claim_index,
                    "raw_integrity_passes": raw_result.status == "REVIEW",
                    "anchored_integrity_passes": anchored_result.status == "REVIEW",
                    "raw_error_codes": list(raw_result.errors),
                    "anchored_error_codes": list(anchored_result.errors),
                }

    expected_origin_keys = set(origins_by_key)
    task_ids = {task["task_id"] for task in bundle["tasks"]}
    if len(task_ids) != len(bundle["tasks"]):
        raise ValueError("blind bundle contains duplicate review tasks")
    answer_task_ids = {row["task_id"] for row in answer_key["task_keys"]}
    if task_ids != answer_task_ids:
        raise ValueError("blind bundle and answer key task mappings do not match")
    answer_case_by_task = {row["task_id"]: row["case_id"] for row in answer_key["task_keys"]}
    if len(answer_case_by_task) != len(answer_key["task_keys"]):
        raise ValueError("answer key contains duplicate task mappings")
    origin_to_task: dict[tuple[int, str, str, int], str] = {}
    mapped_task_ids: set[str] = set()
    for task_row in provenance["task_provenance"]:
        task_id, task_case_id = task_row["task_id"], task_row["case_id"]
        if task_id not in task_ids or task_id in mapped_task_ids:
            raise ValueError("review task provenance is missing, extra, or duplicated")
        mapped_task_ids.add(task_id)
        if answer_case_by_task.get(task_id) != task_case_id:
            raise ValueError("answer key and provenance task cases do not match")
        if not task_row["origins"]:
            raise ValueError("review task has no origin mapping")
        for origin in task_row["origins"]:
            key = (
                origin["receipt_index"],
                origin["model_key"],
                task_case_id,
                origin["claim_index"],
            )
            if key not in expected_origin_keys or key in origin_to_task:
                raise ValueError("review task provenance contains an extra or duplicate origin")
            origin_to_task[key] = task_id
    if mapped_task_ids != task_ids or set(origin_to_task) != expected_origin_keys:
        raise ValueError("review task provenance does not cover every claim origin")

    reviewer_labels: list[dict[str, Any]] = []
    for document in response_documents:
        rows = validate_responses(bundle, document)
        if not rows:
            raise ValueError("response document has no tasks to identify its reviewer")
        reviewer_labels.append(
            {
                "reviewer_id": rows[0]["reviewer_id"],
                "reviewer_kind": rows[0]["reviewer_kind"],
                "labels": {row["task_id"]: row["label"] for row in rows},
            }
        )
    reviewer_labels.sort(key=lambda row: (row["reviewer_id"], row["reviewer_kind"]))

    origin_audit: list[dict[str, Any]] = []
    for key in sorted(origins_by_key, key=lambda item: (item[1], item[2], item[3])):
        origin = origins_by_key[key]
        origin["task_id"] = origin_to_task[key]
        origin_audit.append(
            {
                "model_key": origin["model_key"],
                "case_id": origin["case_id"],
                "claim_index": origin["claim_index"],
                "task_id": origin["task_id"],
                "raw_integrity_passes": origin["raw_integrity_passes"],
                "raw_error_codes": origin["raw_error_codes"],
                "anchored_integrity_passes": origin["anchored_integrity_passes"],
                "anchored_error_codes": origin["anchored_error_codes"],
            }
        )

    by_model: dict[str, dict[str, Any]] = {}
    model_keys = sorted(row["model_key"] for row in receipt_rows)
    for model_key in model_keys:
        by_model[model_key] = {}
        for split in ("dev", "test"):
            cell_origins = [
                origin
                for origin in origins_by_key.values()
                if origin["model_key"] == model_key and split_by_case[origin["case_id"]] == split
            ]
            by_model[model_key][split] = {
                "schema_valid_origin_count": len(cell_origins),
                "filters": {
                    filter_name: _filter_cell(
                        cell_origins, reviewer_labels, filter_name=filter_name
                    )
                    for filter_name in _FILTER_NAMES
                },
                "verifier_error_code_counts": {
                    "raw": _error_counts(cell_origins, "raw_error_codes"),
                    "anchored": _error_counts(cell_origins, "anchored_error_codes"),
                },
            }

    receipt_hashes = []
    for index, receipt_bytes in enumerate(receipt_bytes_sequence):
        receipt_document = json.loads(
            receipt_bytes.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        receipt_hashes.append(
            {
                "model_key": receipt_rows[index]["model_key"],
                "sha256": hashlib.sha256(receipt_bytes).hexdigest(),
                "canonical_sha256": _canonical_hash(receipt_document),
            }
        )

    return {
        "schema_version": 1,
        "seed": seed,
        "dataset_sha256": dataset_sha256,
        "bundle_sha256": bundle["bundle_sha256"],
        "bundle_canonical_sha256": _canonical_hash(bundle),
        "provenance_sha256": provenance["provenance_sha256"],
        "task_mapping_sha256": provenance["task_mapping_sha256"],
        "inputs": {
            "receipts": receipt_hashes,
            "responses": accounting["responses"],
        },
        "accounting": accounting,
        "filter_definitions": dict(_FILTER_DEFINITIONS),
        "limitations": _LIMITATIONS,
        "by_model": by_model,
        "origin_audit": origin_audit,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the offline integrity-selectivity report command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", required=True, nargs="+", type=Path)
    parser.add_argument("--responses", nargs="*", type=Path, default=[])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    inputs = (*arguments.receipts, *arguments.responses)
    try:
        outputs = _preflight((arguments.output,), inputs)
        receipt_bytes = [path.read_bytes() for path in arguments.receipts]
        response_documents = [_read_json(path) for path in arguments.responses]
        report = evaluate_integrity_selectivity(
            receipt_bytes,
            response_documents,
            seed=arguments.seed,
        )
        _write_many(outputs, (report,))
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"evaluate_integrity_selectivity: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
