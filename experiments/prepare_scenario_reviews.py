"""Prepare one blinded, local review bundle from validated scenario receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from experiments import evaluate_scenarios
from experiments.review_annotations import _preflight, _write_many
from tracemem.annotations import _blind_content, _canonical_json, export_review_bundle
from tracemem.challenge import _reject_constant, _unique_object
from tracemem.claims import parse_candidates
from tracemem.scenarios import load_scenarios


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_hash(value: object) -> str:
    return _sha256(_canonical_json(value).encode("utf-8"))


def _parse_anchored_output(value: object) -> list[dict[str, Any]]:
    if type(value) is not str:
        raise ValueError("anchored model output must be text")
    try:
        payload = json.loads(
            value,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, ValueError, RecursionError):
        raise ValueError("anchored output is not valid strict JSON") from None
    parse_candidates(payload)
    assert type(payload) is dict
    return deepcopy(payload["claims"])


def prepare_reviews(
    receipt_bytes_sequence: Sequence[bytes], *, seed: int
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate receipts and return a blind bundle, answer key, and source provenance."""
    if (
        not isinstance(receipt_bytes_sequence, Sequence)
        or isinstance(receipt_bytes_sequence, (str, bytes, bytearray))
        or not receipt_bytes_sequence
    ):
        raise TypeError("receipt_bytes_sequence must be a non-empty sequence of raw receipts")

    cases, manifest, fixture_sha256 = load_scenarios()
    receipt_rows: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    unlabeled_by_case: dict[str, list[dict[str, Any]]] = {case["case_id"]: [] for case in cases}
    visible_origins: dict[str, tuple[str, list[dict[str, Any]]]] = {}
    seen_candidates: dict[str, set[str]] = {case["case_id"]: set() for case in cases}
    seen_receipts: set[str] = set()
    seen_models: set[str] = set()

    for receipt_index, receipt_bytes in enumerate(receipt_bytes_sequence):
        if type(receipt_bytes) is not bytes:
            raise TypeError(f"receipt {receipt_index} must be provided as immutable raw bytes")
        report = evaluate_scenarios.evaluate_receipt(receipt_bytes)
        if report["input_sha256"] != fixture_sha256:
            raise ValueError("validated receipt fixture hash changed during review preparation")
        model_key = report["model"]["key"]
        receipt_sha256 = _sha256(receipt_bytes)
        if receipt_sha256 in seen_receipts:
            raise ValueError("duplicate receipt SHA-256 input")
        if model_key in seen_models:
            raise ValueError(f"more than one receipt for model {model_key}")
        seen_receipts.add(receipt_sha256)
        seen_models.add(model_key)
        receipt_rows.append(
            {
                "receipt_index": receipt_index,
                "receipt_sha256": receipt_sha256,
                "model_key": model_key,
            }
        )

        for case in cases:
            case_id = case["case_id"]
            diagnostic = report["output_diagnostics"][case_id]
            state = diagnostic["status"]
            if state not in {"valid_schema", "repaired", "parse_invalid", "schema_invalid"}:
                raise ValueError(f"receipt {receipt_index} has an unsupported output diagnostic")
            raw_status = state if state in {"parse_invalid", "schema_invalid"} else "valid_schema"
            anchored_status = "unchanged" if state == "valid_schema" else state
            if state in {"parse_invalid", "schema_invalid"}:
                anchored_status = "excluded"
            row: dict[str, Any] = {
                "receipt_index": receipt_index,
                "receipt_sha256": receipt_sha256,
                "model_key": model_key,
                "case_id": case_id,
                "raw_status": raw_status,
                "anchored_status": anchored_status,
            }
            if state in {"parse_invalid", "schema_invalid"}:
                row.update(anchored_status="excluded", candidate_count=0)
                output_rows.append(row)
                exclusions.append({**row, "reason": diagnostic["raw_failure"]})
                continue

            try:
                candidates = _parse_anchored_output(report["anchored_outputs"][case_id])
            except (TypeError, ValueError):
                row.update(anchored_status="excluded", candidate_count=0)
                output_rows.append(row)
                exclusions.append({**row, "reason": "anchored_candidate_invalid"})
                continue

            row["candidate_count"] = len(candidates)
            output_rows.append(row)
            for claim_index, candidate in enumerate(candidates):
                candidate_key = _canonical_json(candidate)
                messages, blinded_candidate = _blind_content(case["messages"], candidate)
                visible_key = _canonical_json(
                    {"messages": messages, "candidate": blinded_candidate}
                )
                origins = visible_origins.setdefault(visible_key, (case_id, []))[1]
                origins.append(
                    {
                        "receipt_index": receipt_index,
                        "receipt_sha256": receipt_sha256,
                        "model_key": model_key,
                        "claim_index": claim_index,
                        "raw_status": raw_status,
                        "anchored_status": anchored_status,
                    }
                )
                if candidate_key not in seen_candidates[case_id]:
                    seen_candidates[case_id].add(candidate_key)
                    unlabeled_by_case[case_id].append(candidate)

    cases_without_authored_targets = deepcopy(cases)
    for case in cases_without_authored_targets:
        case["expected"] = []
    bundle, answer_key = export_review_bundle(
        cases_without_authored_targets,
        fixture_sha256,
        seed=seed,
        case_metadata=manifest["case_metadata"],
        unlabeled_by_case=unlabeled_by_case,
    )

    task_case_ids = {row["task_id"]: row["case_id"] for row in answer_key["task_keys"]}
    task_provenance: list[dict[str, Any]] = []
    for task in bundle["tasks"]:
        task_id = task["task_id"]
        case_id = task_case_ids[task_id]
        content_key = _canonical_json(
            {"messages": task["messages"], "candidate": task["candidate"]}
        )
        matched = visible_origins.get(content_key)
        if matched is None or matched[0] != case_id:
            raise ValueError("blind task has no receipt provenance mapping")
        task_provenance.append({"task_id": task_id, "case_id": case_id, "origins": matched[1]})
    task_provenance.sort(key=lambda row: row["task_id"])

    mapping_hash = _canonical_hash(
        {"bundle_sha256": bundle["bundle_sha256"], "task_provenance": task_provenance}
    )
    provenance: dict[str, Any] = {
        "schema_version": 1,
        "mode": "scenario_review_provenance",
        "fixture_sha256": fixture_sha256,
        "bundle_sha256": bundle["bundle_sha256"],
        "answer_key_sha256": _canonical_hash(answer_key),
        "interpretation": (
            "Generated candidates are unlabeled review inputs. Exact quote anchoring may repair "
            "offsets but does not establish raw citation integrity or semantic support; "
            "no semantic judgment is inferred."
        ),
        "receipts": receipt_rows,
        "outputs": output_rows,
        "exclusions": exclusions,
        "task_provenance": task_provenance,
        "task_mapping_sha256": mapping_hash,
    }
    provenance["provenance_sha256"] = _canonical_hash(provenance)
    return bundle, answer_key, provenance


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", required=True, nargs="+", type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--answer-key", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args(argv)
    outputs = (args.bundle, args.answer_key, args.provenance)
    try:
        paths = _preflight(outputs, args.receipts)
        receipts = [path.read_bytes() for path in args.receipts]
        bundle, answer_key, provenance = prepare_reviews(receipts, seed=args.seed)
        _write_many(paths, (bundle, answer_key, provenance))
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"prepare_scenario_reviews: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
