"""Strict two-axis scorer for synthetic verifier-pair experiments."""

from __future__ import annotations

import json
from typing import Any

from training.verifier_data import validate_verifier_dataset

_SUPPORT_LABELS = {"supported", "unsupported", "uncertain"}
_ELIGIBILITY_LABELS = {"eligible", "ineligible", "uncertain"}


def parse_verdict(raw: str) -> dict[str, str]:
    """Parse one verdict object with exactly the two string-valued axes."""
    if type(raw) is not str:
        raise ValueError("verdict must be JSON text")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite,
        )
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("verdict must be valid JSON") from exc
    if type(value) is not dict or set(value) != {"support", "eligibility"}:
        raise ValueError("verdict must contain exactly support and eligibility")
    support, eligibility = value["support"], value["eligibility"]
    if type(support) is not str or support not in _SUPPORT_LABELS:
        raise ValueError("support must be supported, unsupported, or uncertain")
    if type(eligibility) is not str or eligibility not in _ELIGIBILITY_LABELS:
        raise ValueError("eligibility must be eligible, ineligible, or uncertain")
    return {"support": support, "eligibility": eligibility}


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def score_outputs(dataset: object, split: str, outputs: list[dict[str, Any]]) -> dict[str, Any]:
    """Score one complete dev or test run against the authored pair dataset."""
    cases = validate_verifier_dataset(dataset)
    if type(split) is not str or split not in {"dev", "test"}:
        raise ValueError("split must be dev or test")
    selected = [case for case in cases if case["split"] == split]
    if not selected:
        raise ValueError(f"dataset has no {split} cases")
    expected_ids = {
        f"{case['case_id']}::{candidate['candidate_id']}"
        for case in selected
        for candidate in case["candidates"]
    }
    if len(expected_ids) != sum(len(case["candidates"]) for case in selected):
        raise ValueError("dataset candidate item IDs are ambiguous")
    if type(outputs) is not list:
        raise ValueError("outputs must be a list")
    output_ids: list[str] = []
    for row in outputs:
        if type(row) is not dict or type(row.get("item_id")) is not str:
            raise ValueError("every output must have a string item_id")
        output_ids.append(row["item_id"])
    if len(output_ids) != len(set(output_ids)) or set(output_ids) != expected_ids:
        raise ValueError("output IDs must match split candidates exactly one-to-one")

    output_by_id = {row["item_id"]: row for row in outputs}
    candidate_results: list[dict[str, Any]] = []
    for case in selected:
        for candidate in case["candidates"]:
            item_id = f"{case['case_id']}::{candidate['candidate_id']}"
            row = output_by_id[item_id]
            raw, reason = row.get("raw_output"), row.get("finish_reason")
            if type(raw) is not str:
                raise ValueError(f"output {item_id} raw_output must be text")
            if type(reason) is not str or reason not in {"eos", "length", "unknown", "empty"}:
                raise ValueError(f"output {item_id} has invalid finish_reason")
            tokens = row.get("output_tokens")
            if "output_tokens" in row:
                if type(tokens) is not int or tokens < 0:
                    raise ValueError(
                        f"output {item_id} output_tokens must be a non-negative integer"
                    )
            else:
                tokens = None
            try:
                verdict = parse_verdict(raw)
            except ValueError:
                verdict = None
            is_valid = verdict is not None and reason == "eos"
            candidate_results.append(
                {
                    "item_id": item_id,
                    "case_id": case["case_id"],
                    "candidate_id": candidate["candidate_id"],
                    "phenomenon": case["phenomenon"],
                    "reference": {
                        "support": candidate["support"],
                        "eligibility": candidate["eligibility"],
                    },
                    "verdict": verdict,
                    "valid": is_valid,
                    "malformed": verdict is None,
                    "finish_reason": reason,
                    "output_tokens": tokens,
                }
            )

    pair_results = _pair_results(candidate_results)
    aggregate = _summarize(candidate_results, pair_results)
    aggregate["per_phenomenon"] = {}
    for phenomenon in sorted({row["phenomenon"] for row in candidate_results}):
        rows = [row for row in candidate_results if row["phenomenon"] == phenomenon]
        pairs = [pair for pair in pair_results if pair["phenomenon"] == phenomenon]
        aggregate["per_phenomenon"][phenomenon] = _summarize(rows, pairs)

    control_rows = [
        {
            **row,
            "verdict": {"support": "supported", "eligibility": "eligible"},
            "valid": True,
            "malformed": False,
            "finish_reason": "control",
            "output_tokens": None,
        }
        for row in candidate_results
    ]
    control_pairs = _pair_results(control_rows)
    no_filter = _summarize(control_rows, control_pairs, include_validity=False)
    aggregate["no_filter_control"] = {
        "policy": "predict supported and eligible for every candidate; retain all candidates",
        "retained_candidate_count": len(control_rows),
        "retention_rate": 1.0,
        **no_filter,
    }
    return {
        "split": split,
        "candidate_results": candidate_results,
        "pair_results": pair_results,
        "aggregate": aggregate,
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _validity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    finish_reasons = {reason: 0 for reason in ("eos", "length", "unknown", "empty")}
    for row in rows:
        finish_reasons[row["finish_reason"]] += 1
    valid = sum(row["valid"] for row in rows)
    return {
        "total": len(rows),
        "valid": valid,
        "invalid": len(rows) - valid,
        "rate": _rate(valid, len(rows)),
        "malformed": sum(row["malformed"] for row in rows),
        "truncated": finish_reasons["length"],
        "finish_reasons": finish_reasons,
    }


def _axis_metrics(
    rows: list[dict[str, Any]], axis: str, labels: tuple[str, str, str]
) -> dict[str, Any]:
    predictions = (*labels, "invalid")
    confusion = {reference: {prediction: 0 for prediction in predictions} for reference in labels}
    for row in rows:
        reference = row["reference"][axis]
        prediction = row["verdict"][axis] if row["valid"] else "invalid"
        confusion[reference][prediction] += 1

    decisive = labels[:2]
    decisive_rows = [row for row in rows if row["reference"][axis] in decisive]
    decisive_correct = sum(
        row["valid"] and row["verdict"][axis] == row["reference"][axis] for row in decisive_rows
    )
    uncertain_rows = [row for row in rows if row["reference"][axis] == "uncertain"]
    uncertain_predictions = {
        prediction: confusion["uncertain"][prediction] for prediction in predictions
    }
    uncertain = {
        "total": len(uncertain_rows),
        "exact_correct": confusion["uncertain"]["uncertain"],
        "prediction_counts": uncertain_predictions,
    }

    by_reference: dict[str, dict[str, Any]]
    if axis == "support":
        by_reference = {
            "supported": {
                "total": len([r for r in rows if r["reference"][axis] == "supported"]),
                "retained": confusion["supported"]["supported"],
            },
            "unsupported": {
                "total": len([r for r in rows if r["reference"][axis] == "unsupported"]),
                "caught": confusion["unsupported"]["unsupported"],
            },
            "uncertain": uncertain,
        }
        by_reference["supported"]["retention_rate"] = _rate(
            by_reference["supported"]["retained"], by_reference["supported"]["total"]
        )
        by_reference["unsupported"]["catch_rate"] = _rate(
            by_reference["unsupported"]["caught"], by_reference["unsupported"]["total"]
        )
    else:
        by_reference = {
            reference: {
                "total": sum(row["reference"][axis] == reference for row in rows),
                "correct": confusion[reference][reference],
            }
            for reference in decisive
        }
        for reference in decisive:
            by_reference[reference]["accuracy"] = _rate(
                by_reference[reference]["correct"], by_reference[reference]["total"]
            )
        by_reference["uncertain"] = uncertain

    return {
        "by_reference": by_reference,
        "confusion": confusion,
        "decisive_exact_correct": decisive_correct,
        "decisive_total": len(decisive_rows),
        "decisive_accuracy": _rate(decisive_correct, len(decisive_rows)),
        "uncertain": uncertain,
    }


def _pair_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["case_id"], []).append(row)
    results: list[dict[str, Any]] = []
    for case_id, pair in grouped.items():
        decisive = all(row["reference"]["support"] != "uncertain" for row in pair)
        both_correct = decisive and all(
            row["valid"] and row["verdict"]["support"] == row["reference"]["support"]
            for row in pair
        )
        results.append(
            {
                "case_id": case_id,
                "phenomenon": pair[0]["phenomenon"],
                "candidate_count": len(pair),
                "decisive": decisive,
                "both_support_labels_correct": both_correct if decisive else None,
            }
        )
    return results


def _summarize(
    rows: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    *,
    include_validity: bool = True,
) -> dict[str, Any]:
    decisive_pairs = [pair for pair in pairs if pair["decisive"]]
    correct_pairs = sum(pair["both_support_labels_correct"] is True for pair in decisive_pairs)
    result: dict[str, Any] = {
        "candidate_count": len(rows),
        "support": _axis_metrics(rows, "support", ("supported", "unsupported", "uncertain")),
        "eligibility": _axis_metrics(rows, "eligibility", ("eligible", "ineligible", "uncertain")),
        "support_pairs": {
            "total": len(pairs),
            "decisive_total": len(decisive_pairs),
            "both_support_labels_correct": correct_pairs,
            "exact_rate": _rate(correct_pairs, len(decisive_pairs)),
            "excluded_uncertain": len(pairs) - len(decisive_pairs),
        },
        "telemetry": {
            "output_tokens": sum(row["output_tokens"] or 0 for row in rows),
            "items_with_output_tokens": sum(row["output_tokens"] is not None for row in rows),
        },
    }
    if include_validity:
        result["validity"] = _validity(rows)
    return result
