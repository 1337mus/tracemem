"""Authored synthetic NLI diagnostic; scores support by the source text, not real-world truth.

The authored labels are agent-reviewed diagnostics, not independently human-validated ground truth.
This module is not a runtime acceptance gate.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from importlib.resources import files
from typing import Any, NoReturn

_CASE_FIELDS = frozenset(
    {"case_id", "scenario_id", "split", "tags", "premise", "hypothesis", "supported"}
)
_MAX_ID_CHARS = 128
_MAX_SPLIT_CHARS = 32
_MAX_TAG_CHARS = 64
_MAX_TEXT_CHARS = 2_000
_PROBABILITY_SUM_TOLERANCE = 1e-6
_THRESHOLDS = (0.5, 0.9, 0.99)
_BOOTSTRAP_DRAWS = 1_000
_BOOTSTRAP_SEED = 1_729
_CLASS_NAMES = ("contradiction", "entailment", "neutral")
_LIMITATIONS = [
    "Gold labels are authored and agent-reviewed synthetic judgments, not independent human truth.",
    "A premise supports claims about what its source says, not the real-world truth of that claim.",
    "All pairs use a fictional-record wrapper; simplified narratives may differ from "
    "natural conversation.",
    "Intervals are descriptive scenario-bootstrap intervals for this diagnostic sample only.",
    "Zero observed errors can yield [1, 1] intervals; this is not a >99% population guarantee.",
    "Probability calibration is not assessed.",
    "This diagnostic does not grant Gate A approval and never auto-ACCEPTs.",
]


def load_support_challenge() -> tuple[list[dict[str, Any]], str]:
    """Load the bundled synthetic support challenge and return its raw-file SHA-256."""
    raw = files("tracemem").joinpath("data", "support_challenge.json").read_bytes()
    try:
        payload = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"support challenge JSON is invalid: {exc}") from exc
    if type(payload) is not dict or set(payload) != {"version", "license", "cases"}:
        raise ValueError("support challenge must contain exactly version, license, and cases")
    if type(payload["version"]) is not int or payload["version"] != 1:
        raise ValueError("unsupported support challenge version")
    if payload["license"] != "MIT":
        raise ValueError("support challenge license must be MIT")
    cases = _validate_cases(payload["cases"], require_diagnostic=True)
    if len(cases) != 48 or len({case["scenario_id"] for case in cases}) != 24:
        raise ValueError("support challenge must contain 48 cases across 24 scenarios")
    return cases, hashlib.sha256(raw).hexdigest()


def score_support_cases(
    cases: Sequence[Mapping[str, Any]], outputs: Mapping[str, Sequence[float]]
) -> dict[str, Any]:
    """Score support probabilities as diagnostic data, never as runtime acceptance."""
    validated = _validate_cases(cases, require_diagnostic=True)
    if not isinstance(outputs, Mapping):
        raise TypeError("outputs must map case IDs to probability triplets")
    if any(type(case_id) is not str for case_id in outputs):
        raise TypeError("output case IDs must be strings")
    expected_ids = {case["case_id"] for case in validated}
    missing = sorted(expected_ids.difference(outputs))
    extra = sorted(set(outputs).difference(expected_ids))
    if missing:
        raise ValueError(f"missing outputs for case IDs: {', '.join(missing)}")
    if extra:
        raise ValueError(f"unexpected output IDs: {', '.join(extra)}")

    probabilities = {
        case["case_id"]: _validate_probabilities(outputs[case["case_id"]], case["case_id"])
        for case in validated
    }
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in validated:
        by_scenario[case["scenario_id"]].append(case)
    resamples = _scenario_resamples(sorted(by_scenario))
    threshold_reports: dict[str, Any] = {}
    for threshold in _THRESHOLDS:
        decisions = {
            case["case_id"]: probabilities[case["case_id"]][1] >= threshold for case in validated
        }
        entries = [(case["supported"], decisions[case["case_id"]]) for case in validated]
        metrics = _confusion_metrics(entries)
        interval, defined_draws = _bootstrap_precision(
            by_scenario, probabilities, resamples, threshold
        )
        by_tag: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
        for case in validated:
            for tag in case["tags"]:
                by_tag[tag].append((case["supported"], decisions[case["case_id"]]))
        metrics.update(
            {
                "threshold": threshold,
                "supported_precision_interval_95": interval,
                "precision_interval_defined_draws": defined_draws,
                "by_tag": {tag: _confusion_metrics(by_tag[tag]) for tag in sorted(by_tag)},
            }
        )
        threshold_reports[str(threshold)] = metrics

    case_results = []
    for case in sorted(validated, key=lambda item: item["case_id"]):
        scores = probabilities[case["case_id"]]
        case_results.append(
            {
                "case_id": case["case_id"],
                "scenario_id": case["scenario_id"],
                "split": case["split"],
                "tags": sorted(case["tags"]),
                "supported": case["supported"],
                "probabilities": dict(zip(_CLASS_NAMES, scores, strict=True)),
                "selected_at_thresholds": {
                    str(threshold): scores[1] >= threshold for threshold in _THRESHOLDS
                },
            }
        )
    return {
        "purpose": "Descriptive synthetic support diagnostic only.",
        "case_count": len(validated),
        "scenario_count": len(by_scenario),
        "probability_order": list(_CLASS_NAMES),
        "probability_sum_tolerance": _PROBABILITY_SUM_TOLERANCE,
        "bootstrap": {
            "unit": "scenario",
            "draws": _BOOTSTRAP_DRAWS,
            "seed": _BOOTSTRAP_SEED,
            "percentile_method": "linear interpolation at position fraction*(n-1)",
            "undefined_precision_draws": (
                "excluded from intervals; defined draw count reported per threshold"
            ),
        },
        "thresholds": threshold_reports,
        "case_results": case_results,
        "limitations": list(_LIMITATIONS),
        "gate_a_approval": "not_granted",
        "automatic_accept": False,
    }


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    raise ValueError(f"nonstandard JSON constant: {value}")


def _validate_cases(raw_cases: object, *, require_diagnostic: bool) -> list[dict[str, Any]]:
    if isinstance(raw_cases, (str, bytes)) or not isinstance(raw_cases, Sequence):
        raise TypeError("cases must be a non-empty sequence")
    if not raw_cases:
        raise ValueError("cases must be a non-empty sequence")
    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scenario_splits: dict[str, set[str]] = defaultdict(set)
    for raw_case in raw_cases:
        if not isinstance(raw_case, Mapping):
            raise TypeError("each support case must be a mapping")
        if set(raw_case) != _CASE_FIELDS:
            raise ValueError("support case has missing or unexpected fields")
        case_id = _required_text(raw_case["case_id"], "case_id", _MAX_ID_CHARS)
        if case_id in seen_ids:
            raise ValueError(f"duplicate case_id: {case_id}")
        seen_ids.add(case_id)
        scenario_id = _required_text(raw_case["scenario_id"], "scenario_id", _MAX_ID_CHARS)
        split = _required_text(raw_case["split"], "split", _MAX_SPLIT_CHARS)
        premise = _required_text(raw_case["premise"], "premise", _MAX_TEXT_CHARS)
        hypothesis = _required_text(raw_case["hypothesis"], "hypothesis", _MAX_TEXT_CHARS)
        if type(raw_case["supported"]) is not bool:
            raise TypeError("supported must be a boolean")
        raw_tags = raw_case["tags"]
        if type(raw_tags) is not list or not raw_tags:
            raise ValueError("case tags must be a non-empty list")
        tags = [_required_text(tag, "tag", _MAX_TAG_CHARS) for tag in raw_tags]
        if len(tags) != len(set(tags)):
            raise ValueError("case tags must be unique")
        case = {
            "case_id": case_id,
            "scenario_id": scenario_id,
            "split": split,
            "tags": tags,
            "premise": premise,
            "hypothesis": hypothesis,
            "supported": raw_case["supported"],
        }
        cases.append(case)
        by_scenario[scenario_id].append(case)
        scenario_splits[scenario_id].add(split)

    for scenario_id, pair in by_scenario.items():
        if len(scenario_splits[scenario_id]) != 1:
            raise ValueError(f"scenario appears in multiple splits: {scenario_id}")
        if len(pair) != 2 or {case["supported"] for case in pair} != {True, False}:
            raise ValueError(
                f"scenario must have one supported and one unsupported case: {scenario_id}"
            )
        if pair[0]["premise"] != pair[1]["premise"]:
            raise ValueError(f"scenario pair must share an identical premise: {scenario_id}")
    if require_diagnostic and any(case["split"] != "diagnostic" for case in cases):
        raise ValueError("support challenge scoring accepts diagnostic cases only")
    return cases


def _required_text(value: object, field: str, max_chars: int) -> str:
    if type(value) is not str:
        raise TypeError(f"{field} must be a string")
    if not value.strip():
        raise ValueError(f"{field} must be non-empty")
    if len(value) > max_chars:
        raise ValueError(f"{field} must not exceed {max_chars} characters")
    return value


def _validate_probabilities(value: object, case_id: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"probabilities for {case_id} must be a sequence")
    if len(value) != 3:
        raise ValueError(f"probabilities for {case_id} must contain exactly 3 values")
    scores: list[float] = []
    for probability in value:
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise TypeError(f"probabilities must be numbers for {case_id}")
        try:
            score = float(probability)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"probabilities must be finite for {case_id}") from exc
        if not math.isfinite(score):
            raise ValueError(f"probabilities must be finite for {case_id}")
        if not 0 <= score <= 1:
            raise ValueError(f"probabilities must be between 0 and 1 for {case_id}")
        scores.append(score)
    if not math.isclose(sum(scores), 1.0, rel_tol=0.0, abs_tol=_PROBABILITY_SUM_TOLERANCE):
        raise ValueError(f"probabilities must sum to 1 within tolerance for {case_id}")
    return (scores[0], scores[1], scores[2])


def _confusion_metrics(entries: Sequence[tuple[bool, bool]]) -> dict[str, Any]:
    tp = sum(supported and selected for supported, selected in entries)
    fp = sum(not supported and selected for supported, selected in entries)
    fn = sum(supported and not selected for supported, selected in entries)
    tn = sum(not supported and not selected for supported, selected in entries)
    selected = tp + fp
    supported_total = tp + fn
    unsupported_total = fp + tn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "selected_count": selected,
        "coverage": selected / len(entries) if entries else None,
        "supported_precision": tp / selected if selected else None,
        "supported_recall": tp / supported_total if supported_total else None,
        "unsupported_false_positive_rate": (fp / unsupported_total if unsupported_total else None),
    }


def _scenario_resamples(scenario_ids: Sequence[str]) -> list[list[str]]:
    rng = random.Random(_BOOTSTRAP_SEED)
    return [[rng.choice(scenario_ids) for _ in scenario_ids] for _ in range(_BOOTSTRAP_DRAWS)]


def _bootstrap_precision(
    by_scenario: Mapping[str, Sequence[Mapping[str, Any]]],
    probabilities: Mapping[str, tuple[float, float, float]],
    resamples: Sequence[Sequence[str]],
    threshold: float,
) -> tuple[dict[str, float] | None, int]:
    precisions: list[float] = []
    for sampled_scenarios in resamples:
        tp = selected = 0
        for scenario_id in sampled_scenarios:
            for case in by_scenario[scenario_id]:
                is_selected = probabilities[case["case_id"]][1] >= threshold
                if is_selected:
                    selected += 1
                    tp += case["supported"]
        if selected:
            precisions.append(tp / selected)
    if not precisions:
        return None, 0
    precisions.sort()
    return {
        "lower": _percentile(precisions, 0.025),
        "upper": _percentile(precisions, 0.975),
    }, len(precisions)


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    position = fraction * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


__all__ = ["load_support_challenge", "score_support_cases"]
