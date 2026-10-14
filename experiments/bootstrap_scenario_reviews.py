"""Descriptive scenario-cluster bootstrap for generated-candidate reviews."""

from __future__ import annotations

import argparse
import math
import random
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.prepare_scenario_reviews import prepare_reviews
from experiments.review_annotations import _preflight, _read_json, _write_many
from experiments.summarize_scenario_reviews import summarize_reviews
from tracemem.annotations import validate_responses
from tracemem.scenarios import load_scenarios

_METRICS = (
    "supported_rate_all_reviewed_origins",
    "supported_rate_decisive_origins",
    "uncertain_rate_all_reviewed_origins",
)


def _percentile_interval(values: Sequence[float | None]) -> tuple[float, float] | None:
    """Return linearly interpolated 2.5th and 97.5th percentiles of finite draws."""
    finite = sorted(value for value in values if value is not None and math.isfinite(value))
    if not finite:
        return None

    def percentile(fraction: float) -> float:
        position = fraction * (len(finite) - 1)
        lower = int(position)
        upper = min(lower + 1, len(finite) - 1)
        weight = position - lower
        return finite[lower] * (1 - weight) + finite[upper] * weight

    return percentile(0.025), percentile(0.975)


def bootstrap_reviews(
    receipt_bytes_sequence: Sequence[bytes],
    response_documents: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    bootstrap_seed: int = 1729,
    bootstrap_samples: int = 2000,
) -> dict[str, Any]:
    """Summarize validated reviews and scenario-cluster percentile intervals."""
    _require_integer(seed, "seed")
    _require_integer(bootstrap_seed, "bootstrap_seed")
    if type(bootstrap_samples) is not int or not 1 <= bootstrap_samples <= 100_000:
        raise ValueError("bootstrap_samples must be an integer between 1 and 100000")

    bundle, _, provenance = prepare_reviews(receipt_bytes_sequence, seed=seed)
    accounting_report = summarize_reviews(receipt_bytes_sequence, response_documents, seed=seed)
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

    cases, _, _ = load_scenarios()
    case_ids_by_split = {
        split: sorted(case["case_id"] for case in cases if case["split"] == split)
        for split in ("dev", "test")
    }
    if any(not case_ids for case_ids in case_ids_by_split.values()):
        raise ValueError("scenario fixture must contain cases in both dev and test splits")

    labels_by_task_and_reviewer = {
        reviewer["reviewer_id"]: reviewer["labels"] for reviewer in reviewers
    }
    origins_by_model_case: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for task in provenance["task_provenance"]:
        task_id, case_id = task["task_id"], task["case_id"]
        for origin in task["origins"]:
            model_key = origin["model_key"]
            for reviewer in reviewers:
                origins_by_model_case[(model_key, case_id)][reviewer["reviewer_id"]].append(
                    labels_by_task_and_reviewer[reviewer["reviewer_id"]][task_id]
                )

    rng = random.Random(bootstrap_seed)
    resamples_by_split = {
        split: [[rng.choice(case_ids) for _ in case_ids] for _ in range(bootstrap_samples)]
        for split, case_ids in case_ids_by_split.items()
    }

    intervals_by_model: dict[str, dict[str, Any]] = {}
    for model_key in sorted(accounting_report["by_model"]):
        intervals_by_model[model_key] = {}
        for split in ("dev", "test"):
            accounting_cell = accounting_report["by_model"][model_key][split]
            reviewer_accounting = {row["reviewer_id"]: row for row in accounting_cell["reviewers"]}
            interval_rows: list[dict[str, Any]] = []
            for reviewer in reviewers:
                reviewer_id = reviewer["reviewer_id"]
                point_row = reviewer_accounting[reviewer_id]
                counts_by_case = {
                    case_id: _label_counts(
                        origins_by_model_case[(model_key, case_id)].get(reviewer_id, [])
                    )
                    for case_id in case_ids_by_split[split]
                }
                active_case_count = sum(bool(counts["all"]) for counts in counts_by_case.values())
                replicates_by_metric: dict[str, list[float | None]] = {
                    metric: [] for metric in _METRICS
                }
                sampled_with_origins = 0
                sampled_without_origins = 0
                for sample in resamples_by_split[split]:
                    counts = {key: 0 for key in ("supported", "unsupported", "uncertain")}
                    with_origins = 0
                    for case_id in sample:
                        case_counts = counts_by_case[case_id]
                        if case_counts["all"]:
                            with_origins += 1
                        for label in counts:
                            counts[label] += case_counts[label]
                    sampled_with_origins += with_origins
                    sampled_without_origins += len(sample) - with_origins
                    all_reviewed = counts["supported"] + counts["unsupported"] + counts["uncertain"]
                    decisive = counts["supported"] + counts["unsupported"]
                    estimates = {
                        "supported_rate_all_reviewed_origins": (
                            counts["supported"] / all_reviewed if all_reviewed else None
                        ),
                        "supported_rate_decisive_origins": (
                            counts["supported"] / decisive if decisive else None
                        ),
                        "uncertain_rate_all_reviewed_origins": (
                            counts["uncertain"] / all_reviewed if all_reviewed else None
                        ),
                    }
                    for metric in _METRICS:
                        replicates_by_metric[metric].append(estimates[metric])

                observed_counts = _label_counts(
                    label
                    for case_id in case_ids_by_split[split]
                    for label in origins_by_model_case[(model_key, case_id)].get(reviewer_id, [])
                )
                if (
                    observed_counts["supported"] != point_row["supported_origin_count"]
                    or observed_counts["unsupported"] != point_row["unsupported_origin_count"]
                    or observed_counts["uncertain"] != point_row["uncertain_origin_count"]
                ):
                    raise ValueError("bootstrap origin accounting does not match review summary")
                observed_all = observed_counts["all"]
                point_estimates = {
                    "supported_rate_all_reviewed_origins": point_row[
                        "supported_rate_all_reviewed_origins"
                    ],
                    "supported_rate_decisive_origins": point_row["supported_rate_decisive_origins"],
                    "uncertain_rate_all_reviewed_origins": (
                        observed_counts["uncertain"] / observed_all if observed_all else None
                    ),
                }
                interval_metrics: dict[str, Any] = {}
                for metric in _METRICS:
                    replicates = replicates_by_metric[metric]
                    interval = _percentile_interval(replicates)
                    valid_count = sum(
                        value is not None and math.isfinite(value) for value in replicates
                    )
                    interval_metrics[metric] = {
                        "estimate": point_estimates[metric],
                        "interval_95": (
                            {"lower": interval[0], "upper": interval[1]}
                            if interval is not None
                            else None
                        ),
                        "valid_replicate_count": valid_count,
                        "undefined_replicate_count": bootstrap_samples - valid_count,
                    }
                interval_rows.append(
                    {
                        "reviewer_id": reviewer_id,
                        "reviewer_kind": reviewer["reviewer_kind"],
                        "scenario_sampling": {
                            "scenarios_per_replicate": len(case_ids_by_split[split]),
                            "available_scenarios_with_reviewed_origins": active_case_count,
                            "available_scenarios_without_reviewed_origins": (
                                len(case_ids_by_split[split]) - active_case_count
                            ),
                            "sampled_scenario_draws_with_reviewed_origins": sampled_with_origins,
                            "sampled_scenario_draws_without_reviewed_origins": (
                                sampled_without_origins
                            ),
                        },
                        "metrics": interval_metrics,
                    }
                )
            intervals_by_model[model_key][split] = {"reviewers": interval_rows}

    accounting = dict(accounting_report)
    accounting["exclusions"] = provenance["exclusions"]
    return {
        "version": 1,
        "seed": seed,
        "bootstrap": {
            "unit": "scenario case, sampled with replacement within each split",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "percentile_method": "linear interpolation at rank=(n-1)*q",
            "undefined_replicates": (
                "excluded from intervals; valid and undefined counts are reported"
            ),
        },
        "accounting": accounting,
        "by_model": intervals_by_model,
        "interpretation": (
            "Intervals describe resampling variability across this fictional scenario sample only. "
            "They do not establish independent truth, calibration, acceptance precision, Gate A, "
            "or a model winner. Reviewer identity and independence are not verified. Zero observed "
            "errors can yield [1, 1] intervals; this is not a population guarantee."
        ),
    }


def _label_counts(labels: Iterable[str]) -> dict[str, int]:
    counts = {key: 0 for key in ("supported", "unsupported", "uncertain")}
    for label in labels:
        counts[label] += 1
    counts["all"] = sum(counts.values())
    return counts


def _require_integer(value: object, name: str) -> None:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", required=True, nargs="+", type=Path)
    parser.add_argument("--responses", nargs="*", type=Path, default=[])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--bootstrap-seed", type=int, default=1729)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    inputs = (*args.receipts, *args.responses)
    try:
        outputs = _preflight((args.output,), inputs)
        receipts = [path.read_bytes() for path in args.receipts]
        response_documents = [_read_json(path) for path in args.responses]
        report = bootstrap_reviews(
            receipts,
            response_documents,
            seed=args.seed,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_samples=args.bootstrap_samples,
        )
        _write_many(outputs, (report,))
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"bootstrap_scenario_reviews: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
