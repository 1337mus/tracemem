"""Create a metadata-only audit receipt and optional supervised training export."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from training.verifier_data import render_training_rows, validate_verifier_dataset

_SPLITS = ("train", "dev", "test")
_SUPPORT_LABELS = ("supported", "unsupported", "uncertain")
_ELIGIBILITY_LABELS = ("eligible", "ineligible", "uncertain")
_PHENOMENA = ("attribution", "condition", "actor", "negation", "time", "causality")


def summarize_dataset(payload: object) -> dict[str, Any]:
    """Summarize validated verifier annotation data."""
    cases = validate_verifier_dataset(payload)
    candidates = [candidate for case in cases for candidate in case["candidates"]]
    per_split: dict[str, dict[str, Any]] = {}

    for split in _SPLITS:
        split_cases = [case for case in cases if case["split"] == split]
        split_candidates = [candidate for case in split_cases for candidate in case["candidates"]]
        support_counts = Counter(candidate["support"] for candidate in split_candidates)
        eligibility_counts = Counter(candidate["eligibility"] for candidate in split_candidates)
        phenomenon_counts = Counter(case["phenomenon"] for case in split_cases)
        per_split[split] = {
            "case_count": len(split_cases),
            "scenario_count": len({case["scenario_group"] for case in split_cases}),
            "candidate_count": len(split_candidates),
            "support_counts": {label: support_counts[label] for label in _SUPPORT_LABELS},
            "eligibility_counts": {
                label: eligibility_counts[label] for label in _ELIGIBILITY_LABELS
            },
            "phenomenon_counts": {
                phenomenon: phenomenon_counts[phenomenon] for phenomenon in _PHENOMENA
            },
            "no_filter": _no_filter_counts(split_candidates),
        }

    paired_support_cases = 0
    uncertain_support_cases = 0
    for case in cases:
        labels = [candidate["support"] for candidate in case["candidates"]]
        if labels.count("supported") == 1 and labels.count("unsupported") == 1:
            paired_support_cases += 1
        if "uncertain" in labels:
            uncertain_support_cases += 1

    return {
        "schema_version": 1,
        "annotation_status": "agent_authored",
        "human_reviewers": 0,
        "case_count": len(cases),
        "scenario_count": len({case["scenario_group"] for case in cases}),
        "candidate_count": len(candidates),
        "per_split": per_split,
        "paired_support_cases": paired_support_cases,
        "uncertain_support_cases": uncertain_support_cases,
        # validate_verifier_dataset enforces that scenario, template, and entity groups do not
        # cross splits, so a successfully summarized payload has no split-group overlap.
        "split_group_overlap": 0,
    }


def _no_filter_counts(candidates: list[dict[str, Any]]) -> dict[str, int]:
    """Count all retained candidates while keeping decisive label axes independent."""
    support_counts = Counter(candidate["support"] for candidate in candidates)
    eligibility_counts = Counter(candidate["eligibility"] for candidate in candidates)
    return {
        "candidate_count": len(candidates),
        "all_candidates_retained": len(candidates),
        "supported_retained": support_counts["supported"],
        "supported_total": support_counts["supported"],
        "unsupported_retained": support_counts["unsupported"],
        "unsupported_total": support_counts["unsupported"],
        "unsupported_rejected": 0,
        "uncertain_support_unscored": support_counts["uncertain"],
        "eligible_retained": eligibility_counts["eligible"],
        "eligible_total": eligibility_counts["eligible"],
        "ineligible_retained": eligibility_counts["ineligible"],
        "ineligible_total": eligibility_counts["ineligible"],
        "uncertain_eligibility_unscored": eligibility_counts["uncertain"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-output", type=Path)
    args = parser.parse_args(argv)

    dataset_path = args.dataset
    output_path = args.output
    train_output_path = args.train_output
    try:
        _preflight_paths(dataset_path, output_path, train_output_path)
        source_bytes = dataset_path.read_bytes()
        payload = _parse_json(source_bytes)
        summary = summarize_dataset(payload)
        report = {
            "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "summary": summary,
        }
        report_text = (
            json.dumps(
                report,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

        train_text: str | None = None
        if train_output_path is not None:
            rows = render_training_rows(payload, "train")
            train_text = "".join(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
                for row in rows
            )

        # Catch destinations created while validation and rendering were in progress. File opens
        # below use exclusive creation as the final guard against races after this check.
        _preflight_paths(dataset_path, output_path, train_output_path)
        _create_exclusive(output_path, report_text)
        if train_output_path is not None and train_text is not None:
            _create_exclusive(train_output_path, train_text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


def _preflight_paths(
    dataset_path: Path,
    output_path: Path,
    train_output_path: Path | None,
) -> None:
    dataset_resolved = dataset_path.resolve(strict=False)
    output_resolved = output_path.resolve(strict=False)
    if dataset_resolved == output_resolved:
        raise ValueError("dataset and output paths must be different")
    if train_output_path is not None:
        train_resolved = train_output_path.resolve(strict=False)
        if train_resolved == dataset_resolved:
            raise ValueError("dataset and train-output paths must be different")
        if train_resolved == output_resolved:
            raise ValueError("output and train-output paths must be different")

    for path in (output_path, train_output_path):
        if path is not None and (path.exists() or path.is_symlink()):
            raise ValueError(f"output path already exists: {path}")


def _parse_json(source_bytes: bytes) -> object:
    try:
        source_text = source_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("dataset must be valid UTF-8") from exc
    return json.loads(
        source_text,
        object_pairs_hook=_unique_json_object,
        parse_constant=_reject_non_finite,
        parse_float=_parse_finite_float,
    )


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_non_finite(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number is not allowed")
    return parsed


def _create_exclusive(path: Path, content: str) -> None:
    try:
        with path.open("x", encoding="utf-8", newline="\n") as output_file:
            output_file.write(content)
    except FileExistsError as exc:
        # If this is the train output after the report was created, keep the report receipt.
        raise ValueError(f"output path already exists: {path}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
