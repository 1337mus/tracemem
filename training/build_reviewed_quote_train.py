"""Build a reproducible, source-preserving reviewed quote training corpus."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from training.build_quote_data import build as build_v2
from training.build_quote_v3 import build as build_v3
from training.evaluate_quote_sft import _json_object
from training.quote_data import validate_dataset
from training.review_quote_targets import review_seed_targets

JsonMap = dict[str, Any]
_ID_FIELDS = ("case_id", "scenario_group", "template_group")


def build_reviewed_train(
    *,
    v2_source: bytes,
    v2_review: bytes,
    v3_source: bytes,
    v3_review: bytes,
    historical_train: bytes,
) -> dict[str, Any]:
    """Return the reviewed train dataset and a reproducibility/preservation report."""
    raw_inputs = {
        "v2_source": v2_source,
        "v2_review": v2_review,
        "v3_source": v3_source,
        "v3_review": v3_review,
        "historical_train": historical_train,
    }
    for label, raw in raw_inputs.items():
        if type(raw) is not bytes:
            raise ValueError(f"{label} must be bytes")

    v2_seed = _json_object(v2_source, "v2 source seeds")
    v3_seed = _json_object(v3_source, "v3 source seeds")
    v2_review_obj = _json_object(v2_review, "v2 review")
    v3_review_obj = _json_object(v3_review, "v3 review")
    historical = _json_object(historical_train, "historical train")

    original_v2 = build_v2(v2_seed, "train")
    rebuilt_historical = build_v3(v3_seed, original_v2)
    historical_rebuild_matches = rebuilt_historical == historical
    if not historical_rebuild_matches:
        raise ValueError("historical train does not match deterministic source rebuild")

    v2_gate = review_seed_targets(v2_source, v2_review_obj)
    v3_gate = review_seed_targets(v3_source, v3_review_obj)
    reviewed_v2 = build_v2(v2_gate["seeds"], "train")

    # Compare authored expansions before their builders apply v3 replay namespaces.
    original_v3_authored = build_v2(v3_seed, "train")
    reviewed_v3_authored = build_v2(v3_gate["seeds"], "train")
    v2_non_target_fields_match = _require_non_target_cases_match(
        original_v2["cases"], reviewed_v2["cases"], "v2 source expansion"
    )
    v3_non_target_fields_match = _require_non_target_cases_match(
        original_v3_authored["cases"], reviewed_v3_authored["cases"], "v3 source expansion"
    )

    rebuilt_reviewed = build_v3(v3_gate["seeds"], reviewed_v2)
    final_non_target_fields_match = _require_non_target_cases_match(
        rebuilt_historical["cases"], rebuilt_reviewed["cases"], "historical train"
    )

    historical_by_id = {case["case_id"]: case for case in rebuilt_historical["cases"]}
    reviewed_by_id = {case["case_id"]: case for case in rebuilt_reviewed["cases"]}
    if historical_by_id.keys() != reviewed_by_id.keys():
        raise ValueError("reviewed train changed the source case IDs")
    changed_cases = [
        case_id
        for case_id, old_case in historical_by_id.items()
        if old_case["expected"] != reviewed_by_id[case_id]["expected"]
    ]
    changed_families = sorted(
        {reviewed_by_id[case_id]["scenario_group"] for case_id in changed_cases}
    )

    dataset = {
        **rebuilt_reviewed,
        "cases": [_namespace_case(case) for case in rebuilt_reviewed["cases"]],
    }
    cases = validate_dataset(dataset, "train")
    group_counts = Counter(case["scenario_group"] for case in cases)
    family_count = len(group_counts)
    if (
        len(cases) != 768
        or family_count != 72
        or Counter(group_counts.values()) != Counter({12: 48, 8: 24})
    ):
        raise ValueError("reviewed train does not match the 768-row, 72-family composition")
    empty_target_families = {case["scenario_group"] for case in cases if not case["expected"]}
    old_claim_count = sum(len(case["expected"]) for case in rebuilt_historical["cases"])
    new_claim_count = sum(len(case["expected"]) for case in cases)

    report = {
        "input_sha256": {name: hashlib.sha256(raw).hexdigest() for name, raw in raw_inputs.items()},
        "v2_review_gate": v2_gate["report"],
        "v3_review_gate": v3_gate["report"],
        "row_count": len(cases),
        "family_count": family_count,
        "empty_target_row_count": sum(not case["expected"] for case in cases),
        "empty_target_family_count": len(empty_target_families),
        "old_claim_count": old_claim_count,
        "new_claim_count": new_claim_count,
        "changed_row_count": len(changed_cases),
        "changed_family_count": len(changed_families),
        "changed_family_groups": changed_families,
        "source_preservation": {
            "historical_rebuild_matches": historical_rebuild_matches,
            "v2_non_target_fields_match": v2_non_target_fields_match,
            "v3_non_target_fields_match": v3_non_target_fields_match,
            "non_target_fields_match": final_non_target_fields_match,
        },
        "reviewer_status": "agent-only;zero human reviewers",
        "quality_assessment": "not assessed by this dataset builder",
    }
    return {"dataset": dataset, "report": report}


def _require_non_target_cases_match(old_cases: object, new_cases: object, label: str) -> bool:
    if type(old_cases) is not list or type(new_cases) is not list:
        raise ValueError(f"{label} cases must be lists")
    old_ids = [case["case_id"] for case in old_cases]
    new_ids = [case["case_id"] for case in new_cases]
    if old_ids != new_ids:
        raise ValueError(f"{label} changed case order or IDs")
    if len(set(old_ids)) != len(old_ids):
        raise ValueError(f"{label} has duplicate case IDs")
    old_by_id = {case["case_id"]: case for case in old_cases}
    new_by_id = {case["case_id"]: case for case in new_cases}
    if old_by_id.keys() != new_by_id.keys():
        raise ValueError(f"{label} changed case IDs")
    for case_id, old_case in old_by_id.items():
        if _without_expected(old_case) != _without_expected(new_by_id[case_id]):
            raise ValueError(f"{label} changed non-target fields for {case_id}")
    return True


def _without_expected(case: JsonMap) -> JsonMap:
    return {key: value for key, value in case.items() if key != "expected"}


def _namespace_case(case: JsonMap) -> JsonMap:
    copied = copy.deepcopy(case)
    for field in _ID_FIELDS:
        copied[field] = f"V4T-{case[field]}"
    copied["entity_ids"] = [f"V4T-{value}" for value in case["entity_ids"]]
    message_ids: dict[str, str] = {}
    for message in copied["messages"]:
        message["session_id"] = f"V4T-{message['session_id']}"
        old_message_id = message["message_id"]
        message_ids[old_message_id] = f"V4T-{old_message_id}"
        message["message_id"] = message_ids[old_message_id]
    for claim in copied["expected"]:
        claim["message_id"] = message_ids[claim["message_id"]]
    return copied


def _write_exclusive(path: Path, payload: object) -> None:
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded)
    except OSError:
        path.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-source", type=Path, required=True)
    parser.add_argument("--v2-review", type=Path, required=True)
    parser.add_argument("--v3-source", type=Path, required=True)
    parser.add_argument("--v3-review", type=Path, required=True)
    parser.add_argument("--historical-train", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        result = build_reviewed_train(
            v2_source=args.v2_source.read_bytes(),
            v2_review=args.v2_review.read_bytes(),
            v3_source=args.v3_source.read_bytes(),
            v3_review=args.v3_review.read_bytes(),
            historical_train=args.historical_train.read_bytes(),
        )
        if args.output_dir.exists() or args.output_dir.is_symlink():
            raise ValueError("output directory already exists")
        args.output_dir.mkdir(parents=True, exist_ok=False)
        created: list[Path] = []
        try:
            train_path = args.output_dir / "train.json"
            report_path = args.output_dir / "report.json"
            _write_exclusive(train_path, result["dataset"])
            created.append(train_path)
            _write_exclusive(report_path, result["report"])
            created.append(report_path)
        except OSError:
            for path in created:
                path.unlink(missing_ok=True)
            args.output_dir.rmdir()
            raise
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps({"output_dir": str(args.output_dir), "row_count": result["report"]["row_count"]})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
