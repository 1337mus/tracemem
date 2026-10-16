"""Build quote-v3 training data from source-preserving replays and authored seeds."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

from training.build_quote_data import build as build_from_seeds
from training.quote_data import validate_dataset

_REPLAY_FAMILIES = 48
_SOURCE_VARIANTS_PER_FAMILY = 16
_REPLAY_VARIANTS = 12


def _namespace_case(case: dict[str, Any], namespace: str) -> dict[str, Any]:
    copied = copy.deepcopy(case)
    for field in ("case_id", "scenario_group", "template_group"):
        copied[field] = f"{namespace}-{case[field]}"
    copied["entity_ids"] = [f"{namespace}-{value}" for value in case["entity_ids"]]
    message_ids: dict[str, str] = {}
    for message in copied["messages"]:
        message["session_id"] = f"{namespace}-{message['session_id']}"
        old_message_id = message["message_id"]
        message_ids[old_message_id] = f"{namespace}-{old_message_id}"
        message["message_id"] = message_ids[old_message_id]
    for claim in copied["expected"]:
        claim["message_id"] = message_ids[claim["message_id"]]
    return copied


def build(seed_payload: object, source_dataset: object) -> dict[str, Any]:
    """Keep twelve ordered v2 rows per family, then append namespaced v3 cases."""
    try:
        source_cases = validate_dataset(source_dataset, "train")
    except ValueError as exc:
        raise ValueError(f"invalid v2 training source: {exc}") from exc

    variants_by_family = Counter(case["scenario_group"] for case in source_cases)
    if len(variants_by_family) != _REPLAY_FAMILIES:
        raise ValueError(
            "invalid v2 training source: expected 48 scenario groups, "
            f"found {len(variants_by_family)}"
        )
    invalid_counts = {
        family: count
        for family, count in variants_by_family.items()
        if count != _SOURCE_VARIANTS_PER_FAMILY
    }
    if invalid_counts:
        family, count = next(iter(invalid_counts.items()))
        raise ValueError(
            "invalid v2 training source: scenario group "
            f"{family} has {count} variants, expected {_SOURCE_VARIANTS_PER_FAMILY}"
        )

    generated = build_from_seeds(seed_payload, "train")
    # build_from_seeds validates the seed payload before returning.
    validated_seed_payload = cast(dict[str, Any], seed_payload)
    raw_families = validated_seed_payload["families"]
    if any(not family["family_id"].startswith("V3T") for family in raw_families):
        raise ValueError("quote-v3 family IDs must start with V3T")

    seen: Counter[str] = Counter()
    replay = []
    for case in source_cases:
        family = case["scenario_group"]
        if seen[family] < _REPLAY_VARIANTS:
            replay.append(_namespace_case(case, "V3T-REPLAY"))
        seen[family] += 1
    authored = [_namespace_case(case, "V3T-TRAIN") for case in generated["cases"]]
    result = {**generated, "cases": replay + authored}
    validate_dataset(result, "train")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    data_dir = args.root / "datasets" / "sft-quote-v3"
    source_path = args.root / "datasets" / "sft-quote-v2" / "train.json"
    seeds = json.loads((data_dir / "target-train-seeds.json").read_text(encoding="utf-8"))
    source = json.loads(source_path.read_text(encoding="utf-8"))
    result = build(seeds, source)
    (data_dir / "train.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
