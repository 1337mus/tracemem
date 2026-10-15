"""Deterministically expand the fiction-only SFT training seeds."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tracemem.challenge import _validate_case  # noqa: E402

SEEDS_PATH = ROOT / "datasets/sft-pilot/train-seeds.json"
TRAIN_PATH = ROOT / "datasets/sft-pilot/train.json"
_CLAIM_FIELDS = (
    "subject",
    "predicate",
    "object",
    "speaker_role",
    "attribution",
    "modality",
)


def _render(value: Any, values: dict[str, str]) -> Any:
    if isinstance(value, str):
        return value.format_map(values)
    if isinstance(value, list):
        return [_render(item, values) for item in value]
    if isinstance(value, dict):
        return {key: _render(item, values) for key, item in value.items()}
    return value


def build(payload: dict[str, Any]) -> dict[str, Any]:
    variants = payload["variants_per_family"]
    banks = payload["value_banks"]
    seeds = payload["seeds"]
    if type(variants) is not int or variants < 1 or type(payload["seed"]) is not int:
        raise ValueError("seed and variants_per_family must be integers")
    if not seeds or len({seed["family_id"] for seed in seeds}) != len(seeds):
        raise ValueError("training seeds must have unique family IDs")
    for name, values in banks.items():
        if (
            type(values) is not list
            or len(values) != variants
            or any(type(v) is not str for v in values)
        ):
            raise ValueError(f"value bank {name} must have exactly {variants} strings")

    rng = random.Random(payload["seed"])
    cases: list[dict[str, Any]] = []
    metadata: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        family_id = seed["family_id"]
        family_key = family_id.removeprefix("TRAIN-F")
        order = list(range(variants))
        rng.shuffle(order)
        for variant, source_index in enumerate(order):
            values = {name: bank[source_index] for name, bank in banks.items()}
            case_id = f"TRAIN-CASE-{family_key}-{variant:02d}"
            messages = []
            for index, raw in enumerate(seed["messages"]):
                messages.append(
                    {
                        "session_id": (
                            f"TRAIN-SESSION-{family_key}-{variant:02d}-{raw['session']:02d}"
                        ),
                        "message_id": f"TRAIN-MSG-{family_key}-{variant:02d}-{index:02d}",
                        "role": raw["role"],
                        "text": _render(raw["text"], values),
                        "recorded_at": None,
                        "content_kind": raw.get("content_kind", "conversation"),
                    }
                )

            expected = []
            for raw_claim in seed["claims"]:
                claim = {field: _render(raw_claim[field], values) for field in _CLAIM_FIELDS}
                claim.update(
                    polarity=raw_claim.get("polarity", "positive"),
                    valid_from=_render(raw_claim.get("valid_from"), values),
                    valid_to=_render(raw_claim.get("valid_to"), values),
                    evidence=[],
                )
                for raw_span in raw_claim["evidence"]:
                    message = messages[raw_span["message"]]
                    quote = _render(raw_span["quote"], values)
                    if message["text"].count(quote) != 1:
                        raise ValueError(f"evidence quote is not unique in {case_id}")
                    start = message["text"].index(quote)
                    claim["evidence"].append(
                        {
                            "session_id": message["session_id"],
                            "message_id": message["message_id"],
                            "char_start": start,
                            "char_end": start + len(quote),
                            "quote": quote,
                        }
                    )
                expected.append(claim)

            case = {
                "case_id": case_id,
                "scenario_id": f"TRAIN-SCENARIO-{family_key}",
                "split": "train",
                "tags": seed["tags"],
                "messages": messages,
                "expected": expected,
            }
            _validate_case(case)
            cases.append(case)
            metadata[case_id] = {
                "scenario_group": f"TRAIN-SCENARIO-GROUP-{family_key}",
                "template_group": f"TRAIN-TEMPLATE-GROUP-{family_key}",
                "entity_ids": [f"TRAIN-ENTITY-{family_key}-{variant:02d}"],
            }

    return {
        "version": 1,
        "license": "MIT",
        "annotation_status": "agent_authored",
        "provenance": "newly_authored_fiction",
        "cases": cases,
        "case_metadata": metadata,
    }


def main() -> None:
    payload = json.loads(SEEDS_PATH.read_text(encoding="utf-8"))
    result = build(payload)
    TRAIN_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"seed families={len(payload['seeds'])}; rows={len(result['cases'])}; "
        f"abstentions={sum(not case['expected'] for case in result['cases'])}; "
        f"seed={payload['seed']}"
    )


if __name__ == "__main__":
    main()
