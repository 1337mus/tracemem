"""Build expanded, deterministic quote-v2 training and development fixtures."""

from __future__ import annotations

import argparse
import json
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from training.quote_data import validate_dataset

_SEED_KEYS = {"seed", "split", "variants", "value_banks", "families"}
_BANK_KEYS = {"person", "product", "object", "place", "date_from", "date_to"}
_PREFIX = {"train": "TRAIN", "dev": "DEV"}


def build(seed_payload: object, split: str) -> dict[str, Any]:
    """Expand authored family templates into one validated public dataset split."""
    if split not in _PREFIX:
        raise ValueError("split must be train or dev")
    if type(seed_payload) is not dict or set(seed_payload) != _SEED_KEYS:
        raise ValueError("seed root has missing or unexpected fields")
    if seed_payload["split"] != split:
        raise ValueError("seed split does not match requested split")
    seed = seed_payload["seed"]
    variants = seed_payload["variants"]
    if type(seed) is not int or seed != 23:
        raise ValueError("builder seed must be integer 23")
    if type(variants) is not int or variants < 1:
        raise ValueError("variants must be a positive integer")
    banks = seed_payload["value_banks"]
    if type(banks) is not dict or set(banks) != _BANK_KEYS:
        raise ValueError("value_banks has missing or unexpected fields")
    normalized_banks: dict[str, list[str]] = {}
    for key, raw_values in banks.items():
        if type(raw_values) is not list or len(raw_values) != variants:
            raise ValueError(f"value bank {key} must contain exactly {variants} values")
        if any(type(value) is not str or not value.strip() for value in raw_values):
            raise ValueError(f"value bank {key} must contain non-empty strings")
        normalized_banks[key] = raw_values

    families = seed_payload["families"]
    if type(families) is not list or not families:
        raise ValueError("families must be a non-empty list")
    family_ids: set[str] = set()
    prefix = _PREFIX[split]
    variant_order = list(range(variants))
    random.Random(seed).shuffle(variant_order)
    cases: list[dict[str, Any]] = []
    case_number = 0
    message_number = 0
    for family in families:
        if type(family) is not dict:
            raise ValueError("family must be an object")
        required_family_keys = {"family_id", "tags", "messages", "expected"}
        if not required_family_keys <= set(family) or set(family) - (
            required_family_keys | {"entity_slots"}
        ):
            raise ValueError("family has missing or unexpected fields")
        family_id = _text(family["family_id"], "family_id")
        if family_id in family_ids:
            raise ValueError(f"duplicate family_id: {family_id}")
        family_ids.add(family_id)
        tags = _string_list(family["tags"], "family tags")
        entity_slots = _string_list(
            family.get("entity_slots", ["speaker", "subject"]), "entity_slots"
        )
        raw_messages = family["messages"]
        if type(raw_messages) is not list or not 1 <= len(raw_messages) <= 6:
            raise ValueError(f"family {family_id} must have one to six messages")
        raw_expected = family["expected"]
        if type(raw_expected) is not list:
            raise ValueError(f"family {family_id} expected must be a list")

        for ordinal, bank_index in enumerate(variant_order, start=1):
            case_number += 1
            values = {key: rows[bank_index] for key, rows in normalized_banks.items()}
            case_id = f"{prefix}-CASE-{family_id}-V{ordinal:02d}"
            message_ids_by_ref: dict[str, str] = {}
            messages: list[dict[str, Any]] = []
            for raw_message in raw_messages:
                if type(raw_message) is not dict or set(raw_message) != {
                    "ref",
                    "role",
                    "session",
                    "content_kind",
                    "text",
                }:
                    raise ValueError(f"family {family_id} message has missing or extra fields")
                ref = _text(raw_message["ref"], "message ref")
                if ref in message_ids_by_ref:
                    raise ValueError(f"family {family_id} has duplicate message ref: {ref}")
                if type(raw_message["session"]) is not int or raw_message["session"] < 1:
                    raise ValueError("message session must be a positive integer")
                message_number += 1
                message_id = f"{prefix}-MSG-{message_number:05d}"
                message_ids_by_ref[ref] = message_id
                minutes = message_number
                recorded_at = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=minutes)
                messages.append(
                    {
                        "session_id": (
                            f"{prefix}-SESSION-{case_number:04d}-{raw_message['session']:02d}"
                        ),
                        "message_id": message_id,
                        "role": raw_message["role"],
                        "text": _format(raw_message["text"], values),
                        "recorded_at": recorded_at.isoformat().replace("+00:00", "Z"),
                        "content_kind": raw_message["content_kind"],
                    }
                )
            expected: list[dict[str, Any]] = []
            for raw_claim in raw_expected:
                if type(raw_claim) is not dict or set(raw_claim) != {
                    "claim",
                    "message_ref",
                    "quote",
                    "valid_from",
                    "valid_to",
                }:
                    raise ValueError(
                        f"family {family_id} expected claim has missing or extra fields"
                    )
                message_ref = _text(raw_claim["message_ref"], "expected message_ref")
                if message_ref not in message_ids_by_ref:
                    raise ValueError(
                        f"family {family_id} expected claim references unknown message"
                    )
                expected.append(
                    {
                        "claim": _format(raw_claim["claim"], values),
                        "message_id": message_ids_by_ref[message_ref],
                        "quote": _format(raw_claim["quote"], values),
                        "valid_from": _optional_formatted(raw_claim["valid_from"], values),
                        "valid_to": _optional_formatted(raw_claim["valid_to"], values),
                    }
                )
            cases.append(
                {
                    "case_id": case_id,
                    "split": split,
                    "scenario_group": f"{prefix}-SCENARIO-{family_id}",
                    "template_group": f"{prefix}-TEMPLATE-{family_id}",
                    "entity_ids": [
                        f"{prefix}-ENTITY-{family_id}-V{ordinal:02d}-{slot.upper()}"
                        for slot in entity_slots
                    ],
                    "tags": tags,
                    "messages": messages,
                    "expected": expected,
                }
            )
    payload = {
        "version": 2,
        "license": "MIT",
        "annotation_status": "agent_authored",
        "provenance": "newly_authored_fiction",
        "cases": cases,
    }
    validate_dataset(payload, split)
    return payload


def build_all(train_seeds: object, dev_seeds: object) -> dict[str, dict[str, Any]]:
    """Build both partitions and reject any ID, family, or message-text overlap."""
    train = build(train_seeds, "train")
    dev = build(dev_seeds, "dev")
    train_cases = train["cases"]
    dev_cases = dev["cases"]
    for key in ("case_id", "scenario_group", "template_group"):
        train_values = {case[key] for case in train_cases}
        dev_values = {case[key] for case in dev_cases}
        if train_values & dev_values:
            raise ValueError(f"train/dev {key} overlap")
    train_entities = {value for case in train_cases for value in case["entity_ids"]}
    dev_entities = {value for case in dev_cases for value in case["entity_ids"]}
    if train_entities & dev_entities:
        raise ValueError("train/dev entity_id overlap")
    train_conversations = {_conversation_signature(case) for case in train_cases}
    dev_conversations = {_conversation_signature(case) for case in dev_cases}
    if train_conversations & dev_conversations:
        raise ValueError("train/dev conversation content overlap")
    return {"train": train, "dev": dev}


def _conversation_signature(case: dict[str, Any]) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (message["role"], message["content_kind"], " ".join(message["text"].split()))
        for message in case["messages"]
    )


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _string_list(value: object, label: str) -> list[str]:
    if type(value) is not list or not value:
        raise ValueError(f"{label} must be a non-empty list")
    if any(type(item) is not str or not item.strip() for item in value):
        raise ValueError(f"{label} must contain non-empty strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{label} must contain unique strings")
    return value


def _format(value: object, values: dict[str, str]) -> str:
    text = _text(value, "template")
    try:
        return text.format_map(values)
    except KeyError as exc:
        raise ValueError(f"unknown template value: {exc.args[0]}") from exc


def _optional_formatted(value: object, values: dict[str, str]) -> str | None:
    if value is None:
        return None
    return _format(value, values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    data_dir = args.root / "datasets" / "sft-quote-v2"
    train_seeds = json.loads((data_dir / "train-seeds.json").read_text(encoding="utf-8"))
    dev_seeds = json.loads((data_dir / "dev-seeds.json").read_text(encoding="utf-8"))
    built = build_all(train_seeds, dev_seeds)
    for split, payload in built.items():
        destination = data_dir / f"{split}.json"
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
