"""Build the fictional SFT pilot evaluation fixtures from authored seed files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from tracemem.challenge import _validate_case

ROOT = Path(__file__).resolve().parents[1]
SEED_KEYS = {"version", "license", "annotation_status", "provenance", "families"}
SEED_META = {
    "version": 1,
    "license": "MIT",
    "annotation_status": "agent_authored",
    "provenance": "newly_authored_fiction",
}


def _expand_text(value: Any, values: dict[str, str]) -> Any:
    if isinstance(value, str):
        return value.format_map(values)
    if isinstance(value, list):
        return [_expand_text(item, values) for item in value]
    if isinstance(value, dict):
        return {key: _expand_text(item, values) for key, item in value.items()}
    return value


def expand_family(
    family: dict[str, Any], split: str
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    prefix = {"dev": "EVALDEV", "test": "EVALTEST"}.get(split)
    if prefix is None:
        raise ValueError("split must be dev or test")
    group = family["group_id"]
    if not group.startswith(prefix + "-") or not family["template_group"].startswith(prefix + "-"):
        raise ValueError("family and template IDs must carry the split prefix")
    templates = family["message_templates"]
    if len(templates) != 6 or len(family["variants"]) != 4:
        raise ValueError("each family must have six message templates and four variants")
    cases = []
    metadata = {}
    for variant in family["variants"]:
        values = variant["values"]
        case_id = f"{group}-{variant['case_suffix']}"
        session_id = f"{case_id}-S01"
        messages = []
        for index, template in enumerate(templates, 1):
            if set(template) != {"role", "text", "recorded_at", "content_kind"}:
                raise ValueError("message template has missing or unexpected fields")
            messages.append(
                {
                    "session_id": session_id,
                    "message_id": f"{case_id}-M{index:02d}",
                    "role": template["role"],
                    "text": _expand_text(template["text"], values),
                    "recorded_at": _expand_text(template["recorded_at"], values),
                    "content_kind": _expand_text(template["content_kind"], values),
                }
            )
        expected = []
        for target in family["expected_templates"]:
            claim = {}
            for key, raw in target.items():
                if key != "evidence":
                    claim[key] = _expand_text(raw, values)
                    continue
                spans = []
                for evidence in raw:
                    index = evidence["message_index"]
                    if type(index) is not int or not 0 <= index < len(messages):
                        raise ValueError("evidence message_index is out of range")
                    quote = _expand_text(evidence["quote"], values)
                    text = messages[index]["text"]
                    if text.count(quote) != 1:
                        raise ValueError(f"evidence quote must occur exactly once in {case_id}")
                    start = text.index(quote)
                    spans.append(
                        {
                            "session_id": session_id,
                            "message_id": messages[index]["message_id"],
                            "char_start": start,
                            "char_end": start + len(quote),
                            "quote": quote,
                        }
                    )
                claim["evidence"] = spans
            expected.append(claim)
        case = {
            "case_id": case_id,
            "scenario_id": group,
            "split": split,
            "tags": list(family["tags"]),
            "messages": messages,
            "expected": expected,
        }
        _validate_case(case)
        cases.append(case)
        metadata[case_id] = {
            "scenario_group": group,
            "template_group": family["template_group"],
            "entity_ids": list(family["entity_ids"]),
        }
    return cases, metadata


def build_dataset(seed: dict[str, Any], split: str) -> dict[str, Any]:
    if split not in {"dev", "test"} or set(seed) != SEED_KEYS:
        raise ValueError("seed root or split is invalid")
    if any(seed[key] != value for key, value in SEED_META.items()):
        raise ValueError("seed metadata does not match the evaluation contract")
    families = seed["families"]
    if type(families) is not list or len(families) != 8:
        raise ValueError("each partition must contain exactly eight scenario families")
    prefix = "EVALDEV" if split == "dev" else "EVALTEST"
    cases = []
    case_metadata: dict[str, dict[str, Any]] = {}
    family_ids = set()
    entity_ids: set[str] = set()
    for family in families:
        group = family["group_id"]
        if group in family_ids:
            raise ValueError("scenario group IDs must be unique")
        family_ids.add(group)
        if type(family["entity_ids"]) is not list or not family["entity_ids"]:
            raise ValueError("each family must name at least one entity ID")
        if any(not entity.startswith(prefix + "-") for entity in family["entity_ids"]):
            raise ValueError("entity IDs must carry the split prefix")
        if entity_ids.intersection(family["entity_ids"]):
            raise ValueError("entity IDs must not be reused across scenario families")
        entity_ids.update(family["entity_ids"])
        expanded, metadata = expand_family(family, split)
        if set(case_metadata).intersection(metadata):
            raise ValueError("case IDs must be unique")
        cases.extend(expanded)
        case_metadata.update(metadata)
    return {**SEED_META, "cases": cases, "case_metadata": case_metadata}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError("JSON input root must be an object")
    return cast(dict[str, Any], value)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-seeds", type=Path, help="explicit local-only test seed file")
    parser.add_argument(
        "--test-output",
        type=Path,
        default=Path(".local/sft-pilot-20261015/evaluation-private/test.json"),
    )
    args = parser.parse_args()
    dev = build_dataset(_read_json(ROOT / "datasets/sft-pilot/eval-seeds.json"), "dev")
    _write_json(ROOT / "datasets/sft-pilot/eval.json", dev)
    print(f"dev: {len(dev['cases'])} cases -> datasets/sft-pilot/eval.json")
    if args.test_seeds is not None:
        test = build_dataset(_read_json(args.test_seeds), "test")
        _write_json(args.test_output, test)
        print(f"test: {len(test['cases'])} cases -> {args.test_output}")


if __name__ == "__main__":
    main()
