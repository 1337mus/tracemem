"""Prepare a bounded, offline SFT dataset from explicitly authored fixtures."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tracemem.challenge import _validate_case
from tracemem.prompts import render_explicit_prompt

_ROOT_KEYS = frozenset(
    {"version", "license", "annotation_status", "provenance", "cases", "case_metadata"}
)
_METADATA_KEYS = frozenset({"scenario_group", "template_group", "entity_ids"})
_SPLITS = ("train", "dev", "test")


def prepare_dataset(source_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate authored source payloads and return JSONL text plus a hash manifest.

    Each payload uses the exact root contract documented by the input fixtures. Cases
    retain their declared split. The function performs no file or network operations.
    """
    if type(source_payloads) is not list or not source_payloads:
        raise ValueError("source_payloads must be a non-empty list")
    try:
        canonical_payload = json.dumps(
            source_payloads,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError("source payloads must contain finite, valid JSON values") from exc

    cases: list[dict[str, Any]] = []
    metadata_by_id: dict[str, dict[str, Any]] = {}
    case_ids: set[str] = set()
    message_ids: set[tuple[str, str]] = set()
    content_keys: set[tuple[tuple[str, str], ...]] = set()
    group_splits: dict[tuple[str, str], str] = {}
    splits_seen: set[str] = set()

    for payload in source_payloads:
        _validate_root(payload)
        raw_cases = payload["cases"]
        metadata = payload["case_metadata"]
        if type(raw_cases) is not list or not raw_cases:
            raise ValueError("cases must be a non-empty list")
        if type(metadata) is not dict:
            raise ValueError("case_metadata must be an object")
        for case in raw_cases:
            _validate_case(case)
        ids_in_payload = {case.get("case_id") for case in raw_cases if type(case) is dict}
        if len(ids_in_payload) != len(raw_cases) or set(metadata) != ids_in_payload:
            raise ValueError("case_metadata keys must exactly match payload case IDs")

        for case in raw_cases:
            case_id = case["case_id"]
            split = case["split"]
            if split not in _SPLITS:
                raise ValueError(f"case {case_id} has unsupported split: {split}")
            if case_id in case_ids:
                raise ValueError(f"duplicate case_id: {case_id}")
            case_ids.add(case_id)
            splits_seen.add(split)

            case_metadata = _validate_metadata(metadata[case_id], case_id)
            metadata_by_id[case_id] = case_metadata
            _check_group_split("scenario_id", case["scenario_id"], split, group_splits)
            _check_group_split(
                "scenario_group", case_metadata["scenario_group"], split, group_splits
            )
            _check_group_split(
                "template_group", case_metadata["template_group"], split, group_splits
            )
            for entity_id in case_metadata["entity_ids"]:
                _check_group_split("entity_id", entity_id, split, group_splits)

            ordered_content = tuple(
                (message["role"], " ".join(message["text"].split())) for message in case["messages"]
            )
            if ordered_content in content_keys:
                raise ValueError(f"duplicate conversation content: {case_id}")
            content_keys.add(ordered_content)
            for message in case["messages"]:
                identity = (message["session_id"], message["message_id"])
                if identity in message_ids:
                    raise ValueError(f"duplicate global message identity: {identity}")
                message_ids.add(identity)
            cases.append(case)

    missing_splits = set(_SPLITS).difference(splits_seen)
    if missing_splits:
        raise ValueError(
            f"complete bundle requires train, dev, and test cases; missing {sorted(missing_splits)}"
        )

    files: dict[str, str] = {}
    partition_summary: dict[str, dict[str, Any]] = {}
    character_counts: dict[str, dict[str, int]] = {}
    output_hashes: dict[str, str] = {}
    for split in _SPLITS:
        rows: list[dict[str, Any]] = []
        families: Counter[str] = Counter()
        claims = empty_targets = prompt_chars = completion_chars = 0
        for case in cases:
            if case["split"] != split:
                continue
            case_metadata = metadata_by_id[case["case_id"]]
            prompt = render_explicit_prompt(case)
            completion = json.dumps(
                {"claims": case["expected"]},
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            rows.append(
                {
                    "case_id": case["case_id"],
                    "split": split,
                    "scenario_group": case_metadata["scenario_group"],
                    "template_group": case_metadata["template_group"],
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": completion},
                    ],
                }
            )
            claim_count = len(case["expected"])
            claims += claim_count
            empty_targets += int(claim_count == 0)
            families[case_metadata["template_group"]] += 1
            prompt_chars += len(prompt)
            completion_chars += len(completion)
        filename = f"{split}.jsonl"
        content = "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
            for row in rows
        )
        output_bytes = content.encode("utf-8")
        files[filename] = content
        output_hashes[filename] = hashlib.sha256(output_bytes).hexdigest()
        partition_summary[split] = {
            "rows": len(rows),
            "claims": claims,
            "empty_targets": empty_targets,
            "family_counts": dict(sorted(families.items())),
        }
        character_counts[split] = {
            "prompt_chars": prompt_chars,
            "completion_chars": completion_chars,
            "total_chars": prompt_chars + completion_chars,
        }

    prompt_source_path = inspect.getsourcefile(render_explicit_prompt)
    if prompt_source_path is None:
        raise ValueError("could not locate prompt source for manifest hashing")
    prompt_source = Path(prompt_source_path).read_bytes()
    total_prompt_chars = sum(row["prompt_chars"] for row in character_counts.values())
    total_completion_chars = sum(row["completion_chars"] for row in character_counts.values())
    manifest = {
        "version": 1,
        "source": {
            "hash_kind": "canonical_payload_sha256",
            "sha256": hashlib.sha256(canonical_payload).hexdigest(),
        },
        "prompt_source_sha256": hashlib.sha256(prompt_source).hexdigest(),
        "output_sha256": output_hashes,
        "partitions": partition_summary,
        "character_counts": {
            "by_partition": character_counts,
            "total": {
                "prompt_chars": total_prompt_chars,
                "completion_chars": total_completion_chars,
                "total_chars": total_prompt_chars + total_completion_chars,
            },
            "note": "Character counts are not token counts.",
        },
        "token_count": None,
        "token_count_status": "unavailable_locally",
        "annotation_policy": {
            "support_status": "provisional_agent_authored",
            "gate_a": "not_assessed",
            "human_validity": "not_assessed",
            "empty_target": (
                "An authored empty claims list does not establish truth or evidence sufficiency."
            ),
        },
    }
    return {"files": files, "manifest": manifest}


def _validate_root(payload: object) -> None:
    if type(payload) is not dict or set(payload) != _ROOT_KEYS:
        raise ValueError("source root has missing or unexpected fields")
    if type(payload["version"]) is not int or payload["version"] != 1:
        raise ValueError("source version must be integer 1")
    if type(payload["license"]) is not str or payload["license"] != "MIT":
        raise ValueError("source license must be MIT")
    if (
        type(payload["annotation_status"]) is not str
        or payload["annotation_status"] != "agent_authored"
    ):
        raise ValueError("annotation_status must be agent_authored")
    if type(payload["provenance"]) is not str or payload["provenance"] != "newly_authored_fiction":
        raise ValueError("provenance must be newly_authored_fiction")


def _validate_metadata(value: object, case_id: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _METADATA_KEYS:
        raise ValueError(f"case_metadata for {case_id} has missing or unexpected fields")
    scenario_group = _metadata_text(value["scenario_group"], "scenario_group")
    template_group = _metadata_text(value["template_group"], "template_group")
    entity_ids = value["entity_ids"]
    if (
        type(entity_ids) is not list
        or not entity_ids
        or any(type(entity_id) is not str for entity_id in entity_ids)
    ):
        raise ValueError(f"entity_ids for {case_id} must be a non-empty list of strings")
    clean_entity_ids = [_metadata_text(item, "entity_id") for item in entity_ids]
    if len(set(clean_entity_ids)) != len(clean_entity_ids):
        raise ValueError(f"entity_ids for {case_id} must be unique")
    return {
        "scenario_group": scenario_group,
        "template_group": template_group,
        "entity_ids": clean_entity_ids,
    }


def _metadata_text(value: object, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a non-blank string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError(f"{name} must be valid Unicode") from None
    return value


def _check_group_split(
    kind: str, group: str, split: str, assignments: dict[tuple[str, str], str]
) -> None:
    key = (kind, group)
    previous = assignments.setdefault(key, split)
    if previous != split:
        raise ValueError(f"{kind} {group} appears in multiple splits")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _read_payload(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise ValueError(f"{path.name} is not valid strict JSON: {exc}") from exc
    if type(payload) is not dict:
        raise ValueError(f"{path.name} root must be an object")
    return payload, raw


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for split in _SPLITS:
        parser.add_argument(
            f"--{split}", required=True, type=Path, help=f"explicit {split} JSON input"
        )
    parser.add_argument("--output-dir", required=True, type=Path, help="new output directory")
    args = parser.parse_args(argv)
    output_dir: Path = args.output_dir
    if output_dir.exists() or output_dir.is_symlink():
        print("error: output directory already exists", file=sys.stderr)
        return 2

    try:
        payloads: list[dict[str, Any]] = []
        raw_hashes: dict[str, str] = {}
        for split in _SPLITS:
            payload, raw = _read_payload(getattr(args, split))
            payloads.append(payload)
            raw_hashes[split] = hashlib.sha256(raw).hexdigest()
        bundle = prepare_dataset(payloads)
        for split, payload in zip(_SPLITS, payloads, strict=True):
            if any(case["split"] != split for case in payload["cases"]):
                raise ValueError(f"--{split} input contains a case assigned to another split")
        bundle["manifest"]["source"] = {
            "hash_kind": "original_input_bytes_sha256_by_partition",
            "sha256_by_partition": raw_hashes,
        }
        manifest_bytes = (
            json.dumps(
                bundle["manifest"], ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
            )
            + "\n"
        ).encode("utf-8")
        output_dir.mkdir(parents=True, exist_ok=False)
        for filename, content in bundle["files"].items():
            (output_dir / filename).write_text(content, encoding="utf-8", newline="")
        (output_dir / "manifest.json").write_bytes(manifest_bytes)
    except (OSError, TypeError, ValueError, RecursionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
