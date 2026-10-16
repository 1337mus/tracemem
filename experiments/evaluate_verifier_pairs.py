"""Validate and score frozen two-axis verifier-pair receipts without model imports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from experiments.verifier_pairs import score_outputs
from training.verifier_data import (
    render_candidate_prompt,
    render_training_rows,
    validate_verifier_dataset,
)

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = _ROOT / "examples/synthetic/verifier-pairs-v1.json"
_DATASET_KEY = "examples/synthetic/verifier-pairs-v1.json"
_TRAINING_ROWS = {"baseline": 0, "train": 48}
_TOKEN_FIELDS = {
    "input_token_ids",
    "input_tokens",
    "output_token_ids",
    "output_tokens",
    "ended_with_eos",
    "hit_max_new_tokens",
}


def _strict_json(raw: bytes, name: str) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains a duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"{name} contains a non-finite JSON constant: {value}")

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{name} is not strict UTF-8 JSON: {exc}") from exc


def _canonical_sha(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _output_rows(value: object, name: str) -> list[dict[str, Any]]:
    if type(value) is not list or any(type(row) is not dict for row in value):
        raise ValueError(f"receipt {name} must be a list of output objects")
    return value


def _validate_output_metadata(
    arms: dict[str, list[dict[str, Any]]], receipt: dict[str, Any], settings: dict[str, Any]
) -> None:
    rows = [row for outputs in arms.values() for row in outputs]
    if not any(_TOKEN_FIELDS.intersection(row) for row in rows):
        return
    if any(not _TOKEN_FIELDS <= set(row) for row in rows):
        raise ValueError("token IDs, counts, and finish flags must be present consistently")
    tokenizer = receipt.get("tokenizer")
    eos_id = tokenizer.get("eos_token_id") if type(tokenizer) is dict else None
    max_new_tokens = settings.get("max_new_tokens")
    if type(eos_id) is not int or eos_id < 0:
        raise ValueError("tokenizer eos_token_id is required when output token metadata is present")
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("settings max_new_tokens is invalid")
    input_ids_by_arm: dict[str, dict[str, list[int]]] = {}
    for arm_name, outputs in arms.items():
        arm_input_ids: dict[str, list[int]] = {}
        input_ids_by_arm[arm_name] = arm_input_ids
        for row in outputs:
            item_id = row.get("item_id")
            if type(item_id) is not str:
                raise ValueError("output item_id must be a string when token metadata is present")
            for ids_key in ("input_token_ids", "output_token_ids"):
                ids = row[ids_key]
                if type(ids) is not list or any(
                    type(token) is not int or token < 0 for token in ids
                ):
                    raise ValueError(f"output {item_id} {ids_key} is invalid")
            input_count, output_count = row["input_tokens"], row["output_tokens"]
            if (
                type(input_count) is not int
                or input_count != len(row["input_token_ids"])
                or type(output_count) is not int
                or output_count != len(row["output_token_ids"])
                or output_count > max_new_tokens
            ):
                raise ValueError(f"output {item_id} token counts are inconsistent")
            ended, hit_limit, reason = (
                row["ended_with_eos"],
                row["hit_max_new_tokens"],
                row.get("finish_reason"),
            )
            if type(ended) is not bool or type(hit_limit) is not bool:
                raise ValueError(f"output {item_id} finish flags must be booleans")
            has_eos = bool(row["output_token_ids"] and row["output_token_ids"][-1] == eos_id)
            if ended != has_eos or ended != (reason == "eos"):
                raise ValueError(f"output {item_id} EOS metadata is inconsistent")
            if hit_limit != (output_count >= max_new_tokens):
                raise ValueError(f"output {item_id} generation-limit flag is inconsistent")
            if reason == "length" and (ended or not hit_limit):
                raise ValueError(f"output {item_id} length finish metadata is inconsistent")
            if reason == "unknown" and (ended or hit_limit):
                raise ValueError(f"output {item_id} unknown finish metadata is inconsistent")
            if reason == "empty" and (output_count or ended or hit_limit):
                raise ValueError(f"output {item_id} empty finish metadata is inconsistent")
            if item_id in arm_input_ids:
                raise ValueError(f"output {item_id} is duplicated in {arm_name} token metadata")
            arm_input_ids[item_id] = row["input_token_ids"]

    if "tuned" in input_ids_by_arm:
        baseline_ids, tuned_ids = input_ids_by_arm["baseline"], input_ids_by_arm["tuned"]
        if baseline_ids.keys() != tuned_ids.keys() or any(
            baseline_ids[item_id] != tuned_ids[item_id] for item_id in baseline_ids
        ):
            raise ValueError("baseline and tuned input token IDs must match by item_id")


def evaluate_report(dataset_raw: bytes, payload_raw: bytes, receipt_raw: bytes) -> dict[str, Any]:
    """Validate frozen provenance and score every output arm in one receipt."""
    dataset = _strict_json(dataset_raw, "dataset")
    payload = _strict_json(payload_raw, "payload")
    receipt = _strict_json(receipt_raw, "receipt")
    if type(payload) is not dict or type(receipt) is not dict:
        raise ValueError("payload and receipt must be JSON objects")
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise ValueError("payload schema_version must be integer 1")
    mode, split = payload.get("mode"), payload.get("split")
    if (
        type(mode) is not str
        or mode not in _TRAINING_ROWS
        or type(split) is not str
        or split != "dev"
    ):
        raise ValueError("only baseline/train dev payloads are supported")
    cases = validate_verifier_dataset(dataset)
    source_hashes = payload.get("source_sha256")
    dataset_sha = hashlib.sha256(dataset_raw).hexdigest()
    if type(source_hashes) is not dict or source_hashes.get(_DATASET_KEY) != dataset_sha:
        raise ValueError("payload dataset source SHA-256 does not match raw dataset bytes")
    items = [
        {
            "item_id": f"{case['case_id']}::{candidate['candidate_id']}",
            "prompt": render_candidate_prompt(case, candidate["candidate_id"]),
        }
        for case in cases
        if case["split"] == split
        for candidate in case["candidates"]
    ]
    if payload.get("items") != items:
        raise ValueError("payload item IDs or prompts do not match the selected dataset split")
    expected_train_rows = render_training_rows(dataset, "train") if mode == "train" else []
    if payload.get("train_rows") != expected_train_rows:
        raise ValueError("payload train_rows do not match the exact rendered train split")

    payload_sha = _canonical_sha(payload)
    if receipt.get("payload_sha256") != payload_sha:
        raise ValueError("receipt payload SHA-256 does not match canonical payload")
    for key in ("schema_version", "mode", "split", "runner_sha256", "model", "settings"):
        if receipt.get(key) != payload.get(key):
            raise ValueError(f"receipt {key} does not match payload")
    if receipt.get("status") != "success":
        raise ValueError("receipt status must be success")
    if type(payload.get("model")) is not dict or type(payload.get("settings")) is not dict:
        raise ValueError("payload model and settings must be objects")
    runner_hash = payload.get("runner_sha256")
    if (
        type(runner_hash) is not str
        or len(runner_hash) != 64
        or any(char not in "0123456789abcdef" for char in runner_hash)
    ):
        raise ValueError("payload runner_sha256 is invalid")

    arms = {"baseline": _output_rows(receipt.get("baseline_outputs"), "baseline_outputs")}
    if mode == "train":
        arms["tuned"] = _output_rows(receipt.get("tuned_outputs"), "tuned_outputs")
    elif "tuned_outputs" in receipt:
        raise ValueError("baseline receipt must not contain tuned_outputs")
    _validate_output_metadata(arms, receipt, payload["settings"])

    receipt_sha = hashlib.sha256(receipt_raw).hexdigest()
    scored_arms = {
        name: {
            "receipt_sha256": receipt_sha,
            "score": score_outputs(dataset, split, outputs),
        }
        for name, outputs in arms.items()
    }
    return {
        "schema_version": 1,
        "mode": mode,
        "split": split,
        "dataset_sha256": dataset_sha,
        "payload_sha256": payload_sha,
        "receipt_sha256": receipt_sha,
        "runner_sha256": payload["runner_sha256"],
        "model": payload["model"],
        "settings": payload["settings"],
        "arms": scored_arms,
    }


def _write_exclusive(path: Path, report: dict[str, Any]) -> None:
    content = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    os.fchmod(descriptor, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--payload", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = evaluate_report(
            args.dataset.read_bytes(), args.payload.read_bytes(), args.receipt.read_bytes()
        )
        _write_exclusive(args.output, report)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
