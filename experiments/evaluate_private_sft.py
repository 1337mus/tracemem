"""Validate and score a frozen private SFT receipt offline."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from experiments import private_sft
from experiments.evaluate_private_verifier import _candidate_refs, _outputs_by_id, _score_arm
from experiments.private_sft_remote import _build_sampling_plan
from training.private_verifier_data import validate_private_verifier_dataset

_SHA256 = re.compile(r"[0-9a-f]{64}")
_OUTPUT_FIELDS = {
    "item_id",
    "raw_output",
    "finish_reason",
    "output_tokens",
    "input_token_ids",
    "input_tokens",
    "input_token_ids_sha256",
    "output_token_ids",
    "output_token_ids_sha256",
}
_TRAINING_FIELDS = {
    "row_count",
    "microsteps",
    "optimizer_steps",
    "input_tokens",
    "supervised_tokens",
    "mean_loss",
    "optimizer_step_losses",
    "trainable_parameters",
    "adapter_hash_before_training",
    "adapter_hash_after_training",
    "adapter_changed",
    "seconds",
}
_SAMPLING_FIELDS = {"row_index_sha256", "exposure_counts", "unique_source_counts"}
_RECEIPT_FIELDS = {
    "schema_version",
    "status",
    "mode",
    "training_performed",
    "payload_sha256",
    "runner_sha256",
    "model",
    "settings",
    "prior_adapter",
    "source_sha256",
    "provenance",
    "training",
    "artifact_manifest",
    "base_outputs",
    "prior_outputs",
    "trained_outputs",
}
_ARMS = {"base": "base_outputs", "prior": "prior_outputs", "trained": "trained_outputs"}
_TERMINAL_EOS = "<|im_end|>"
_QWEN_EOS_TOKEN_ID = 151645


def _canonical_sha(value: object, name: str) -> str:
    try:
        encoded = json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError(f"{name} must be canonical JSON data") from exc
    return hashlib.sha256(encoded).hexdigest()


def _expected_sampling_metrics(train_rows: object, sampling_plan: object) -> dict[str, Any]:
    labels = ("supported", "unsupported", "uncertain")
    if (
        type(train_rows) is not list
        or not 1 <= len(train_rows) <= 256
        or type(sampling_plan) is not dict
        or set(sampling_plan) != {"strategy", "seed", "row_indices"}
        or type(sampling_plan.get("seed")) is not int
        or type(sampling_plan.get("row_indices")) is not list
        or len(sampling_plan["row_indices"]) != 256
        or any(
            type(index) is not int or not 0 <= index < len(train_rows)
            for index in sampling_plan["row_indices"]
        )
    ):
        raise ValueError("payload sampling plan is invalid")
    strategy = sampling_plan["strategy"]
    if strategy == "support-balanced-v1":
        if len(train_rows) != 256:
            raise ValueError("payload sampling plan is invalid")
        try:
            expected_plan = _build_sampling_plan(train_rows, sampling_plan["seed"])
        except (TypeError, ValueError):
            raise ValueError("payload sampling plan is invalid") from None
        if sampling_plan != expected_plan:
            raise ValueError("payload sampling plan differs from the deterministic schedule")
    elif strategy == "explicit-exposures-v1":
        if sampling_plan["seed"] != private_sft._SETTINGS["seed"]:
            raise ValueError("payload sampling plan is invalid")
        seen_prompts: set[str] = set()
        for row in train_rows:
            if type(row) is not dict or set(row) != {"prompt", "completion"}:
                raise ValueError("payload sampling rows are invalid")
            prompt = row["prompt"]
            if type(prompt) is not str or not prompt.strip() or prompt in seen_prompts:
                raise ValueError("payload sampling rows must have unique prompts")
            seen_prompts.add(prompt)
    else:
        raise ValueError("payload sampling plan is invalid")

    exposure_counts = {label: 0 for label in labels}
    unique_sources: dict[str, set[int]] = {label: set() for label in labels}
    for index in sampling_plan["row_indices"]:
        try:
            verdict = json.loads(train_rows[index]["completion"])
        except (KeyError, TypeError, ValueError, UnicodeError):
            raise ValueError("payload sampling rows are invalid") from None
        if type(verdict) is not dict or verdict.get("support") not in labels:
            raise ValueError("payload sampling rows are invalid")
        support = verdict["support"]
        exposure_counts[support] += 1
        unique_sources[support].add(index)
    row_indices = json.dumps(
        sampling_plan["row_indices"], separators=(",", ":"), allow_nan=False
    ).encode("ascii")
    return {
        "row_index_sha256": hashlib.sha256(row_indices).hexdigest(),
        "exposure_counts": exposure_counts,
        "unique_source_counts": {label: len(unique_sources[label]) for label in labels},
    }


def _validate_training(
    value: object,
    row_count: int,
    accumulation: int,
    *,
    sampling_plan: object = None,
    train_rows: object = None,
) -> dict[str, Any]:
    expected_sampling = None
    expected_microsteps = row_count
    if sampling_plan is not None:
        expected_sampling = _expected_sampling_metrics(train_rows, sampling_plan)
        if (
            type(train_rows) is not list
            or type(row_count) is not int
            or row_count != len(train_rows)
        ):
            raise ValueError("receipt training row_count does not match payload")
        expected_microsteps = 256
    expected_fields = _TRAINING_FIELDS | ({"sampling"} if sampling_plan is not None else set())
    if type(value) is not dict or set(value) != expected_fields:
        raise ValueError("receipt training statistics have an invalid shape")
    training: dict[str, Any] = value
    if type(training["row_count"]) is not int or training["row_count"] != row_count:
        raise ValueError("receipt training row_count does not match payload")
    if (
        training["microsteps"] != expected_microsteps
        or training["optimizer_steps"] != (expected_microsteps + accumulation - 1) // accumulation
    ):
        raise ValueError("receipt training steps do not match one-epoch settings")
    for field in (
        "microsteps",
        "optimizer_steps",
        "input_tokens",
        "supervised_tokens",
        "trainable_parameters",
    ):
        if type(training[field]) is not int or training[field] < (
            1 if field in {"microsteps", "optimizer_steps", "trainable_parameters"} else 0
        ):
            raise ValueError(f"receipt training {field} is invalid")
    if training["supervised_tokens"] > training["input_tokens"]:
        raise ValueError("receipt supervised_tokens exceeds input_tokens")
    losses = training["optimizer_step_losses"]
    if (
        type(losses) is not list
        or len(losses) != training["optimizer_steps"]
        or any(
            type(loss) not in {int, float} or not math.isfinite(loss) or loss < 0 for loss in losses
        )
    ):
        raise ValueError("receipt optimizer_step_losses are invalid")
    if (
        type(training["mean_loss"]) not in {int, float}
        or not math.isfinite(training["mean_loss"])
        or training["mean_loss"] < 0
    ):
        raise ValueError("receipt mean_loss is invalid")
    if (
        type(training["seconds"]) not in {int, float}
        or not math.isfinite(training["seconds"])
        or training["seconds"] < 0
    ):
        raise ValueError("receipt training seconds are invalid")
    for field in ("adapter_hash_before_training", "adapter_hash_after_training"):
        if type(training[field]) is not str or _SHA256.fullmatch(training[field]) is None:
            raise ValueError(f"receipt {field} is invalid")
    changed = training["adapter_hash_before_training"] != training["adapter_hash_after_training"]
    if type(training["adapter_changed"]) is not bool or training["adapter_changed"] != changed:
        raise ValueError("receipt adapter_changed disagrees with adapter hashes")
    if not training["adapter_changed"]:
        raise ValueError("receipt adapter did not change during training")
    if expected_sampling is not None:
        sampling = training["sampling"]
        if type(sampling) is not dict or set(sampling) != _SAMPLING_FIELDS:
            raise ValueError("receipt training sampling statistics have an invalid shape")
        if (
            type(sampling["row_index_sha256"]) is not str
            or _SHA256.fullmatch(sampling["row_index_sha256"]) is None
            or sampling["row_index_sha256"] != expected_sampling["row_index_sha256"]
        ):
            raise ValueError("receipt training sampling row-index hash is invalid")
        for field in ("exposure_counts", "unique_source_counts"):
            counts = sampling[field]
            if (
                type(counts) is not dict
                or set(counts) != set(expected_sampling[field])
                or any(type(count) is not int or count < 0 for count in counts.values())
                or counts != expected_sampling[field]
            ):
                raise ValueError(f"receipt training sampling {field} are invalid")
    return training


def _validate_manifest(value: object, output_volume: dict[str, Any], payload_sha: str) -> None:
    if type(value) is not dict or set(value) != {
        "volume_name",
        "volume_mount",
        "relative_path",
        "artifact_sha256",
    }:
        raise ValueError("receipt artifact_manifest has an invalid shape")
    if (
        value["volume_name"] != output_volume["name"]
        or value["volume_mount"] != output_volume["mount"]
    ):
        raise ValueError("receipt artifact_manifest volume differs from payload")
    if value["relative_path"] != f"runs/{payload_sha}/adapter":
        raise ValueError("receipt artifact_manifest path does not match payload")
    hashes = value["artifact_sha256"]
    required = {"adapter_config.json", "adapter_model.safetensors"}
    if (
        type(hashes) is not dict
        or not required <= set(hashes)
        or not set(hashes) <= required | {"README.md"}
    ):
        raise ValueError("receipt artifact_manifest hashes have an invalid shape")
    if any(
        type(digest) is not str or _SHA256.fullmatch(digest) is None for digest in hashes.values()
    ):
        raise ValueError("receipt artifact_manifest contains an invalid hash")


def _validate_outputs(
    value: object, expected_ids: list[str], arm: str
) -> dict[str, dict[str, Any]]:
    outputs = _outputs_by_id(value, expected_ids, arm)
    for row in outputs.values():
        if set(row) != _OUTPUT_FIELDS:
            raise ValueError(f"receipt {arm} output row has an invalid shape")
        if type(row["raw_output"]) is not str or row["finish_reason"] not in {
            "eos",
            "length",
            "unknown",
            "empty",
        }:
            raise ValueError(f"receipt {arm} output text or finish_reason is invalid")
        input_ids, output_ids = row["input_token_ids"], row["output_token_ids"]
        for ids, field in ((input_ids, "input"), (output_ids, "output")):
            if type(ids) is not list or any(type(token) is not int or token < 0 for token in ids):
                raise ValueError(f"receipt {arm} {field}_token_ids are invalid")
        if not input_ids:
            raise ValueError(f"receipt {arm} input_token_ids must be non-empty")
        for ids_field, sha_field in (
            ("input_token_ids", "input_token_ids_sha256"),
            ("output_token_ids", "output_token_ids_sha256"),
        ):
            digest = row[sha_field]
            if (
                type(digest) is not str
                or _SHA256.fullmatch(digest) is None
                or digest != _canonical_sha(row[ids_field], sha_field)
            ):
                raise ValueError(f"receipt {arm} {sha_field} does not match token IDs")
        if type(row["output_tokens"]) is not int or row["output_tokens"] != len(output_ids):
            raise ValueError(f"receipt {arm} output_tokens does not match output_token_ids")
        if type(row["input_tokens"]) is not int or row["input_tokens"] != len(input_ids):
            raise ValueError(f"receipt {arm} input_tokens does not match input_token_ids")
    return outputs


def _normalize_terminal_eos(
    outputs: dict[str, dict[str, dict[str, Any]]], tokenizer: object
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, int]]:
    normalized: dict[str, dict[str, dict[str, Any]]] = {}
    removed_by_arm: dict[str, int] = {}
    for arm, rows in outputs.items():
        normalized[arm] = {}
        removed = 0
        for item_id, row in rows.items():
            copied = dict(row)
            raw = row["raw_output"]
            if raw.endswith(_TERMINAL_EOS):
                if (
                    type(tokenizer) is not dict
                    or type(tokenizer.get("eos_token_id")) is not int
                    or tokenizer["eos_token_id"] != _QWEN_EOS_TOKEN_ID
                    or row["finish_reason"] != "eos"
                    or not row["output_token_ids"]
                    or row["output_token_ids"][-1] != _QWEN_EOS_TOKEN_ID
                ):
                    raise ValueError("terminal EOS marker does not match tokenizer and token IDs")
                copied["raw_output"] = raw[: -len(_TERMINAL_EOS)]
                removed += 1
            normalized[arm][item_id] = copied
        removed_by_arm[arm] = removed
    return normalized, removed_by_arm


def evaluate_report(
    dataset: object,
    source_records: object,
    payload: object,
    receipt: object,
    *,
    approved_runner_sha256: str,
    synthetic_replay_dataset: object | None = None,
) -> dict[str, Any]:
    """Validate a frozen training receipt and return aggregate dev-set metrics only."""
    if type(approved_runner_sha256) is not str or _SHA256.fullmatch(approved_runner_sha256) is None:
        raise ValueError("approved_runner_sha256 must be 64 lowercase hex characters")
    sampling_strategy = (
        "support-balanced-v1" if type(payload) is dict and "sampling_plan" in payload else None
    )
    try:
        if type(payload) is dict and "execution_id" in payload:
            expected = private_sft.build_payload(
                dataset,
                source_records,
                synthetic_replay_dataset,
                sampling_strategy=sampling_strategy,
                execution_id=payload["execution_id"],
            )
        else:
            expected = private_sft.build_payload(
                dataset,
                source_records,
                synthetic_replay_dataset,
                sampling_strategy=sampling_strategy,
            )
    except (TypeError, ValueError, UnicodeError):
        raise ValueError("dataset and source records failed private exporter validation") from None
    if type(payload) is not dict:
        raise ValueError("payload does not match exact exporter output")
    payload_sha = _canonical_sha(payload, "payload")
    if payload_sha != _canonical_sha(expected, "expected payload"):
        raise ValueError("payload does not match exact exporter output")
    payload_obj: dict[str, Any] = payload
    if type(receipt) is not dict or not _RECEIPT_FIELDS <= set(receipt):
        raise ValueError("receipt is missing required private SFT fields")
    receipt_obj: dict[str, Any] = receipt
    if (
        type(receipt_obj["schema_version"]) is not int
        or receipt_obj["schema_version"] != 1
        or receipt_obj["status"] != "success"
        or receipt_obj["mode"] != "private_sft"
        or receipt_obj["training_performed"] is not True
    ):
        raise ValueError("receipt is not a successful private SFT result")
    if receipt_obj["payload_sha256"] != payload_sha:
        raise ValueError("receipt payload SHA-256 does not match payload")
    if receipt_obj["runner_sha256"] != approved_runner_sha256:
        raise ValueError("receipt runner SHA-256 does not match approved runner")
    if any(
        receipt_obj[field] != payload_obj[field]
        for field in ("model", "settings", "prior_adapter", "source_sha256", "provenance")
    ):
        raise ValueError("receipt metadata does not match payload")
    accumulation = payload_obj["settings"].get("gradient_accumulation_steps")
    if type(accumulation) is not int or accumulation < 1:
        raise ValueError("payload gradient_accumulation_steps is invalid")
    training = _validate_training(
        receipt_obj["training"],
        len(payload_obj["train_rows"]),
        accumulation,
        sampling_plan=payload_obj.get("sampling_plan"),
        train_rows=payload_obj["train_rows"] if "sampling_plan" in payload_obj else None,
    )
    _validate_manifest(receipt_obj["artifact_manifest"], payload_obj["output_volume"], payload_sha)

    cases = validate_private_verifier_dataset(dataset, source_records)
    refs = _candidate_refs(cases)
    if len(refs) != len(payload_obj["dev_prompts"]):
        raise ValueError("payload dev prompt count does not match exportable references")
    item_ids = [f"dev_{index:03d}" for index in range(len(payload_obj["dev_prompts"]))]
    mapped = {
        arm: _validate_outputs(receipt_obj[field], item_ids, arm) for arm, field in _ARMS.items()
    }
    for item_id in item_ids:
        base_ids = mapped["base"][item_id]["input_token_ids"]
        if any(mapped[arm][item_id]["input_token_ids"] != base_ids for arm in ("prior", "trained")):
            raise ValueError("input token IDs differ across scoring arms")
    raw_arms = {arm: _score_arm(refs, item_ids, mapped[arm]) for arm in _ARMS}
    content_outputs, normalized_counts = _normalize_terminal_eos(
        mapped, receipt_obj.get("tokenizer")
    )
    arms = {arm: _score_arm(refs, item_ids, content_outputs[arm]) for arm in _ARMS}
    training_report = {
        field: training[field]
        for field in (
            "row_count",
            "microsteps",
            "optimizer_steps",
            "input_tokens",
            "supervised_tokens",
            "mean_loss",
            "trainable_parameters",
            "adapter_changed",
            "seconds",
        )
    }
    if "sampling" in training:
        training_report["sampling"] = training["sampling"]
    return {
        "schema_version": 1,
        "payload_sha256": payload_sha,
        "receipt_sha256": _canonical_sha(receipt_obj, "receipt"),
        "runner_sha256": approved_runner_sha256,
        "scoring_convention": "content_after_verified_terminal_eos_removal",
        "normalization": {
            "terminal_eos_markers_removed": sum(normalized_counts.values()),
            "by_arm": normalized_counts,
        },
        "raw_validity": {arm: raw_arms[arm]["summary"]["validity"] for arm in _ARMS},
        "scope": {
            "split": "dev",
            "case_count": len({case["case_id"] for case, _ in refs}),
            "candidate_count": len(refs),
            "session_count": len({case["session_id"] for case, _ in refs}),
        },
        "training": training_report,
        "arms": arms,
    }
