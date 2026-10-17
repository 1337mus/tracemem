"""Pure payload construction for the bounded private verifier SFT experiment."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any

from experiments import modal_private_verifier
from experiments.modal_verifier_v2 import _MODEL

_PRIOR_VOLUME_NAME = "tracemem-verifier-v2"
_PRIOR_VOLUME_MOUNT = "/mnt/tracemem-verifier-v2"
_OUTPUT_VOLUME_NAME = "tracemem-private-sft"
_OUTPUT_VOLUME_MOUNT = "/mnt/tracemem-private-sft"
_PROVENANCE = {
    "data_origin": "user_provided_private_conversations",
    "permitted_use": "explicit_user_authorized_private_fine_tuning_only",
    "license": "private_use_only_no_redistribution_license_asserted",
    "raw_content_fields": ["train_rows", "dev_prompts"],
}
_SETTINGS: dict[str, Any] = {
    "do_sample": False,
    "enable_thinking": False,
    "max_new_tokens": 128,
    "max_input_tokens": 2048,
    "attn_implementation": "eager",
    "seed": 17,
    "optimizer": "AdamW",
    "learning_rate": 0.00002,
    "gradient_accumulation_steps": 4,
    "micro_batch_size": 1,
    "epochs": 1,
    "gradient_checkpointing": True,
    "completion_only_loss": True,
    "completion_eos_loss": True,
}
_MAX_TRAIN_ROWS = 256
_MAX_DEV_ITEMS = 32
_MAX_PROMPT_CHARS = 12_000
_MAX_COMPLETION_CHARS = 4_000
_MAX_PAYLOAD_BYTES = 4_500_000


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        raise ValueError("private SFT inputs are not canonical JSON") from None


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def payload_sha256(payload: dict[str, Any]) -> str:
    """Return the exact canonical hash used for local and remote approval."""
    return _sha256(payload)


def _prior_adapter() -> dict[str, Any]:
    manifest = modal_private_verifier._EXPECTED_CANDIDATE_MANIFEST
    return {
        "manifest": deepcopy(manifest),
        "volume_name": _PRIOR_VOLUME_NAME,
        "volume_mount": _PRIOR_VOLUME_MOUNT,
        "relative_path": manifest["adapter_relative_path"],
        "artifact_sha256": deepcopy(manifest["adapter_artifact_sha256"]),
    }


def _validate_rows(
    train_rows: object, dev_prompts: object
) -> tuple[list[dict[str, str]], list[str]]:
    if type(train_rows) is not list or not train_rows:
        raise ValueError("private SFT training export must be a non-empty list")
    if type(dev_prompts) is not list or not dev_prompts:
        raise ValueError("private SFT dev export must be a non-empty list")
    if len(train_rows) > _MAX_TRAIN_ROWS or len(dev_prompts) > _MAX_DEV_ITEMS:
        raise ValueError("private SFT row or dev-item limit exceeded")
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in train_rows:
        if type(row) is not dict or set(row) != {"prompt", "completion"}:
            raise ValueError("private SFT train rows must contain only prompt and completion")
        prompt, completion = row["prompt"], row["completion"]
        if (
            type(prompt) is not str
            or not prompt.strip()
            or len(prompt) > _MAX_PROMPT_CHARS
            or "\x00" in prompt
            or prompt in seen
            or type(completion) is not str
            or not completion.strip()
            or len(completion) > _MAX_COMPLETION_CHARS
            or "\x00" in completion
        ):
            raise ValueError("private SFT train row is invalid or duplicates another prompt")
        seen.add(prompt)
        normalized.append({"prompt": prompt, "completion": completion})
    normalized_dev: list[str] = []
    for prompt in dev_prompts:
        if (
            type(prompt) is not str
            or not prompt.strip()
            or len(prompt) > _MAX_PROMPT_CHARS
            or "\x00" in prompt
            or prompt in seen
        ):
            raise ValueError("private SFT dev prompt is invalid or overlaps a train prompt")
        seen.add(prompt)
        normalized_dev.append(prompt)
    return normalized, normalized_dev


def build_payload(
    private_dataset: object,
    source_records: object,
    synthetic_replay_dataset: object | None = None,
) -> dict[str, Any]:
    """Build a deterministic payload from the existing validated exporters only."""
    from training.private_verifier_data import (
        render_private_eval_prompts,
        render_private_training_rows,
        validate_private_verifier_dataset,
    )

    cases = validate_private_verifier_dataset(private_dataset, source_records)
    if type(cases) is not list:
        raise ValueError("private SFT dataset validation returned an invalid case list")
    if type(private_dataset) is not dict or type(private_dataset.get("cases")) is not list:
        raise ValueError("private SFT dataset cases are invalid")
    if len(cases) != len(private_dataset["cases"]):
        raise ValueError("private SFT dataset validation changed its case count")
    private_rows: list[dict[str, str]] = []
    train_sessions: set[str] = set()
    dev_prompts: list[str] = []
    dev_sessions: set[str] = set()
    for case, raw_case in zip(cases, private_dataset["cases"], strict=True):
        if type(case) is not dict or type(raw_case) is not dict:
            raise ValueError("private SFT dataset contains an invalid case")
        split, session_id = case.get("split"), case.get("session_id")
        if split not in {"train", "dev", "test"} or type(session_id) is not str or not session_id:
            raise ValueError("private SFT validated case metadata is invalid")
        subset = {**private_dataset, "cases": [raw_case]}
        if split == "train":
            rows = render_private_training_rows(subset, source_records)
            if rows:
                private_rows.extend(rows)
                train_sessions.add(session_id)
        elif split == "dev":
            prompts = render_private_eval_prompts(subset, source_records, "dev")
            if prompts:
                dev_prompts.extend(prompts)
                dev_sessions.add(session_id)
    if (
        not private_rows
        or len(train_sessions) < 2
        or not dev_prompts
        or not dev_sessions
        or train_sessions & dev_sessions
    ):
        raise ValueError(
            "private SFT requires exportable train rows from two sessions "
            "and a separate dev session"
        )
    replay_rows: list[dict[str, str]] = []
    replay_sha256: str | None = None
    if synthetic_replay_dataset is not None:
        from training.verifier_data import render_training_rows, validate_verifier_dataset

        validate_verifier_dataset(synthetic_replay_dataset)
        replay_rows = render_training_rows(synthetic_replay_dataset, "train")
        replay_sha256 = _sha256(synthetic_replay_dataset)
    train_rows, dev_prompts = _validate_rows([*private_rows, *replay_rows], dev_prompts)
    payload = {
        "schema_version": 1,
        "mode": "private_sft",
        "provenance": deepcopy(_PROVENANCE),
        "source_sha256": {
            "private_dataset": _sha256(private_dataset),
            "source_records": _sha256(source_records),
            "synthetic_replay_dataset": replay_sha256,
        },
        "model": deepcopy(_MODEL),
        "settings": deepcopy(_SETTINGS),
        "prior_adapter": _prior_adapter(),
        "output_volume": {
            "name": _OUTPUT_VOLUME_NAME,
            "mount": _OUTPUT_VOLUME_MOUNT,
        },
        "train_rows": train_rows,
        "dev_prompts": dev_prompts,
    }
    if len(_canonical_bytes(payload)) > _MAX_PAYLOAD_BYTES:
        raise ValueError("private SFT payload exceeds its byte limit")
    return payload
