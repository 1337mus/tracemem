"""Strict local scoring for the frozen verifier-v2 continuation receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from experiments import modal_verifier_v2
from experiments.evaluate_verifier_pairs import (
    _output_rows,
    _strict_json,
    _validate_output_metadata,
    _write_exclusive,
)
from experiments.verifier_pairs import score_outputs
from training.verifier_data import validate_verifier_dataset

_ROOT = Path(__file__).resolve().parents[1]
_CRITERIA_PATH = _ROOT / "docs/verifier-v2-criteria.json"
_CRITERIA_SHA256 = "053e9bf5d26846b2d6f9d3d9329bfe8e331329a1ceb8fb1ac888df0ef831f958"
_HISTORICAL_TRANSFER_PATH = _ROOT / "examples/synthetic/verifier-transfer-remote-v1.json"
_HISTORICAL_TRANSFER_SHA256 = "070ad8665bfd34fc7192c67943b0e27141c9cf714dfff83abb42a1de7ed4766f"
_DATASETS = {
    "new_dev": (
        _ROOT / "examples/synthetic/verifier-v2-train-dev.json",
        "examples/synthetic/verifier-v2-train-dev.json",
    ),
    "transfer_regression": (
        _ROOT / "examples/synthetic/verifier-transfer-v1.json",
        "examples/synthetic/verifier-transfer-v1.json",
    ),
    "original_paired_dev": (
        _ROOT / "examples/synthetic/verifier-pairs-v1.json",
        "examples/synthetic/verifier-pairs-v1.json",
    ),
}
_ARMS = ("base", "prior", "continued")
_OUTPUT_FIELDS = {
    "item_id",
    "raw_output",
    "input_token_ids",
    "input_tokens",
    "output_token_ids",
    "output_tokens",
    "ended_with_eos",
    "hit_max_new_tokens",
    "finish_reason",
}
_REPLAY_FIELDS = tuple(sorted(_OUTPUT_FIELDS))
_PINNED_RUNTIME_VERSIONS = {
    "torch": "2.7.1",
    "transformers": "4.51.3",
    "tokenizers": "0.21.1",
    "accelerate": "1.6.0",
    "peft": "0.15.2",
    "bitsandbytes": "0.45.5",
    "safetensors": "0.5.3",
}
_A10_GPU_NAMES = {"NVIDIA A10", "NVIDIA A10G"}
_TIMING_FIELDS = (
    "model_load_seconds",
    "baseline_seconds",
    "prior_seconds",
    "training_seconds",
    "tuned_reload_seconds",
    "tuned_seconds",
    "total_seconds",
)


def _valid_sha256(value: object) -> bool:
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _canonical_sha(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_frozen_sources(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    criteria_raw = _CRITERIA_PATH.read_bytes()
    criteria_sha = hashlib.sha256(criteria_raw).hexdigest()
    if criteria_sha != _CRITERIA_SHA256:
        raise ValueError("verifier v2 criteria SHA-256 is not the frozen value")
    criteria = _strict_json(criteria_raw, "verifier v2 criteria")
    if type(criteria) is not dict or criteria.get("schema_version") != 1:
        raise ValueError("verifier v2 criteria schema is invalid")

    source_hashes = payload.get("source_sha256")
    if type(source_hashes) is not dict:
        raise ValueError("payload source_sha256 must be an object")
    datasets: dict[str, Any] = {}
    dataset_hashes: dict[str, str] = {}
    for cohort, (path, source_key) in _DATASETS.items():
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if source_hashes.get(source_key) != digest:
            raise ValueError(f"payload source SHA-256 does not match {source_key}")
        parsed = _strict_json(raw, cohort + " dataset")
        validate_verifier_dataset(parsed)
        datasets[cohort] = parsed
        dataset_hashes[cohort] = digest

    historical_raw = _HISTORICAL_TRANSFER_PATH.read_bytes()
    historical_sha = hashlib.sha256(historical_raw).hexdigest()
    if historical_sha != _HISTORICAL_TRANSFER_SHA256:
        raise ValueError("historical transfer receipt SHA-256 is not the frozen value")
    if source_hashes.get("examples/synthetic/verifier-transfer-remote-v1.json") != historical_sha:
        raise ValueError("payload historical transfer receipt SHA-256 does not match raw bytes")
    historical = _strict_json(historical_raw, "historical transfer receipt")
    if type(historical) is not dict:
        raise ValueError("historical transfer receipt must be an object")
    return {**datasets, "criteria": criteria, "historical": historical}, {
        **dataset_hashes,
        "criteria": criteria_sha,
        "historical_transfer_receipt": historical_sha,
    }


def _validate_payload(payload_raw: bytes) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    payload = _strict_json(payload_raw, "payload")
    if type(payload) is not dict:
        raise ValueError("payload must be a JSON object")
    expected = modal_verifier_v2.build_payload()
    if _canonical_sha(payload) != modal_verifier_v2.payload_sha256(expected):
        raise ValueError("payload does not exactly match the current frozen verifier v2 payload")
    sources, source_digests = _load_frozen_sources(payload)
    runner_sha = payload.get("runner_sha256")
    if not _valid_sha256(runner_sha):
        raise ValueError("payload runner_sha256 is invalid")
    return payload, sources, source_digests


def _validate_runtime(receipt: dict[str, Any]) -> dict[str, Any]:
    versions = receipt.get("versions")
    if type(versions) is not dict or any(
        versions.get(package) != expected for package, expected in _PINNED_RUNTIME_VERSIONS.items()
    ):
        raise ValueError("receipt runtime version does not match the frozen image")
    python_version = versions.get("python")
    if type(python_version) is not str or not python_version.startswith("3.12."):
        raise ValueError("receipt runtime version does not match the frozen Python 3.12 image")
    if type(versions.get("modal")) is not str or not versions["modal"]:
        raise ValueError("receipt runtime version is missing Modal provenance")

    tokenizer = receipt.get("tokenizer")
    if (
        type(tokenizer) is not dict
        or type(tokenizer.get("class")) is not str
        or not tokenizer["class"]
        or type(tokenizer.get("is_fast")) is not bool
        or not _valid_sha256(tokenizer.get("chat_template_sha256"))
        or type(tokenizer.get("eos_token_id")) is not int
        or tokenizer["eos_token_id"] < 0
        or type(tokenizer.get("pad_token_id")) is not int
        or tokenizer["pad_token_id"] < 0
    ):
        raise ValueError("receipt tokenizer provenance is invalid")

    timings = receipt.get("timings")
    if type(timings) is not dict or any(
        type(timings.get(key)) not in (int, float)
        or not math.isfinite(timings[key])
        or timings[key] < 0
        for key in _TIMING_FIELDS
    ):
        raise ValueError("receipt runtime timings are invalid")
    if timings["total_seconds"] > 1800:
        raise ValueError("receipt runtime exceeded the 1800 second training limit")
    stage_total = sum(timings[key] for key in _TIMING_FIELDS if key != "total_seconds")
    if timings["total_seconds"] + 0.01 < stage_total:
        raise ValueError("receipt total timing is shorter than its recorded stages")

    memory = receipt.get("memory")
    if (
        type(memory) is not dict
        or type(memory.get("gpu_name")) is not str
        or memory["gpu_name"] not in _A10_GPU_NAMES
        or type(memory.get("peak_allocated_bytes")) is not int
        or memory["peak_allocated_bytes"] < 0
    ):
        raise ValueError("receipt runtime hardware metadata is invalid")
    return {"versions": versions, "tokenizer": tokenizer, "timings": timings, "memory": memory}


def _expected_prior_hashes(payload: dict[str, Any]) -> dict[str, str]:
    adapter = payload.get("prior_adapter")
    if type(adapter) is not dict:
        raise ValueError("payload prior_adapter is invalid")
    config_sha256 = adapter.get("config_sha256")
    model_sha256 = adapter.get("safetensors_sha256")
    if (
        type(config_sha256) is not str
        or not _valid_sha256(config_sha256)
        or type(model_sha256) is not str
        or not _valid_sha256(model_sha256)
    ):
        raise ValueError("payload prior adapter artifact hashes are invalid")
    return {
        "adapter_config.json": config_sha256,
        "adapter_model.safetensors": model_sha256,
    }


def _validate_training(
    receipt: dict[str, Any], payload: dict[str, Any], payload_sha: str
) -> dict[str, Any]:
    training = receipt.get("training")
    if type(training) is not dict:
        raise ValueError("receipt training metadata must be an object")
    settings, counts = payload["settings"], payload["counts"]
    expected_counts = {
        "row_count": 192,
        "epochs": 2,
        "microsteps": 384,
        "optimizer_steps": 96,
    }
    if any(
        type(training.get(key)) is not int or training[key] != value
        for key, value in expected_counts.items()
    ):
        raise ValueError("receipt training counts do not match the frozen 192-row schedule")
    if counts.get("train_rows") != 192 or settings.get("learning_rate") != 0.00005:
        raise ValueError("payload training counts or learning rate changed")
    if training.get("learning_rate") != 0.00005:
        raise ValueError("receipt training learning rate does not match the frozen value")

    for key, expected_length in (("microstep_losses", 384), ("optimizer_step_losses", 96)):
        losses = training.get(key)
        if (
            type(losses) is not list
            or len(losses) != expected_length
            or any(
                type(loss) not in (int, float) or not math.isfinite(loss) or loss < 0
                for loss in losses
            )
        ):
            raise ValueError(f"receipt training {key} are invalid")

    names = training.get("trainable_parameter_names")
    target_modules = settings.get("lora", {}).get("target_modules")
    if (
        type(names) is not list
        or not names
        or len(names) > 4096
        or len(names) != len(set(names))
        or type(target_modules) is not list
    ):
        raise ValueError("receipt trainable LoRA parameter names are invalid")
    allowed_name = re.compile(
        r"(?:^|\.)(?:"
        + "|".join(re.escape(name) for name in target_modules)
        + r")\.lora_[AB](?:\.[^.]+)?\.weight$"
    )
    if any(
        type(name) is not str or len(name) > 512 or not allowed_name.search(name) for name in names
    ):
        raise ValueError("receipt contains a trainable parameter outside the frozen LoRA targets")
    parameter_count = training.get("trainable_parameter_count")
    if type(parameter_count) is not int or parameter_count != 33_030_144:
        raise ValueError("receipt trainable parameter count does not match the frozen LoRA shape")
    adapter_path = training.get("adapter_path")
    expected_path = f"/mnt/tracemem-verifier-v2/runs/{payload_sha}/adapter"
    if type(adapter_path) is not str or adapter_path != expected_path:
        raise ValueError("receipt new adapter path does not match the approved run")

    expected_prior = _expected_prior_hashes(payload)
    prior_before = receipt.get("prior_adapter_artifact_sha256")
    prior_after = receipt.get("prior_adapter_artifact_sha256_after_training")
    if prior_before != expected_prior or prior_after != expected_prior:
        raise ValueError("receipt prior adapter artifact hashes changed during continuation")

    artifacts = training.get("adapter_artifact_sha256")
    core_artifacts = set(expected_prior)
    allowed_artifacts = core_artifacts | {"README.md"}
    if (
        type(artifacts) is not dict
        or not core_artifacts <= set(artifacts)
        or not set(artifacts) <= allowed_artifacts
        or any(not _valid_sha256(value) for value in artifacts.values())
        or artifacts["adapter_model.safetensors"] == expected_prior["adapter_model.safetensors"]
    ):
        raise ValueError("receipt new adapter artifact SHA-256 values are invalid or unchanged")
    if training.get("adapter_artifact_sha256_after_reload") != artifacts:
        raise ValueError(
            "receipt reloaded adapter artifact SHA-256 values do not match the saved adapter"
        )
    mutation = training.get("adapter_mutation")
    if (
        type(mutation) is not dict
        or not _valid_sha256(mutation.get("initial_state_sha256"))
        or not _valid_sha256(mutation.get("final_state_sha256"))
        or mutation.get("changed") is not True
        or mutation["initial_state_sha256"] == mutation["final_state_sha256"]
    ):
        raise ValueError("receipt does not prove the new adapter changed during training")
    return training


def _validate_outputs(
    receipt: dict[str, Any], payload: dict[str, Any], settings: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    fields = {
        "base": "baseline_outputs",
        "prior": "prior_outputs",
        "continued": "tuned_outputs",
    }
    arms = {arm: _output_rows(receipt.get(key), key) for arm, key in fields.items()}
    if any(len(rows) != 108 for rows in arms.values()):
        raise ValueError("each receipt arm must contain exactly 108 output rows")
    if any(any(not _OUTPUT_FIELDS <= set(row) for row in rows) for rows in arms.values()):
        raise ValueError(
            "token IDs, counts, finish flags, and raw text are required on every output"
        )

    # Reuse the v1 validator for token/count/EOS consistency, then extend its pairwise
    # comparison to all three arms.
    _validate_output_metadata(
        {"baseline": arms["base"], "prior": arms["prior"], "tuned": arms["continued"]},
        receipt,
        settings,
    )
    input_ids = {
        arm: {row["item_id"]: row["input_token_ids"] for row in rows} for arm, rows in arms.items()
    }
    if any(input_ids[arm] != input_ids["base"] for arm in ("prior", "continued")):
        raise ValueError("input token IDs must match by item_id across all three arms")
    max_input_tokens = settings.get("max_input_tokens")
    if type(max_input_tokens) is not int or max_input_tokens < 1:
        raise ValueError("payload max_input_tokens is invalid")
    if any(
        not 1 <= row["input_tokens"] <= max_input_tokens for rows in arms.values() for row in rows
    ):
        raise ValueError("input token count is outside the frozen inference limits")
    return arms


def _replay_differences(
    arms: dict[str, list[dict[str, Any]]], payload: dict[str, Any]
) -> list[dict[str, str]]:
    expected_rows = payload.get("prior_reference_outputs")
    if type(expected_rows) is not dict:
        raise ValueError("payload prior_reference_outputs is invalid")
    references = {
        "base": expected_rows.get("baseline_outputs"),
        "prior": expected_rows.get("tuned_outputs"),
    }
    transfer_ids = {
        item["item_id"] for item in payload["items"] if item["cohort"] == "transfer_regression"
    }
    differences: list[dict[str, str]] = []
    for arm, reference in references.items():
        if type(reference) is not list:
            raise ValueError("payload historical transfer reference arm is invalid")
        expected_by_id = {row.get("item_id"): row for row in reference}
        actual_by_id = {row["item_id"]: row for row in arms[arm] if row["item_id"] in transfer_ids}
        if expected_by_id.keys() != transfer_ids or actual_by_id.keys() != transfer_ids:
            raise ValueError("historical transfer reference IDs do not match the frozen cohort")
        for item_id in sorted(transfer_ids):
            expected, actual = expected_by_id[item_id], actual_by_id[item_id]
            if any(actual.get(field) != expected.get(field) for field in _REPLAY_FIELDS):
                differences.append({"arm": arm, "item_id": item_id})
    return differences


def _observed(
    score: dict[str, Any], outputs: list[dict[str, Any]], max_new_tokens: int
) -> dict[str, int | bool]:
    aggregate = score["aggregate"]
    support = aggregate["support"]
    unsupported = support["confusion"]["unsupported"]
    eligibility = aggregate["eligibility"]
    valid = aggregate["validity"]
    candidate_results = score["candidate_results"]
    return {
        "candidate_count": aggregate["candidate_count"],
        "supported_retained": support["by_reference"]["supported"]["retained"],
        "supported_total": support["by_reference"]["supported"]["total"],
        "unsupported_rejected": support["by_reference"]["unsupported"]["caught"],
        "unsupported_total": support["by_reference"]["unsupported"]["total"],
        "unsupported_accepted": unsupported["supported"],
        "unsupported_uncertain": unsupported["uncertain"],
        "supported_eligible_retained": sum(
            row["valid"]
            and row["reference"] == {"support": "supported", "eligibility": "eligible"}
            and row["verdict"] == {"support": "supported", "eligibility": "eligible"}
            for row in candidate_results
        ),
        "supported_eligible_total": sum(
            row["reference"] == {"support": "supported", "eligibility": "eligible"}
            for row in candidate_results
        ),
        "unsupported_eligible_accepted": sum(
            row["valid"]
            and row["reference"] == {"support": "unsupported", "eligibility": "eligible"}
            and row["verdict"] == {"support": "supported", "eligibility": "eligible"}
            for row in candidate_results
        ),
        "unsupported_eligible_total": sum(
            row["reference"] == {"support": "unsupported", "eligibility": "eligible"}
            for row in candidate_results
        ),
        "decisive_eligibility_exact": eligibility["decisive_exact_correct"],
        "decisive_eligibility_total": eligibility["decisive_total"],
        "uncertain_eligibility_exact": eligibility["uncertain"]["exact_correct"],
        "uncertain_eligibility_total": eligibility["uncertain"]["total"],
        "valid_count": valid["valid"],
        "invalid_count": valid["invalid"],
        "truncated_count": valid["truncated"],
        "valid_untruncated": valid["valid"] == len(outputs)
        and valid["truncated"] == 0
        and all(
            not row["hit_max_new_tokens"] and row["output_tokens"] < max_new_tokens
            for row in outputs
        ),
    }


def _eligibility_sensitivity(score: dict[str, Any]) -> dict[str, int | str]:
    rows = [row for row in score["candidate_results"] if row["case_id"] != "case_transfer_04"]
    decisive = [
        row for row in rows if row["reference"]["eligibility"] in {"eligible", "ineligible"}
    ]
    uncertain = [row for row in rows if row["reference"]["eligibility"] == "uncertain"]
    return {
        "excluded_case_id": "case_transfer_04",
        "candidate_count": len(rows),
        "decisive_eligibility_exact": sum(
            row["valid"] and row["verdict"]["eligibility"] == row["reference"]["eligibility"]
            for row in decisive
        ),
        "decisive_eligibility_total": len(decisive),
        "uncertain_eligibility_exact": sum(
            row["valid"] and row["verdict"]["eligibility"] == "uncertain" for row in uncertain
        ),
        "uncertain_eligibility_total": len(uncertain),
    }


def _criterion(
    arms: dict[str, dict[str, Any]],
    criteria: dict[str, Any],
    replay_differences: list[dict[str, str]],
) -> dict[str, Any]:
    continued = arms["continued"]["cohorts"]
    prior = arms["prior"]["cohorts"]
    thresholds = criteria["cohorts"]
    observed = {
        arm: {cohort: data["observed"] for cohort, data in entry["cohorts"].items()}
        for arm, entry in arms.items()
    }
    checks: dict[str, bool] = {
        "all_arms_all_outputs_valid_untruncated": all(
            data["observed"]["valid_untruncated"]
            for entry in arms.values()
            for data in entry["cohorts"].values()
        ),
        "historical_transfer_replay_matches": not replay_differences,
    }
    new = continued["new_dev"]["observed"]
    new_limits = thresholds["new_dev"]
    checks.update(
        {
            "new_dev_supported_retained": new["supported_retained"]
            >= new_limits["supported_retained_min"],
            "new_dev_unsupported_rejected": new["unsupported_rejected"]
            >= new_limits["unsupported_rejected_min"],
            "new_dev_supported_eligible_retained": new["supported_eligible_retained"]
            >= new_limits["supported_eligible_retained_min"],
            "new_dev_unsupported_eligible_accepted": new["unsupported_eligible_accepted"]
            <= new_limits["unsupported_eligible_accepted_max"],
            "new_dev_decisive_eligibility_exact": new["decisive_eligibility_exact"]
            >= new_limits["decisive_eligibility_exact_min"],
            "new_dev_uncertain_eligibility_exact": new["uncertain_eligibility_exact"]
            >= new_limits["uncertain_eligibility_exact_min"],
        }
    )
    transfer = continued["transfer_regression"]["observed"]
    transfer_limits = thresholds["transfer_regression"]
    checks.update(
        {
            "transfer_supported_retained": transfer["supported_retained"]
            >= transfer_limits["supported_retained_min"],
            "transfer_unsupported_rejected": transfer["unsupported_rejected"]
            >= transfer_limits["unsupported_rejected_min"],
            "transfer_supported_eligible_retained": transfer["supported_eligible_retained"]
            >= transfer_limits["supported_eligible_retained_min"],
            "transfer_decisive_eligibility_exact": transfer["decisive_eligibility_exact"]
            >= transfer_limits["decisive_eligibility_exact_min"],
            "transfer_uncertain_eligibility_exact": transfer["uncertain_eligibility_exact"]
            >= transfer_limits["uncertain_eligibility_exact_min"],
        }
    )
    original = continued["original_paired_dev"]["observed"]
    original_limits = thresholds["original_paired_dev"]
    checks.update(
        {
            "original_paired_supported_retained": original["supported_retained"]
            >= original_limits["supported_retained_min"],
            "original_paired_unsupported_rejected": original["unsupported_rejected"]
            >= original_limits["unsupported_rejected_min"],
        }
    )
    for cohort in thresholds:
        checks[f"{cohort}_unsupported_false_accepts_no_increase"] = (
            arms["continued"]["cohorts"][cohort]["observed"]["unsupported_accepted"]
            <= prior[cohort]["observed"]["unsupported_accepted"]
        )
    return {
        "passed": all(checks.values()),
        "thresholds": thresholds,
        "checks": checks,
        "observed": observed,
        "historical_transfer_replay_differences": replay_differences,
        "eligibility_sensitivity_excluding_case_transfer_04": {
            arm: entry["cohorts"]["transfer_regression"][
                "eligibility_sensitivity_excluding_case_transfer_04"
            ]
            for arm, entry in arms.items()
        },
    }


def evaluate_report(payload_raw: bytes, receipt_raw: bytes) -> dict[str, Any]:
    """Validate exact frozen provenance, score each development cohort, and report all gates."""
    payload, sources, source_digests = _validate_payload(payload_raw)
    receipt = _strict_json(receipt_raw, "receipt")
    if type(receipt) is not dict:
        raise ValueError("receipt must be a JSON object")
    payload_sha = modal_verifier_v2.payload_sha256(payload)
    if receipt.get("payload_sha256") != payload_sha:
        raise ValueError("receipt payload SHA-256 does not match canonical payload")
    for key in (
        "schema_version",
        "mode",
        "split",
        "runner_sha256",
        "source_sha256",
        "model",
        "settings",
        "prior_adapter",
        "counts",
    ):
        if receipt.get(key) != payload.get(key):
            raise ValueError(f"receipt {key} does not match payload")
    if receipt.get("status") != "success":
        raise ValueError("receipt status must be success")
    runtime = _validate_runtime(receipt)
    training = _validate_training(receipt, payload, payload_sha)
    outputs = _validate_outputs(receipt, payload, payload["settings"])

    cohort_for_item = {item["item_id"]: item["cohort"] for item in payload["items"]}
    if len(cohort_for_item) != 108 or set(cohort_for_item.values()) != set(_DATASETS):
        raise ValueError("payload evaluation items do not match the frozen cohort inventory")
    receipt_sha = hashlib.sha256(receipt_raw).hexdigest()
    arms: dict[str, dict[str, Any]] = {}
    for arm, rows in outputs.items():
        rows_by_cohort = {
            cohort: [row for row in rows if cohort_for_item.get(row.get("item_id")) == cohort]
            for cohort in _DATASETS
        }
        cohorts: dict[str, dict[str, Any]] = {}
        for cohort, dataset in ((name, sources[name]) for name in _DATASETS):
            score = score_outputs(dataset, "dev", rows_by_cohort[cohort])
            cohorts[cohort] = {
                "score": score,
                "observed": _observed(
                    score, rows_by_cohort[cohort], payload["settings"]["max_new_tokens"]
                ),
            }
        cohorts["transfer_regression"]["eligibility_sensitivity_excluding_case_transfer_04"] = (
            _eligibility_sensitivity(cohorts["transfer_regression"]["score"])
        )
        arms[arm] = {"receipt_sha256": receipt_sha, "cohorts": cohorts}

    replay_differences = _replay_differences(outputs, payload)
    criteria = sources["criteria"]
    criterion = _criterion(arms, criteria, replay_differences)
    prior_adapter = payload["prior_adapter"]
    return {
        "schema_version": 1,
        "mode": payload["mode"],
        "split": payload["split"],
        "dataset_sha256": source_digests,
        "payload_sha256": payload_sha,
        "receipt_sha256": receipt_sha,
        "runner_sha256": payload["runner_sha256"],
        "source_sha256": payload["source_sha256"],
        "model": payload["model"],
        "settings": payload["settings"],
        "prior_adapter": prior_adapter,
        "training": training,
        "runtime": runtime,
        "arms": arms,
        "criterion": criterion,
        "candidate": {
            "model": payload["model"],
            "settings": payload["settings"],
            "training_payload_sha256": payload_sha,
            "training_receipt_sha256": receipt_sha,
            "adapter_relative_path": f"runs/{payload_sha}/adapter",
            "adapter_artifact_sha256": {
                key: training["adapter_artifact_sha256"][key]
                for key in ("adapter_config.json", "adapter_model.safetensors")
            },
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = evaluate_report(args.payload.read_bytes(), args.receipt.read_bytes())
        _write_exclusive(args.output, report)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
