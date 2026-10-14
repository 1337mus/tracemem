"""Validate and score raw scenario receipts without changing their outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from experiments import scenario_payload
from tracemem.anchoring import anchor_unique_quotes
from tracemem.challenge import (
    _case_records,
    evaluate_outputs,
)
from tracemem.claims import parse_candidates

_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_RECEIPT_BYTES = 10_000_000
_RECEIPT_KEYS = {
    "schema_version",
    "mode",
    "input_sha256",
    "prompt_source_sha256",
    "payload_sha256",
    "prompt_sha256",
    "prompt_version",
    "model",
    "package_versions",
    "gpu_compute_seconds",
    "model_load_seconds",
    "remote_function_elapsed_seconds",
    "peak_gpu_memory_bytes",
    "python_version",
    "torch_cuda_version",
    "gpu_name",
    "modal_function_call_id",
    "modal_app_id",
    "client_call_wall_seconds_including_image_and_startup",
    "runner_sha256",
    "generation_eos_token_ids",
    "generation_pad_token_id",
    "case_count",
    "generated_outputs",
}
_INTERPRETATION_NOTE = (
    "Exact structured extraction diagnostics only. Quote anchoring repairs only uniquely located "
    "exact quotes and does not establish entailment. Labels are agent-authored provisional labels, "
    "not independent human annotations."
)


def _strict_json_loads(text: str) -> object:
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _digest(value: object, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"receipt {name} is invalid")
    return value


def _nonnegative_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"receipt {name} must be a finite nonnegative number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"receipt {name} must be a finite nonnegative number")
    return number


def _provenance(receipt: dict[str, Any]) -> dict[str, Any]:
    versions = receipt["package_versions"]
    if (
        type(versions) is not dict
        or not versions
        or any(
            type(name) is not str or not name or type(version) is not str or not version
            for name, version in versions.items()
        )
    ):
        raise ValueError("receipt package versions are invalid")
    timings = {
        name: _nonnegative_number(receipt[name], name)
        for name in (
            "gpu_compute_seconds",
            "model_load_seconds",
            "remote_function_elapsed_seconds",
            "client_call_wall_seconds_including_image_and_startup",
        )
    }
    memory = receipt["peak_gpu_memory_bytes"]
    if type(memory) is not int or memory < 0:
        raise ValueError("receipt peak GPU memory is invalid")
    text_values: dict[str, str] = {}
    for name in ("python_version", "gpu_name", "modal_function_call_id", "modal_app_id"):
        value = receipt[name]
        if type(value) is not str or not value or len(value) > 256:
            raise ValueError(f"receipt {name} is invalid")
        text_values[name] = value
    cuda_version = receipt["torch_cuda_version"]
    if cuda_version is not None and (
        type(cuda_version) is not str or not cuda_version or len(cuda_version) > 128
    ):
        raise ValueError("receipt torch CUDA version is invalid")
    runner_hash = _digest(receipt["runner_sha256"], "runner SHA-256")
    eos_ids = receipt["generation_eos_token_ids"]
    if (
        type(eos_ids) is not list
        or not eos_ids
        or any(type(value) is not int or value < 0 for value in eos_ids)
        or len(set(eos_ids)) != len(eos_ids)
    ):
        raise ValueError("receipt generation EOS token IDs are invalid")
    pad_id = receipt["generation_pad_token_id"]
    if type(pad_id) is not int or pad_id < 0:
        raise ValueError("receipt generation pad token ID is invalid")
    return {
        "package_versions": versions,
        **timings,
        "peak_gpu_memory_bytes": memory,
        **text_values,
        "torch_cuda_version": cuda_version,
        "runner_sha256": runner_hash,
        "generation_eos_token_ids": eos_ids,
        "generation_pad_token_id": pad_id,
    }


def _validate_receipt(
    receipt: object, cases: list[dict[str, Any]], fixture_hash: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if type(receipt) is not dict or set(receipt) != _RECEIPT_KEYS:
        raise ValueError("receipt keys do not match the scenario schema")
    if type(receipt["schema_version"]) is not int or receipt["schema_version"] != 1:
        raise ValueError("receipt schema version is unsupported")
    if receipt["mode"] != "scenarios":
        raise ValueError("receipt mode is not scenario inference")
    if receipt["prompt_version"] != "explicit-v2":
        raise ValueError("receipt prompt version is not explicit-v2")
    input_hash = _digest(receipt["input_sha256"], "input SHA-256")
    payload_hash = _digest(receipt["payload_sha256"], "payload SHA-256")
    prompt_hash = _digest(receipt["prompt_sha256"], "prompt SHA-256")
    if input_hash != fixture_hash:
        raise ValueError("receipt input hash does not match the current scenario fixture")
    model = receipt["model"]
    if type(model) is not dict or type(model.get("key")) is not str:
        raise ValueError("receipt model provenance is invalid")
    payload = scenario_payload.build_payload(model["key"])
    if model != payload["model"]:
        raise ValueError("receipt model provenance does not match the pinned model settings")
    expected_payload_hash = scenario_payload.payload_sha256(payload)
    if payload_hash != expected_payload_hash:
        raise ValueError("receipt payload hash does not match the current fixture payload")
    if prompt_hash != payload["prompt_sha256"]:
        raise ValueError("receipt prompt hash does not match the current rendered prompts")
    prompt_source_hash = _digest(receipt["prompt_source_sha256"], "prompt source SHA-256")
    if prompt_source_hash != payload["prompt_source_sha256"]:
        raise ValueError("receipt prompt source hash does not match the frozen renderer")
    if type(receipt["case_count"]) is not int or receipt["case_count"] != 48:
        raise ValueError("receipt case count is invalid")
    provenance = _provenance(receipt)

    rows = receipt["generated_outputs"]
    if type(rows) is not list or len(rows) != 48:
        raise ValueError("receipt generated output count is invalid")
    expected_ids = {case["case_id"] for case in cases}
    output_ids: set[str] = set()
    outputs: dict[str, str] = {}
    input_total = generated_total = maxed_out = 0
    for row in rows:
        if type(row) is not dict or set(row) != {
            "case_id",
            "raw_output",
            "input_tokens",
            "generated_tokens",
            "hit_max_new_tokens",
            "output_characters",
            "finish_reason",
            "generated_last_token_id",
        }:
            raise ValueError("receipt generated output has an unsupported shape")
        case_id = row["case_id"]
        if type(case_id) is not str or case_id in output_ids:
            raise ValueError("receipt has a duplicate or invalid output case ID")
        output_ids.add(case_id)
        raw_output = row["raw_output"]
        if type(raw_output) is not str or len(raw_output) > 100_000:
            raise ValueError("receipt raw output is invalid or too large")
        input_tokens, generated_tokens = row["input_tokens"], row["generated_tokens"]
        if (
            type(input_tokens) is not int
            or input_tokens < 1
            or input_tokens > model["max_prompt_tokens"]
            or type(generated_tokens) is not int
            or generated_tokens < 1
            or generated_tokens > model["max_new_tokens"]
        ):
            raise ValueError("receipt output token count is outside the pinned limits")
        hit_limit = row["hit_max_new_tokens"]
        if type(hit_limit) is not bool or hit_limit != (
            generated_tokens >= model["max_new_tokens"]
        ):
            raise ValueError("receipt max-new-tokens flag does not match the generated length")
        char_count = row["output_characters"]
        if type(char_count) is not int or char_count != len(raw_output):
            raise ValueError("receipt raw output character count is invalid")
        end_id = row["generated_last_token_id"]
        if type(end_id) is not int or end_id < 0:
            raise ValueError("receipt generated final token ID is invalid")
        finish_reason = row["finish_reason"]
        if finish_reason == "eos":
            if end_id not in provenance["generation_eos_token_ids"]:
                raise ValueError("receipt eos finish reason does not match the configured EOS IDs")
        elif finish_reason == "length":
            if (
                generated_tokens != model["max_new_tokens"]
                or end_id in provenance["generation_eos_token_ids"]
            ):
                raise ValueError("receipt length finish reason is inconsistent")
        else:
            raise ValueError("receipt finish reason is unsupported")
        outputs[case_id] = raw_output
        input_total += input_tokens
        generated_total += generated_tokens
        maxed_out += int(hit_limit)
    if output_ids != expected_ids:
        missing, extra = expected_ids - output_ids, output_ids - expected_ids
        if missing:
            raise ValueError(f"receipt is missing output case IDs: {', '.join(sorted(missing))}")
        raise ValueError(f"receipt has unexpected output case IDs: {', '.join(sorted(extra))}")
    totals = {
        "input_tokens": input_total,
        "generated_tokens": generated_total,
        "outputs_at_generation_limit": maxed_out,
    }
    return payload, provenance, {"outputs": outputs, "totals": totals}


def _anchor_outputs(
    cases: list[dict[str, Any]], raw_outputs: dict[str, str]
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    anchored: dict[str, str] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for case in cases:
        case_id, raw_output = case["case_id"], raw_outputs[case["case_id"]]
        try:
            payload = _strict_json_loads(raw_output)
        except (json.JSONDecodeError, ValueError, RecursionError):
            anchored[case_id] = raw_output
            diagnostics[case_id] = {
                "status": "parse_invalid",
                "raw_failure": "invalid_json",
                "claims": [],
            }
            continue
        try:
            claims = parse_candidates(payload)
        except (TypeError, ValueError, RecursionError):
            anchored[case_id] = raw_output
            diagnostics[case_id] = {
                "status": "schema_invalid",
                "raw_failure": "invalid_candidate_schema",
                "claims": [],
            }
            continue
        records = _case_records(case)
        anchored_claims: list[dict[str, Any]] = []
        claim_rows: list[dict[str, Any]] = []
        changed = False
        for index, claim in enumerate(claims):
            result = anchor_unique_quotes(claim, records)
            anchored_claims.append(asdict(result.claim))
            repairs = [asdict(repair) for repair in result.repairs]
            changed = changed or bool(repairs)
            claim_rows.append(
                {"claim_index": index, "repairs": repairs, "unresolved": list(result.unresolved)}
            )
        anchored[case_id] = (
            json.dumps(
                {"claims": anchored_claims},
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            if changed
            else raw_output
        )
        diagnostics[case_id] = {
            "status": "repaired" if changed else "valid_schema",
            "raw_failure": None,
            "claims": claim_rows,
        }
    return anchored, diagnostics


def evaluate_receipt(receipt_bytes: bytes) -> dict[str, Any]:
    """Validate a raw receipt, score both splits, and preserve raw model outputs."""
    if type(receipt_bytes) is not bytes:
        raise TypeError("receipt must be provided as raw bytes")
    if not receipt_bytes or len(receipt_bytes) > _MAX_RECEIPT_BYTES:
        raise ValueError("receipt is empty or exceeds the size limit")
    try:
        receipt = _strict_json_loads(receipt_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError):
        raise ValueError("receipt is not valid UTF-8 JSON") from None
    if type(receipt) is not dict:
        raise ValueError("receipt must be a JSON object")
    cases, manifest, fixture_hash = scenario_payload.load_scenarios()
    if type(cases) is not list or len(cases) != 48 or type(manifest) is not dict:
        raise ValueError("current scenario fixture or manifest has an invalid shape")
    payload, provenance, result = _validate_receipt(receipt, cases, fixture_hash)
    raw_outputs: dict[str, str] = result["outputs"]
    anchored_outputs, output_diagnostics = _anchor_outputs(cases, raw_outputs)
    metrics: dict[str, Any] = {}
    for split in ("dev", "test"):
        selected = [case for case in cases if case["split"] == split]
        if len(selected) != 24:
            raise ValueError(f"current scenario {split} split must contain exactly 24 cases")
        selected_ids = {case["case_id"] for case in selected}
        metrics[split] = {
            "raw": evaluate_outputs(
                selected, {case_id: raw_outputs[case_id] for case_id in selected_ids}
            ),
            "unique_quote_anchored": evaluate_outputs(
                selected, {case_id: anchored_outputs[case_id] for case_id in selected_ids}
            ),
        }
    receipt_hash = hashlib.sha256(receipt_bytes).hexdigest()
    return {
        "schema_version": 1,
        "mode": "scenario_evaluation",
        "prompt_version": "explicit-v2",
        "input_sha256": payload["input_sha256"],
        "prompt_sha256": payload["prompt_sha256"],
        "prompt_source_sha256": payload["prompt_source_sha256"],
        "payload_sha256": scenario_payload.payload_sha256(payload),
        "raw_receipt_sha256": receipt_hash,
        "model": payload["model"],
        "provenance": provenance,
        "dataset": {
            "version": manifest.get("version"),
            "license": manifest.get("license"),
            "annotation_status": manifest.get("annotation_status"),
            "case_count": len(cases),
            "split_counts": {split: 24 for split in ("dev", "test")},
        },
        "token_totals": result["totals"],
        "interpretation_note": _INTERPRETATION_NOTE,
        "metrics": metrics,
        "raw_outputs": raw_outputs,
        "anchored_outputs": anchored_outputs,
        "output_diagnostics": output_diagnostics,
        "raw_generated_outputs": receipt["generated_outputs"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="completed raw receipt JSON")
    parser.add_argument(
        "--output", required=True, type=Path, help="exclusive evaluation report path"
    )
    args = parser.parse_args(argv)
    try:
        if args.input.resolve() == args.output.resolve():
            raise ValueError("evaluation output must not point to the input receipt")
        if args.input.exists() and args.output.exists() and args.input.samefile(args.output):
            raise ValueError("evaluation output must not alias the input receipt")
        report = evaluate_receipt(args.input.read_bytes())
        serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
