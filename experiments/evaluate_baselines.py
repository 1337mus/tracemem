"""Evaluate local receipts produced by the synthetic-only Modal baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

from experiments.modal_baseline import build_payload, payload_sha256
from tracemem import EvidenceStore
from tracemem.benchmark import load_synthetic_data, run_benchmark
from tracemem.remote_vectors import PrecomputedEmbedder

_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_RECEIPT_BYTES = 10_000_000
_COMMON_KEYS = {
    "schema_version",
    "mode",
    "input_sha256",
    "payload_sha256",
    "model",
    "package_versions",
    "gpu_compute_seconds",
    "model_load_seconds",
    "remote_function_elapsed_seconds",
    "peak_gpu_memory_bytes",
}
_OPTIONAL_KEYS = {
    "client_call_wall_seconds_including_image_and_startup",
    "modal_function_call_id",
    "modal_app_id",
    "prompt_version",
    "runner_sha256",
    "python_version",
    "torch_cuda_version",
    "gpu_name",
}
_EXTRACTION_KEYS = {"case_count", "generated_outputs"}
_EMBEDDING_KEYS = {"vectors", "vector_count"}
_DIAGNOSTIC_NOTE = (
    "Exact structured extraction diagnostics only; scores do not establish semantic "
    "entailment or claim support."
)


def evaluate_receipt(receipt_bytes: bytes, *, prompt_version: str = "v1") -> dict[str, Any]:
    """Validate a runner receipt and evaluate its outputs against current fixtures."""
    if not isinstance(receipt_bytes, bytes):
        raise TypeError("receipt must be provided as raw bytes")
    if not receipt_bytes or len(receipt_bytes) > _MAX_RECEIPT_BYTES:
        raise ValueError("receipt is empty or exceeds the size limit")
    try:
        receipt = json.loads(
            receipt_bytes.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise ValueError("receipt is not valid UTF-8 JSON") from None
    if type(receipt) is not dict:
        raise ValueError("receipt must be a JSON object")

    mode = receipt.get("mode")
    if type(mode) is not str or mode not in {"embeddings", "extraction"}:
        raise ValueError("receipt mode is not supported")
    expected_payload = build_payload(mode, prompt_version=prompt_version)
    if "prompt_version" in receipt and receipt["prompt_version"] != prompt_version:
        raise ValueError("receipt prompt version does not match the configured version")
    provenance = _validate_receipt_header(receipt, expected_payload)
    receipt_hash = hashlib.sha256(receipt_bytes).hexdigest()

    if mode == "extraction":
        result = _evaluate_extraction(receipt, expected_payload)
        return {
            "schema_version": 1,
            "mode": mode,
            "prompt_version": prompt_version,
            "input_sha256": expected_payload["input_sha256"],
            "payload_sha256": payload_sha256(expected_payload),
            "raw_receipt_sha256": receipt_hash,
            "model": expected_payload["model"],
            "provenance": provenance,
            "token_totals": result["token_totals"],
            "truncated_cases": result["truncated_cases"],
            "diagnostic_note": _DIAGNOSTIC_NOTE,
            "evaluation_scope": "prompt-development diagnostics; not an independent evaluation",
            "result": result["report"],
        }

    benchmark = _evaluate_embeddings(receipt, expected_payload)
    return {
        "schema_version": 1,
        "mode": mode,
        "prompt_version": prompt_version,
        "input_sha256": expected_payload["input_sha256"],
        "payload_sha256": payload_sha256(expected_payload),
        "raw_receipt_sha256": receipt_hash,
        "model": expected_payload["model"],
        "provenance": provenance,
        "result": benchmark,
    }


def _validate_receipt_header(
    receipt: dict[str, Any], expected_payload: dict[str, Any]
) -> dict[str, Any]:
    mode = expected_payload["mode"]
    mode_keys = _EXTRACTION_KEYS if mode == "extraction" else _EMBEDDING_KEYS
    if set(receipt) - _OPTIONAL_KEYS != _COMMON_KEYS | mode_keys:
        raise ValueError("receipt keys do not match the selected mode schema")
    if type(receipt.get("schema_version")) is not int or receipt["schema_version"] != 1:
        raise ValueError("receipt schema version is unsupported")
    if receipt["mode"] != mode:
        raise ValueError("receipt mode does not match the fixture payload")
    for name in ("input_sha256", "payload_sha256"):
        value = receipt.get(name)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError(f"receipt {name} is invalid")
    if receipt["input_sha256"] != expected_payload["input_sha256"]:
        raise ValueError("receipt input hash does not match current fixtures")
    expected_digest = payload_sha256(expected_payload)
    if receipt["payload_sha256"] != expected_digest:
        raise ValueError("receipt payload hash does not match current payload")
    if type(receipt.get("model")) is not dict or _canonical(receipt["model"]) != _canonical(
        expected_payload["model"]
    ):
        raise ValueError("receipt model does not match the pinned fixture payload")
    return _provenance(receipt)


def _provenance(receipt: dict[str, Any]) -> dict[str, Any]:
    versions = receipt.get("package_versions")
    if (
        type(versions) is not dict
        or not versions
        or any(
            type(key) is not str or not key or type(value) is not str or not value
            for key, value in versions.items()
        )
    ):
        raise ValueError("receipt package versions are invalid")
    for name in (
        "gpu_compute_seconds",
        "model_load_seconds",
        "remote_function_elapsed_seconds",
    ):
        _nonnegative_number(receipt.get(name), name)
    memory = receipt.get("peak_gpu_memory_bytes")
    if type(memory) is not int or memory < 0:
        raise ValueError("receipt peak GPU memory is invalid")
    provenance: dict[str, Any] = {
        "package_versions": versions,
        "gpu_compute_seconds": receipt["gpu_compute_seconds"],
        "model_load_seconds": receipt["model_load_seconds"],
        "remote_function_elapsed_seconds": receipt["remote_function_elapsed_seconds"],
        "peak_gpu_memory_bytes": memory,
    }
    optional_wall = "client_call_wall_seconds_including_image_and_startup"
    if optional_wall in receipt:
        _nonnegative_number(receipt[optional_wall], optional_wall)
        provenance[optional_wall] = receipt[optional_wall]
    for name in ("modal_function_call_id", "modal_app_id", "python_version", "gpu_name"):
        if name in receipt:
            value = receipt[name]
            if type(value) is not str or not value or len(value) > 256:
                raise ValueError(f"receipt {name} is invalid")
            provenance[name] = value
    if "runner_sha256" in receipt:
        runner_hash = receipt["runner_sha256"]
        if type(runner_hash) is not str or _SHA256.fullmatch(runner_hash) is None:
            raise ValueError("receipt runner SHA-256 is invalid")
        provenance["runner_sha256"] = runner_hash
    if "torch_cuda_version" in receipt:
        cuda_version = receipt["torch_cuda_version"]
        if cuda_version is not None and (
            type(cuda_version) is not str or not cuda_version or len(cuda_version) > 128
        ):
            raise ValueError("receipt torch CUDA version is invalid")
        provenance["torch_cuda_version"] = cuda_version
    return provenance


def _evaluate_extraction(
    receipt: dict[str, Any], expected_payload: dict[str, Any]
) -> dict[str, Any]:
    from tracemem.challenge import evaluate_outputs, load_challenge

    cases, fixture_hash = load_challenge()
    if fixture_hash != expected_payload["input_sha256"]:
        raise ValueError("challenge fixture hash changed during evaluation")
    payload_ids = [item["case_id"] for item in expected_payload["items"]]
    payload_id_set = set(payload_ids)
    selected_cases = [case for case in cases if case["case_id"] in payload_id_set]
    if len(payload_ids) != 24 or len(selected_cases) != len(payload_ids):
        raise ValueError("extraction payload must cover all 24 current challenge cases")
    if type(receipt.get("case_count")) is not int or receipt["case_count"] != len(payload_ids):
        raise ValueError("receipt extraction case count is invalid")

    raw_rows = receipt.get("generated_outputs")
    if type(raw_rows) is not list:
        raise ValueError("receipt generated outputs must be a list")
    expected_ids = set(payload_ids)
    outputs: dict[str, str] = {}
    input_tokens = generated_tokens = truncated_cases = 0
    model_info = expected_payload["model"]
    for row in raw_rows:
        if type(row) is not dict or set(row) != {
            "case_id",
            "raw_output",
            "input_tokens",
            "generated_tokens",
            "hit_max_new_tokens",
        }:
            raise ValueError("receipt extraction output has an unsupported shape")
        case_id = row["case_id"]
        if type(case_id) is not str or not case_id:
            raise ValueError("receipt extraction case ID is invalid")
        if case_id in outputs:
            raise ValueError(f"duplicate extraction output for case_id: {case_id}")
        raw_output = row["raw_output"]
        if type(raw_output) is not str or len(raw_output) > 100_000:
            raise ValueError("receipt raw output is invalid or too large")
        prompt_tokens = _token_count(row["input_tokens"], "input_tokens")
        output_tokens = _token_count(row["generated_tokens"], "generated_tokens")
        if prompt_tokens > model_info["max_prompt_tokens"]:
            raise ValueError("receipt input token count exceeds the pinned limit")
        if output_tokens > model_info["max_new_tokens"]:
            raise ValueError("receipt generated token count exceeds the pinned limit")
        hit_limit = output_tokens >= model_info["max_new_tokens"]
        if type(row["hit_max_new_tokens"]) is not bool or row["hit_max_new_tokens"] != hit_limit:
            raise ValueError("receipt truncation flag does not match the generated token count")
        outputs[case_id] = raw_output
        input_tokens += prompt_tokens
        generated_tokens += output_tokens
        truncated_cases += int(hit_limit)

    duplicate_or_unexpected = set(outputs) - expected_ids
    missing = expected_ids - set(outputs)
    if duplicate_or_unexpected:
        identifiers = ", ".join(sorted(duplicate_or_unexpected))
        raise ValueError(f"unexpected extraction output IDs: {identifiers}")
    if missing:
        identifiers = ", ".join(sorted(missing))
        raise ValueError(f"missing extraction outputs for case IDs: {identifiers}")
    report = evaluate_outputs(selected_cases, outputs)
    return {
        "report": report,
        "token_totals": {"input": input_tokens, "generated": generated_tokens},
        "truncated_cases": truncated_cases,
    }


def _evaluate_embeddings(
    receipt: dict[str, Any], expected_payload: dict[str, Any]
) -> dict[str, Any]:
    records, queries, fixture_hash = load_synthetic_data()
    if fixture_hash != expected_payload["input_sha256"]:
        raise ValueError("embedding fixture hash changed during evaluation")
    expected_hashes = {item["text_sha256"] for item in expected_payload["items"]}
    vectors = receipt.get("vectors")
    if type(vectors) is not dict or set(vectors) != expected_hashes:
        raise ValueError("receipt vectors do not match the fixture text hashes")
    count = receipt.get("vector_count")
    if type(count) is not int or count != len(expected_hashes):
        raise ValueError("receipt vector count is invalid")
    model_info = expected_payload["model"]
    embedder = PrecomputedEmbedder(
        model_id=model_info["model_id"],
        revision=model_info["revision"],
        dimensions=model_info["dimensions"],
        vectors=vectors,
    )
    with tempfile.TemporaryDirectory(prefix="tracemem-evaluate-") as directory:
        with EvidenceStore(Path(directory) / "evidence.sqlite") as store:
            store.ingest(records)
            return run_benchmark(store, queries, embedder=embedder, input_sha256=fixture_hash)


def _token_count(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"receipt {name} must be a nonnegative integer")
    return value


def _nonnegative_number(value: object, name: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"receipt {name} must be a finite nonnegative number")
    try:
        finite = math.isfinite(value)
    except (OverflowError, ValueError):
        finite = False
    if not finite:
        raise ValueError(f"receipt {name} must be a finite nonnegative number")
    if value < 0:
        raise ValueError(f"receipt {name} must be a finite nonnegative number")
    return value


def _canonical(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("receipt contains duplicate JSON keys")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"receipt contains nonstandard JSON constant: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evaluate-baselines")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="write JSON here instead of stdout")
    parser.add_argument("--prompt-version", choices=("v1", "explicit-v2"), default="v1")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.output is not None and args.input.resolve() == args.output.resolve():
            raise ValueError("input and output paths must differ")
        raw = args.input.read_bytes()
        report = evaluate_receipt(raw, prompt_version=args.prompt_version)
        output = json.dumps(
            report, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
        if args.output is None:
            print(output)
        else:
            args.output.write_text(output + "\n", encoding="utf-8")
        return 0
    except (OSError, TypeError, ValueError) as exc:
        print(f"evaluate-baselines failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
