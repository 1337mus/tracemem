"""Analyze receipts from the frozen, source-only synthetic QA diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

from experiments.modal_synthetic_qa import _canonical_bytes, build_payload
from experiments.prepare_synthetic_qa import prepare_synthetic_qa

_REQUEST_SHA256 = "85d1334d0401ef2bdc920bf316f066bc3e74e9423ee5bec0aa63f4a559be2084"
_REFERENCE_SHA256 = "70769ec36ac4c1e6a226d7432ea88241225e35d91d803cdc0a678fe2589226d2"
_FIXTURE_SHA256 = "a4d94c8e727c71e8c8b723e97c9674b1d0008ac9f35ba675485a7ea796349d78"
_PAYLOAD_SHA256 = "80e058c629de8f8a06ca4caa8f1093a8308a299c976b018ab7e8c5c62493b2ac"
_MODES = ("bm25", "vector", "hybrid")
_CEILINGS = (0, 512, 2048)
_SPLITS = ("dev", "test")


def _fixed_data() -> tuple[
    list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any], dict[str, str]
]:
    requests, references = prepare_synthetic_qa()
    payload = build_payload()
    if (
        requests.get("fixture_sha256") != _FIXTURE_SHA256
        or hashlib.sha256(_canonical_bytes(requests)).hexdigest() != _REQUEST_SHA256
        or hashlib.sha256(_canonical_bytes(references)).hexdigest() != _REFERENCE_SHA256
        or hashlib.sha256(_canonical_bytes(payload)).hexdigest() != _PAYLOAD_SHA256
    ):
        raise ValueError("fixed synthetic QA inputs differ from the reviewed bundles")
    reference_by_id: dict[str, dict[str, Any]] = {}
    for split in _SPLITS:
        for reference in references["questions"][split]:
            reference_by_id[reference["question_id"]] = reference
    hashes = {
        "request_bundle_sha256": _REQUEST_SHA256,
        "reference_bundle_sha256": _REFERENCE_SHA256,
        "fixture_sha256": _FIXTURE_SHA256,
    }
    return requests["requests"], reference_by_id, payload, hashes


def _agreement_cell(
    rows: list[dict[str, Any]],
    references: dict[str, dict[str, Any]],
    outputs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    answerable_matches = 0
    unknown_matches = 0
    for row in rows:
        reference = references[row["question_id"]]
        matched = _normalize(outputs[row["request_id"]]["raw_output"]) == _normalize(
            reference["answer"]
        )
        if reference["is_abstention"]:
            unknown_matches += int(matched)
        else:
            answerable_matches += int(matched)
    answerable = sum(not references[row["question_id"]]["is_abstention"] for row in rows)
    unknown = len(rows) - answerable
    return {
        "all": {"numerator": answerable_matches + unknown_matches, "denominator": len(rows)},
        "answerable": {"numerator": answerable_matches, "denominator": answerable},
        "unknown": {"numerator": unknown_matches, "denominator": unknown},
    }


def _zero_context_unknown(
    rows: list[dict[str, Any]], outputs: dict[str, dict[str, Any]], ceiling: int
) -> dict[str, int]:
    is_zero_context = ceiling == 0
    return {
        "numerator": sum(
            _normalize(outputs[row["request_id"]]["raw_output"]) == "unknown" for row in rows
        )
        if is_zero_context
        else 0,
        "denominator": len(rows) if is_zero_context else 0,
    }


def _token_stats(
    rows: list[dict[str, Any]], outputs: dict[str, dict[str, Any]], key: str
) -> dict[str, int]:
    values = [outputs[row["request_id"]][key] for row in rows]
    return {"min": min(values), "max": max(values), "sum": sum(values)}


def _outcomes(
    requests: list[dict[str, Any]],
    references: dict[str, dict[str, Any]],
    outputs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    outcomes = []
    for request in requests:
        output = outputs[request["request_id"]]
        reference = references[request["question_id"]]
        normalized = _normalize(output["raw_output"])
        outcomes.append(
            {
                "request_id": request["request_id"],
                "question_id": request["question_id"],
                "split": request["split"],
                "mode": request["mode"],
                "byte_ceiling": request["byte_ceiling"],
                "is_answerable": not reference["is_abstention"],
                "literal_agreement": normalized == _normalize(reference["answer"]),
                "zero_context_unknown": normalized == "unknown"
                if request["byte_ceiling"] == 0
                else None,
                "stop_reason": output["stop_reason"],
                "hit_max_new_tokens": output["hit_max_new_tokens"],
                "truncated": output["truncated"],
                "empty": normalized == "",
                "input_tokens": output["input_tokens"],
                "generated_tokens": output["generated_tokens"],
            }
        )
    return outcomes


def _output_map(raw_outputs: Any, requests: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    expected = {request["request_id"] for request in requests}
    if type(raw_outputs) is not list:
        raise ValueError("receipt outputs are invalid")
    ids = [row.get("request_id") if type(row) is dict else None for row in raw_outputs]
    if (
        len(raw_outputs) != len(expected)
        or any(type(request_id) is not str for request_id in ids)
        or len(set(ids)) != len(expected)
        or set(ids) != expected
    ):
        raise ValueError("receipt output request IDs are duplicate, missing, or extra")
    return {row["request_id"]: row for row in raw_outputs}


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _empty_cells(
    requests: list[dict[str, Any]],
    references: dict[str, dict[str, Any]],
    outputs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    cells = []
    for split in _SPLITS:
        for mode in _MODES:
            for ceiling in _CEILINGS:
                rows = [
                    request
                    for request in requests
                    if request["split"] == split
                    and request["mode"] == mode
                    and request["byte_ceiling"] == ceiling
                ]
                answerable = sum(
                    not references[row["question_id"]]["is_abstention"] for row in rows
                )
                cells.append(
                    {
                        "split": split,
                        "mode": mode,
                        "byte_ceiling": ceiling,
                        "n": len(rows),
                        "answerable_n": answerable,
                        "unknown_n": len(rows) - answerable,
                        "agreement": _agreement_cell(rows, references, outputs),
                        "zero_context_unknown": _zero_context_unknown(rows, outputs, ceiling),
                        "empty_count": sum(
                            _normalize(outputs[row["request_id"]]["raw_output"]) == ""
                            for row in rows
                        ),
                        "finish_counts": {
                            reason: sum(
                                outputs[row["request_id"]]["stop_reason"] == reason for row in rows
                            )
                            for reason in ("eos", "length", "other")
                        },
                        "hit_max_new_tokens_count": sum(
                            outputs[row["request_id"]]["hit_max_new_tokens"] for row in rows
                        ),
                        "truncated_count": sum(
                            outputs[row["request_id"]]["truncated"] for row in rows
                        ),
                        "input_tokens": _token_stats(rows, outputs, "input_tokens"),
                        "generated_tokens": _token_stats(rows, outputs, "generated_tokens"),
                    }
                )
    return cells


def analyze_receipt(
    receipt: dict[str, Any],
    *,
    approved_runner_sha256: str,
    approved_template_sha256: str,
) -> dict[str, Any]:
    """Validate a receipt and calculate the frozen literal-agreement report."""
    requests, references, payload, source_hashes = _fixed_data()
    outputs = _validate_receipt(
        receipt, approved_runner_sha256, approved_template_sha256, requests, payload
    )
    try:
        receipt_hash = hashlib.sha256(_canonical_bytes(receipt)).hexdigest()
    except (TypeError, ValueError):
        raise ValueError("receipt is not finite JSON data") from None
    provenance = {
        "source_receipt_sha256": receipt_hash,
        **source_hashes,
        "payload_sha256": _PAYLOAD_SHA256,
        "runner_sha256": receipt["runner_sha256"],
        "template_sha256": receipt["result"]["template_sha256"],
        "config_sha256": receipt["result"]["config_sha256"],
        "analyzer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    return {
        "schema_version": 1,
        "label": "provisional agent-only",
        "question_count": 8,
        "matched_request_count": 72,
        "limitations": [
            "No semantic QA judging; literal reference comparison only.",
            "No matched-token budget; byte ceilings are UTF-8 bytes.",
            "No independent-scenario claim; cells repeat the same eight questions.",
            "References are provisional agent-authored and not independently reviewed.",
            "No pooled dev/test score or model ranking.",
            "Not evidence of benchmark compatibility, calibration, cost per correct answer, "
            "or training readiness.",
        ],
        "provenance": provenance,
        "cells": _empty_cells(requests, references, outputs),
        "outcomes": _outcomes(requests, references, outputs),
    }


def _validate_receipt(
    receipt: Any,
    approved_runner: str,
    approved_template: str,
    requests: list[dict[str, Any]],
    payload: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if not _is_sha256(approved_runner) or not _is_sha256(approved_template):
        raise ValueError("approval hashes must be lowercase SHA-256 values")
    envelope = {
        "schema_version",
        "payload_sha256",
        "runner_sha256",
        "client_elapsed_seconds",
        "result",
    }
    if type(receipt) is not dict or set(receipt) != envelope:
        raise ValueError("receipt envelope is malformed")
    expected_payload = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    if (
        type(receipt["schema_version"]) is not int
        or receipt["schema_version"] != 1
        or receipt["payload_sha256"] != expected_payload
        or receipt["runner_sha256"] != approved_runner
    ):
        raise ValueError("receipt provenance does not match the reviewed approvals")
    elapsed = receipt["client_elapsed_seconds"]
    try:
        valid_elapsed = type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0
    except OverflowError:
        valid_elapsed = False
    if not valid_elapsed:
        raise ValueError("receipt timing is invalid")
    result = receipt["result"]
    required = {
        "schema_version",
        "payload_sha256",
        "model_id",
        "revision",
        "config_sha256",
        "template_sha256",
        "package_versions",
        "request_count",
        "outputs",
        "modal_call_id",
    }
    if (
        type(result) is not dict
        or not required <= set(result)
        or "error" in result
        or "failure" in result
        or result.get("status", "success") not in ("success", "ok")
    ):
        raise ValueError("receipt result is malformed")
    config_hash = hashlib.sha256(_canonical_bytes(payload["model"])).hexdigest()
    versions = result["package_versions"]
    if (
        type(result["schema_version"]) is not int
        or result["schema_version"] != 1
        or result["payload_sha256"] != expected_payload
        or result["model_id"] != "Qwen/Qwen3-4B"
        or result["revision"] != "1cfa9a7208912126459214e8b04321603b3df60c"
        or result["config_sha256"] != config_hash
        or result["template_sha256"] != approved_template
        or type(versions) is not dict
        or re.fullmatch(r"2\.7\.1(?:\+cu[0-9]+)?", str(versions.get("torch"))) is None
        or versions.get("transformers") != "4.51.3"
        or versions.get("accelerate") != "1.6.0"
        or type(result["modal_call_id"]) is not str
        or not result["modal_call_id"].strip()
    ):
        raise ValueError("receipt runtime provenance is invalid")
    if type(result["request_count"]) is not int or result["request_count"] != 72:
        raise ValueError("receipt request count is invalid")
    raw_outputs = result["outputs"]
    if type(raw_outputs) is not list or len(raw_outputs) != 72:
        raise ValueError("receipt output count is invalid")
    fields = {
        "request_id",
        "raw_output",
        "input_tokens",
        "generated_tokens",
        "stop_reason",
        "hit_max_new_tokens",
        "truncated",
    }
    if any(type(row) is not dict or set(row) != fields for row in raw_outputs):
        raise ValueError("receipt output rows are malformed")
    for row in raw_outputs:
        input_tokens, generated, reason = (
            row["input_tokens"],
            row["generated_tokens"],
            row["stop_reason"],
        )
        hit, truncated = row["hit_max_new_tokens"], row["truncated"]
        if (
            type(row["request_id"]) is not str
            or type(row["raw_output"]) is not str
            or type(input_tokens) is not int
            or not 1 <= input_tokens <= 2048
            or type(generated) is not int
            or not 0 <= generated <= 128
            or type(reason) is not str
            or reason not in ("eos", "length", "other")
            or type(hit) is not bool
            or type(truncated) is not bool
            or hit != (generated == 128)
            or (reason == "length" and (generated != 128 or not truncated))
            or (reason == "eos" and (generated < 1 or truncated))
        ):
            raise ValueError("receipt output values are invalid")
    return _output_map(raw_outputs, requests)


def _is_sha256(value: Any) -> bool:
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON object key")
        value[key] = item
    return value


def _reject_constant(_: str) -> None:
    raise ValueError("non-finite JSON number")


def _read_receipt(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as source:
            raw = source.read(2 * 1024 * 1024 + 1)
        if len(raw) > 2 * 1024 * 1024:
            raise ValueError("receipt exceeds the 2 MiB limit")
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError):
        raise ValueError("receipt input is invalid") from None
    if type(value) is not dict:
        raise ValueError("receipt input is invalid")
    return value


def _write_new(path: Path, content: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(content)
    except OSError:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            path.unlink()
        except OSError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--approve-runner-sha256", required=True)
    parser.add_argument("--approve-template-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = analyze_receipt(
            _read_receipt(args.receipt),
            approved_runner_sha256=args.approve_runner_sha256,
            approved_template_sha256=args.approve_template_sha256,
        )
        encoded = (
            json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        _write_new(args.output, encoded)
        return 0
    except (OSError, TypeError, ValueError, RecursionError):
        print("synthetic QA analysis failed: invalid input or output", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
