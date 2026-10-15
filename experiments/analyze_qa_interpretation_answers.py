"""Offline literal analysis for the pinned QA interpretation answer receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

from experiments import prepare_qa_interpretation_controls as _preparer
from experiments import qa_interpretation_answer_receipt as _receipt

_ROOT = Path(__file__).resolve().parents[1]
_EXPECTED_REQUEST_SHA256 = "df91a152242ed6b2942802b5957dc7531a58bad3bffce67a284ce4526f90b871"
_EXPECTED_PAYLOAD_SHA256 = "36a03f56b69500674feb219fc40f8479965982bc9041de7a74b4464cace5bed1"
_EXPECTED_REFERENCES_SHA256 = "7669d8bcae3f9dffe71c1cdd19d4b8c2c5a8fa150435675d7128a24f26d43894"
_EXPECTED_RUNNER_SHA256 = "9ee4f7fbb440d166eff415c978239ca71da69a6e607fe63f98cc47fcf9a8568c"
_EXPECTED_CONTRACT_SHA256 = "8f03a1c2b7767cdc5095fc51d28e066fa32750c127954ec0f6946862c8ee7278"
_EXPECTED_PREPARER_SHA256 = "62c1775ffe6d295e5da39d2743fb7c97d67ccb356c19be1b012f7b01b1ebec5b"
_MAX_RECEIPT_BYTES = 8 * 1024 * 1024
_VARIANTS = ("full_all_message_ids", "support_omitted", "empty")
_RECEIPT_FIELDS = {
    "schema_version",
    "request_sha256",
    "runner_sha256",
    "client_elapsed_seconds",
    "result",
}
_CELL_FIELDS = {
    "cell_id",
    "scenario_id",
    "question_id",
    "variant_id",
    "prompt_id",
    "selected_message_ids",
    "full_source_expected_answer",
    "expected_prompt_answer",
    "is_answerable",
}


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_receipt._canonical_bytes(value)).hexdigest()


def _source_sha256(path: Path, expected: str, name: str) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(f"pinned {name} source hash differs")
    return digest


def _verify_sources() -> dict[str, str]:
    runner = _ROOT / "experiments/modal_qa_interpretation_answer.py"
    contract = Path(_receipt.__file__)
    preparer = Path(_preparer.__file__)
    return {
        "runner_sha256": _source_sha256(runner, _EXPECTED_RUNNER_SHA256, "runner"),
        "contract_sha256": _source_sha256(contract, _EXPECTED_CONTRACT_SHA256, "contract"),
        "preparer_sha256": _source_sha256(preparer, _EXPECTED_PREPARER_SHA256, "preparer"),
    }


def _metric(numerator: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "metrics": {
            "prompt": _metric(sum(row["matches"]["prompt"] for row in rows), len(rows)),
            "full_source": _metric(sum(row["matches"]["full_source"] for row in rows), len(rows)),
        },
        "exact_unknown_abstentions": sum(row["exact_unknown_abstention"] for row in rows),
        "empty_outputs": sum(row["empty_output"] for row in rows),
        "stop_counts": {
            stop: sum(row["stop_reason"] == stop for row in rows)
            for stop in ("eos_token", "max_new_tokens")
        },
        "cap_count": sum(row["hit_max_new_tokens"] for row in rows),
        "unique_prompt_input_tokens": sum(row["input_tokens"] for row in rows),
        "unique_prompt_generated_tokens": sum(row["generated_tokens"] for row in rows),
        "unique_prompts": len({row["prompt_id"] for row in rows}),
    }


def _pinned_references(
    request: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str]:
    payload, references, audit = _preparer.prepare_controls()
    payload_hash = _canonical_sha256(payload)
    references_hash = _canonical_sha256(references)
    if payload_hash != _EXPECTED_PAYLOAD_SHA256:
        raise ValueError("prepared controls payload differs from its frozen pin")
    if references_hash != _EXPECTED_REFERENCES_SHA256:
        raise ValueError("prepared controls references differ from their frozen pin")
    if (
        references.get("split") != "dev"
        or references.get("test") != []
        or references.get("human_label_count") != 0
        or references.get("cells_are_independent_trials") is not False
        or audit.get("counts")
        != {
            "scenario_groups": 6,
            "sessions": 7,
            "messages": 22,
            "questions": 12,
            "full_source_answerable": 10,
            "full_source_unknown": 2,
            "cells": 36,
            "unique_prompts": 36,
        }
    ):
        raise ValueError("prepared controls are outside the frozen dev-only scope")
    request_items = {item["prompt_id"]: item["messages"] for item in request["items"]}
    payload_items = {item["prompt_id"]: item["messages"] for item in payload["items"]}
    if request_items != payload_items:
        raise ValueError("request prompts differ from the pinned reference payload")
    baseline = audit["always_unknown_reference_baseline"]["total"]
    if baseline != {
        "prompt": {"matches": 26, "denominator": 36},
        "full_source": {"matches": 6, "denominator": 36},
    }:
        raise ValueError("always-UNKNOWN reference baseline differs from its frozen counts")
    return references, audit, references_hash


def analyze_receipt(receipt: Any) -> dict[str, Any]:
    """Validate one successful pinned receipt and report literal dev-label matches."""
    if type(receipt) is not dict or set(receipt) != _RECEIPT_FIELDS:
        raise ValueError("receipt wrapper fields are invalid")
    if (
        type(receipt["schema_version"]) is not int
        or receipt["schema_version"] != 1
        or type(receipt["request_sha256"]) is not str
        or type(receipt["runner_sha256"]) is not str
        or not _receipt._nonnegative_finite(receipt["client_elapsed_seconds"])
        or type(receipt["result"]) is not dict
    ):
        raise ValueError("receipt wrapper types or elapsed time are invalid")
    pins = _verify_sources()
    request = _receipt.build_request()
    request_hash = _receipt.request_sha256(request)
    if request_hash != _EXPECTED_REQUEST_SHA256 or receipt["request_sha256"] != request_hash:
        raise ValueError("receipt request hash differs from its canonical pin")
    if receipt["runner_sha256"] != pins["runner_sha256"]:
        raise ValueError("receipt runner hash differs from its frozen source")
    result = _receipt.validate_result(receipt["result"], request, pins["runner_sha256"])
    receipt_hash = _canonical_sha256(receipt)
    result_hash = _canonical_sha256(result)
    references, audit, references_hash = _pinned_references(request)
    cells = references.get("cells")
    if type(cells) is not list or len(cells) != 36:
        raise ValueError("pinned references must contain all 36 cells")
    cells_by_prompt: dict[str, dict[str, Any]] = {}
    for cell in cells:
        prompt_id = cell.get("prompt_id") if type(cell) is dict else None
        if (
            type(cell) is not dict
            or set(cell) != _CELL_FIELDS
            or type(prompt_id) is not str
            or prompt_id in cells_by_prompt
            or cell["variant_id"] not in _VARIANTS
        ):
            raise ValueError("pinned reference cell mapping is invalid")
        cells_by_prompt[prompt_id] = cell
    outputs_by_prompt = {output["prompt_id"]: output for output in result["outputs"]}
    if len(outputs_by_prompt) != 36 or set(outputs_by_prompt) != set(cells_by_prompt):
        raise ValueError("receipt outputs do not map one-to-one to the 36 frozen cells")

    per_cell = []
    by_variant = {variant: [] for variant in _VARIANTS}
    for cell in cells:
        output = outputs_by_prompt[cell["prompt_id"]]
        normalized = _normalize(output["raw_output"])
        row = {
            "cell_id": cell["cell_id"],
            "scenario_id": cell["scenario_id"],
            "question_id": cell["question_id"],
            "variant_id": cell["variant_id"],
            "prompt_id": cell["prompt_id"],
            "raw_output": output["raw_output"],
            "normalized_output": normalized,
            "matches": {
                "prompt": normalized == _normalize(cell["expected_prompt_answer"]),
                "full_source": normalized == _normalize(cell["full_source_expected_answer"]),
            },
            "exact_unknown_abstention": normalized == "unknown",
            "empty_output": normalized == "",
            "stop_reason": output["stop_reason"],
            "hit_max_new_tokens": output["hit_max_new_tokens"],
            "input_tokens": output["input_tokens"],
            "generated_tokens": output["generated_tokens"],
        }
        per_cell.append(row)
        by_variant[cell["variant_id"]].append(row)
    variant_summaries = {variant: _summary(rows) for variant, rows in by_variant.items()}
    total = _summary(per_cell)
    return {
        "schema_version": 1,
        "kind": "qa-interpretation-controls-answer-analysis",
        "scope": {
            "split": "dev",
            "test": [],
            "human_label_count": 0,
            "scenario_groups": 6,
            "questions": 12,
            "cells": 36,
            "unique_prompts": 36,
            "correlated_groups": 6,
            "cells_are_independent_trials": False,
        },
        "provenance": {
            "request_sha256": request_hash,
            "receipt_sha256": receipt_hash,
            "result_sha256": result_hash,
            **pins,
            "payload_sha256": _EXPECTED_PAYLOAD_SHA256,
            "references_sha256": references_hash,
        },
        "interpretation": (
            "Literal exact matching against provisional agent-authored dev labels only; "
            "not semantic QA accuracy or calibration. Cells are correlated within six groups; "
            "no independent-cell confidence intervals are reported."
        ),
        "metrics": total["metrics"],
        "by_variant": variant_summaries,
        "counts": {
            key: total[key]
            for key in (
                "exact_unknown_abstentions",
                "empty_outputs",
                "stop_counts",
                "cap_count",
                "unique_prompt_input_tokens",
                "unique_prompt_generated_tokens",
                "unique_prompts",
            )
        },
        "baseline": {
            "kind": "deterministic-always-unknown-reference-baseline",
            "prediction": "UNKNOWN",
            "model_run": False,
            "metrics": {
                "prompt": _metric(26, 36),
                "full_source": _metric(6, 36),
            },
            "correlated_dev_only": True,
            "not_model_quality": True,
        },
        "per_cell": per_cell,
    }


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("receipt JSON contains duplicate object keys")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"receipt JSON contains nonfinite constant {value}")


def read_receipt(path: Path) -> dict[str, Any]:
    """Read one bounded regular JSON file with duplicate keys and NaN rejected."""
    if not isinstance(path, Path):
        raise TypeError("receipt path must be a pathlib.Path")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > _MAX_RECEIPT_BYTES:
            raise ValueError("receipt must be a regular file no larger than 8 MiB")
        raw = bytearray()
        while len(raw) <= _MAX_RECEIPT_BYTES:
            chunk = os.read(descriptor, min(65_536, _MAX_RECEIPT_BYTES + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        if len(raw) > _MAX_RECEIPT_BYTES:
            raise ValueError("receipt exceeds the 8 MiB size limit")
    finally:
        os.close(descriptor)
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError):
        raise ValueError("receipt is not strict JSON") from None
    if type(value) is not dict:
        raise ValueError("receipt JSON root must be an object")
    return value


def write_analysis(path: Path, analysis: dict[str, Any]) -> None:
    """Write a new private analysis file without replacing existing paths."""
    if not isinstance(path, Path):
        raise TypeError("analysis path must be a pathlib.Path")
    data = (
        json.dumps(analysis, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            descriptor = -1
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="analyze-qa-interpretation-answers")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        analysis = analyze_receipt(read_receipt(args.receipt))
        write_analysis(args.output, analysis)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "references_sha256": analysis["provenance"]["references_sha256"],
                    "cells": analysis["scope"]["cells"],
                    "remote_started": False,
                },
                sort_keys=True,
            )
        )
        return 0
    except (OSError, TypeError, ValueError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
