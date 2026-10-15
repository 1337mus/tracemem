"""Offline descriptive analysis for the frozen fictional QA contrast receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

_PAYLOAD_SHA256 = "80bd1b68eb0f4fabc7262f00ba17ddfbeeaa4079b924d1228881253029bc7b4e"
_REFERENCES_SHA256 = "ad628ce021099887dec30acf4885bee53adeddbf63442cd099834a1db6c47c02"
_RUNNER_SHA256 = "1e1b7409c7a19bc4c0d51db72eb9256e6b1667ebc311af01d257e395e734a3db"
_PREPARER_SHA256 = "59270049e744629bdbbda16670160120c3135c6028c933af2a84fa8979462fe7"
_VALIDATOR_SHA256 = "63e9bcfa37447f9b6e81b091d4f928bb1ba36668426260c5d99239156a7b0461"
_MODES = ("bm25", "vector", "hybrid")
_CEILINGS = (128, 256, 512, 1024)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _finite_nonnegative(value: Any) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def _literal_match(left: str, right: str) -> bool:
    """Compare only after casefolding and collapsing whitespace."""
    return " ".join(left.casefold().split()) == " ".join(right.casefold().split())


def _fixed_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    from experiments import prepare_qa_contrast_requests as preparer

    source_hash = _sha256(Path(preparer.__file__).read_bytes())
    if source_hash != _PREPARER_SHA256:
        raise ValueError("QA contrast request preparer source differs from the frozen source")
    payload, references, audit = preparer.prepare_requests()
    if _sha256(_canonical_bytes(payload)) != _PAYLOAD_SHA256:
        raise ValueError("QA contrast payload differs from the frozen payload hash")
    if _sha256(_canonical_bytes(references)) != _REFERENCES_SHA256:
        raise ValueError("QA contrast references differ from the frozen reference hash")
    mapping = audit.get("cell_mapping")
    questions = references.get("questions")
    if (
        type(mapping) is not list
        or len(mapping) != 72
        or type(questions) is not list
        or len(questions) != 6
        or audit.get("test") != []
        or references.get("test") != []
        or audit.get("human_label_count") != 0
        or references.get("human_label_count") != 0
        or audit.get("dev_only") is not True
        or references.get("dev_only") is not True
    ):
        raise ValueError("frozen QA contrast audit or reference scope is invalid")
    unique_prompt_tokens = sum(item["prompt_tokens"] for item in payload["items"])
    if len(payload["items"]) != 46 or unique_prompt_tokens != 23533:
        raise ValueError("frozen QA contrast unique prompt inventory is invalid")
    if sum(row["prompt_tokens"] for row in mapping) != 28944:
        raise ValueError("frozen QA contrast cell mapping token inventory is invalid")
    return payload, references, audit, {"preparer_source_sha256": source_hash}


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def _group_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in rows if row["is_answerable"]]
    unknown = [row for row in rows if not row["is_answerable"]]
    explicit = sum(row["retrieval_diagnostics"]["explicit_label_count"] for row in rows)
    found = sum(row["retrieval_diagnostics"]["found_label_count"] for row in rows)
    return {
        "n": len(rows),
        "denominators": {"all": len(rows), "answerable": len(answerable), "unknown": len(unknown)},
        "exact_match": {
            "all_reference": _rate(sum(row["literal_exact_match"] for row in rows), len(rows)),
            "answerable": _rate(
                sum(row["literal_exact_match"] for row in answerable), len(answerable)
            ),
            "unknown": _rate(sum(row["unknown_abstention"] for row in unknown), len(unknown)),
        },
        "unknown_abstentions": sum(row["unknown_abstention"] for row in unknown),
        "false_unknown_on_answerable": sum(row["false_unknown"] for row in answerable),
        "non_unknown_outputs_on_unknown": sum(row["non_unknown_on_unknown"] for row in unknown),
        "empty_count": sum(row["empty"] for row in rows),
        "eos_stop_count": sum(row["stop_reason"] == "eos_token" for row in rows),
        "length_stop_count": sum(row["stop_reason"] == "max_new_tokens" for row in rows),
        "hit_max_new_tokens_count": sum(row["hit_max_new_tokens"] for row in rows),
        "truncated_count": sum(row["truncated"] for row in rows),
        "token_totals": {
            "prompt_tokens": sum(row["prompt_tokens"] for row in rows),
            "input_tokens": sum(row["input_tokens"] for row in rows),
            "generated_tokens": sum(row["generated_tokens"] for row in rows),
        },
        "retrieval_diagnostics": {
            "explicit_label_count": explicit,
            "found_label_count": found,
            "label_recall": found / explicit if explicit else None,
        },
    }


def analyze_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    """Validate the complete approved receipt and produce frozen literal metrics."""
    from experiments import qa_contrast_answer_receipt as validator

    validator_hash = _sha256(Path(validator.__file__).read_bytes())
    if validator_hash != _VALIDATOR_SHA256:
        raise ValueError("QA contrast answer validator source differs from the frozen source")
    payload, references, audit, source_hashes = _fixed_inputs()
    runner_path = Path(__file__).with_name("modal_qa_contrast_answer.py")
    runner_hash = _sha256(runner_path.read_bytes())
    if runner_hash != _RUNNER_SHA256:
        raise ValueError("QA contrast answer runner differs from the frozen reviewed source")
    expected_fields = {
        "schema_version",
        "payload_sha256",
        "runner_sha256",
        "client_elapsed_seconds",
        "result",
    }
    if type(receipt) is not dict or set(receipt) != expected_fields:
        raise ValueError("receipt envelope fields are invalid")
    elapsed = receipt["client_elapsed_seconds"]
    if (
        type(receipt["schema_version"]) is not int
        or receipt["schema_version"] != 1
        or receipt["payload_sha256"] != _PAYLOAD_SHA256
        or receipt["runner_sha256"] != _RUNNER_SHA256
        or not _finite_nonnegative(elapsed)
    ):
        raise ValueError("receipt envelope provenance or timing is invalid")
    try:
        validator.validate_result(receipt["result"], payload, _RUNNER_SHA256)
    except OverflowError:
        raise ValueError("receipt runtime timing must be finite and nonnegative") from None
    try:
        receipt_hash = _sha256(_canonical_bytes(receipt))
    except (TypeError, ValueError):
        raise ValueError("receipt is not finite JSON data") from None

    references_by_id = {row["question_id"]: row for row in references["questions"]}
    if (
        len(references_by_id) != 6
        or sum(not row["is_abstention"] for row in references_by_id.values()) != 4
        or sum(row["is_abstention"] for row in references_by_id.values()) != 2
    ):
        raise ValueError("frozen questions must contain four answerable and two unknown cases")
    output_by_id = {row["prompt_id"]: row for row in receipt["result"]["outputs"]}
    cells: list[dict[str, Any]] = []
    seen: set[str] = set()
    for mapping in audit["cell_mapping"]:
        question_id, prompt_id = mapping["question_id"], mapping["prompt_id"]
        if (
            mapping.get("split") != "dev"
            or question_id not in references_by_id
            or prompt_id not in output_by_id
            or mapping["cell_id"] in seen
        ):
            raise ValueError("QA contrast audit contains an invalid or duplicate cell")
        seen.add(mapping["cell_id"])
        reference = references_by_id[question_id]
        output = output_by_id[prompt_id]
        raw, answer = output["raw_output"], reference["answer"]
        normalized_unknown = " ".join(raw.casefold().split()) == "unknown"
        answerable = not reference["is_abstention"]
        scoring = mapping["scoring"]
        cells.append(
            {
                "cell_id": mapping["cell_id"],
                "question_id": question_id,
                "mode": mapping["mode"],
                "ceiling": mapping["ceiling"],
                "prompt_id": prompt_id,
                "is_answerable": answerable,
                "prompt_tokens": mapping["prompt_tokens"],
                "reference": answer,
                "raw_output": raw,
                "literal_exact_match": _literal_match(raw, answer),
                "unknown_abstention": not answerable and normalized_unknown,
                "false_unknown": answerable and normalized_unknown,
                "non_unknown_on_unknown": not answerable and not normalized_unknown,
                "empty": " ".join(raw.casefold().split()) == "",
                "input_tokens": output["input_tokens"],
                "generated_tokens": output["generated_tokens"],
                "stop_reason": output["stop_reason"],
                "hit_max_new_tokens": output["hit_max_new_tokens"],
                "truncated": output["truncated"],
                "retrieval_diagnostics": {
                    "explicit_label_count": scoring["explicit_label_count"],
                    "found_label_count": scoring["found_label_count"],
                    "label_recall": scoring["label_recall"],
                    "selected_citation_count": len(mapping["selected_citation_ids"]),
                    "selected_record_count": len(mapping["selected_record_ids"]),
                },
            }
        )
    if len(cells) != 72:
        raise ValueError("QA contrast audit must map exactly 72 cells")
    groups = []
    for mode in _MODES:
        for ceiling in _CEILINGS:
            rows = [row for row in cells if row["mode"] == mode and row["ceiling"] == ceiling]
            if (
                len(rows) != 6
                or sum(row["is_answerable"] for row in rows) != 4
                or sum(not row["is_answerable"] for row in rows) != 2
            ):
                raise ValueError("each retrieval mode and ceiling group must map 4/2 questions")
            groups.append({"mode": mode, "ceiling": ceiling, **_group_report(rows)})
    result = receipt["result"]
    runtime = result["runtime"]
    return {
        "schema_version": 1,
        "label": "provisional agent-only development analysis",
        "dev_only": True,
        "provisional": True,
        "test": [],
        "human_label_count": 0,
        "question_count": 6,
        "cell_count": 72,
        "unique_prompt_count": 46,
        "reused_cell_count": 26,
        "answer_reuse": {
            "same_46_answers_reused_across_72_cells": True,
            "independent_trials": False,
        },
        "limitations": [
            "Exact match is literal equality after casefold and whitespace collapse only.",
            (
                "Punctuation, aliases, paraphrases, substring matches, and semantic judging "
                "are excluded."
            ),
            (
                "The six fictional questions are dev examples with provisional agent-authored "
                "references."
            ),
            (
                "No semantic accuracy, calibrated abstention, test performance, or model "
                "superiority follows."
            ),
            "No billing or cost-per-correct-answer inference is supported.",
        ],
        "provenance": {
            "analyzer_source_sha256": _sha256(Path(__file__).read_bytes()),
            "receipt_sha256": receipt_hash,
            "payload_sha256": _PAYLOAD_SHA256,
            "references_sha256": _REFERENCES_SHA256,
            "audit_sha256": _sha256(_canonical_bytes(audit)),
            "runner_sha256": _RUNNER_SHA256,
            "validator_source_sha256": validator_hash,
            **source_hashes,
        },
        "unique_prompt_token_totals": {
            "input_tokens": sum(item["prompt_tokens"] for item in payload["items"]),
            "generated_tokens": sum(row["generated_tokens"] for row in result["outputs"]),
        },
        "runtime": {
            "gpu_name": runtime["gpu_name"],
            "gpu_peak_memory_bytes": runtime["gpu_peak_memory_bytes"],
            "model_load_seconds": runtime["model_load_seconds"],
            "gpu_compute_seconds": runtime["gpu_compute_seconds"],
            "elapsed_seconds": runtime["elapsed_seconds"],
            "client_elapsed_seconds": elapsed,
        },
        "groups": groups,
        "cells": cells,
    }


def _write_new(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as output:
        os.fchmod(output.fileno(), 0o600)
        output.write(data)
        output.flush()
        os.fsync(output.fileno())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="analyze-qa-contrast-answers")
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--output", type=Path, help="write to a new mode-0600 JSON file")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
        report = analyze_receipt(receipt)
        data = json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        if args.output is None:
            print(data, end="")
        else:
            _write_new(args.output, data.encode("utf-8"))
    except (OSError, ValueError, TypeError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
