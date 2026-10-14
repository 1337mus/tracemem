"""Compare raw and quote-anchored outputs from a completed extraction receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from experiments import evaluate_baselines
from tracemem.anchoring import anchor_unique_quotes
from tracemem.challenge import _case_records, evaluate_outputs, load_challenge
from tracemem.claims import parse_candidates

_INTERPRETATION_NOTE = (
    "Offset repair restores uniquely located exact quotes only; it does not establish "
    "entailment, and repaired claims must never be auto-accepted."
)


def evaluate_anchoring(receipt_bytes: bytes, *, prompt_version: str = "v1") -> dict[str, Any]:
    """Evaluate raw and conservatively anchored outputs without changing the receipt."""
    baseline = evaluate_baselines.evaluate_receipt(receipt_bytes, prompt_version=prompt_version)
    if baseline["mode"] != "extraction":
        raise ValueError("anchoring evaluation requires an extraction receipt")

    receipt = _load_receipt(receipt_bytes)
    cases, _ = load_challenge()
    raw_outputs = {row["case_id"]: row["raw_output"] for row in receipt["generated_outputs"]}
    anchored_outputs: dict[str, str] = {}
    case_diagnostics: dict[str, dict[str, Any]] = {}

    for case in cases:
        case_id = case["case_id"]
        raw_output = raw_outputs[case_id]
        try:
            payload = _strict_json_loads(raw_output)
        except (json.JSONDecodeError, ValueError, RecursionError):
            anchored_outputs[case_id] = raw_output
            case_diagnostics[case_id] = {
                "status": "parse_invalid",
                "output_error": "invalid_json",
                "claims": [],
            }
            continue

        try:
            claims = parse_candidates(payload)
        except (TypeError, ValueError, RecursionError):
            anchored_outputs[case_id] = raw_output
            case_diagnostics[case_id] = {
                "status": "schema_invalid",
                "output_error": "invalid_candidate_schema",
                "claims": [],
            }
            continue

        records = _case_records(case)
        anchored_claims: list[dict[str, Any]] = []
        claim_diagnostics: list[dict[str, Any]] = []
        changed = False
        for claim_index, claim in enumerate(claims):
            result = anchor_unique_quotes(claim, records)
            anchored_claims.append(asdict(result.claim))
            repairs = [asdict(repair) for repair in result.repairs]
            changed = changed or bool(repairs)
            claim_diagnostics.append(
                {
                    "claim_index": claim_index,
                    "repairs": repairs,
                    "unresolved": list(result.unresolved),
                }
            )

        anchored_outputs[case_id] = (
            json.dumps(
                {"claims": anchored_claims},
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            if changed
            else raw_output
        )
        case_diagnostics[case_id] = {"status": "anchored", "claims": claim_diagnostics}

    return {
        "schema_version": 1,
        "mode": "extraction",
        "prompt_version": prompt_version,
        "input_sha256": baseline["input_sha256"],
        "payload_sha256": baseline["payload_sha256"],
        "raw_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "model": baseline["model"],
        "provenance": baseline["provenance"],
        "token_totals": baseline["token_totals"],
        "truncated_cases": baseline["truncated_cases"],
        "evaluation_scope": baseline["evaluation_scope"],
        "interpretation_note": _INTERPRETATION_NOTE,
        "raw_result": baseline["result"],
        "anchored_result": evaluate_outputs(cases, anchored_outputs),
        "cases": case_diagnostics,
        "anchored_outputs": anchored_outputs,
    }


def _load_receipt(receipt_bytes: bytes) -> dict[str, Any]:
    value = _strict_json_loads(receipt_bytes.decode("utf-8"))
    if type(value) is not dict:
        raise ValueError("validated receipt unexpectedly is not an object")
    return value


def _strict_json_loads(value: str) -> object:
    return json.loads(value, object_pairs_hook=_unique_object, parse_constant=_reject_constant)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="completed receipt JSON")
    parser.add_argument("--output", required=True, type=Path, help="write the evaluation report")
    parser.add_argument("--prompt-version", choices=("v1", "explicit-v2"), default="v1")
    args = parser.parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        parser.error("--output must not point to the input receipt")
    try:
        report = evaluate_anchoring(args.input.read_bytes(), prompt_version=args.prompt_version)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
