"""Score matched base and tuned outputs for the provisional quote-v2 dev set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from training.evaluate_sft import _canonical_job_hash, _token_ids
from training.quote_data import render_quote_prompt, validate_claims, validate_dataset
from training.train_sft import MODEL_ID, MODEL_REVISION

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "datasets/sft-quote-v2/dev.json"
DEFAULT_JOB = ROOT / ".local/sft-quote-v2-20261015/training-job-v1.json"
DEFAULT_RECEIPT = ROOT / ".local/sft-quote-v2-20261015/remote-training-v1.json"
DEFAULT_OUTPUT = ROOT / ".local/sft-quote-v2-20261015/dev-report/quote-v2-dev-report.json"
GENERATION = {"max_new_tokens": 512, "do_sample": False}
CLAIM_FIELDS = {"claim", "message_id", "quote", "valid_from", "valid_to"}
JsonMap = dict[str, Any]


def _json_value(raw: bytes, label: str) -> Any:
    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"invalid JSON constant: {value}")

    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=unique_pairs, parse_constant=reject_constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    return value


def _json_object(raw: bytes, label: str) -> dict[str, Any]:
    value = _json_value(raw, label)
    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return value


def _strict_schema_claims(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    if set(parsed) != {"claims"} or type(parsed["claims"]) is not list:
        raise ValueError("response must contain exactly a claims list")
    claims: list[dict[str, Any]] = []
    for item in parsed["claims"]:
        if type(item) is not dict or set(item) != CLAIM_FIELDS:
            raise ValueError("claim has missing or unexpected fields")
        if any(type(item[key]) is not str for key in ("claim", "message_id", "quote")):
            raise ValueError("claim, message_id, and quote must be strings")
        if any(
            item[key] is not None and type(item[key]) is not str
            for key in ("valid_from", "valid_to")
        ):
            raise ValueError("valid_from and valid_to must be strings or null")
        claims.append(item)
    return claims


def _validated_outputs(raw: object, expected_ids: set[str], label: str) -> dict[str, JsonMap]:
    if type(raw) is not list:
        raise ValueError(f"{label} outputs must be a list")
    outputs: dict[str, dict[str, Any]] = {}
    for row in raw:
        if type(row) is not dict:
            raise ValueError(f"{label} output must be an object")
        case_id = row.get("case_id")
        if type(case_id) is not str or case_id not in expected_ids or case_id in outputs:
            raise ValueError(f"{label} output IDs must match the dev prompts one-to-one")
        prompt_ids = _token_ids(row.get("prompt_token_ids"), f"{label} prompt_token_ids")
        if not prompt_ids:
            raise ValueError(f"{label} prompt_token_ids must not be empty")
        completion_ids = _token_ids(
            row.get("completion_token_ids"), f"{label} completion_token_ids"
        )
        generated = row.get("generated_tokens")
        if type(generated) is not int or not 0 <= generated <= GENERATION["max_new_tokens"]:
            raise ValueError(f"{label} generated_tokens is invalid")
        if len(completion_ids) != generated:
            raise ValueError(f"{label} completion token count does not match generated_tokens")
        if (
            row.get("max_new_tokens") != GENERATION["max_new_tokens"]
            or type(row.get("max_new_tokens")) is not int
        ):
            raise ValueError(f"{label} generation settings differ from the contract")
        truncated = row.get("reached_generation_limit")
        if type(truncated) is not bool or (truncated and generated != GENERATION["max_new_tokens"]):
            raise ValueError(f"{label} truncation metadata is inconsistent")
        if type(row.get("response_text")) is not str or type(row.get("raw_text")) is not str:
            raise ValueError(f"{label} raw response fields must be strings")
        outputs[case_id] = row
    if set(outputs) != expected_ids:
        raise ValueError(f"{label} output IDs must match the dev prompts one-to-one")
    return outputs


def _normalized_statement(value: str) -> str:
    return " ".join(value.casefold().split())


def _reference_key(claim: dict[str, Any]) -> tuple[str, str | None, str | None, str]:
    return (
        _normalized_statement(claim["claim"]),
        claim["valid_from"],
        claim["valid_to"],
        claim["message_id"],
    )


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _reference_scores(
    expected: list[dict[str, Any]], predicted: list[dict[str, Any]]
) -> dict[str, Any]:
    expected_counts = Counter(_reference_key(item) for item in expected)
    predicted_counts = Counter(_reference_key(item) for item in predicted)
    matched = sum((expected_counts & predicted_counts).values())
    false_positive = len(predicted) - matched
    false_negative = len(expected) - matched
    precision = _ratio(matched, matched + false_positive)
    recall = _ratio(matched, matched + false_negative)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "matched_reference_strings": matched,
        "predicted_claims": len(predicted),
        "expected_claims": len(expected),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _arm_case(case: JsonMap, output: JsonMap) -> JsonMap:
    response = output["response_text"]
    parsed: dict[str, Any] | None = None
    schema_claims: list[dict[str, Any]] = []
    parse_error: str | None = None
    schema_error: str | None = None
    raw_claim_count = 0
    try:
        parsed_value = _json_value(response.encode("utf-8"), "model response")
    except ValueError as exc:
        parse_error = str(exc)
    else:
        if type(parsed_value) is dict:
            parsed = parsed_value
        else:
            schema_error = "response must be a JSON object"
    if parsed is not None:
        if type(parsed.get("claims")) is list:
            raw_claim_count = len(parsed["claims"])
        try:
            schema_claims = _strict_schema_claims(parsed)
        except ValueError as exc:
            schema_error = str(exc)

    citation_valid: list[dict[str, Any]] = []
    citation_errors: Counter[str] = Counter()
    for claim in schema_claims:
        try:
            checked = validate_claims({"claims": [claim]}, case["messages"])
        except ValueError as exc:
            citation_errors[str(exc)] += 1
        else:
            citation_valid.extend(checked)

    expected = case["expected"]
    scores = _reference_scores(expected, schema_claims)
    target_empty = not expected
    valid_abstention = (
        target_empty
        and not schema_claims
        and parse_error is None
        and schema_error is None
        and not citation_errors
        and not output["reached_generation_limit"]
    )
    return {
        "raw_text": output["raw_text"],
        "response_text": response,
        "generated_tokens": output["generated_tokens"],
        "truncated": output["reached_generation_limit"],
        "raw_claim_count": raw_claim_count,
        "schema_valid_claims": len(schema_claims),
        "parsed_claims": schema_claims,
        "parse_failure": parse_error is not None,
        "parse_error": parse_error,
        "schema_failure": schema_error is not None,
        "schema_error": schema_error,
        "citation_valid_claims": len(citation_valid),
        "citation_invalid_claims": len(schema_claims) - len(citation_valid),
        "citation_errors": dict(sorted(citation_errors.items())),
        "reference_string_scores": scores,
        "valid_empty_target_abstention": valid_abstention,
        "missed_nonempty_target": bool(expected) and not schema_claims,
    }


def _summarize_arm(cases: list[JsonMap], outputs: dict[str, JsonMap]) -> JsonMap:
    rows = {case["case_id"]: _arm_case(case, outputs[case["case_id"]]) for case in cases}
    expected_claims = [claim for case in cases for claim in case["expected"]]
    predicted_claims = [claim for row in rows.values() for claim in row["parsed_claims"]]
    score = _reference_scores(expected_claims, predicted_claims)
    valid_citations = sum(row["citation_valid_claims"] for row in rows.values())
    attempted_citations = sum(row["schema_valid_claims"] for row in rows.values())
    parse_failures = sum(row["parse_failure"] for row in rows.values())
    schema_failures = sum(row["schema_failure"] for row in rows.values())
    citation_failures = sum(row["citation_invalid_claims"] > 0 for row in rows.values())
    empty_targets = sum(not case["expected"] for case in cases)
    abstentions = sum(row["valid_empty_target_abstention"] for row in rows.values())
    missed_nonempty = sum(row["missed_nonempty_target"] for row in rows.values())

    by_tag: dict[str, list[dict[str, Any]]] = {}
    cases_by_id = {case["case_id"]: case for case in cases}
    for case_id, row in rows.items():
        for tag in cases_by_id[case_id]["tags"]:
            by_tag.setdefault(tag, []).append(row)
    tag_scores: dict[str, Any] = {}
    for tag, tagged_rows in sorted(by_tag.items()):
        tagged_expected = [
            claim for case in cases if tag in case["tags"] for claim in case["expected"]
        ]
        tagged_predicted = [claim for row in tagged_rows for claim in row["parsed_claims"]]
        tag_scores[tag] = {
            "case_count": len(tagged_rows),
            "reference_string_scores": _reference_scores(tagged_expected, tagged_predicted),
            "parse_failures": sum(row["parse_failure"] for row in tagged_rows),
            "schema_failures": sum(row["schema_failure"] for row in tagged_rows),
            "citation_valid_claims": sum(row["citation_valid_claims"] for row in tagged_rows),
            "citation_invalid_claims": sum(row["citation_invalid_claims"] for row in tagged_rows),
        }

    return {
        "response_count": len(cases),
        "parse_failures": parse_failures,
        "schema_failures": schema_failures,
        "citation_validation_failure_cases": citation_failures,
        "citation_integrity": {
            "claims_checked": attempted_citations,
            "valid_claims": valid_citations,
            "invalid_claims": attempted_citations - valid_citations,
            "interpretation": "exact source, quote, and explicit-date validation only",
        },
        "reference_string_scores": score,
        "abstention": {
            "empty_target_cases": empty_targets,
            "valid_empty_target_abstentions": abstentions,
            "missed_nonempty_target_cases": missed_nonempty,
        },
        "truncated_case_ids": sorted(case_id for case_id, row in rows.items() if row["truncated"]),
        "per_tag_breakdown": {
            "overlap": "cases with multiple tags appear in each applicable tag",
            "tags": tag_scores,
        },
        "cases": rows,
    }


def evaluate_report(
    job_path: Path,
    receipt_path: Path,
    *,
    fixture_path: Path = DEFAULT_FIXTURE,
) -> dict[str, Any]:
    """Validate all frozen inputs before scoring the matched 24-case comparison."""
    job_bytes = job_path.read_bytes()
    receipt_bytes = receipt_path.read_bytes()
    fixture_bytes = fixture_path.read_bytes()
    job = _json_object(job_bytes, "job")
    receipt = _json_object(receipt_bytes, "receipt")
    fixture_payload = _json_object(fixture_bytes, "dev fixture")

    job_hash = job.get("job_sha256")
    if type(job_hash) is not str or job_hash != _canonical_job_hash(job):
        raise ValueError("job hash does not match job contents")
    request = job.get("request")
    if type(request) is not dict or request.get("mode") != "train":
        raise ValueError("job must contain a train request")
    if request.get("profile") != "quote-v2":
        raise ValueError("job request profile must be quote-v2")
    prompts = request.get("dev_prompts")
    if type(prompts) is not list or len(prompts) != 24:
        raise ValueError("job must contain exactly 24 dev prompts")
    prompt_by_id: dict[str, str] = {}
    for row in prompts:
        if type(row) is not dict or set(row) != {"case_id", "prompt"}:
            raise ValueError("each job dev prompt must contain only case_id and prompt")
        case_id, prompt_text = row["case_id"], row["prompt"]
        if type(case_id) is not str or not case_id or case_id in prompt_by_id:
            raise ValueError("job dev prompt IDs must be unique non-empty strings")
        if type(prompt_text) is not str or not prompt_text:
            raise ValueError("job dev prompts must be non-empty strings")
        prompt_by_id[case_id] = prompt_text

    source_hashes = job.get("source_sha256")
    if type(source_hashes) is not dict or type(source_hashes.get("dev")) is not str:
        raise ValueError("job dev fixture hash is missing")
    fixture_hash = hashlib.sha256(fixture_bytes).hexdigest()
    if source_hashes["dev"] != fixture_hash:
        raise ValueError("job dev fixture hash does not match fixture bytes")
    if receipt.get("mode") != "train":
        raise ValueError("receipt mode must be train")
    if receipt.get("job_sha256") != job_hash:
        raise ValueError("receipt job hash does not match job")

    cases = validate_dataset(fixture_payload, "dev")
    if len(cases) != 24:
        raise ValueError("quote-v2 dev fixture must contain exactly 24 cases")
    case_by_id = {case["case_id"]: case for case in cases}
    expected_ids = set(case_by_id)
    if len(case_by_id) != len(cases) or set(prompt_by_id) != expected_ids:
        raise ValueError("job prompt IDs must match all 24 dev fixture cases")
    for case_id, prompt_text in prompt_by_id.items():
        if render_quote_prompt(case_by_id[case_id]) != prompt_text:
            raise ValueError(f"job prompt does not match rendered dev case {case_id}")

    result = receipt.get("result")
    if type(result) is not dict or result.get("status") != "completed":
        raise ValueError("receipt result is not completed")
    if result.get("model_id") != MODEL_ID or result.get("revision") != MODEL_REVISION:
        raise ValueError("receipt model or revision does not match the pinned training model")
    generation = result.get("generation")
    if type(generation) is not dict:
        raise ValueError("receipt generation settings are missing")
    if (
        generation.get("max_new_tokens") != GENERATION["max_new_tokens"]
        or type(generation.get("max_new_tokens")) is not int
    ):
        raise ValueError("receipt max_new_tokens setting does not match the contract")
    if generation.get("do_sample") is not GENERATION["do_sample"]:
        raise ValueError("receipt do_sample setting does not match the contract")
    base = _validated_outputs(generation.get("base_outputs"), expected_ids, "base")
    tuned = _validated_outputs(generation.get("tuned_outputs"), expected_ids, "tuned")
    for case_id in expected_ids:
        if base[case_id]["prompt_token_ids"] != tuned[case_id]["prompt_token_ids"]:
            raise ValueError(f"base and tuned prompt token IDs differ for {case_id}")

    ordered_cases = [case_by_id[row["case_id"]] for row in prompts]
    base_summary = _summarize_arm(ordered_cases, base)
    tuned_summary = _summarize_arm(ordered_cases, tuned)
    return {
        "schema_version": 1,
        "report_type": "provisional quote-v2 development diagnostic",
        "label": "agent-authored provisional; zero human reviewers",
        "label_provenance": "agent-authored provisional",
        "human_reviewers": 0,
        "held_out": False,
        "private_test_used": False,
        "private_test_read": False,
        "case_count": len(ordered_cases),
        "generation": dict(GENERATION),
        "model": {"model_id": MODEL_ID, "revision": MODEL_REVISION},
        "comparison": {
            "base_and_tuned_use_same_prompts": True,
            "base_and_tuned_use_same_generation_settings": True,
            "prior_pilot_comparison_caveat": "Representation, data, and steps changed vs pilot; "
            "effects are confounded.",
            "reference_string_metric_caveat": "Casefold/whitespace statement matches include exact "
            "date bounds and source ID; they do not prove semantic support or ground truth.",
            "semantic_support": "Not evaluated; unmatched outputs are not automatically "
            "unsupported. Use a separate blind review.",
            "receipt_provenance_limitations": "Completion counts are checked, but token IDs are "
            "not decoded and remote worker provenance/hash is not independently verified.",
            "failed_output_retention": "All cases, including parse/schema/source failures and "
            "truncations, retain both raw_text and response_text.",
        },
        "input_sha256": {
            "job": hashlib.sha256(job_bytes).hexdigest(),
            "receipt": hashlib.sha256(receipt_bytes).hexdigest(),
            "dev_fixture": fixture_hash,
        },
        "base": base_summary,
        "tuned": tuned_summary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, default=DEFAULT_JOB)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        report = evaluate_report(args.job, args.receipt, fixture_path=args.fixture)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except (OSError, TypeError, ValueError) as exc:
        print(f"Quote-v2 evaluation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"output": str(args.output), "case_count": report["case_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
