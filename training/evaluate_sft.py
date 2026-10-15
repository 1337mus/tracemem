"""Score the prepared SFT pilot receipt against its authored dev fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from tracemem.challenge import evaluate_outputs
from tracemem.prompts import render_explicit_prompt

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_JOB = _ROOT / ".local/sft-pilot-20261015/training-job-v1.json"
_DEFAULT_RECEIPT = _ROOT / ".local/sft-pilot-20261015/remote-training-v1.json"
_DEFAULT_FIXTURE = _ROOT / "datasets/sft-pilot/eval.json"
_DEFAULT_OUTPUT = _ROOT / ".local/sft-pilot-20261015/dev-report/sft-dev-report-v1.json"
_EXPECTED_GENERATION = {"max_new_tokens": 1024, "do_sample": False}
_EXACT_MATCH_CAVEAT = (
    "Targets are authored, selective, and non-exhaustive; extra predictions count as false "
    "positives under exact matching. The score does not establish semantic support."
)


def _json_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return value


def _canonical_job_hash(job: dict[str, Any]) -> str:
    unsigned = {key: value for key, value in job.items() if key != "job_sha256"}
    canonical = json.dumps(
        unsigned, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _token_ids(value: object, label: str) -> list[int]:
    if type(value) is not list or any(type(token) is not int or token < 0 for token in value):
        raise ValueError(f"{label} must be a list of non-negative token IDs")
    return value


def _validated_outputs(
    raw: object, expected_ids: set[str], generation: dict[str, Any], label: str
) -> dict[str, dict[str, Any]]:
    if type(raw) is not list:
        raise ValueError(f"{label} outputs must be a list")
    outputs: dict[str, dict[str, Any]] = {}
    for row in raw:
        if type(row) is not dict:
            raise ValueError(f"{label} output must be an object")
        case_id = row.get("case_id")
        if type(case_id) is not str or case_id not in expected_ids or case_id in outputs:
            raise ValueError(f"{label} output IDs must match the job prompts one-to-one")
        _token_ids(row.get("prompt_token_ids"), f"{label} prompt_token_ids")
        if type(row.get("response_text")) is not str:
            raise ValueError(f"{label} response_text must be text")
        if (
            type(row.get("max_new_tokens")) is not int
            or row["max_new_tokens"] != generation["max_new_tokens"]
        ):
            raise ValueError(f"{label} generation settings differ from the receipt settings")
        if (
            type(row.get("generated_tokens")) is not int
            or not 0 <= row["generated_tokens"] <= generation["max_new_tokens"]
        ):
            raise ValueError(f"{label} generated_tokens is invalid")
        if type(row.get("reached_generation_limit")) is not bool:
            raise ValueError(f"{label} truncation flag is invalid")
        if row["reached_generation_limit"] and row["generated_tokens"] != row["max_new_tokens"]:
            raise ValueError(f"{label} truncation metadata is inconsistent")
        outputs[case_id] = row
    if set(outputs) != expected_ids:
        raise ValueError(f"{label} output IDs must match the job prompts one-to-one")
    return outputs


def evaluate_report(job_path: Path, receipt_path: Path) -> dict[str, Any]:
    """Validate inputs, then score exactly the eight dev prompts in the job."""
    job_bytes, receipt_bytes = job_path.read_bytes(), receipt_path.read_bytes()
    fixture_bytes = _DEFAULT_FIXTURE.read_bytes()
    job = _json_object(job_bytes, "job")
    receipt = _json_object(receipt_bytes, "receipt")
    fixture = _json_object(fixture_bytes, "dev fixture")

    request = job.get("request")
    if type(request) is not dict or request.get("mode") != "train":
        raise ValueError("job must contain a train request")
    prompts = request.get("dev_prompts")
    if type(prompts) is not list or len(prompts) != 8:
        raise ValueError("job must contain exactly eight dev prompts")
    prompt_by_id: dict[str, str] = {}
    for prompt in prompts:
        if type(prompt) is not dict or set(prompt) != {"case_id", "prompt"}:
            raise ValueError("each job dev prompt must contain only case_id and prompt")
        case_id, prompt_text = prompt["case_id"], prompt["prompt"]
        if type(case_id) is not str or not case_id or case_id in prompt_by_id:
            raise ValueError("job dev prompt IDs must be unique non-empty strings")
        if type(prompt_text) is not str or not prompt_text:
            raise ValueError("job dev prompts must be non-empty strings")
        prompt_by_id[case_id] = prompt_text
    expected_job_hash = job.get("job_sha256")
    if type(expected_job_hash) is not str or expected_job_hash != _canonical_job_hash(job):
        raise ValueError("job hash does not match job contents")
    source_hashes = job.get("source_sha256")
    if type(source_hashes) is not dict or type(source_hashes.get("dev")) is not str:
        raise ValueError("job dev fixture hash is missing")
    if source_hashes["dev"] != hashlib.sha256(fixture_bytes).hexdigest():
        raise ValueError("job dev fixture hash does not match fixture bytes")
    if receipt.get("job_sha256") != expected_job_hash:
        raise ValueError("receipt job hash does not match job")

    result = receipt.get("result")
    if type(result) is not dict or result.get("status") != "completed":
        raise ValueError("receipt result is not completed")
    generation = result.get("generation")
    if type(generation) is not dict:
        raise ValueError("receipt generation settings are missing")
    max_new_tokens = generation.get("max_new_tokens")
    if generation.get("max_new_tokens") != _EXPECTED_GENERATION["max_new_tokens"]:
        raise ValueError("receipt max_new_tokens setting does not match the training contract")
    if generation.get("do_sample") is not _EXPECTED_GENERATION["do_sample"]:
        raise ValueError("receipt do_sample setting does not match the training contract")

    cases = fixture.get("cases")
    if type(cases) is not list:
        raise ValueError("dev fixture cases must be a list")
    fixture_cases: dict[str, dict[str, Any]] = {}
    for case in cases:
        if type(case) is not dict or type(case.get("case_id")) is not str:
            continue
        if case["case_id"] in prompt_by_id:
            if case["case_id"] in fixture_cases:
                raise ValueError("dev fixture contains duplicate selected case IDs")
            fixture_cases[case["case_id"]] = case
    if set(fixture_cases) != set(prompt_by_id):
        raise ValueError("job prompt IDs do not match cases in the dev fixture")
    selected_cases: list[dict[str, Any]] = []
    for case_id, prompt_text in prompt_by_id.items():
        case = fixture_cases[case_id]
        if case.get("split") != "dev":
            raise ValueError("job prompts must select dev fixture cases only")
        if render_explicit_prompt(case) != prompt_text:
            raise ValueError(f"job prompt does not match rendered dev case {case_id}")
        selected_cases.append(case)

    base = _validated_outputs(generation.get("base_outputs"), set(prompt_by_id), generation, "base")
    tuned = _validated_outputs(
        generation.get("tuned_outputs"), set(prompt_by_id), generation, "tuned"
    )
    for case_id in prompt_by_id:
        if base[case_id]["prompt_token_ids"] != tuned[case_id]["prompt_token_ids"]:
            raise ValueError(f"base and tuned prompt token IDs differ for {case_id}")

    base_text = {case_id: row["response_text"] for case_id, row in base.items()}
    tuned_text = {case_id: row["response_text"] for case_id, row in tuned.items()}
    base_diagnostic = evaluate_outputs(selected_cases, base_text)
    tuned_diagnostic = evaluate_outputs(selected_cases, tuned_text)
    per_case = {
        case_id: {
            "base": {
                "response_text": base[case_id]["response_text"],
                "truncated": base[case_id]["reached_generation_limit"],
            },
            "tuned": {
                "response_text": tuned[case_id]["response_text"],
                "truncated": tuned[case_id]["reached_generation_limit"],
            },
        }
        for case_id in prompt_by_id
    }
    return {
        "schema_version": 1,
        "label": "agent-authored provisional dev diagnostic",
        "label_provenance": "agent-authored provisional",
        "human_reviewers": 0,
        "held_out": False,
        "case_count": len(selected_cases),
        "generation": {"max_new_tokens": max_new_tokens, "do_sample": generation["do_sample"]},
        "input_sha256": {
            "job": hashlib.sha256(job_bytes).hexdigest(),
            "receipt": hashlib.sha256(receipt_bytes).hexdigest(),
            "dev_fixture": hashlib.sha256(fixture_bytes).hexdigest(),
        },
        "exact_match_caveat": _EXACT_MATCH_CAVEAT,
        "base": {
            "diagnostic": base_diagnostic,
            "truncated_cases": sum(row["reached_generation_limit"] for row in base.values()),
            "truncated_case_ids": sorted(
                case_id for case_id, row in base.items() if row["reached_generation_limit"]
            ),
        },
        "tuned": {
            "diagnostic": tuned_diagnostic,
            "truncated_cases": sum(row["reached_generation_limit"] for row in tuned.values()),
            "truncated_case_ids": sorted(
                case_id for case_id, row in tuned.items() if row["reached_generation_limit"]
            ),
        },
        "cases": per_case,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, default=_DEFAULT_JOB)
    parser.add_argument("--receipt", type=Path, default=_DEFAULT_RECEIPT)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        report = evaluate_report(args.job, args.receipt)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            args.output,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except (OSError, TypeError, ValueError) as exc:
        print(f"SFT evaluation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"output": str(args.output), "case_count": report["case_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
