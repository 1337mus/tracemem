"""Offline evaluator for the provisional quote-v3 SFT comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from training.evaluate_quote_sft import _json_object, _summarize_arm, _validated_outputs
from training.evaluate_sft import _canonical_job_hash
from training.quote_data import render_quote_prompt, validate_dataset
from training.train_sft import _QUOTE_V3_PRIOR_ADAPTER, MODEL_ID, MODEL_REVISION

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JOB = ROOT / ".local/sft-quote-v3-20261015/training-job-v1.json"
DEFAULT_RECEIPT = ROOT / ".local/sft-quote-v3-20261015/remote-training-v1.json"
DEFAULT_REGRESSION = ROOT / "datasets/sft-quote-v2/dev.json"
DEFAULT_DEV = ROOT / "datasets/sft-quote-v3/dev.json"
DEFAULT_TEST = ROOT / "datasets/sft-quote-v3/test.json"
DEFAULT_OUTPUT = ROOT / ".local/sft-quote-v3-20261015/evaluator/quote-v3-report.json"
GENERATION = {"max_new_tokens": 512, "do_sample": False}
COHORT_SIZES = {"regression": 24, "dev": 24, "test": 48}
ARMS = ("base", "prior", "tuned")
_SHA256_LENGTH = 64
JsonMap = dict[str, Any]


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_fixture(path: Path, label: str, split: str, count: int) -> tuple[bytes, list[JsonMap]]:
    raw = path.read_bytes()
    payload = _json_object(raw, label)
    cases = validate_dataset(payload, split)
    if len(cases) != count:
        raise ValueError(f"{label} must contain exactly {count} cases")
    return raw, cases


def _strip_outer_json_fence(text: str) -> tuple[str, bool]:
    candidate = text.strip()
    opening, closing = "```json\n", "\n```"
    if candidate.startswith(opening) and candidate.endswith(closing):
        return candidate[len(opening) : -len(closing)], True
    return text, False


def _fence_diagnostic(cases: list[JsonMap], outputs: dict[str, JsonMap]) -> JsonMap:
    diagnostic_outputs: dict[str, JsonMap] = {}
    removed_ids: list[str] = []
    for case in cases:
        case_id = case["case_id"]
        output = dict(outputs[case_id])
        response, removed = _strip_outer_json_fence(output["response_text"])
        output["response_text"] = response
        diagnostic_outputs[case_id] = output
        if removed:
            removed_ids.append(case_id)

    summary = _summarize_arm(cases, diagnostic_outputs)
    for case_id, row in summary["cases"].items():
        original = outputs[case_id]
        row["original_raw_text"] = original["raw_text"]
        row["original_response_text"] = original["response_text"]
        row["diagnostic_response_text"] = diagnostic_outputs[case_id]["response_text"]
        row["outer_json_fence_removed"] = case_id in removed_ids
    return {
        "method": (
            "strip only an outer exact ```json newline ... newline ``` fence after text.strip()"
        ),
        "removed_case_ids": removed_ids,
        "summary": summary,
    }


def evaluate_report(
    job_path: Path,
    receipt_path: Path,
    *,
    regression_path: Path,
    dev_path: Path,
    test_path: Path,
) -> dict[str, Any]:
    """Validate the frozen job and score regression, dev, and test independently."""
    job_bytes, receipt_bytes = job_path.read_bytes(), receipt_path.read_bytes()
    job = _json_object(job_bytes, "job")
    receipt = _json_object(receipt_bytes, "receipt")
    if job.get("job_sha256") != _canonical_job_hash(job):
        raise ValueError("job hash does not match job contents")
    request = job.get("request")
    if type(request) is not dict or request.get("mode") != "train":
        raise ValueError("job must contain a train request")
    if request.get("profile") != "quote-v3":
        raise ValueError("job request profile must be quote-v3")
    if request.get("prior_adapter") != _QUOTE_V3_PRIOR_ADAPTER:
        raise ValueError("job prior adapter does not match the pinned quote-v2 checkpoint")

    train_rows = request.get("train_rows")
    if type(train_rows) is not list or len(train_rows) != 768:
        raise ValueError("quote-v3 job must contain exactly 768 training rows")
    train_ids: set[str] = set()
    for row in train_rows:
        if type(row) is not dict or row.get("split") != "train":
            raise ValueError("quote-v3 training rows must be train objects")
        case_id = row.get("case_id")
        if type(case_id) is not str or not case_id.strip() or case_id in train_ids:
            raise ValueError("quote-v3 training row IDs must be unique non-empty strings")
        train_ids.add(case_id)

    prompts = request.get("dev_prompts")
    if type(prompts) is not list or len(prompts) != 96:
        raise ValueError("quote-v3 job must contain exactly 96 prompt-only evaluation prompts")
    prompt_by_id: dict[str, str] = {}
    for row in prompts:
        if type(row) is not dict or set(row) != {"case_id", "prompt"}:
            raise ValueError("each quote-v3 prompt must contain only case_id and prompt")
        case_id, prompt_text = row["case_id"], row["prompt"]
        if type(case_id) is not str or not case_id.strip() or case_id in prompt_by_id:
            raise ValueError("quote-v3 prompt IDs must be unique non-empty strings")
        if type(prompt_text) is not str or not prompt_text.strip():
            raise ValueError("quote-v3 prompts must be non-empty strings")
        prompt_by_id[case_id] = prompt_text

    raw_fixtures: dict[str, bytes] = {}
    cases_by_cohort: dict[str, list[JsonMap]] = {}
    for name, path, split in (
        ("regression", regression_path, "dev"),
        ("dev", dev_path, "dev"),
        ("test", test_path, "test"),
    ):
        raw, cases = _load_fixture(path, f"{name} fixture", split, COHORT_SIZES[name])
        raw_fixtures[name] = raw
        cases_by_cohort[name] = cases

    all_cases = [case for name in COHORT_SIZES for case in cases_by_cohort[name]]
    fixture_ids = [case["case_id"] for case in all_cases]
    if len(set(fixture_ids)) != len(fixture_ids):
        raise ValueError("quote-v3 fixtures contain duplicate evaluation case IDs")
    if train_ids.intersection(fixture_ids):
        raise ValueError("quote-v3 evaluation IDs must be disjoint from training rows")
    case_by_id = {case["case_id"]: case for case in all_cases}
    if set(prompt_by_id) != set(case_by_id):
        raise ValueError("job prompt IDs must match the three evaluation fixtures")
    for case_id, prompt_text in prompt_by_id.items():
        if render_quote_prompt(case_by_id[case_id]) != prompt_text:
            raise ValueError(f"job prompt does not match rendered case {case_id}")

    cohorts = request.get("eval_cohorts")
    if type(cohorts) is not dict or set(cohorts) != set(COHORT_SIZES):
        raise ValueError("quote-v3 eval_cohorts must include regression, dev, and test")
    cohort_ids: list[str] = []
    for name, expected_count in COHORT_SIZES.items():
        ids = cohorts[name]
        if type(ids) is not list or len(ids) != expected_count:
            raise ValueError(f"quote-v3 {name} cohort must contain {expected_count} case IDs")
        if ids != [case["case_id"] for case in cases_by_cohort[name]]:
            raise ValueError(f"quote-v3 {name} cohort IDs do not match its fixture order")
        cohort_ids.extend(ids)
    if cohort_ids != [row["case_id"] for row in prompts]:
        raise ValueError("quote-v3 prompts must follow the ordered cohort IDs exactly")

    source_hashes = job.get("source_sha256")
    if type(source_hashes) is not dict or set(source_hashes) != {"train", *COHORT_SIZES}:
        raise ValueError("quote-v3 source hashes must include train, regression, dev, and test")
    for name, value in source_hashes.items():
        if not _is_sha256(value):
            raise ValueError(f"quote-v3 {name} source hash is invalid")
    fixture_hashes = {name: hashlib.sha256(raw_fixtures[name]).hexdigest() for name in COHORT_SIZES}
    for name, digest in fixture_hashes.items():
        if source_hashes[name] != digest:
            raise ValueError(f"job {name} fixture hash does not match fixture bytes")

    if not _is_sha256(job.get("runner_sha256")):
        raise ValueError("job runner hash is missing or invalid")
    module_hashes = job.get("module_sha256")
    if (
        type(module_hashes) is not dict
        or not module_hashes
        or any(not _is_sha256(value) for value in module_hashes.values())
    ):
        raise ValueError("job module hashes are missing or invalid")
    job_hash = job["job_sha256"]
    if receipt.get("job_sha256") != job_hash:
        raise ValueError("receipt job hash does not match job")
    if receipt.get("runner_sha256") != job["runner_sha256"]:
        raise ValueError("receipt runner hash does not match job")

    result = receipt.get("result")
    if (
        type(result) is not dict
        or result.get("mode") != "train"
        or result.get("status") != "completed"
    ):
        raise ValueError("receipt result is not a completed training run")
    if result.get("model_id") != MODEL_ID or result.get("revision") != MODEL_REVISION:
        raise ValueError("receipt model or revision does not match the pinned training model")
    comparison = result.get("comparison")
    comparison_checks = (
        "verified_before",
        "verified_after",
        "unloaded_without_merge",
        "rng_restored",
        "no_prior_lora_parameters",
    )
    if (
        type(comparison) is not dict
        or comparison.get("prior_adapter") != request["prior_adapter"]
        or any(comparison.get(field) is not True for field in comparison_checks)
    ):
        raise ValueError("receipt prior adapter comparison metadata is invalid")
    training = result.get("training")
    if type(training) is not dict or any(
        type(training.get(field)) is not int or training[field] != expected
        for field, expected in (
            ("max_steps", 192),
            ("optimizer_steps", 192),
            ("examples_consumed", 768),
        )
    ):
        raise ValueError("receipt training counts do not match the quote-v3 contract")
    generation = result.get("generation")
    if type(generation) is not dict:
        raise ValueError("receipt generation settings are missing")
    output_arms = {key for key in generation if key.endswith("_outputs")}
    if output_arms != {f"{arm}_outputs" for arm in ARMS}:
        raise ValueError("receipt output arms must be exactly base, prior, and tuned")
    if (
        type(generation.get("max_new_tokens")) is not int
        or generation["max_new_tokens"] != GENERATION["max_new_tokens"]
        or generation.get("do_sample") is not GENERATION["do_sample"]
    ):
        raise ValueError("receipt generation settings do not match the quote-v3 contract")

    expected_ids = set(case_by_id)
    outputs = {
        arm: _validated_outputs(generation.get(f"{arm}_outputs"), expected_ids, arm) for arm in ARMS
    }
    for case_id in expected_ids:
        prompt_ids = outputs["base"][case_id]["prompt_token_ids"]
        if any(outputs[arm][case_id]["prompt_token_ids"] != prompt_ids for arm in ARMS[1:]):
            raise ValueError(f"cross-arm prompt token IDs differ for {case_id}")
    cohorts_report: dict[str, JsonMap] = {}
    for name, cases in cases_by_cohort.items():
        ids = {case["case_id"] for case in cases}
        arms: dict[str, JsonMap] = {}
        for arm in ARMS:
            selected_outputs = {case_id: outputs[arm][case_id] for case_id in ids}
            arms[arm] = {
                "strict_raw_metrics": _summarize_arm(cases, selected_outputs),
                "fence_diagnostic": _fence_diagnostic(cases, selected_outputs),
            }
        cohorts_report[name] = {
            "case_count": len(cases),
            "case_ids": [case["case_id"] for case in cases],
            "arms": arms,
        }

    return {
        "schema_version": 1,
        "report_type": "provisional quote-v3 cohort development diagnostic",
        "label": "agent-authored provisional; zero human reviewers",
        "label_provenance": "agent-authored provisional",
        "human_reviewers": 0,
        "semantic_support": "not yet evaluated",
        "reference_match_caveat": (
            "Reference-string matches are not accuracy or proof of semantic support; labels are "
            "provisional, selective, and non-exhaustive."
        ),
        "generation": dict(GENERATION),
        "model": {"model_id": MODEL_ID, "revision": MODEL_REVISION},
        "comparison": {
            "arms": list(ARMS),
            "same_prompts_and_generation_settings": True,
            "prior_adapter_verification": (
                "recorded by the remote run; local evaluator checks the receipt fields"
            ),
            "remote_provenance_limit": (
                "Job module and runner hashes are format-checked and receipt-bound; the evaluator "
                "cannot independently attest the remote host or training process."
            ),
            "strict_metrics": "Score original response_text without fence stripping or repair.",
            "fence_diagnostic": (
                "Separate diagnostic strips only an outer exact ```json newline ... newline ``` "
                "fence after text.strip(); it does not repair response content."
            ),
            "failed_output_retention": (
                "Original raw_text, response_text, generated token counts, parse and schema "
                "failures, "
                "citation failures, and truncations remain in strict metrics."
            ),
        },
        "input_sha256": {
            "job": hashlib.sha256(job_bytes).hexdigest(),
            "receipt": hashlib.sha256(receipt_bytes).hexdigest(),
            "fixtures": fixture_hashes,
        },
        "cohorts": cohorts_report,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, default=DEFAULT_JOB)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--regression-fixture", type=Path, default=DEFAULT_REGRESSION)
    parser.add_argument("--dev-fixture", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--test-fixture", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        report = evaluate_report(
            args.job,
            args.receipt,
            regression_path=args.regression_fixture,
            dev_path=args.dev_fixture,
            test_path=args.test_fixture,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except (OSError, TypeError, ValueError) as exc:
        print(f"Quote-v3 evaluation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"output": str(args.output), "cohort_counts": COHORT_SIZES}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
