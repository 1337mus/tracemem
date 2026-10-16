"""Score bounded CPU-only quote-verifier receipts against frozen dev labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from experiments.quote_verifier import parse_verdicts, select_supported_candidates
from experiments.quote_verifier_payload import build_preparation, payload_sha256, validate_payload
from training.evaluate_quote_sft import _json_object, _json_value
from training.quote_data import validate_claims

_ROOT = Path(__file__).resolve().parents[1]
_REVIEW = _ROOT / "docs/quote-baseline-4b-adjudicated-review.json"
_MAP = _ROOT / "examples/synthetic/quote-baseline-4b-curator-map-v1.json"
_REVIEW_SHA256 = "c85a884fd55f23e8bd5f26eb8cb83e550ca09f630beecbe198e1aa2364597822"
_MAP_SHA256 = "6b69263b4023fd0b857dae9b0def583c489774341719abef5cb226d2ec56ec67"
_MAX_RECEIPT_BYTES = 5_000_000


def load_receipt_bytes(raw: bytes) -> dict[str, Any]:
    """Load one bounded receipt object with duplicate-key and NaN rejection."""
    if type(raw) is not bytes or len(raw) > _MAX_RECEIPT_BYTES:
        raise ValueError("receipt must be bytes no larger than 5000000 bytes")
    return _json_object(raw, "receipt")


def _labels() -> tuple[dict[tuple[str, int], str], dict[str, str]]:
    raw = {"adjudicated_review": _REVIEW.read_bytes(), "quote_v2_curator_map": _MAP.read_bytes()}
    hashes = {key: hashlib.sha256(value).hexdigest() for key, value in raw.items()}
    if hashes != {"adjudicated_review": _REVIEW_SHA256, "quote_v2_curator_map": _MAP_SHA256}:
        raise ValueError("frozen quote-v2 label file SHA-256 changed")
    review = _json_object(raw["adjudicated_review"], "frozen review")
    curator = _json_object(raw["quote_v2_curator_map"], "quote-v2 curator map")
    if (
        review.get("human_reviewers") != 0
        or curator.get("model") != "qwen3-4b"
        or curator.get("packet_sha256") != review.get("packet_sha256")
    ):
        raise ValueError("frozen quote-v2 label metadata is invalid")
    tasks = review.get("tasks")
    task_map = curator.get("task_map")
    if type(tasks) is not list or type(task_map) is not dict:
        raise ValueError("frozen quote-v2 label tables are invalid")
    by_task = {row["task_id"]: row for row in tasks}
    labels: dict[tuple[str, int], str] = {}
    good = bad = 0
    for task_id, source in task_map.items():
        if source["prompt_version"] != "quote-v2":
            continue
        task = by_task[task_id]
        case_id = source["case_id"]
        if task["reviewable"] is not True or source["item_id"] != f"{case_id}::quote-v2":
            raise ValueError("quote-v2 label task does not align with its source item")
        for row in task["claims"]:
            index = row["claim_index"]
            key = (case_id, index)
            if key in labels:
                raise ValueError("quote-v2 frozen labels are duplicate or malformed")
            label = "good" if row["label"] == "supported" and row["eligible"] is True else "bad"
            labels[key] = label
            good += label == "good"
            bad += label == "bad"
    if (len(labels), good, bad) != (43, 36, 7):
        raise ValueError("frozen quote-v2 label counts changed")
    return labels, hashes


def _prompt_data(prompt: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        before, candidate_json = prompt.rsplit("\nCandidates: ", 1)
        _, message_json = before.rsplit("\nMessages: ", 1)
        messages, candidates = (
            _json_value(message_json.encode(), "prompt messages"),
            _json_value(candidate_json.encode(), "prompt candidates"),
        )
    except (ValueError, UnicodeEncodeError) as exc:
        raise ValueError(
            "prepared prompt cannot be read as source messages and candidates"
        ) from exc
    if type(messages) is not list or type(candidates) is not list:
        raise ValueError("prepared prompt messages and candidates must be lists")
    return messages, candidates


def _preparation(
    preparation: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if type(preparation) is not dict or set(preparation) != {"payload", "audit"}:
        raise ValueError("preparation must contain exactly payload and audit")
    payload, audit = preparation["payload"], preparation["audit"]
    validate_payload(payload)
    if payload_sha256(preparation) != payload_sha256(build_preparation()):
        raise ValueError("preparation does not match fresh pinned preparation")
    cases = audit.get("cases")
    if type(cases) is not list or len(cases) != 12:
        raise ValueError("local selection audit must contain 12 cases")
    items = payload["items"]
    item_by_case = {item["case_id"]: item for item in items}
    cases_by_id: dict[str, dict[str, Any]] = {}
    for row in cases:
        case_id, indices, candidates = (
            row.get("case_id"),
            row.get("candidate_indices"),
            row.get("candidates"),
        )
        if (
            type(case_id) is not str
            or case_id in cases_by_id
            or row.get("source_item_id") != f"{case_id}::quote-v2"
            or type(indices) is not list
            or type(candidates) is not list
            or len(indices) != len(candidates)
            or any(type(i) is not int or i < 0 for i in indices)
            or indices != sorted(set(indices))
            or type(row.get("rejected")) is not list
        ):
            raise ValueError("local audit candidate provenance is invalid")
        item = item_by_case.get(case_id)
        if candidates:
            if (
                row.get("status") != "semantic-ready"
                or item is None
                or item.get("item_id") != f"{case_id}::quote-verifier-v1"
                or item.get("candidate_count") != len(candidates)
            ):
                raise ValueError("semantic payload item does not match its audited candidates")
            messages, prompt_candidates = _prompt_data(item["prompt"])
            if prompt_candidates != candidates:
                raise ValueError("prompt candidates differ from the original audited candidates")
            validate_claims({"claims": candidates}, messages)
        elif item is not None:
            raise ValueError("empty audit case unexpectedly has a semantic payload item")
        cases_by_id[case_id] = row
    if set(item_by_case) != {case_id for case_id, row in cases_by_id.items() if row["candidates"]}:
        raise ValueError("payload and audit semantic case IDs differ")
    return payload, cases


def _metadata_error(
    row: dict[str, Any], settings: dict[str, Any], eos_ids: list[int]
) -> str | None:
    cap, prompt_cap = settings["max_new_tokens"], settings["max_prompt_tokens"]
    input_ids, output_ids = row.get("input_token_ids"), row.get("output_token_ids")
    inputs, outputs = row.get("input_tokens"), row.get("output_tokens")
    if (
        type(input_ids) is not list
        or any(type(x) is not int or x < 0 for x in input_ids)
        or type(inputs) is not int
        or inputs != len(input_ids)
        or not 1 <= inputs <= prompt_cap
    ):
        return "input token IDs must be non-negative and counts within the prompt cap"
    if (
        type(output_ids) is not list
        or any(type(x) is not int or x < 0 for x in output_ids)
        or type(outputs) is not int
        or outputs != len(output_ids)
        or not 1 <= outputs <= cap
    ):
        return "output token IDs must be non-negative and counts within the output cap"
    hit, ended, last, finish = (
        row.get("hit_max_new_tokens"),
        row.get("ended_with_eos"),
        row.get("last_token_id"),
        row.get("finish_reason"),
    )
    if type(hit) is not bool or hit != (outputs >= cap):
        return "hit_max_new_tokens disagrees with output token count"
    if (
        type(ended) is not bool
        or type(last) is not int
        or last != output_ids[-1]
        or ended != (last in eos_ids)
    ):
        return "EOS flag or last token disagrees with output token IDs"
    expected_finish = "eos" if ended else "length" if hit else "unknown"
    if finish != expected_finish:
        return "finish_reason disagrees with EOS and output-limit metadata"
    if hit:
        return "output reached max_new_tokens"
    if not ended:
        return "output did not end with EOS"
    if type(row.get("raw_output")) is not str:
        return "raw_output must be text"
    return None


def evaluate(receipt: dict[str, Any], preparation: dict[str, Any]) -> dict[str, Any]:
    """Validate and score one strict receipt, retaining original candidate provenance."""
    payload, audit_cases = _preparation(preparation)
    if type(receipt) is not dict:
        raise ValueError("receipt must be a JSON object")
    expected = {
        "schema_version": 1,
        "mode": "quote-verifier-semantic",
        "payload_sha256": payload_sha256(payload),
        "runner_sha256": payload["runner_source_sha256"],
    }
    for key, value in expected.items():
        if (key == "schema_version" and type(receipt.get(key)) is not int) or receipt.get(
            key
        ) != value:
            raise ValueError(f"receipt {key} does not match the prepared payload")
    for key in ("model", "settings"):
        actual = receipt.get(key)
        if type(actual) is not dict or payload_sha256(actual) != payload_sha256(payload[key]):
            raise ValueError(f"receipt {key} does not match the prepared payload")
    for key in (
        "input_sha256",
        "generator_receipt_sha256",
        "candidate_sha256",
        "items_sha256",
        "renderer_source_sha256",
        "payload_source_sha256",
    ):
        if receipt.get(key) != payload[key]:
            raise ValueError(f"receipt {key} does not match the prepared payload")
    items, outputs = payload["items"], receipt.get("generated_outputs")
    if (
        type(outputs) is not list
        or type(receipt.get("item_count")) is not int
        or receipt.get("item_count") != len(items)
        or type(receipt.get("case_count")) is not int
        or receipt.get("case_count") != len(items)
        or len(outputs) != len(items)
    ):
        raise ValueError("receipt output counts do not match the prepared batches")
    eos_ids = receipt.get("generation_eos_token_ids")
    if (
        type(eos_ids) is not list
        or not eos_ids
        or any(type(x) is not int or x < 0 for x in eos_ids)
        or len(set(eos_ids)) != len(eos_ids)
    ):
        raise ValueError("receipt EOS token IDs are invalid")
    expected_ids = {item["item_id"]: item["case_id"] for item in items}
    output_by_case: dict[str, dict[str, Any]] = {}
    for row in outputs:
        if (
            type(row) is not dict
            or type(row.get("case_id")) is not str
            or type(row.get("item_id")) is not str
        ):
            raise ValueError("receipt output IDs are malformed")
        case_id, item_id = row["case_id"], row["item_id"]
        if expected_ids.get(item_id) != case_id or case_id in output_by_case:
            raise ValueError("receipt output IDs must match prepared items one-to-one")
        output_by_case[case_id] = row
    if set(output_by_case) != set(expected_ids.values()):
        raise ValueError("receipt output IDs must match prepared items one-to-one")

    labels, label_hashes = _labels()
    semantic_labels = [
        (row["case_id"], index, labels.get((row["case_id"], index)))
        for row in audit_cases
        for index in row["candidate_indices"]
    ]
    if len(semantic_labels) != 39 or any(label is None for _, _, label in semantic_labels):
        raise ValueError("semantic candidates do not align with all 39 frozen quote-v2 labels")
    results: list[dict[str, Any]] = []
    accepted_counts = {
        "accepted_good": 0,
        "accepted_bad": 0,
        "rejected_good": 0,
        "rejected_bad": 0,
        "unscored_good": 0,
        "unscored_bad": 0,
    }
    breakdown = {
        s: {"eligible": 0, "ineligible": 0} for s in ("supported", "unsupported", "uncertain")
    }
    valid_batches = failed_batches = valid_abstentions = 0
    generated_tokens = sum(
        row["output_tokens"]
        for row in outputs
        if type(row) is dict and type(row.get("output_tokens")) is int and row["output_tokens"] >= 0
    )
    for audit in audit_cases:
        case_id = audit["case_id"]
        candidates, indices = audit["candidates"], audit["candidate_indices"]
        base: dict[str, Any] = {
            "case_id": case_id,
            "source_item_id": audit["source_item_id"],
            "status": audit["status"],
            "generator_error": audit["generator_error"],
            "truncated": audit["truncated"],
            "finish_reason": audit["finish_reason"],
            "original_indices": indices,
            "batch_valid": None,
            "batch_failure_reason": None,
            "verdicts": [],
            "accepted": [],
            "rejected": [],
            "outcome": "no_candidates",
        }
        if not candidates:
            base["structural_rejections"] = audit["rejected"]
            results.append(base)
            continue
        output = output_by_case[case_id]
        error = _metadata_error(output, payload["settings"], eos_ids)
        verdicts: list[dict[str, Any]] = []
        if error is None:
            try:
                verdicts = parse_verdicts(output["raw_output"], len(candidates))
            except ValueError as exc:
                error = f"invalid verdict response: {exc}"
        if error is not None:
            failed_batches += 1
            base.update(batch_valid=False, batch_failure_reason=error, outcome="invalid_batch")
            for index, candidate in zip(indices, candidates, strict=True):
                label = labels[(case_id, index)]
                accepted_counts[f"unscored_{label}"] += 1
                base["rejected"].append(
                    {
                        "original_index": index,
                        "candidate": candidate,
                        "label": label,
                        "verdict": None,
                        "reason": f"batch failed: {error}",
                    }
                )
            results.append(base)
            continue
        selected = select_supported_candidates(candidates, verdicts)
        kept_positions = [
            i for i, v in enumerate(verdicts) if v["support"] == "supported" and v["eligible"]
        ]
        if selected != [candidates[i] for i in kept_positions]:
            raise ValueError("supported-candidate selector changed original candidate provenance")
        valid_batches += 1
        accepted_by_position = set(kept_positions)
        for position, (index, candidate, verdict) in enumerate(
            zip(indices, candidates, verdicts, strict=True)
        ):
            label = labels[(case_id, index)]
            accepted = position in accepted_by_position
            accepted_counts[f"{'accepted' if accepted else 'rejected'}_{label}"] += 1
            breakdown[verdict["support"]]["eligible" if verdict["eligible"] else "ineligible"] += 1
            entry = {
                "original_index": index,
                "candidate": candidate,
                "label": label,
                "verdict": verdict,
            }
            base["accepted" if accepted else "rejected"].append(entry)
            base["verdicts"].append({"original_index": index, **verdict})
        base.update(
            batch_valid=True, outcome="valid_retained" if kept_positions else "valid_abstention"
        )
        valid_abstentions += not kept_positions
        results.append(base)

    total_good = sum(label == "good" for _, _, label in semantic_labels)
    total_bad = len(semantic_labels) - total_good
    semantic = {
        "total": len(semantic_labels),
        **accepted_counts,
        "support_eligibility": breakdown,
        "valid_batches": valid_batches,
        "failed_batches": failed_batches,
    }
    criterion = {
        "all_three_bad_rejected": accepted_counts["accepted_bad"] == 0
        and accepted_counts["rejected_bad"] == 3,
        "at_least_35_good_retained": accepted_counts["accepted_good"] >= 35,
        "all_ten_batches_valid_untruncated": valid_batches == 10 and failed_batches == 0,
        "met": accepted_counts["accepted_bad"] == 0
        and accepted_counts["rejected_bad"] == 3
        and accepted_counts["accepted_good"] >= 35
        and valid_batches == 10
        and failed_batches == 0,
    }
    metrics = {
        key: receipt[key]
        for key in (
            "model_load_seconds",
            "remote_function_elapsed_seconds",
            "gpu_compute_seconds",
            "client_call_wall_seconds_including_image_and_startup",
            "peak_gpu_memory_bytes",
            "modal_function_call_id",
        )
        if key in receipt
    }
    for key, value in metrics.items():
        if key == "modal_function_call_id":
            if type(value) is not str:
                raise ValueError("receipt call ID must be text")
        elif type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"receipt metric {key} must be a finite non-negative number")
    metrics["generated_output_tokens"] = generated_tokens
    if "cost_usd" in receipt:
        cost = receipt["cost_usd"]
        if type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0:
            raise ValueError("receipt cost_usd must be a finite non-negative number")
        metrics["cost_usd"] = cost
    return {
        "schema_version": 1,
        "payload_sha256": expected["payload_sha256"],
        "source_hashes": {
            key: payload[key]
            for key in (
                "input_sha256",
                "generator_receipt_sha256",
                "candidate_sha256",
                "items_sha256",
            )
        },
        "runner_sha256": payload["runner_source_sha256"],
        "model": payload["model"],
        "settings": payload["settings"],
        "label_file_sha256": label_hashes,
        "case_results": results,
        "aggregate": {
            "deterministic_only": {
                "total": 39,
                "accepted_good": total_good,
                "accepted_bad": total_bad,
                "rejected_good": 0,
                "rejected_bad": 0,
            },
            "semantic": semantic,
            "no_candidate_cases": sum(not row["candidates"] for row in audit_cases),
            "gate_status_counts": {
                status: sum(row["status"] == status for row in audit_cases)
                for status in sorted({row["status"] for row in audit_cases})
            },
            "citation_rejection_count": sum(len(row["rejected"]) for row in audit_cases),
            "valid_abstention_cases": valid_abstentions,
        },
        "criterion": criterion,
        "receipt_metrics": metrics,
        "interpretation": {
            "human_reviewers": 0,
            "labels": "provisional same-family agent labels on development cases",
            "scope": "development experiment only; not fact recall, QA benefit, or ground truth",
            "selected_fact_retention": "pending separate audit",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument(
        "--output", required=True, type=Path, help="exclusive path for the score report"
    )
    args = parser.parse_args()
    try:
        preparation = build_preparation()
        report = evaluate(load_receipt_bytes(args.receipt.read_bytes()), preparation)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
