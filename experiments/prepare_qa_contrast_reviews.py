"""Prepare and validate separate offline qualitative QA answer reviews."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from tracemem.annotations import (
    _require_nonblank_text,
    _require_sha256,
    _validate_utc_timestamp,
)
from tracemem.challenge import _reject_constant, _unique_object

_BUNDLE_KIND = "qa-contrast-answer-review"
_KEY_KIND = "qa-contrast-answer-review-key"
_RESPONSE_KIND = "qa-contrast-answer-review-responses"
_TASK_NAMESPACE = "tracemem.qa-contrast-answer-review.task.v1"
_SOURCE_ANSWER_LABELS = {"correct", "incorrect", "uncertain"}
_PROMPT_SUPPORT_LABELS = {"supported", "unsupported", "uncertain"}
_RESPONSE_FIELDS = {
    "task_id",
    "source_answer",
    "prompt_support",
    "rationale",
    "reviewer_id",
    "reviewer_kind",
    "timestamp",
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )


def _fixture_content() -> tuple[dict[str, dict[str, Any]], str]:
    from experiments import prepare_qa_contrast_requests as request_preparer

    fixture_path = request_preparer._FIXTURE_PATH
    fixture_raw = fixture_path.read_bytes()
    fixture_sha256 = _sha256(fixture_raw)
    if fixture_sha256 != request_preparer._FIXTURE_SHA256:
        raise ValueError("QA contrast development fixture differs from the pinned fixture")
    fixture = json.loads(
        fixture_raw.decode("utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    if (
        type(fixture) is not dict
        or type(fixture["cases"]) is not list
        or len(fixture["cases"]) != 6
    ):
        raise ValueError("pinned QA contrast fixture structure is invalid")

    cases: dict[str, dict[str, Any]] = {}
    for row in fixture["cases"]:
        if type(row) is not dict or type(row.get("item")) is not dict:
            raise ValueError("QA contrast fixture case is invalid")
        item = row["item"]
        question_id = item.get("question_id")
        if type(question_id) is not str or not question_id or question_id in cases:
            raise ValueError("QA contrast fixture question ID is invalid or duplicated")
        dates = item.get("haystack_dates")
        sessions = item.get("haystack_sessions")
        if (
            type(dates) is not list
            or type(sessions) is not list
            or not sessions
            or len(dates) != len(sessions)
        ):
            raise ValueError("QA contrast fixture source sessions are invalid")
        source_sessions: list[dict[str, Any]] = []
        for date, messages in zip(dates, sessions, strict=True):
            if type(date) is not str or type(messages) is not list:
                raise ValueError("QA contrast fixture source session is invalid")
            cleaned_messages: list[dict[str, str]] = []
            for message in messages:
                if (
                    type(message) is not dict
                    or type(message.get("role")) is not str
                    or type(message.get("content")) is not str
                ):
                    raise ValueError("QA contrast fixture source message is invalid")
                cleaned_messages.append({"role": message["role"], "content": message["content"]})
            source_sessions.append({"date": date, "messages": cleaned_messages})
        question, question_date = item.get("question"), item.get("question_date")
        if type(question) is not str or type(question_date) is not str:
            raise ValueError("QA contrast fixture question content is invalid")
        cases[question_id] = {
            "question": question,
            "question_date": question_date,
            "source_sessions": source_sessions,
        }
    return cases, fixture_sha256


def _task_id(visible_content: dict[str, Any]) -> str:
    payload = {"namespace": _TASK_NAMESPACE, "content": visible_content}
    return "qa_task_" + _sha256(_canonical_bytes(payload))


def prepare_reviews(
    receipt: dict[str, Any], *, seed: int
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return a blind QA bundle, separate provenance key, and blank response template."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    from experiments.analyze_qa_contrast_answers import _fixed_inputs, analyze_receipt

    analysis = analyze_receipt(receipt)
    payload, references, audit, source_hashes = _fixed_inputs()
    fixture_by_id, fixture_sha256 = _fixture_content()
    if (
        analysis["dev_only"] is not True
        or analysis["provisional"] is not True
        or analysis["test"] != []
        or analysis["human_label_count"] != 0
        or analysis["question_count"] != 6
        or analysis["cell_count"] != 72
        or analysis["unique_prompt_count"] != 46
    ):
        raise ValueError("QA contrast analysis scope is invalid for review preparation")
    literal_analysis_report = (
        json.dumps(analysis, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")

    question_refs = {row["question_id"]: row for row in references["questions"]}
    prompt_items = {row["prompt_id"]: row for row in payload["items"]}
    output_by_prompt: dict[str, str] = {}
    for row in receipt["result"]["outputs"]:
        prompt_id = row["prompt_id"]
        raw_output = row["raw_output"]
        if prompt_id in output_by_prompt or type(raw_output) is not str:
            raise ValueError("validated QA receipt has duplicate prompts or invalid output")
        output_by_prompt[prompt_id] = raw_output

    cells_by_prompt: dict[str, list[dict[str, Any]]] = {prompt_id: [] for prompt_id in prompt_items}
    seen_cells: set[str] = set()
    question_by_prompt: dict[str, str] = {}
    for cell in audit["cell_mapping"]:
        prompt_id, question_id, cell_id = (
            cell["prompt_id"],
            cell["question_id"],
            cell["cell_id"],
        )
        if (
            prompt_id not in prompt_items
            or prompt_id not in output_by_prompt
            or question_id not in question_refs
            or question_id not in fixture_by_id
            or cell_id in seen_cells
        ):
            raise ValueError("QA contrast cell mapping is invalid or duplicated")
        if prompt_id in question_by_prompt and question_by_prompt[prompt_id] != question_id:
            raise ValueError("one frozen prompt maps to multiple fictional questions")
        question_by_prompt[prompt_id] = question_id
        seen_cells.add(cell_id)
        cells_by_prompt[prompt_id].append(
            {"cell_id": cell_id, "mode": cell["mode"], "ceiling": cell["ceiling"]}
        )
    if (
        len(seen_cells) != 72
        or set(prompt_items) != set(output_by_prompt)
        or set(prompt_items) != set(question_by_prompt)
        or len(prompt_items) != 46
    ):
        raise ValueError("frozen QA prompt and cell mapping coverage is invalid")

    tasks: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    seen_task_ids: set[str] = set()
    for prompt_id, prompt_item in prompt_items.items():
        question_id = question_by_prompt[prompt_id]
        reference = question_refs[question_id]
        fixture_case = fixture_by_id[question_id]
        messages = prompt_item["messages"]
        visible_content = {
            "question": fixture_case["question"],
            "question_date": fixture_case["question_date"],
            "prompt_messages": [dict(message) for message in messages],
            "source_sessions": fixture_case["source_sessions"],
            "raw_output": output_by_prompt[prompt_id],
        }
        task_id = _task_id(visible_content)
        if task_id in seen_task_ids:
            raise ValueError("QA review task ID collision")
        seen_task_ids.add(task_id)
        tasks.append({"task_id": task_id, **visible_content})
        key_rows.append(
            {
                "task_id": task_id,
                "prompt_id": prompt_id,
                "question_id": question_id,
                "authored_reference": reference["answer"],
                "is_answerable": not reference["is_abstention"],
                "cells": cells_by_prompt[prompt_id],
            }
        )

    random.Random(seed).shuffle(tasks)
    body = {"version": 1, "kind": _BUNDLE_KIND, "tasks": tasks}
    bundle_sha256 = _sha256(_canonical_bytes(body))
    bundle = {**body, "bundle_sha256": bundle_sha256}
    key_rows.sort(key=lambda row: row["task_id"])
    key = {
        "version": 1,
        "kind": _KEY_KIND,
        "bundle_sha256": bundle_sha256,
        "fixture_sha256": fixture_sha256,
        "receipt_sha256": analysis["provenance"]["receipt_sha256"],
        "review_preparer_source_sha256": _sha256(Path(__file__).read_bytes()),
        "literal_analysis_report_sha256": _sha256(literal_analysis_report),
        "frozen_input_sha256": {
            "payload_sha256": analysis["provenance"]["payload_sha256"],
            "references_sha256": analysis["provenance"]["references_sha256"],
            "audit_sha256": analysis["provenance"]["audit_sha256"],
            "analyzer_source_sha256": analysis["provenance"]["analyzer_source_sha256"],
            "preparer_source_sha256": source_hashes["preparer_source_sha256"],
        },
        "seed": seed,
        "dev_only": True,
        "provisional": True,
        "test": [],
        "human_label_count": 0,
        "label_provenance": "authored-agent-only; not human judgment",
        "question_count": 6,
        "prompt_count": 46,
        "cell_count": 72,
        "independent_trials": False,
        "tasks": key_rows,
    }
    template = {
        "version": 1,
        "kind": _RESPONSE_KIND,
        "bundle_sha256": bundle_sha256,
        "responses": [
            {
                "task_id": task["task_id"],
                "source_answer": "",
                "prompt_support": "",
                "rationale": "",
                "reviewer_id": "",
                "reviewer_kind": "",
                "timestamp": "",
            }
            for task in tasks
        ],
    }
    return bundle, key, template


def validate_responses(
    receipt: dict[str, Any], response_document: dict[str, Any], *, seed: int
) -> list[dict[str, Any]]:
    """Validate one complete QA review response and return rows in bundle order."""
    bundle, _key, _template = prepare_reviews(receipt, seed=seed)
    if type(response_document) is not dict or set(response_document) != {
        "version",
        "kind",
        "bundle_sha256",
        "responses",
    }:
        raise ValueError("response document has missing or unexpected fields")
    if (
        type(response_document["version"]) is not int
        or response_document["version"] != 1
        or response_document["kind"] != _RESPONSE_KIND
    ):
        raise ValueError("unsupported QA review response document kind or version")
    response_hash = _require_sha256(response_document["bundle_sha256"], "bundle_sha256")
    if response_hash != bundle["bundle_sha256"]:
        raise ValueError("response bundle hash does not match regenerated QA review bundle")
    raw_rows = response_document["responses"]
    if type(raw_rows) is not list:
        raise TypeError("responses must be a list")
    expected_ids = [task["task_id"] for task in bundle["tasks"]]
    expected_set = set(expected_ids)
    rows: dict[str, dict[str, Any]] = {}
    reviewer_ids: set[str] = set()
    reviewer_kinds: set[str] = set()
    for row in raw_rows:
        if type(row) is not dict or set(row) != _RESPONSE_FIELDS:
            raise ValueError("response row has missing or unexpected fields")
        task_id = row["task_id"]
        if type(task_id) is not str or task_id not in expected_set:
            raise ValueError("response has an unknown task ID")
        if task_id in rows:
            raise ValueError("response has a duplicate task ID")
        if (
            type(row["source_answer"]) is not str
            or row["source_answer"] not in _SOURCE_ANSWER_LABELS
        ):
            raise ValueError("source_answer label is invalid")
        if (
            type(row["prompt_support"]) is not str
            or row["prompt_support"] not in _PROMPT_SUPPORT_LABELS
        ):
            raise ValueError("prompt_support label is invalid")
        _require_nonblank_text(row["rationale"], "rationale", 4_000)
        reviewer_id = _require_nonblank_text(row["reviewer_id"], "reviewer_id", 256)
        reviewer_kind = row["reviewer_kind"]
        if type(reviewer_kind) is not str or reviewer_kind not in {"human", "agent"}:
            raise ValueError("reviewer_kind must be human or agent")
        _validate_utc_timestamp(row["timestamp"])
        reviewer_ids.add(reviewer_id)
        reviewer_kinds.add(reviewer_kind)
        rows[task_id] = dict(row)
    if len(reviewer_ids) > 1 or len(reviewer_kinds) > 1:
        raise ValueError("one response document must have a single reviewer ID and kind")
    if set(rows) != expected_set:
        missing = sorted(expected_set.difference(rows))
        raise ValueError(f"missing response task ID(s): {', '.join(missing)}")
    return [rows[task_id] for task_id in expected_ids]


def _write_json(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _preflight_output_dir(output_dir: Path, receipt_path: Path) -> None:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    input_path = receipt_path.resolve()
    outputs = [
        (output_dir / name).resolve()
        for name in ("bundle.json", "key.json", "response-template.json")
    ]
    if input_path in outputs:
        raise ValueError("an output path aliases the receipt input")
    if len(set(outputs)) != len(outputs):
        raise ValueError("output paths alias each other")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prepare-qa-contrast-reviews")
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="write blind review files to a new directory")
    export.add_argument("receipt", type=Path)
    export.add_argument("--seed", required=True, type=int)
    export.add_argument("--output-dir", required=True, type=Path)
    validate = commands.add_parser("validate", help="validate a complete response document")
    validate.add_argument("receipt", type=Path)
    validate.add_argument("--seed", required=True, type=int)
    validate.add_argument("--responses", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = _read_json(args.receipt)
        if args.command == "export":
            _preflight_output_dir(args.output_dir, args.receipt)
            bundle, key, template = prepare_reviews(receipt, seed=args.seed)
            from experiments import prepare_qa_contrast_requests as request_preparer

            request_preparer._write_new_output(
                args.output_dir,
                {
                    "bundle.json": _write_json(bundle),
                    "key.json": _write_json(key),
                    "response-template.json": _write_json(template),
                },
            )
        else:
            responses = _read_json(args.responses)
            rows = validate_responses(receipt, responses, seed=args.seed)
            first = rows[0]
            summary = {
                "validated_response_count": len(rows),
                "reviewer_kind_self_attested": first["reviewer_kind"],
                "source_answer_label_counts": dict(
                    sorted(Counter(row["source_answer"] for row in rows).items())
                ),
                "prompt_support_label_counts": dict(
                    sorted(Counter(row["prompt_support"] for row in rows).items())
                ),
            }
            print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"prepare_qa_contrast_reviews: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
