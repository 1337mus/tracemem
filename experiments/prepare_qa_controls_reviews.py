"""Prepare an offline blinded review bundle for fictional QA controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

from experiments import analyze_qa_interpretation_answers as _analysis
from experiments import prepare_qa_interpretation_controls as _preparer
from experiments import qa_interpretation_answer_receipt as _receipt

_ANALYZER_SHA256 = "b3b19ca9163ab77855b6005bfcb7dd7d6fda925b77000edc56292f6f14984fca"
_BUNDLE_KIND = "qa-interpretation-controls-answer-review"
_KEY_KIND = "qa-interpretation-controls-answer-review-key"
_RESPONSE_KIND = "qa-interpretation-controls-answer-review-responses"
_TASK_NAMESPACE = "tracemem.qa-interpretation-controls-answer-review.task.v1"
_SCOPE = {
    "split": "dev",
    "test": [],
    "human_label_count": 0,
    "scenario_groups": 6,
    "questions": 12,
    "cells": 36,
    "unique_prompts": 36,
    "correlated_groups": 6,
    "cells_are_independent_trials": False,
}


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256(_receipt._canonical_bytes(value))


def _verify_analyzer_source() -> str:
    digest = _sha256(Path(_analysis.__file__).read_bytes())
    if digest != _ANALYZER_SHA256:
        raise ValueError("pinned answer analyzer source hash differs")
    return digest


def _task_id(visible_content: dict[str, Any]) -> str:
    payload = {"namespace": _TASK_NAMESPACE, "content": visible_content}
    return "qa_controls_task_" + _canonical_sha256(payload)


def _source_by_question() -> tuple[dict[str, tuple[dict[str, Any], dict[str, Any]]], str]:
    raw = _preparer._FIXTURE_PATH.read_bytes()
    fixture_sha256 = _sha256(raw)
    if fixture_sha256 != _preparer._FIXTURE_SHA256:
        raise ValueError("fictional controls source fixture differs from its frozen pin")
    groups, counts = _preparer._parse_fixture(raw)
    if counts != {
        "scenario_groups": 6,
        "sessions": 7,
        "messages": 22,
        "questions": 12,
        "full_source_answerable": 10,
        "full_source_unknown": 2,
    }:
        raise ValueError("fictional controls source scope differs from its frozen counts")
    result = {}
    for group in groups:
        for question in group["questions"]:
            result[question["question_id"]] = (group, question)
    return result, fixture_sha256


def prepare_reviews(
    receipt: Any, *, seed: int
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build a blind controls review bundle, separate key, and blank response."""
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    analyzer_sha256 = _verify_analyzer_source()
    analysis = _analysis.analyze_receipt(receipt)
    if analysis["scope"] != _SCOPE:
        raise ValueError("answer analysis is outside the frozen dev-only controls scope")
    payload, references, _audit = _preparer.prepare_controls()
    source_by_question, fixture_sha256 = _source_by_question()
    review_preparer_sha256 = _sha256(Path(__file__).read_bytes())
    request = _receipt.build_request()
    prompt_by_id = {item["prompt_id"]: item for item in request["items"]}
    output_by_id = {row["prompt_id"]: row for row in analysis["per_cell"]}
    cells = references.get("cells")
    if (
        type(cells) is not list
        or len(cells) != 36
        or len(prompt_by_id) != 36
        or set(prompt_by_id) != set(output_by_id)
        or set(prompt_by_id) != {cell["prompt_id"] for cell in cells}
    ):
        raise ValueError("frozen prompts, outputs, and cells do not map one-to-one")
    if payload["items"] != request["items"]:
        raise ValueError("frozen source payload differs from the actual prompt request")

    tasks = []
    task_keys = []
    for cell in cells:
        prompt_id = cell["prompt_id"]
        try:
            group, question = source_by_question[cell["question_id"]]
            output = output_by_id[prompt_id]
            prompt = prompt_by_id[prompt_id]
        except KeyError:
            raise ValueError("frozen cell references unknown source or prompt data") from None
        visible = {
            "question": question["question"],
            "prompt_messages": prompt["messages"],
            "source_sessions": group["source_sessions"],
            "raw_output": output["raw_output"],
        }
        task_id = _task_id(visible)
        tasks.append({"task_id": task_id, **visible})
        task_keys.append(
            {
                "task_id": task_id,
                "cell_id": cell["cell_id"],
                "scenario_id": cell["scenario_id"],
                "question_id": cell["question_id"],
                "variant_id": cell["variant_id"],
                "prompt_id": prompt_id,
                "expected_prompt_answer": cell["expected_prompt_answer"],
                "full_source_expected_answer": cell["full_source_expected_answer"],
            }
        )
    if len({task["task_id"] for task in tasks}) != 36:
        raise ValueError("content-derived review task IDs are not unique")

    random.Random(seed).shuffle(tasks)
    body = {"version": 1, "kind": _BUNDLE_KIND, "tasks": tasks}
    bundle_sha256 = _canonical_sha256(body)
    bundle = {**body, "bundle_sha256": bundle_sha256}
    report_sha256 = _canonical_sha256(analysis)
    key = {
        "version": 1,
        "kind": _KEY_KIND,
        "bundle_sha256": bundle_sha256,
        "seed": seed,
        "scope": {**_SCOPE, "test": list(_SCOPE["test"])},
        "source_pins": {
            "analyzer_source_sha256": analyzer_sha256,
            "review_preparer_source_sha256": review_preparer_sha256,
            "fixture_sha256": fixture_sha256,
            **analysis["provenance"],
            "report_sha256": report_sha256,
        },
        "task_keys": sorted(task_keys, key=lambda row: row["task_id"]),
    }
    blank_response = {
        "version": 1,
        "kind": _RESPONSE_KIND,
        "bundle_sha256": bundle_sha256,
        "allowed_choices": {
            "source_answer": ["correct", "incorrect", "uncertain"],
            "prompt_support": ["supported", "unsupported", "uncertain"],
            "reviewer_kind": ["human", "agent"],
        },
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
    return bundle, key, blank_response


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prepare-qa-controls-reviews")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = _analysis.read_receipt(args.receipt)
        bundle, key, blank_response = prepare_reviews(receipt, seed=args.seed)
        if args.output_dir.exists() or args.output_dir.is_symlink():
            raise FileExistsError(
                f"refusing to overwrite existing output directory: {args.output_dir}"
            )
        args.output_dir.mkdir(mode=0o700)
        args.output_dir.chmod(0o700)
        documents = (
            ("bundle.json", bundle),
            ("key.json", key),
            ("response-template.json", blank_response),
        )
        for name, document in documents:
            _analysis.write_analysis(args.output_dir / name, document)
        print(
            json.dumps(
                {
                    "output_dir": str(args.output_dir),
                    "bundle_sha256": bundle["bundle_sha256"],
                    "tasks": len(bundle["tasks"]),
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
