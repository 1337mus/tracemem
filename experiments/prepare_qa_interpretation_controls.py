"""Prepare pinned, fictional QA interpretation controls without model inference."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from experiments import prepare_synthetic_qa
from experiments.prepare_synthetic_qa import (
    _SYSTEM_PROMPT,
    _USER_TEMPLATE,
    _canonical_bytes,
    _sha256,
)

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE_PATH = _ROOT / "examples/synthetic/qa-interpretation-controls-dev-v1.json"
_FIXTURE_SHA256 = "09698c216d3f8f9751383bef8e984bf4385701b7e199fa7413b70f23fae31d53"
_VARIANTS = ("full_all_message_ids", "support_omitted", "empty")
_EXPECTED_COUNTS = {"scenario_groups": 6, "sessions": 7, "messages": 22, "questions": 12}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("fixture contains duplicate object keys")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"fixture contains invalid JSON constant: {value}")


def _require_keys(value: object, keys: Iterable[str], name: str) -> dict[str, Any]:
    expected = set(keys)
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"fixture {name} fields are invalid")
    return value


def _nonblank(value: object, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"fixture {name} must be nonblank text")
    return value


def _parse_fixture(raw: bytes) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if type(raw) is not bytes or len(raw) > 1024 * 1024:
        raise ValueError("fixture must be at most 1 MiB of bytes")
    try:
        fixture = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        raise ValueError("fixture is not valid strict JSON") from None
    except ValueError as error:
        raise ValueError(str(error)) from None

    envelope = _require_keys(
        fixture,
        (
            "kind",
            "version",
            "provenance",
            "split",
            "provisional",
            "human_labels",
            "scenario_groups",
            "license",
        ),
        "envelope",
    )
    if (
        envelope["kind"] != "qa-interpretation-controls-candidate"
        or type(envelope["version"]) is not int
        or envelope["version"] != 1
        or envelope["provenance"] != "authored-fictional-agent-only"
        or envelope["split"] != "dev"
        or envelope["provisional"] is not True
        or type(envelope["human_labels"]) is not int
        or envelope["human_labels"] != 0
        or envelope["license"] != "MIT"
    ):
        raise ValueError("fixture envelope is not the provisional fictional dev candidate")
    groups = envelope["scenario_groups"]
    if type(groups) is not list or len(groups) != _EXPECTED_COUNTS["scenario_groups"]:
        raise ValueError("fixture must contain exactly six scenario groups")

    scenario_ids: set[str] = set()
    session_ids: set[str] = set()
    message_ids: set[str] = set()
    question_ids: set[str] = set()
    counts = {key: 0 for key in _EXPECTED_COUNTS}
    answerable = unknown = 0
    for group in groups:
        _require_keys(group, ("scenario_id", "source_sessions", "questions"), "scenario group")
        scenario_id = _nonblank(group["scenario_id"], "scenario_id")
        if scenario_id in scenario_ids:
            raise ValueError("fixture scenario ids must be unique")
        scenario_ids.add(scenario_id)
        sessions = group["source_sessions"]
        questions = group["questions"]
        if type(sessions) is not list or not sessions or type(questions) is not list:
            raise ValueError("fixture source sessions and questions must be lists")
        ordered_sessions: list[tuple[str, dict[str, Any]]] = []
        group_message_ids: set[str] = set()
        for session in sessions:
            _require_keys(session, ("session_id", "date", "messages"), "source session")
            session_id = _nonblank(session["session_id"], "session_id")
            if session_id in session_ids:
                raise ValueError("fixture session ids must be unique")
            session_ids.add(session_id)
            date_value = _nonblank(session["date"], "session date")
            try:
                session_date = dt.date.fromisoformat(date_value)
            except ValueError:
                raise ValueError("fixture session dates must be ISO calendar dates") from None
            if session_date.isoformat() != date_value:
                raise ValueError("fixture session dates must be canonical ISO dates")
            messages = session["messages"]
            if type(messages) is not list or not messages:
                raise ValueError("fixture sessions must contain messages")
            ordered_sessions.append((date_value, session))
            for message in messages:
                _require_keys(message, ("message_id", "role", "content"), "source message")
                message_id = _nonblank(message["message_id"], "message_id")
                if message_id in message_ids:
                    raise ValueError("fixture message ids must be globally unique")
                message_ids.add(message_id)
                role = message["role"]
                if type(role) is not str or role not in {"user", "assistant"}:
                    raise ValueError("fixture message role must be user or assistant")
                _nonblank(message["content"], "message content")
                group_message_ids.add(message_id)
                counts["messages"] += 1
            counts["sessions"] += 1
        ordered_sessions.sort(key=lambda row: row[0])
        ordered_message_ids = [
            message["message_id"]
            for _, session in ordered_sessions
            for message in session["messages"]
        ]

        for question in questions:
            _require_keys(
                question,
                (
                    "question_id",
                    "question",
                    "authored_answer",
                    "full_source_expected_answer",
                    "is_answerable",
                    "supporting_message_ids",
                    "rationale",
                    "context_variants",
                ),
                "question",
            )
            question_id = _nonblank(question["question_id"], "question_id")
            if question_id in question_ids:
                raise ValueError("fixture question ids must be globally unique")
            question_ids.add(question_id)
            _nonblank(question["question"], "question text")
            full_answer = _nonblank(question["full_source_expected_answer"], "full-source answer")
            if question["authored_answer"] != full_answer:
                raise ValueError("fixture authored answer differs from the full-source reference")
            is_answerable = question["is_answerable"]
            if type(is_answerable) is not bool or is_answerable != (full_answer != "UNKNOWN"):
                raise ValueError("fixture answerability must match the full-source UNKNOWN label")
            answerable += is_answerable
            unknown += not is_answerable
            supporting_ids = question["supporting_message_ids"]
            if (
                type(supporting_ids) is not list
                or any(type(value) is not str for value in supporting_ids)
                or len(supporting_ids) != len(set(supporting_ids))
                or not set(supporting_ids) <= group_message_ids
                or bool(supporting_ids) != is_answerable
            ):
                raise ValueError("fixture supporting-message references are invalid")
            variants = _require_keys(question["context_variants"], _VARIANTS, "context variants")
            for variant_id in _VARIANTS:
                variant = _require_keys(
                    variants[variant_id],
                    ("message_ids", "expected_prompt_answer", "rationale"),
                    "context variant",
                )
                selected = variant["message_ids"]
                expected = _nonblank(variant["expected_prompt_answer"], "prompt answer")
                if (
                    type(selected) is not list
                    or any(type(value) is not str for value in selected)
                    or len(selected) != len(set(selected))
                    or not set(selected) <= group_message_ids
                ):
                    raise ValueError("fixture context-variant message references are invalid")
                if variant_id == "full_all_message_ids" and selected != ordered_message_ids:
                    raise ValueError(
                        "fixture full-context reference must preserve all source messages"
                    )
                if variant_id == "support_omitted" and set(selected) & set(supporting_ids):
                    raise ValueError("fixture support-omitted context contains supporting evidence")
                if variant_id == "empty" and (selected or expected != "UNKNOWN"):
                    raise ValueError(
                        "fixture empty context must have no messages and expect UNKNOWN"
                    )
                if variant_id == "full_all_message_ids" and expected != full_answer:
                    raise ValueError(
                        "fixture full-context answer differs from the full-source answer"
                    )
            counts["questions"] += 1
        counts["scenario_groups"] += 1

    if counts != _EXPECTED_COUNTS or (answerable, unknown) != (10, 2):
        raise ValueError("fixture counts differ from the reviewed 6/7/22/12 and 10/2 totals")
    return groups, {**counts, "full_source_answerable": answerable, "full_source_unknown": unknown}


def _context_text(selected: list[dict[str, str]]) -> str:
    return "\n\n".join(
        f"[date={message['date']} session_id={message['session_id']} "
        f"message_id={message['message_id']} role={message['role']}]\n{message['content']}"
        for message in selected
    )


def _normalize_answer(value: str) -> str:
    return " ".join(value.casefold().split())


def _baseline(cells: list[dict[str, Any]]) -> dict[str, Any]:
    axes = {"prompt": "expected_prompt_answer", "full_source": "full_source_expected_answer"}
    totals = {axis: {"matches": 0, "denominator": 0} for axis in axes}
    by_variant: dict[str, dict[str, dict[str, int]]] = {}
    for variant_id in _VARIANTS:
        rows = [cell for cell in cells if cell["variant_id"] == variant_id]
        by_variant[variant_id] = {}
        for axis, field in axes.items():
            count = sum(_normalize_answer(row[field]) == "unknown" for row in rows)
            by_variant[variant_id][axis] = {"matches": count, "denominator": len(rows)}
            totals[axis]["matches"] += count
            totals[axis]["denominator"] += len(rows)
    return {
        "kind": "deterministic-always-unknown-reference-baseline",
        "prediction": "UNKNOWN",
        "model_run": False,
        "by_variant": by_variant,
        "total": totals,
        "correlated_dev_only": True,
        "not_model_quality": True,
    }


def prepare_controls() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return the source-only payload, separate references, and preparation audit."""
    fixture_raw = _FIXTURE_PATH.read_bytes()
    fixture_sha256 = _sha256(fixture_raw)
    if fixture_sha256 != _FIXTURE_SHA256:
        raise ValueError("QA interpretation controls fixture differs from the pinned fixture")
    groups, counts = _parse_fixture(fixture_raw)
    items_by_prompt: dict[str, dict[str, Any]] = {}
    cells: list[dict[str, Any]] = []
    for group in groups:
        sessions = sorted(group["source_sessions"], key=lambda row: row["date"])
        source = [
            {
                "date": session["date"],
                "session_id": session["session_id"],
                "message_id": message["message_id"],
                "role": message["role"],
                "content": message["content"],
            }
            for session in sessions
            for message in session["messages"]
        ]
        question_date = sessions[-1]["date"]
        for question in group["questions"]:
            for variant_id in _VARIANTS:
                variant = question["context_variants"][variant_id]
                selected_ids = set(variant["message_ids"])
                selected = [message for message in source if message["message_id"] in selected_ids]
                messages = [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": _USER_TEMPLATE.format(
                            question_date=question_date,
                            question=question["question"],
                            context=_context_text(selected),
                        ),
                    },
                ]
                prompt_id = _sha256(_canonical_bytes(messages))
                item = {"prompt_id": prompt_id, "messages": messages}
                previous = items_by_prompt.setdefault(prompt_id, item)
                if previous != item:
                    raise ValueError("canonical prompt hash collision")
                cells.append(
                    {
                        "cell_id": (
                            f"{group['scenario_id']}::{question['question_id']}::{variant_id}"
                        ),
                        "scenario_id": group["scenario_id"],
                        "question_id": question["question_id"],
                        "variant_id": variant_id,
                        "prompt_id": prompt_id,
                        "selected_message_ids": [message["message_id"] for message in selected],
                        "full_source_expected_answer": question["full_source_expected_answer"],
                        "expected_prompt_answer": variant["expected_prompt_answer"],
                        "is_answerable": question["is_answerable"],
                    }
                )
    if len(cells) != 36 or len(items_by_prompt) != 36:
        raise ValueError("reviewed controls must generate 36 cells and 36 unique prompts")
    payload = {
        "schema_version": 1,
        "mode": "qa-interpretation-controls-dev-source-only-v1",
        "items": [items_by_prompt[key] for key in sorted(items_by_prompt)],
    }
    payload_sha256 = _sha256(_canonical_bytes(payload))
    references = {
        "schema_version": 1,
        "kind": "tracemem-qa-interpretation-controls-references",
        "provenance": "authored-fictional-agent-only",
        "fixture_sha256": fixture_sha256,
        "payload_sha256": payload_sha256,
        "split": "dev",
        "dev_only": True,
        "test": [],
        "held_out_test_cases": False,
        "provisional": True,
        "label_provenance": "agent-authored-provisional",
        "human_label_count": 0,
        "cells_are_independent_trials": False,
        "scenario_groups_are_derived_controls": True,
        "template_reuse": (
            "All cells use the shared QA prompt template; variants are matched within question."
        ),
        "cells": cells,
    }
    references_sha256 = _sha256(_canonical_bytes(references))
    preparer_source_sha256 = _sha256(Path(__file__).read_bytes())
    shared_prompt_source_sha256 = _sha256(Path(prepare_synthetic_qa.__file__).read_bytes())
    prompt_template_sha256 = _sha256(
        _canonical_bytes({"system": _SYSTEM_PROMPT, "user": _USER_TEMPLATE})
    )
    audit = {
        "schema_version": 1,
        "kind": "offline-qa-interpretation-controls-audit",
        "provenance": {
            "fixture_sha256": fixture_sha256,
            "preparer_source_sha256": preparer_source_sha256,
            "shared_prompt_source_sha256": shared_prompt_source_sha256,
            "prompt_template_sha256": prompt_template_sha256,
            "payload_sha256": payload_sha256,
            "references_sha256": references_sha256,
        },
        "dev_only": True,
        "provisional": True,
        "human_label_count": 0,
        "test": [],
        "held_out_test_cases": False,
        "counts": {**counts, "cells": len(cells), "unique_prompts": len(items_by_prompt)},
        "token_count": {
            "status": "UNMEASURED",
            "note": "No tokenizer was invoked; UTF-8 byte lengths are not token counts.",
        },
        "inference_performed": False,
        "model_outputs_included": False,
        "design_limits": {
            "scenario_groups_are_derived_controls": True,
            "cells_are_independent_trials": False,
            "template_reuse": (
                "One shared system/user template; three matched context variants per question."
            ),
        },
        "scoring_policy": {
            "literal_comparison": (
                "Casefold strings, split on whitespace, and rejoin with one ASCII space."
            ),
            "match": "Exact equality after the stated normalization.",
            "unknown": "UNKNOWN only when the normalized string equals exactly 'unknown'.",
            "axes": ["full_source_expected_answer", "expected_prompt_answer"],
            "axes_are_reported_separately": True,
            "future_semantic_reviews_are_distinct": True,
        },
        "always_unknown_reference_baseline": _baseline(cells),
    }
    return payload, references, audit


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _summary(
    payload: dict[str, Any], references: dict[str, Any], audit: dict[str, Any]
) -> dict[str, Any]:
    return {
        "payload_sha256": _sha256(_canonical_bytes(payload)),
        "references_sha256": _sha256(_canonical_bytes(references)),
        "audit_sha256": _sha256(_canonical_bytes(audit)),
        **{
            key: audit["counts"][key]
            for key in (
                "scenario_groups",
                "sessions",
                "messages",
                "questions",
                "cells",
                "unique_prompts",
            )
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prepare-qa-interpretation-controls")
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload, references, audit = prepare_controls()
        if args.output_dir is not None:
            from experiments.prepare_qa_contrast_requests import _write_new_output

            _write_new_output(
                args.output_dir,
                {
                    "payload.json": _json_bytes(payload),
                    "references.json": _json_bytes(references),
                    "audit.json": _json_bytes(audit),
                },
            )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(_summary(payload, references, audit), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
