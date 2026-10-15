from __future__ import annotations

import builtins
import contextlib
import hashlib
import importlib
import io
import json
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments import prepare_qa_interpretation_controls as controls
from experiments import prepare_synthetic_qa
from experiments.prepare_synthetic_qa import _SYSTEM_PROMPT, _USER_TEMPLATE

_FIXTURE_PATH = Path(__file__).resolve().parents[1] / (
    "examples/synthetic/qa-interpretation-controls-dev-v1.json"
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _independent_reconstruction() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    fixture = json.loads(_FIXTURE_PATH.read_bytes())
    expected_items: dict[str, dict[str, object]] = {}
    expected_cells: list[dict[str, object]] = []
    variants = ("full_all_message_ids", "support_omitted", "empty")
    for group in fixture["scenario_groups"]:
        sessions = sorted(group["source_sessions"], key=lambda session: session["date"])
        source: list[dict[str, str]] = []
        for session in sessions:
            for message in session["messages"]:
                source.append(
                    {
                        "date": session["date"],
                        "session_id": session["session_id"],
                        "message_id": message["message_id"],
                        "role": message["role"],
                        "content": message["content"],
                    }
                )
        source_by_id = {message["message_id"]: message for message in source}
        question_date = sessions[-1]["date"]
        for question in group["questions"]:
            for variant_id in variants:
                variant = question["context_variants"][variant_id]
                selected = set(variant["message_ids"])
                selected_messages = [
                    message for message in source if message["message_id"] in selected
                ]
                context = "\n\n".join(
                    "[date={date} session_id={session_id} message_id={message_id} role={role}]\n"
                    "{content}".format(**message)
                    for message in selected_messages
                )
                messages = [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": _USER_TEMPLATE.format(
                            question_date=question_date,
                            question=question["question"],
                            context=context,
                        ),
                    },
                ]
                prompt_id = hashlib.sha256(_canonical_bytes(messages)).hexdigest()
                expected_items[prompt_id] = {"prompt_id": prompt_id, "messages": messages}
                selected_ids = [message["message_id"] for message in selected_messages]
                if variant_id == "empty":
                    assert selected_ids == []
                assert set(selected_ids) <= set(source_by_id)
                expected_cells.append(
                    {
                        "cell_id": (
                            f"{group['scenario_id']}::{question['question_id']}::{variant_id}"
                        ),
                        "scenario_id": group["scenario_id"],
                        "question_id": question["question_id"],
                        "variant_id": variant_id,
                        "prompt_id": prompt_id,
                        "selected_message_ids": selected_ids,
                        "full_source_expected_answer": question["full_source_expected_answer"],
                        "expected_prompt_answer": variant["expected_prompt_answer"],
                        "is_answerable": question["is_answerable"],
                    }
                )
    return [expected_items[key] for key in sorted(expected_items)], expected_cells


class QAInterpretationControlsTests(unittest.TestCase):
    def test_fixture_rejects_duplicate_json_keys(self) -> None:
        raw = b'{"kind":"candidate","kind":"duplicate"}'
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "duplicate.json"
            changed.write_bytes(raw)
            with (
                patch.object(controls, "_FIXTURE_PATH", changed),
                patch.object(controls, "_FIXTURE_SHA256", hashlib.sha256(raw).hexdigest()),
            ):
                with self.assertRaisesRegex(ValueError, "duplicate object keys"):
                    controls.prepare_controls()

    def test_fixture_rejects_unknown_context_message_reference(self) -> None:
        fixture = json.loads(_FIXTURE_PATH.read_bytes())
        fixture["scenario_groups"][0]["questions"][0]["context_variants"]["full_all_message_ids"][
            "message_ids"
        ][0] = "missing_message"
        raw = _canonical_bytes(fixture)
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "unknown-reference.json"
            changed.write_bytes(raw)
            with (
                patch.object(controls, "_FIXTURE_PATH", changed),
                patch.object(controls, "_FIXTURE_SHA256", hashlib.sha256(raw).hexdigest()),
            ):
                with self.assertRaisesRegex(ValueError, "context-variant message references"):
                    controls.prepare_controls()

    def test_prepare_controls_imports_no_model_or_tokenizer_runtime(self) -> None:
        original_import = builtins.__import__
        attempted: list[str] = []

        def reject_runtime_import(name: str, *args: object, **kwargs: object) -> object:
            if name.split(".", 1)[0] in {"modal", "torch", "transformers", "tokenizers"}:
                attempted.append(name)
                raise AssertionError(f"unexpected model runtime import: {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=reject_runtime_import):
            importlib.reload(prepare_synthetic_qa)
            importlib.reload(controls)
            controls.prepare_controls()
            with tempfile.TemporaryDirectory() as directory:
                controls.main(["--output-dir", str(Path(directory) / "private-output")])
        self.assertEqual(attempted, [])

    def test_cli_writes_new_private_outputs_and_refuses_existing_output_directory(self) -> None:
        payload, references, audit = controls.prepare_controls()
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "new"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(controls.main(["--output-dir", str(output_dir)]), 0)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(
                set(summary),
                {
                    "payload_sha256",
                    "references_sha256",
                    "audit_sha256",
                    "scenario_groups",
                    "sessions",
                    "messages",
                    "questions",
                    "cells",
                    "unique_prompts",
                },
            )
            self.assertEqual(stat.S_IMODE(output_dir.stat().st_mode), 0o700)
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {"payload.json", "references.json", "audit.json"},
            )
            outputs = {path.name: json.loads(path.read_bytes()) for path in output_dir.iterdir()}
            self.assertEqual(
                outputs,
                {
                    "payload.json": payload,
                    "references.json": references,
                    "audit.json": audit,
                },
            )
            self.assertEqual(
                {path.name: stat.S_IMODE(path.stat().st_mode) for path in output_dir.iterdir()},
                {"payload.json": 0o600, "references.json": 0o600, "audit.json": 0o600},
            )

            existing = Path(directory) / "existing"
            existing.mkdir()
            sentinel = existing / "keep.txt"
            sentinel.write_bytes(b"preserve")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
                controls.main(["--output-dir", str(existing)])
            self.assertEqual(error.exception.code, 2)
            self.assertIn("File exists", stderr.getvalue())
            self.assertEqual(sentinel.read_bytes(), b"preserve")

    def test_outputs_are_deterministic_and_audit_binds_sources_hashes_and_limits(self) -> None:
        payload, references, audit = controls.prepare_controls()
        again = controls.prepare_controls()
        self.assertEqual((payload, references, audit), again)
        payload_hash = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
        references_hash = hashlib.sha256(_canonical_bytes(references)).hexdigest()
        provenance = audit["provenance"]
        self.assertEqual(references["payload_sha256"], payload_hash)
        self.assertEqual(provenance["payload_sha256"], payload_hash)
        self.assertEqual(provenance["references_sha256"], references_hash)
        self.assertEqual(
            provenance["fixture_sha256"], hashlib.sha256(_FIXTURE_PATH.read_bytes()).hexdigest()
        )
        self.assertEqual(
            provenance["preparer_source_sha256"],
            hashlib.sha256(Path(controls.__file__).read_bytes()).hexdigest(),
        )
        self.assertEqual(
            provenance["shared_prompt_source_sha256"],
            hashlib.sha256(Path(prepare_synthetic_qa.__file__).read_bytes()).hexdigest(),
        )
        self.assertEqual(
            provenance["prompt_template_sha256"],
            hashlib.sha256(
                _canonical_bytes({"system": _SYSTEM_PROMPT, "user": _USER_TEMPLATE})
            ).hexdigest(),
        )
        self.assertEqual(
            audit["counts"],
            {
                "scenario_groups": 6,
                "sessions": 7,
                "messages": 22,
                "questions": 12,
                "full_source_answerable": 10,
                "full_source_unknown": 2,
                "cells": 36,
                "unique_prompts": 36,
            },
        )
        self.assertEqual(references["test"], [])
        self.assertTrue(references["dev_only"] and references["provisional"])
        self.assertEqual(references["human_label_count"], 0)
        self.assertFalse(references["cells_are_independent_trials"])
        self.assertTrue(references["scenario_groups_are_derived_controls"])
        self.assertEqual(audit["token_count"]["status"], "UNMEASURED")
        self.assertIn("not token counts", audit["token_count"]["note"])
        self.assertFalse(audit["inference_performed"])
        self.assertFalse(audit["model_outputs_included"])
        self.assertEqual(
            set(audit["scoring_policy"]["axes"]),
            {"full_source_expected_answer", "expected_prompt_answer"},
        )
        self.assertTrue(audit["scoring_policy"]["axes_are_reported_separately"])
        self.assertTrue(audit["scoring_policy"]["future_semantic_reviews_are_distinct"])
        self.assertNotIn("expected_prompt_answer", _canonical_bytes(payload).decode("utf-8"))
        self.assertNotIn("full_source_expected_answer", _canonical_bytes(payload).decode("utf-8"))
        self.assertNotIn("rationale", _canonical_bytes(payload).decode("utf-8"))

    def test_fixture_rejects_non_string_message_roles_as_invalid_fixture(self) -> None:
        fixture = json.loads(_FIXTURE_PATH.read_bytes())
        fixture["scenario_groups"][0]["source_sessions"][0]["messages"][0]["role"] = []
        raw = _canonical_bytes(fixture)
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "changed.json"
            changed.write_bytes(raw)
            with (
                patch.object(controls, "_FIXTURE_PATH", changed),
                patch.object(controls, "_FIXTURE_SHA256", hashlib.sha256(raw).hexdigest()),
            ):
                with self.assertRaisesRegex(ValueError, "role"):
                    controls.prepare_controls()

    def test_always_unknown_reference_baseline_reports_both_axes_and_denominators(self) -> None:
        _, _, audit = controls.prepare_controls()
        baseline = audit["always_unknown_reference_baseline"]

        self.assertEqual(baseline["kind"], "deterministic-always-unknown-reference-baseline")
        self.assertEqual(baseline["prediction"], "UNKNOWN")
        self.assertIs(baseline["model_run"], False)
        self.assertIs(baseline["not_model_quality"], True)
        self.assertEqual(
            baseline["by_variant"],
            {
                "full_all_message_ids": {
                    "prompt": {"matches": 2, "denominator": 12},
                    "full_source": {"matches": 2, "denominator": 12},
                },
                "support_omitted": {
                    "prompt": {"matches": 12, "denominator": 12},
                    "full_source": {"matches": 2, "denominator": 12},
                },
                "empty": {
                    "prompt": {"matches": 12, "denominator": 12},
                    "full_source": {"matches": 2, "denominator": 12},
                },
            },
        )
        self.assertEqual(
            baseline["total"],
            {
                "prompt": {"matches": 26, "denominator": 36},
                "full_source": {"matches": 6, "denominator": 36},
            },
        )
        self.assertIs(baseline["correlated_dev_only"], True)

    def test_prepare_controls_reconstructs_all_source_only_cells_and_mappings(self) -> None:
        payload, references, _ = controls.prepare_controls()
        expected_items, expected_cells = _independent_reconstruction()

        self.assertEqual(set(payload), {"schema_version", "mode", "items"})
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["mode"], "qa-interpretation-controls-dev-source-only-v1")
        self.assertEqual(payload["items"], expected_items)
        self.assertEqual(references["cells"], expected_cells)
        self.assertEqual(len(payload["items"]), 36)
        self.assertEqual(len(references["cells"]), 36)
        self.assertTrue(all(set(item) == {"prompt_id", "messages"} for item in payload["items"]))
        self.assertTrue(
            all(
                set(cell)
                == {
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
                for cell in references["cells"]
            )
        )

    def test_prepare_controls_rejects_changed_pinned_fixture_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "changed.json"
            changed.write_bytes(controls._FIXTURE_PATH.read_bytes() + b" ")
            with patch.object(controls, "_FIXTURE_PATH", changed):
                with self.assertRaisesRegex(ValueError, "pinned fixture"):
                    controls.prepare_controls()
