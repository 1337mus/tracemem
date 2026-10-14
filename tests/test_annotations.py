"""Blind candidate-review export and response tests."""

from __future__ import annotations

import copy
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock


def _sample_claim(object_text: str = "quiet rooms") -> dict[str, Any]:
    quote = "I like quiet rooms."
    return {
        "subject": "the user",
        "predicate": "PREFERS",
        "object": object_text,
        "speaker_role": "user",
        "attribution": "direct_user_statement",
        "modality": "preference",
        "polarity": "positive",
        "valid_from": None,
        "valid_to": None,
        "evidence": [
            {
                "session_id": "private-session",
                "message_id": "private-message",
                "char_start": 0,
                "char_end": len(quote),
                "quote": quote,
            }
        ],
    }


def _sample_case() -> dict[str, Any]:
    return {
        "case_id": "private-case",
        "scenario_id": "private-group",
        "split": "test",
        "tags": ["preference"],
        "messages": [
            {
                "session_id": "private-session",
                "message_id": "private-message",
                "role": "user",
                "text": "I like quiet rooms.",
                "recorded_at": "2026-10-01T12:00:00Z",
                "content_kind": "conversation",
            }
        ],
        "expected": [_sample_claim()],
    }


class AnnotationExportTests(unittest.TestCase):
    def test_export_hides_identity_and_gold_membership_while_remapping_evidence_ids(self) -> None:
        try:
            module = importlib.import_module("tracemem.annotations")
        except ModuleNotFoundError:
            self.fail("annotation module must provide blind export")
        exporter = getattr(module, "export_review_bundle", None)
        self.assertTrue(callable(exporter), "export_review_bundle must be implemented")

        case = _sample_case()
        dataset_sha = "a" * 64
        blind, answer_key = exporter(
            [case],
            dataset_sha,
            {case["case_id"]: [_sample_claim("noisy rooms")]},
            seed=17,
            case_metadata={
                case["case_id"]: {
                    "scenario_group": "private-scenario-group",
                    "template_group": "private-template-group",
                    "entity_ids": ["private-entity"],
                }
            },
        )

        self.assertEqual(blind["dataset_sha256"], dataset_sha)
        self.assertEqual(len(blind["bundle_sha256"]), 64)
        self.assertEqual(len(blind["tasks"]), 2)
        public_json = json.dumps(blind, ensure_ascii=False)
        for hidden in (
            "private-case",
            "private-group",
            "private-session",
            "private-message",
            "private-scenario-group",
            "private-template-group",
            "private-entity",
            "expected",
            "gold",
            "foil",
            '"split"',
        ):
            self.assertNotIn(hidden, public_json)

        task = blind["tasks"][0]
        self.assertEqual(set(task), {"task_id", "messages", "candidate"})
        public_message = task["messages"][0]
        span = task["candidate"]["evidence"][0]
        self.assertNotEqual(public_message["session_id"], "private-session")
        self.assertNotEqual(public_message["message_id"], "private-message")
        self.assertEqual(span["session_id"], public_message["session_id"])
        self.assertEqual(span["message_id"], public_message["message_id"])
        self.assertEqual(span["quote"], "I like quiet rooms.")
        self.assertEqual(span["char_start"], 0)
        self.assertEqual(span["char_end"], len(span["quote"]))

        memberships = {row["membership"] for row in answer_key["task_keys"]}
        self.assertEqual(memberships, {"gold", "foil"})
        self.assertEqual(
            {row["reference_label"] for row in answer_key["task_keys"]},
            {"supported", "unsupported"},
        )
        self.assertEqual(answer_key["bundle_sha256"], blind["bundle_sha256"])
        self.assertNotIn("private-case", public_json)

    def test_export_is_repeatable_and_dataset_hash_changes_task_identity(self) -> None:
        exporter = importlib.import_module("tracemem.annotations").export_review_bundle
        case = _sample_case()
        first = exporter([case], "d" * 64, seed=23)
        repeated = exporter([case], "d" * 64, seed=23)
        changed_dataset = exporter([case], "e" * 64, seed=23)

        self.assertEqual(first, repeated)
        first_ids = {task["task_id"] for task in first[0]["tasks"]}
        changed_ids = {task["task_id"] for task in changed_dataset[0]["tasks"]}
        self.assertTrue(first_ids.isdisjoint(changed_ids))
        self.assertNotEqual(first[0]["bundle_sha256"], changed_dataset[0]["bundle_sha256"])

    def test_export_rejects_gold_foil_membership_conflicts(self) -> None:
        exporter = importlib.import_module("tracemem.annotations").export_review_bundle
        case = _sample_case()
        identical_foil = copy.deepcopy(case["expected"][0])

        with self.assertRaisesRegex(ValueError, "duplicate candidate task"):
            exporter(
                [case],
                "f" * 64,
                {case["case_id"]: [identical_foil]},
                seed=2,
            )

    def test_generated_candidates_remain_unlabeled_in_the_private_answer_key(self) -> None:
        exporter = importlib.import_module("tracemem.annotations").export_review_bundle
        case = _sample_case()
        generated = _sample_claim("generated candidate")

        blind, answer_key = exporter(
            [case],
            "6" * 64,
            seed=9,
            unlabeled_by_case={case["case_id"]: [generated]},
        )

        self.assertEqual(len(blind["tasks"]), 2)
        self.assertNotIn("unlabeled", json.dumps(blind))
        generated_task = next(
            task for task in blind["tasks"] if task["candidate"]["object"] == "generated candidate"
        )
        generated_key = next(
            row for row in answer_key["task_keys"] if row["task_id"] == generated_task["task_id"]
        )
        self.assertEqual(generated_key["membership"], "unlabeled")
        self.assertIsNone(generated_key["reference_label"])

    def test_empty_candidate_cases_are_keyed_but_do_not_create_tasks(self) -> None:
        exporter = importlib.import_module("tracemem.annotations").export_review_bundle
        positive = _sample_case()
        empty = _sample_case()
        empty["case_id"] = "private-empty-case"
        empty["scenario_id"] = "private-empty-group"
        empty["expected"] = []
        blind, answer_key = exporter([positive, empty], "8" * 64, seed=5)

        self.assertEqual(len(blind["tasks"]), 1)
        self.assertEqual(set(answer_key["case_ids"]), {positive["case_id"], empty["case_id"]})
        self.assertNotIn(empty["case_id"], {row["case_id"] for row in answer_key["task_keys"]})

    def test_export_remaps_even_unresolved_foil_evidence_for_human_review(self) -> None:
        exporter = importlib.import_module("tracemem.annotations").export_review_bundle
        case = _sample_case()
        foil = _sample_claim("unknown preference")
        foil["evidence"][0].update(
            {
                "session_id": "unlisted-session",
                "message_id": "unlisted-message",
                "char_start": 500,
                "char_end": 509,
                "quote": "fabricated",
            }
        )
        blind, _ = exporter([case], "9" * 64, {case["case_id"]: [foil]}, seed=4)
        task = next(
            task for task in blind["tasks"] if task["candidate"]["object"] == "unknown preference"
        )
        span = task["candidate"]["evidence"][0]

        self.assertNotIn(
            (span["session_id"], span["message_id"]),
            {(message["session_id"], message["message_id"]) for message in task["messages"]},
        )
        self.assertEqual(span["char_start"], 500)
        self.assertEqual(span["char_end"], 509)
        self.assertEqual(span["quote"], "fabricated")

    def test_export_does_not_prejudge_gold_evidence_span_integrity(self) -> None:
        exporter = importlib.import_module("tracemem.annotations").export_review_bundle
        case = _sample_case()
        case["expected"][0]["evidence"][0].update(
            {
                "session_id": "unlisted-session",
                "message_id": "unlisted-message",
                "char_start": 500,
                "char_end": 509,
                "quote": "fabricated",
            }
        )

        blind, answer_key = exporter([case], "7" * 64, seed=7)
        self.assertEqual(len(blind["tasks"]), 1)
        self.assertEqual(answer_key["task_keys"][0]["reference_label"], "supported")

    def test_response_import_requires_exact_one_to_one_task_coverage(self) -> None:
        module = importlib.import_module("tracemem.annotations")
        exporter = module.export_review_bundle
        importer = getattr(module, "validate_responses", None)
        self.assertTrue(callable(importer), "validate_responses must be implemented")
        case = _sample_case()
        blind, _ = exporter(
            [case], "b" * 64, {case["case_id"]: [_sample_claim("alternate preference")]}, seed=3
        )
        responses = {
            "version": 1,
            "bundle_sha256": blind["bundle_sha256"],
            "responses": [
                {
                    "task_id": task["task_id"],
                    "label": "supported",
                    "rationale": "The transcript directly supports this candidate.",
                    "reviewer_id": "reviewer-human-1",
                    "reviewer_kind": "human",
                    "timestamp": "2026-10-14T19:00:00Z",
                }
                for task in blind["tasks"]
            ],
        }

        imported = importer(blind, responses)
        self.assertEqual(len(imported), len(blind["tasks"]))
        self.assertEqual({row["reviewer_id"] for row in imported}, {"reviewer-human-1"})

        with self.assertRaisesRegex(ValueError, "missing task"):
            importer(blind, {**responses, "responses": responses["responses"][:-1]})
        with self.assertRaisesRegex(ValueError, "duplicate task"):
            importer(
                blind,
                {**responses, "responses": responses["responses"] + responses["responses"][:1]},
            )
        with self.assertRaisesRegex(ValueError, "unknown task"):
            importer(
                blind,
                {
                    **responses,
                    "responses": [
                        {**responses["responses"][0], "task_id": "task_unknown"},
                        *responses["responses"][1:],
                    ],
                },
            )
        invalid_kind = copy.deepcopy(responses)
        invalid_kind["responses"][0]["reviewer_kind"] = "unverified-human"
        with self.assertRaisesRegex(ValueError, "reviewer_kind"):
            importer(blind, invalid_kind)
        invalid_timestamp = copy.deepcopy(responses)
        invalid_timestamp["responses"][0]["timestamp"] = "2026-10-14T19:00:00+00:00"
        with self.assertRaisesRegex(ValueError, "canonical UTC"):
            importer(blind, invalid_timestamp)
        mixed_identity = copy.deepcopy(responses)
        mixed_identity["responses"][1]["reviewer_id"] = "reviewer-human-2"
        with self.assertRaisesRegex(ValueError, "single reviewer ID"):
            importer(blind, mixed_identity)
        mixed_kind = copy.deepcopy(responses)
        mixed_kind["responses"][1]["reviewer_kind"] = "agent"
        with self.assertRaisesRegex(ValueError, "single reviewer ID and kind"):
            importer(blind, mixed_kind)
        wrong_hash = {**responses, "bundle_sha256": "0" * 64}
        with self.assertRaisesRegex(ValueError, "bundle hash does not match"):
            importer(blind, wrong_hash)
        tampered_bundle = copy.deepcopy(blind)
        tampered_bundle["tasks"][0]["candidate"]["subject"] = "changed after export"
        with self.assertRaisesRegex(ValueError, "bundle hash mismatch"):
            importer(tampered_bundle, responses)

    def test_report_separates_human_and_agent_descriptive_agreement(self) -> None:
        module = importlib.import_module("tracemem.annotations")
        try:
            report_module = importlib.import_module("tracemem.annotation_report")
        except ModuleNotFoundError:
            self.fail("annotation report module must provide descriptive comparisons")
        reporter = getattr(report_module, "build_review_report", None)
        self.assertTrue(callable(reporter), "build_review_report must be implemented")
        case = _sample_case()
        blind, answer_key = module.export_review_bundle(
            [case],
            "c" * 64,
            {case["case_id"]: [_sample_claim("noisy rooms")]},
            seed=11,
            case_metadata={
                case["case_id"]: {
                    "scenario_group": "scenario-a",
                    "template_group": "template-a",
                    "entity_ids": ["entity-a"],
                }
            },
        )
        answer_key_before = json.dumps(answer_key, sort_keys=True)
        labels_by_task = {row["task_id"]: row["reference_label"] for row in answer_key["task_keys"]}

        def document(
            reviewer_id: str, reviewer_kind: str, labels: dict[str, str]
        ) -> dict[str, Any]:
            return {
                "version": 1,
                "bundle_sha256": blind["bundle_sha256"],
                "responses": [
                    {
                        "task_id": task["task_id"],
                        "label": labels[task["task_id"]],
                        "rationale": "A concise evidence-based explanation.",
                        "reviewer_id": reviewer_id,
                        "reviewer_kind": reviewer_kind,
                        "timestamp": "2026-10-14T19:00:00Z",
                    }
                    for task in blind["tasks"]
                ],
            }

        human = document("human-1", "human", labels_by_task)
        agent = document("agent-1", "agent", {task_id: "uncertain" for task_id in labels_by_task})
        report = reporter(blind, answer_key, [human, agent])

        self.assertEqual(report["by_reviewer_kind"]["human"]["agreement"], 2)
        self.assertEqual(report["by_reviewer_kind"]["human"]["disagreement"], 0)
        self.assertEqual(report["by_reviewer_kind"]["agent"]["uncertain"], 2)
        self.assertEqual(report["by_reviewer_kind"]["agent"]["agreement"], 0)
        self.assertIn("self-attested", report["human_identity_policy"])
        self.assertIn("never modifies or promotes", report["reference_label_policy"])
        self.assertEqual(json.dumps(answer_key, sort_keys=True), answer_key_before)
        with self.assertRaisesRegex(ValueError, "duplicate reviewer"):
            reporter(blind, answer_key, [human, human])
        unknown_group = copy.deepcopy(answer_key)
        unknown_group["case_metadata"]["unlisted-case"] = {
            "scenario_group": "scenario-b",
            "template_group": "template-b",
            "entity_ids": ["entity-b"],
        }
        with self.assertRaisesRegex(ValueError, "unknown case_id"):
            reporter(blind, unknown_group, [human])

    def test_report_counts_generated_labels_without_including_them_in_agreement(self) -> None:
        annotations = importlib.import_module("tracemem.annotations")
        reporter = importlib.import_module("tracemem.annotation_report").build_review_report
        case = _sample_case()
        blind, answer_key = annotations.export_review_bundle(
            [case],
            "5" * 64,
            {case["case_id"]: [_sample_claim("foil candidate")]},
            seed=13,
            unlabeled_by_case={case["case_id"]: [_sample_claim("generated candidate")]},
        )
        task_by_membership = {
            row["membership"]: next(
                task for task in blind["tasks"] if task["task_id"] == row["task_id"]
            )
            for row in answer_key["task_keys"]
        }

        def document(
            reviewer_id: str, reviewer_kind: str, labels: dict[str, str]
        ) -> dict[str, Any]:
            return {
                "version": 1,
                "bundle_sha256": blind["bundle_sha256"],
                "responses": [
                    {
                        "task_id": task["task_id"],
                        "label": labels[membership],
                        "rationale": "A concise evidence-based explanation.",
                        "reviewer_id": reviewer_id,
                        "reviewer_kind": reviewer_kind,
                        "timestamp": "2026-10-14T19:00:00Z",
                    }
                    for membership, task in task_by_membership.items()
                ],
            }

        human = document(
            "human-generated-1",
            "human",
            {"gold": "supported", "foil": "unsupported", "unlabeled": "supported"},
        )
        agent = document(
            "agent-generated-1",
            "agent",
            {"gold": "uncertain", "foil": "supported", "unlabeled": "unsupported"},
        )

        report = reporter(blind, answer_key, [human, agent])
        human_stats = report["by_reviewer_kind"]["human"]
        agent_stats = report["by_reviewer_kind"]["agent"]
        self.assertEqual(human_stats["response_supported"], 2)
        self.assertEqual(human_stats["response_unsupported"], 1)
        self.assertEqual(human_stats["reference_labeled_responses"], 2)
        self.assertEqual(human_stats["unlabeled_responses"], 1)
        self.assertEqual(human_stats["agreement"], 2)
        self.assertEqual(human_stats["decisive_responses"], 2)
        self.assertEqual(agent_stats["response_supported"], 1)
        self.assertEqual(agent_stats["response_unsupported"], 1)
        self.assertEqual(agent_stats["uncertain"], 1)
        self.assertEqual(agent_stats["response_uncertain"], 1)
        self.assertEqual(agent_stats["reference_labeled_responses"], 2)
        self.assertEqual(agent_stats["unlabeled_responses"], 1)
        self.assertEqual(agent_stats["agreement"], 0)
        self.assertEqual(agent_stats["disagreement"], 1)
        self.assertEqual(agent_stats["decisive_responses"], 1)
        self.assertEqual(agent_stats["agreement_denominator"], 1)
        self.assertIn("unlabeled candidates are excluded", report["agreement_scope"])

    def test_report_rejects_authored_labels_for_unlabeled_candidates(self) -> None:
        exporter = importlib.import_module("tracemem.annotations").export_review_bundle
        reporter = importlib.import_module("tracemem.annotation_report").build_review_report
        case = _sample_case()
        blind, answer_key = exporter(
            [case],
            "3" * 64,
            seed=19,
            unlabeled_by_case={case["case_id"]: [_sample_claim("generated candidate")]},
        )
        invalid = copy.deepcopy(answer_key)
        unlabeled_key = next(
            row for row in invalid["task_keys"] if row["membership"] == "unlabeled"
        )
        unlabeled_key["reference_label"] = "unsupported"

        with self.assertRaisesRegex(ValueError, "invalid membership or authored label"):
            reporter(blind, invalid, [])

    def test_cli_rejects_aliases_and_existing_outputs_without_overwriting(self) -> None:
        try:
            cli = importlib.import_module("experiments.review_annotations")
        except ModuleNotFoundError:
            self.fail("review_annotations CLI must be implemented")
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            aliased_input = root / "same.json"
            aliased_input.write_text("preserve me", encoding="utf-8")
            result = cli.main(
                [
                    "report",
                    "--bundle",
                    str(aliased_input),
                    "--answer-key",
                    str(aliased_input),
                    "--responses",
                    str(aliased_input),
                    "--output",
                    str(aliased_input),
                ]
            )
            self.assertEqual(result, 2)
            self.assertEqual(aliased_input.read_text(encoding="utf-8"), "preserve me")

            existing_output = root / "existing-report.json"
            existing_output.write_text("keep unchanged", encoding="utf-8")
            result = cli.main(
                [
                    "report",
                    "--bundle",
                    str(root / "missing-bundle.json"),
                    "--answer-key",
                    str(root / "missing-key.json"),
                    "--responses",
                    str(root / "missing-responses.json"),
                    "--output",
                    str(existing_output),
                ]
            )
            self.assertEqual(result, 2)
            self.assertEqual(existing_output.read_text(encoding="utf-8"), "keep unchanged")

    def test_cli_rolls_back_its_first_output_when_second_creation_fails(self) -> None:
        cli = importlib.import_module("experiments.review_annotations")
        with tempfile.TemporaryDirectory() as temporary_directory:
            first = Path(temporary_directory) / "blind.json"
            second = Path(temporary_directory) / "answer-key.json"
            path_open = Path.open

            def fail_second(path: Path, *args: Any, **kwargs: Any) -> Any:
                if path == second:
                    raise FileExistsError("simulated concurrent output creation")
                return path_open(path, *args, **kwargs)

            with mock.patch.object(Path, "open", fail_second):
                with self.assertRaisesRegex(FileExistsError, "simulated"):
                    cli._write_many((first, second), ({"tasks": []}, {"task_keys": []}))

            self.assertFalse(first.exists())
            self.assertFalse(second.exists())

    def test_cli_exports_model_candidates_as_unlabeled(self) -> None:
        cli = importlib.import_module("experiments.review_annotations")
        case = _sample_case()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            generated_path = root / "generated.json"
            generated_path.write_text(
                json.dumps({case["case_id"]: [_sample_claim("generated by model")]}),
                encoding="utf-8",
            )
            bundle_path = root / "blind.json"
            key_path = root / "answer-key.json"
            with mock.patch.object(
                cli,
                "load_scenarios",
                return_value=([case], {"case_metadata": {}}, "4" * 64),
            ):
                result = cli.main(
                    [
                        "export",
                        "--bundle",
                        str(bundle_path),
                        "--answer-key",
                        str(key_path),
                        "--seed",
                        "15",
                        "--unlabeled",
                        str(generated_path),
                    ]
                )

            self.assertEqual(result, 0)
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            answer_key = json.loads(key_path.read_text(encoding="utf-8"))
            self.assertEqual(len(bundle["tasks"]), 2)
            generated_task = next(
                task
                for task in bundle["tasks"]
                if task["candidate"]["object"] == "generated by model"
            )
            generated_key = next(
                row
                for row in answer_key["task_keys"]
                if row["task_id"] == generated_task["task_id"]
            )
            self.assertEqual(generated_key["membership"], "unlabeled")
            self.assertIsNone(generated_key["reference_label"])


if __name__ == "__main__":
    unittest.main()
