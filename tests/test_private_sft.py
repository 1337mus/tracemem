"""Offline private SFT payload contract tests using fictional inputs only."""

from __future__ import annotations

import hashlib
import json
import unittest
from copy import deepcopy
from typing import Any

from experiments import private_sft
from training import private_verifier_data, verifier_data


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_record(record_id: str, session_id: str, text: str) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "revision": f"revision-{record_id}",
        "session_id": session_id,
        "message_id": f"message-{record_id}",
        "text": text,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "source_locator": f"fiction/{record_id}.jsonl",
        "line_number": 1,
        "role": "user",
        "recorded_at": "2026-10-01T12:00:00Z",
        "content_kind": "conversation",
    }


def _case(record: dict[str, Any], case_id: str, split: str) -> dict[str, Any]:
    message = {
        key: record[key]
        for key in (
            "message_id",
            "record_id",
            "revision",
            "session_id",
            "text",
            "text_sha256",
            "source_locator",
            "line_number",
            "role",
            "recorded_at",
            "content_kind",
        )
    }
    message["source_record_id"] = message.pop("record_id")
    quote = "the blue glass buttons"
    start = record["text"].index(quote)
    candidates = []
    for suffix, support, eligibility in (
        ("one", "uncertain", "uncertain"),
        ("two", "supported", "ineligible"),
    ):
        candidates.append(
            {
                "candidate_id": f"candidate-{case_id}-{suffix}",
                "claim": f"A fictional claim about {quote} ({suffix}).",
                "message_id": message["message_id"],
                "source_record_id": record["record_id"],
                "quote": quote,
                "char_start": start,
                "char_end": start + len(quote),
                "valid_from": None,
                "valid_to": None,
                "support": support,
                "eligibility": eligibility,
                "support_reason": "fictional reviewer rationale",
                "eligibility_reason": "fictional scope rationale",
                "label_status": "agent_provisional",
                "annotator": {"reviewer_id": "fictional-agent", "reviewer_type": "agent"},
                "reviews": [],
            }
        )
    return {
        "case_id": case_id,
        "session_id": record["session_id"],
        "split": split,
        "window_id": f"window-{case_id}",
        "privacy_clearance": {
            "status": "cleared",
            "reviewer_id": "fictional-owner",
            "reviewer_type": "human",
            "reviewed_at": "2026-10-01T12:00:00Z",
        },
        "messages": [message],
        "candidates": candidates,
    }


def _private_fixture(
    *, include_test: bool = False
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    rows = [
        (
            "train-a",
            "session-train-a",
            "train",
            "I might catalog the blue glass buttons after a fictional festival.",
        ),
        (
            "train-b",
            "session-train-b",
            "train",
            "I might catalog the blue glass buttons after a paper moon parade.",
        ),
        (
            "dev-a",
            "session-dev-a",
            "dev",
            "I might catalog the blue glass buttons after an imaginary observatory tour.",
        ),
    ]
    if include_test:
        rows.append(
            (
                "test-a",
                "session-test-a",
                "test",
                "TEST_TEXT_SENTINEL mentions the blue glass buttons.",
            )
        )
    sources = {
        record_id: _source_record(record_id, session, text) for record_id, session, _, text in rows
    }
    cases = [
        _case(sources[record_id], f"case-{record_id}", split) for record_id, _, split, _ in rows
    ]
    if include_test:
        test_candidate = cases[-1]["candidates"][0]
        test_candidate["claim"] = "TEST_LABEL_SENTINEL"
        test_candidate["support"] = "unsupported"
        test_candidate["eligibility"] = "eligible"
        test_candidate["support_reason"] = "RATIONALE_SENTINEL"
        test_candidate["annotator"]["reviewer_id"] = "REVIEWER_SENTINEL"
    dataset = {
        "version": 1,
        "license": "private-not-for-redistribution",
        "annotation_status": "agent_provisional",
        "provenance": "private_source",
        "scope_policy": "personal-memory-research-v1",
        "cases": cases,
    }
    return dataset, sources


def _synthetic_replay_fixture() -> dict[str, Any]:
    cases = []
    for split in ("train", "dev", "test"):
        marker = f"SYNTH_{split.upper()}_SENTINEL"
        message_id = f"synthetic-message-{split}"
        text = f"{marker}: I plan to arrange paper moons before the gallery opens."
        cases.append(
            {
                "case_id": f"synthetic-case-{split}",
                "split": split,
                "scenario_group": f"synthetic-scenario-{split}",
                "template_group": f"synthetic-template-{split}",
                "entity_ids": [f"synthetic-entity-{split}"],
                "phenomenon": "time",
                "messages": [
                    {
                        "session_id": f"synthetic-session-{split}",
                        "message_id": message_id,
                        "role": "user",
                        "text": text,
                        "recorded_at": "2026-10-01T12:00:00Z",
                        "content_kind": "conversation",
                    }
                ],
                "candidates": [
                    {
                        "candidate_id": f"synthetic-candidate-{split}-supported",
                        "claim": "I plan to arrange paper moons before the gallery opens.",
                        "message_id": message_id,
                        "quote": text,
                        "valid_from": None,
                        "valid_to": None,
                        "support": "supported",
                        "eligibility": "eligible",
                        "support_reason": "The quoted text explicitly states this plan.",
                        "eligibility_reason": "Gallery planning fits personal-memory scope.",
                    },
                    {
                        "candidate_id": f"synthetic-candidate-{split}-unsupported",
                        "claim": "I already packed the paper moons.",
                        "message_id": message_id,
                        "quote": text,
                        "valid_from": None,
                        "valid_to": None,
                        "support": "unsupported",
                        "eligibility": "eligible",
                        "support_reason": "The source gives a plan, not completed packing.",
                        "eligibility_reason": "The claim concerns ordinary preparation.",
                    },
                ],
            }
        )
    return {
        "version": 1,
        "license": "MIT",
        "annotation_status": "agent_authored",
        "provenance": "newly_authored_fiction",
        "scope_policy": "personal-memory-research-v1",
        "cases": cases,
    }


def _support_balanced_replay_fixture() -> dict[str, Any]:
    dataset = _synthetic_replay_fixture()
    template = deepcopy(dataset["cases"][0])
    labels = ["supported"] * 115 + ["unsupported"] * 116 + ["uncertain"] * 21
    train_cases: list[dict[str, Any]] = []
    for case_index in range(126):
        case = deepcopy(template)
        case_id = f"balanced-train-{case_index}"
        message_id = f"balanced-message-{case_index}"
        text = f"Fictional balanced source {case_index} mentions a paper moon."
        case["case_id"] = case_id
        case["scenario_group"] = f"balanced-scenario-{case_index}"
        case["template_group"] = f"balanced-template-{case_index}"
        case["entity_ids"] = [f"balanced-entity-{case_index}"]
        message = case["messages"][0]
        message["session_id"] = f"balanced-session-{case_index}"
        message["message_id"] = message_id
        message["text"] = text
        for candidate_index, candidate in enumerate(case["candidates"]):
            candidate["candidate_id"] = f"balanced-candidate-{case_index}-{candidate_index}"
            candidate["claim"] = f"Fictional balanced claim {case_index}-{candidate_index}."
            candidate["message_id"] = message_id
            candidate["quote"] = text
            candidate["support"] = labels[case_index * 2 + candidate_index]
            candidate["eligibility"] = "eligible"
        train_cases.append(case)
    dataset["cases"] = [*dataset["cases"][1:], *train_cases]
    return dataset


class PrivateSftPayloadTests(unittest.TestCase):
    def test_default_payload_keeps_its_original_canonical_hash(self) -> None:
        dataset, sources = _private_fixture()

        payload = private_sft.build_payload(dataset, sources)
        explicit_none = private_sft.build_payload(dataset, sources, sampling_strategy=None)

        self.assertEqual(payload, explicit_none)
        self.assertNotIn("sampling_plan", payload)
        self.assertEqual(
            _canonical_sha256(payload),
            "608bce605996636969d48ca20773538ca0e162118baaed42c73d8cd1f2342dee",
        )

    def test_explicit_execution_id_changes_only_payload_metadata(self) -> None:
        dataset, sources = _private_fixture()
        original = private_sft.build_payload(dataset, sources)

        payload = private_sft.build_payload(dataset, sources, execution_id="retry-2")

        self.assertEqual(payload["execution_id"], "retry-2")
        payload_without_id = dict(payload)
        del payload_without_id["execution_id"]
        self.assertEqual(payload_without_id, original)
        self.assertNotEqual(_canonical_sha256(payload), _canonical_sha256(original))

    def test_execution_id_rejects_values_outside_safe_format(self) -> None:
        dataset, sources = _private_fixture()

        for invalid_id in (True, "", "Upper", "-retry", "retry_id", "a" * 65):
            with self.subTest(execution_id=invalid_id):
                with self.assertRaisesRegex(ValueError, "execution ID is invalid"):
                    private_sft.build_payload(dataset, sources, execution_id=invalid_id)

    def test_payload_includes_only_the_requested_balanced_sampling_plan(self) -> None:
        private_dataset, sources = _private_fixture()
        replay = _support_balanced_replay_fixture()

        payload = private_sft.build_payload(
            private_dataset,
            sources,
            replay,
            sampling_strategy="support-balanced-v1",
        )

        self.assertEqual(len(payload["train_rows"]), 256)
        self.assertEqual(len({row["prompt"] for row in payload["train_rows"]}), 256)
        self.assertEqual(
            payload["sampling_plan"],
            private_sft.build_sampling_plan(payload["train_rows"], seed=17),
        )
        self.assertEqual(set(payload["sampling_plan"]), {"strategy", "seed", "row_indices"})

    def test_support_balanced_plan_uses_exact_exposures_without_duplicate_rows(self) -> None:
        labels = ["supported"] * 117 + ["unsupported"] * 116 + ["uncertain"] * 23
        rows = [
            {
                "prompt": f"fictional prompt {index}",
                "completion": json.dumps(
                    {"support": label, "eligibility": "eligible"}, separators=(",", ":")
                ),
            }
            for index, label in enumerate(labels)
        ]

        plan = private_sft.build_sampling_plan(rows, seed=17)

        self.assertEqual(set(plan), {"strategy", "seed", "row_indices"})
        self.assertEqual(plan["strategy"], "support-balanced-v1")
        self.assertEqual(plan["seed"], 17)
        indices = plan["row_indices"]
        self.assertEqual(len(indices), 256)
        self.assertEqual(
            {label: sum(labels[index] == label for index in indices) for label in set(labels)},
            {"supported": 85, "unsupported": 85, "uncertain": 86},
        )
        self.assertEqual(len({row["prompt"] for row in rows}), 256)
        self.assertEqual(len(set(indices)), 193)
        self.assertEqual(plan, private_sft.build_sampling_plan(rows, seed=17))

    def test_support_balanced_plan_rejects_missing_classes_wrong_size_and_boolean_seed(
        self,
    ) -> None:
        labels = ["supported"] * 117 + ["unsupported"] * 116 + ["uncertain"] * 23
        rows = [
            {
                "prompt": f"fictional prompt {index}",
                "completion": json.dumps(
                    {"support": label, "eligibility": "eligible"}, separators=(",", ":")
                ),
            }
            for index, label in enumerate(labels)
        ]

        with self.assertRaisesRegex(ValueError, "inputs are invalid"):
            private_sft.build_sampling_plan(rows[:-1], seed=17)
        with self.assertRaisesRegex(ValueError, "inputs are invalid"):
            private_sft.build_sampling_plan(rows, seed=True)

        missing_class_rows = [
            {
                **row,
                "completion": json.dumps(
                    {"support": "supported", "eligibility": "eligible"},
                    separators=(",", ":"),
                ),
            }
            if json.loads(row["completion"])["support"] == "uncertain"
            else row
            for row in rows
        ]
        with self.assertRaisesRegex(ValueError, "requires all support classes"):
            private_sft.build_sampling_plan(missing_class_rows, seed=17)

    def test_synthetic_replay_appends_only_validated_train_rows(self) -> None:
        dataset, sources = _private_fixture()
        replay = _synthetic_replay_fixture()

        payload = private_sft.build_payload(dataset, sources, replay)

        private_rows = private_verifier_data.render_private_training_rows(dataset, sources)
        replay_rows = verifier_data.render_training_rows(replay, "train")
        self.assertEqual(payload["train_rows"], private_rows + replay_rows)
        self.assertEqual(
            payload["source_sha256"]["synthetic_replay_dataset"], _canonical_sha256(replay)
        )
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        self.assertNotIn("SYNTH_DEV_SENTINEL", serialized)
        self.assertNotIn("SYNTH_TEST_SENTINEL", serialized)

    def test_synthetic_replay_with_private_provenance_is_rejected(self) -> None:
        dataset, sources = _private_fixture()
        replay = _synthetic_replay_fixture()
        replay["provenance"] = "private_source"

        with self.assertRaisesRegex(ValueError, "provenance must be newly_authored_fiction"):
            private_sft.build_payload(dataset, sources, replay)

    def test_pending_or_rejected_train_session_does_not_count_toward_export_floor(self) -> None:
        dataset, sources = _private_fixture()
        dataset["cases"][1]["candidates"][0]["label_status"] = "pending"
        dataset["cases"][1]["candidates"][1]["label_status"] = "rejected"

        with self.assertRaisesRegex(ValueError, "exportable train rows from two sessions"):
            private_sft.build_payload(dataset, sources)

    def test_payload_uses_only_exported_train_and_label_free_dev_fields(self) -> None:
        dataset, sources = _private_fixture(include_test=True)

        payload = private_sft.build_payload(dataset, sources)

        self.assertEqual(
            payload["train_rows"],
            private_verifier_data.render_private_training_rows(dataset, sources),
        )
        self.assertEqual(
            payload["dev_prompts"],
            private_verifier_data.render_private_eval_prompts(dataset, sources, "dev"),
        )
        self.assertEqual(len(payload["train_rows"]), 4)
        self.assertEqual(len(payload["dev_prompts"]), 2)
        self.assertTrue(all(set(row) == {"prompt", "completion"} for row in payload["train_rows"]))
        self.assertTrue(all(type(prompt) is str for prompt in payload["dev_prompts"]))
        self.assertNotIn(
            {"support": "unsupported", "eligibility": "eligible"},
            [json.loads(row["completion"]) for row in payload["train_rows"]],
        )
        self.assertEqual(
            payload["source_sha256"],
            {
                "private_dataset": _canonical_sha256(dataset),
                "source_records": _canonical_sha256(sources),
                "synthetic_replay_dataset": None,
            },
        )
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        for excluded in (
            "TEST_TEXT_SENTINEL",
            "TEST_LABEL_SENTINEL",
            "RATIONALE_SENTINEL",
            "REVIEWER_SENTINEL",
            "fiction/test-a.jsonl",
        ):
            self.assertNotIn(excluded, serialized)


if __name__ == "__main__":
    unittest.main()
