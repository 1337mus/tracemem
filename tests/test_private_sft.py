"""Offline private SFT payload contract tests using fictional inputs only."""

from __future__ import annotations

import hashlib
import json
import unittest
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


class PrivateSftPayloadTests(unittest.TestCase):
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
