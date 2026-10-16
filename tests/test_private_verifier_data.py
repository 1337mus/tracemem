"""Private verifier schema/export contract tests using invented source text only."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import unittest
from copy import deepcopy
from typing import Any

from training import verifier_data


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
        "message_id": record["message_id"],
        "source_record_id": record["record_id"],
        "revision": record["revision"],
        "session_id": record["session_id"],
        "text": record["text"],
        "text_sha256": record["text_sha256"],
        "source_locator": record["source_locator"],
        "line_number": record["line_number"],
        "role": record["role"],
        "recorded_at": record["recorded_at"],
        "content_kind": record["content_kind"],
    }
    quote = "the blue glass buttons"
    start = record["text"].index(quote)
    return {
        "case_id": case_id,
        "session_id": record["session_id"],
        "split": split,
        "window_id": f"window-{case_id}",
        "privacy_clearance": {
            "status": "cleared",
            "reviewer_id": "owner-fiction",
            "reviewer_type": "human",
            "reviewed_at": "2026-10-16T12:00:00Z",
        },
        "messages": [message],
        "candidates": [
            {
                "candidate_id": f"candidate-{case_id}-one",
                "claim": "The speaker is uncertain about cataloging the buttons.",
                "message_id": message["message_id"],
                "source_record_id": record["record_id"],
                "quote": quote,
                "char_start": start,
                "char_end": start + len(quote),
                "valid_from": None,
                "valid_to": None,
                "support": "uncertain",
                "eligibility": "uncertain",
                "support_reason": "private-rationale-marker",
                "eligibility_reason": "private-scope-rationale-marker",
                "label_status": "agent_provisional",
                "annotator": {"reviewer_id": "agent-run-fiction", "reviewer_type": "agent"},
                "reviews": [],
            },
            {
                "candidate_id": f"candidate-{case_id}-two",
                "claim": "The speaker mentions the blue glass buttons.",
                "message_id": message["message_id"],
                "source_record_id": record["record_id"],
                "quote": quote,
                "char_start": start,
                "char_end": start + len(quote),
                "valid_from": None,
                "valid_to": None,
                "support": "supported",
                "eligibility": "ineligible",
                "support_reason": "A fictional exercise is described.",
                "eligibility_reason": "The detail is exercise metadata.",
                "label_status": "agent_provisional",
                "annotator": {"reviewer_id": "agent-run-fiction", "reviewer_type": "agent"},
                "reviews": [],
            },
        ],
    }


class PrivateVerifierTrainingExportTests(unittest.TestCase):
    def test_private_train_export_keeps_uncertainty_and_uses_shared_label_free_prompt(self) -> None:
        spec = importlib.util.find_spec("training.private_verifier_data")
        self.assertIsNotNone(spec, "private verifier train export behavior is not implemented yet")
        from training import private_verifier_data

        train_source = _source_record(
            "record-train-fiction",
            "session-train-fiction",
            "I might catalog the blue glass buttons after the invented festival.",
        )
        dev_source = _source_record(
            "record-dev-fiction",
            "session-dev-fiction",
            "I might catalog the blue glass buttons after the imaginary parade.",
        )
        train_case = _case(train_source, "case-train-fiction", "train")
        dev_case = _case(dev_source, "case-dev-fiction", "dev")
        payload = {
            "version": 1,
            "license": "private-not-for-redistribution",
            "annotation_status": "agent_provisional",
            "provenance": "private_source",
            "scope_policy": "personal-memory-research-v1",
            "cases": [train_case, dev_case],
        }
        sources = {train_source["record_id"]: train_source, dev_source["record_id"]: dev_source}

        rows = private_verifier_data.render_private_training_rows(payload, sources)

        self.assertEqual(len(rows), 2)
        self.assertEqual(set(rows[0]), {"prompt", "completion"})
        self.assertEqual(
            [json.loads(row["completion"]) for row in rows],
            [
                {"support": "uncertain", "eligibility": "uncertain"},
                {"support": "supported", "eligibility": "ineligible"},
            ],
        )
        shared_case = {
            "messages": [
                {
                    key: train_case["messages"][0][key]
                    for key in ("message_id", "role", "text", "recorded_at", "content_kind")
                }
            ]
        }
        shared_candidate = train_case["candidates"][0]
        self.assertEqual(
            rows[0]["prompt"],
            verifier_data._render_validated_candidate_prompt(shared_case, shared_candidate),
        )
        for row in rows:
            self.assertNotIn("private-rationale-marker", row["prompt"])
            self.assertNotIn("private-scope-rationale-marker", row["prompt"])
            self.assertNotIn("owner-fiction", row["prompt"])
            self.assertNotIn("private_source", row["prompt"])
            self.assertNotIn(train_source["source_locator"], row["prompt"])

    def test_rejects_normalized_long_messages_reused_across_splits(self) -> None:
        spec = importlib.util.find_spec("training.private_verifier_data")
        self.assertIsNotNone(spec, "private verifier validation behavior is not implemented yet")
        from training import private_verifier_data

        long_text = (
            "I might catalog the blue glass buttons after the invented festival, while comparing "
            "fictional labels and sorting imaginary ribbons for a paper display. "
        ) * 2
        normalized_duplicate = long_text.replace("while comparing", "while  comparing")
        train_source = _source_record("record-long-train", "session-long-train", long_text)
        dev_source = _source_record("record-long-dev", "session-long-dev", normalized_duplicate)
        payload = {
            "version": 1,
            "license": "private-not-for-redistribution",
            "annotation_status": "agent_provisional",
            "provenance": "private_source",
            "scope_policy": "personal-memory-research-v1",
            "cases": [
                _case(train_source, "case-long-train", "train"),
                _case(dev_source, "case-long-dev", "dev"),
            ],
        }
        sources = {
            train_source["record_id"]: train_source,
            dev_source["record_id"]: dev_source,
        }

        with self.assertRaisesRegex(ValueError, "long message overlap across splits"):
            private_verifier_data.validate_private_verifier_dataset(payload, sources)

    def test_windows_from_one_source_session_must_share_a_split(self) -> None:
        spec = importlib.util.find_spec("training.private_verifier_data")
        self.assertIsNotNone(spec, "private verifier module is not implemented yet")
        from training import private_verifier_data

        source = _source_record(
            "record-shared-session-fiction",
            "session-shared-fiction",
            "I might catalog the blue glass buttons.",
        )
        train_case = _case(source, "case-shared-train-fiction", "train")
        test_case = deepcopy(train_case)
        test_case["case_id"] = "case-shared-test-fiction"
        test_case["window_id"] = "window-shared-test-fiction"
        test_case["split"] = "test"
        test_case["candidates"][0]["candidate_id"] = "candidate-shared-test-one"
        test_case["candidates"][1]["candidate_id"] = "candidate-shared-test-two"
        payload = {
            "version": 1,
            "license": "private-not-for-redistribution",
            "annotation_status": "agent_provisional",
            "provenance": "private_source",
            "scope_policy": "personal-memory-research-v1",
            "cases": [train_case, test_case],
        }

        with self.assertRaisesRegex(ValueError, "source session .* overlaps splits"):
            private_verifier_data.validate_private_verifier_dataset(
                payload, {source["record_id"]: source}
            )

    def test_eval_renderer_returns_only_cleared_label_free_dev_prompts(self) -> None:
        spec = importlib.util.find_spec("training.private_verifier_data")
        self.assertIsNotNone(spec, "private verifier module is not implemented yet")
        from training import private_verifier_data

        dev_source = _source_record(
            "record-eval-fiction", "session-eval-fiction", "I might catalog the blue glass buttons."
        )
        pending_source = _source_record(
            "record-pending-fiction",
            "session-pending-fiction",
            "I might catalog the blue glass buttons after a paper parade.",
        )
        dev_case = _case(dev_source, "case-eval-fiction", "dev")
        dev_case["candidates"][1]["label_status"] = "rejected"
        pending_case = _case(pending_source, "case-pending-fiction", "dev")
        pending_case["privacy_clearance"]["status"] = "pending"
        payload = {
            "version": 1,
            "license": "private-not-for-redistribution",
            "annotation_status": "agent_provisional",
            "provenance": "private_source",
            "scope_policy": "personal-memory-research-v1",
            "cases": [dev_case, pending_case],
        }
        sources = {
            dev_source["record_id"]: dev_source,
            pending_source["record_id"]: pending_source,
        }

        self.assertTrue(
            hasattr(private_verifier_data, "render_private_eval_prompts"),
            "private eval prompt rendering behavior is not implemented yet",
        )
        prompts = private_verifier_data.render_private_eval_prompts(payload, sources, "dev")

        self.assertEqual(len(prompts), 1)
        self.assertIsInstance(prompts[0], str)
        self.assertIn(dev_case["candidates"][0]["claim"], prompts[0])
        self.assertNotIn(dev_case["candidates"][1]["claim"], prompts[0])
        for private_marker in (
            "private-rationale-marker",
            "private-scope-rationale-marker",
            "owner-fiction",
            "agent-run-fiction",
            "private_source",
        ):
            self.assertNotIn(private_marker, prompts[0])

    def test_training_export_excludes_pending_and_rejected_rows(self) -> None:
        spec = importlib.util.find_spec("training.private_verifier_data")
        self.assertIsNotNone(spec, "private verifier module is not implemented yet")
        from training import private_verifier_data

        source = _source_record(
            "record-status-fiction",
            "session-status-fiction",
            "I might catalog the blue glass buttons.",
        )
        case = _case(source, "case-status-fiction", "train")
        pending_review = {
            "reviewer_id": "reviewer-pending-fiction",
            "reviewer_type": "human",
            "status": "pending",
            "reviewed_at": None,
            "support": None,
            "eligibility": None,
            "support_reason": None,
            "eligibility_reason": None,
        }
        case["candidates"][0]["reviews"] = [pending_review]
        case["candidates"][1]["label_status"] = "pending"
        rejected = deepcopy(case["candidates"][1])
        rejected["candidate_id"] = "candidate-status-rejected"
        rejected["claim"] = "Rejected fictional claim about the buttons."
        rejected["label_status"] = "rejected"
        exportable = deepcopy(case["candidates"][0])
        exportable["candidate_id"] = "candidate-status-exportable"
        exportable["claim"] = "Provisional fictional claim about the buttons."
        exportable["reviews"] = []
        case["candidates"].extend([rejected, exportable])
        payload = {
            "version": 1,
            "license": "private-not-for-redistribution",
            "annotation_status": "agent_provisional",
            "provenance": "private_source",
            "scope_policy": "personal-memory-research-v1",
            "cases": [case],
        }

        rows = private_verifier_data.render_private_training_rows(
            payload, {source["record_id"]: source}
        )

        self.assertEqual(len(rows), 1)
        self.assertIn(exportable["claim"], rows[0]["prompt"])
        self.assertNotIn("Rejected fictional claim", rows[0]["prompt"])

    def test_provisional_review_disagreement_is_preserved_but_not_exported(self) -> None:
        spec = importlib.util.find_spec("training.private_verifier_data")
        self.assertIsNotNone(spec, "private verifier module is not implemented yet")
        from training import private_verifier_data

        source = _source_record(
            "record-disagreement-fiction",
            "session-disagreement-fiction",
            "I might catalog the blue glass buttons.",
        )
        case = _case(source, "case-disagreement-fiction", "train")
        first = case["candidates"][0]
        first["reviews"] = [
            {
                "reviewer_id": "independent-reviewer-fiction",
                "reviewer_type": "human",
                "status": "accepted",
                "reviewed_at": "2026-10-16T12:30:00Z",
                "support": "unsupported",
                "eligibility": "ineligible",
                "support_reason": "The reviewer sees no entailment in this fictional source.",
                "eligibility_reason": "The reviewer classifies the subject as an exercise detail.",
            }
        ]
        case["candidates"][1]["label_status"] = "pending"
        payload = {
            "version": 1,
            "license": "private-not-for-redistribution",
            "annotation_status": "agent_provisional",
            "provenance": "private_source",
            "scope_policy": "personal-memory-research-v1",
            "cases": [case],
        }

        validated = private_verifier_data.validate_private_verifier_dataset(
            payload, {source["record_id"]: source}
        )
        rows = private_verifier_data.render_private_training_rows(
            payload, {source["record_id"]: source}
        )

        self.assertEqual(validated[0]["candidates"][0]["support"], "uncertain")
        self.assertEqual(validated[0]["candidates"][0]["reviews"][0]["support"], "unsupported")
        self.assertEqual(rows, [])

    def test_rejects_source_role_content_kind_pair_not_emitted_by_adapters(self) -> None:
        spec = importlib.util.find_spec("training.private_verifier_data")
        self.assertIsNotNone(spec, "private verifier module is not implemented yet")
        from training import private_verifier_data

        source = _source_record(
            "record-role-fiction", "session-role-fiction", "I might catalog the blue glass buttons."
        )
        source["content_kind"] = "tool_output"
        payload = {
            "version": 1,
            "license": "private-not-for-redistribution",
            "annotation_status": "agent_provisional",
            "provenance": "private_source",
            "scope_policy": "personal-memory-research-v1",
            "cases": [_case(source, "case-role-fiction", "train")],
        }

        with self.assertRaisesRegex(ValueError, "user source cannot be tool_output"):
            private_verifier_data.validate_private_verifier_dataset(
                payload, {source["record_id"]: source}
            )

    def test_accepts_explicit_agent_privacy_clearance_for_provisional_pilot(self) -> None:
        spec = importlib.util.find_spec("training.private_verifier_data")
        self.assertIsNotNone(spec, "private verifier module is not implemented yet")
        from training import private_verifier_data

        source = _source_record(
            "record-agent-clearance-fiction",
            "session-agent-clearance-fiction",
            "I might catalog the blue glass buttons.",
        )
        case = _case(source, "case-agent-clearance-fiction", "train")
        case["privacy_clearance"]["reviewer_id"] = "agent-clearance-run-fiction"
        case["privacy_clearance"]["reviewer_type"] = "agent"
        payload = {
            "version": 1,
            "license": "private-not-for-redistribution",
            "annotation_status": "agent_provisional",
            "provenance": "private_source",
            "scope_policy": "personal-memory-research-v1",
            "cases": [case],
        }

        validated = private_verifier_data.validate_private_verifier_dataset(
            payload, {source["record_id"]: source}
        )

        self.assertEqual(validated[0]["privacy_clearance"]["reviewer_type"], "agent")

    def test_rejects_offsets_that_extend_past_the_local_source_text(self) -> None:
        spec = importlib.util.find_spec("training.private_verifier_data")
        self.assertIsNotNone(spec, "private verifier module is not implemented yet")
        from training import private_verifier_data

        source = _source_record(
            "record-offset-fiction",
            "session-offset-fiction",
            "I might catalog the blue glass buttons.",
        )
        case = _case(source, "case-offset-fiction", "train")
        candidate = case["candidates"][0]
        candidate["quote"] = source["text"][candidate["char_start"] :]
        candidate["char_end"] = len(source["text"]) + 20
        payload = {
            "version": 1,
            "license": "private-not-for-redistribution",
            "annotation_status": "agent_provisional",
            "provenance": "private_source",
            "scope_policy": "personal-memory-research-v1",
            "cases": [case],
        }

        with self.assertRaisesRegex(ValueError, "candidate offsets must end within source text"):
            private_verifier_data.validate_private_verifier_dataset(
                payload, {source["record_id"]: source}
            )

    def test_accepted_review_cannot_disagree_with_final_labels(self) -> None:
        spec = importlib.util.find_spec("training.private_verifier_data")
        self.assertIsNotNone(spec, "private verifier module is not implemented yet")
        from training import private_verifier_data

        source = _source_record(
            "record-review-fiction",
            "session-review-fiction",
            "I might catalog the blue glass buttons.",
        )
        case = _case(source, "case-review-fiction", "train")
        candidate = case["candidates"][0]
        candidate["label_status"] = "accepted"
        candidate["reviews"] = [
            {
                "reviewer_id": "independent-reviewer-fiction",
                "reviewer_type": "human",
                "status": "accepted",
                "reviewed_at": "2026-10-16T12:30:00Z",
                "support": "supported",
                "eligibility": "ineligible",
                "support_reason": "The independent fictional review differs.",
                "eligibility_reason": "The reviewer marks the exercise out of scope.",
            }
        ]
        payload = {
            "version": 1,
            "license": "private-not-for-redistribution",
            "annotation_status": "agent_provisional",
            "provenance": "private_source",
            "scope_policy": "personal-memory-research-v1",
            "cases": [case],
        }

        with self.assertRaisesRegex(ValueError, "accepted review labels must match final labels"):
            private_verifier_data.validate_private_verifier_dataset(
                payload, {source["record_id"]: source}
            )
        candidate["reviews"][0]["support"] = "uncertain"
        candidate["reviews"][0]["eligibility"] = "uncertain"

        validated = private_verifier_data.validate_private_verifier_dataset(
            payload, {source["record_id"]: source}
        )

        self.assertEqual(validated[0]["candidates"][0]["label_status"], "accepted")

    def test_preserves_unknown_source_recorded_at_as_null(self) -> None:
        spec = importlib.util.find_spec("training.private_verifier_data")
        self.assertIsNotNone(spec, "private verifier module is not implemented yet")
        from training import private_verifier_data

        source = _source_record(
            "record-unknown-time-fiction",
            "session-unknown-time-fiction",
            "I might catalog the blue glass buttons.",
        )
        source["recorded_at"] = None
        payload = {
            "version": 1,
            "license": "private-not-for-redistribution",
            "annotation_status": "agent_provisional",
            "provenance": "private_source",
            "scope_policy": "personal-memory-research-v1",
            "cases": [_case(source, "case-unknown-time-fiction", "train")],
        }

        validated = private_verifier_data.validate_private_verifier_dataset(
            payload, {source["record_id"]: source}
        )

        self.assertIsNone(validated[0]["messages"][0]["recorded_at"])

    def test_source_text_hash_must_match_local_record_content(self) -> None:
        spec = importlib.util.find_spec("training.private_verifier_data")
        self.assertIsNotNone(spec, "private verifier module is not implemented yet")
        from training import private_verifier_data

        source = _source_record(
            "record-hash-fiction", "session-hash-fiction", "I might catalog the blue glass buttons."
        )
        source["text"] += " This ending was changed after hashing."
        payload = {
            "version": 1,
            "license": "private-not-for-redistribution",
            "annotation_status": "agent_provisional",
            "provenance": "private_source",
            "scope_policy": "personal-memory-research-v1",
            "cases": [_case(source, "case-hash-fiction", "train")],
        }

        with self.assertRaisesRegex(ValueError, "source text hash does not match"):
            private_verifier_data.validate_private_verifier_dataset(
                payload, {source["record_id"]: source}
            )


if __name__ == "__main__":
    unittest.main()
