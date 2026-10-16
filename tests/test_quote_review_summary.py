"""Synthetic-only behavioral tests for deterministic review tallies."""

from __future__ import annotations

import hashlib
import json
import unittest
from typing import Any

from experiments.quote_review_summary import summarize

ARMS = ("base", "prior", "tuned")


def _json_bytes(value: Any, *, compact: bool = False) -> bytes:
    kwargs = {"ensure_ascii": False, "allow_nan": False}
    if compact:
        kwargs["separators"] = (",", ":")
        text = json.dumps(value, **kwargs)
    else:
        text = json.dumps(value, indent=2, **kwargs) + "\n"
    return text.encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _refresh_packet_hash(packet: list[dict[str, Any]], mapping: dict[str, Any]) -> None:
    mapping["packet_sha256"] = _sha(_json_bytes(packet))


def _refresh_source_hash(
    packet: list[dict[str, Any]], mapping: dict[str, Any], task_id: str
) -> None:
    task = next(row for row in packet if row["task_id"] == task_id)
    curator = next(row for row in mapping["tasks"] if row["task_id"] == task_id)
    curator["source_sha256"]["messages"] = _sha(_json_bytes(task["messages"], compact=True))
    curator["source_sha256"]["raw_text"] = _sha(task["raw_text"].encode())
    curator["source_sha256"]["response_text"] = _sha(task["response_text"].encode())
    _refresh_packet_hash(packet, mapping)


def _inputs() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    packet: list[dict[str, Any]] = []
    curator_tasks: list[dict[str, Any]] = []
    review_tasks: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    case_cohorts = (("dev", "D-CASE", 2), ("test", "T-CASE", 0))
    labels = {"base": "supported", "prior": "uncertain", "tuned": "unsupported"}
    matched = {"base": [0], "prior": [0], "tuned": [0, 1]}

    for cohort, case_id, expected_count in case_cohorts:
        for arm in ARMS:
            task_id = f"{cohort}-{arm}"
            response_text = (
                '{"claims":[{"claim":"synthetic claim"}]}' if expected_count else '{"claims":[]}'
            )
            messages = [{"role": "user", "text": f"synthetic {case_id}"}]
            task = {
                "task_id": task_id,
                "messages": messages,
                "raw_text": f"raw {task_id}",
                "response_text": response_text,
                "truncated": False,
                "support_policy": "synthetic only",
            }
            packet.append(task)
            curator_tasks.append(
                {
                    "task_id": task_id,
                    "cohort": cohort,
                    "case_id": case_id,
                    "arm": arm,
                    "source_sha256": {
                        "messages": _sha(_json_bytes(messages, compact=True)),
                        "raw_text": _sha(task["raw_text"].encode()),
                        "response_text": _sha(response_text.encode()),
                    },
                }
            )
            reviews = (
                []
                if not expected_count
                else [
                    {
                        "claim_index": 0,
                        "claim": "synthetic claim",
                        "label": labels[arm],
                        "reason": "synthetic reason",
                        "quote_context_note": "optional note",
                    }
                ]
            )
            review_tasks.append(
                {
                    "task_id": task_id,
                    "readable": True,
                    "truncated": False,
                    "claim_reviews": reviews,
                    "source_offers_eligible_facts": False,
                    "notes": "synthetic row note",
                }
            )
            matches = matched[arm] if expected_count else []
            coverage.append(
                {
                    "task_id": task_id,
                    "expected_fact_count": expected_count,
                    "matched_fact_indices": matches,
                    "complete_required_facts": expected_count > 0
                    and len(matches) == expected_count,
                }
            )

    review = {"human_reviewers": 0, "tasks": review_tasks, "notes": "synthetic packet"}
    mapping = {
        "packet_sha256": _sha(_json_bytes(packet)),
        "tasks": curator_tasks,
    }
    return packet, mapping, review, coverage


class ReviewSummaryTests(unittest.TestCase):
    def test_counts_responses_and_labels_per_cohort_and_arm(self) -> None:
        packet, mapping, review, coverage = _inputs()

        result = summarize(packet, mapping, review, coverage)

        self.assertEqual(result["cohorts"]["dev"]["arms"]["base"]["responses"], 1)

    def test_counts_labels_and_adverse_responses_per_arm(self) -> None:
        packet, mapping, review, coverage = _inputs()

        result = summarize(packet, mapping, review, coverage)

        dev = result["cohorts"]["dev"]["arms"]
        self.assertEqual(dev["base"]["label_counts"]["supported"], 1)
        self.assertEqual(dev["prior"]["label_counts"]["uncertain"], 1)
        self.assertEqual(dev["tuned"]["adverse_responses"], 1)

    def test_rejects_missing_review_task(self) -> None:
        packet, mapping, review, coverage = _inputs()
        review["tasks"].pop()

        with self.assertRaisesRegex(ValueError, "review task ids"):
            summarize(packet, mapping, review, coverage)

    def test_rejects_stale_packet_hash(self) -> None:
        packet, mapping, review, coverage = _inputs()
        mapping["packet_sha256"] = "0" * 64

        with self.assertRaisesRegex(ValueError, "packet hash"):
            summarize(packet, mapping, review, coverage)

    def test_rejects_packet_and_curator_task_id_mismatch(self) -> None:
        packet, mapping, review, coverage = _inputs()
        mapping["tasks"].pop()

        with self.assertRaisesRegex(ValueError, "packet and curator task ids"):
            summarize(packet, mapping, review, coverage)

    def test_rejects_duplicate_packet_task_ids(self) -> None:
        packet, mapping, review, coverage = _inputs()
        packet[1]["task_id"] = packet[0]["task_id"]
        _refresh_packet_hash(packet, mapping)

        with self.assertRaisesRegex(ValueError, "packet task ids"):
            summarize(packet, mapping, review, coverage)

    def test_rejects_duplicate_curator_task_ids(self) -> None:
        packet, mapping, review, coverage = _inputs()
        mapping["tasks"][1]["task_id"] = mapping["tasks"][0]["task_id"]

        with self.assertRaisesRegex(ValueError, "curator task ids must be unique"):
            summarize(packet, mapping, review, coverage)

    def test_rejects_stale_source_hash(self) -> None:
        packet, mapping, review, coverage = _inputs()
        mapping["tasks"][0]["source_sha256"]["raw_text"] = "0" * 64

        with self.assertRaisesRegex(ValueError, "source hash"):
            summarize(packet, mapping, review, coverage)

    def test_rejects_unbalanced_case_arm_mapping(self) -> None:
        packet, mapping, review, coverage = _inputs()
        mapping["tasks"][0]["arm"] = "tuned"

        with self.assertRaisesRegex(ValueError, "each case must map once to every arm"):
            summarize(packet, mapping, review, coverage)

    def test_rejects_nonsequential_claim_indices(self) -> None:
        packet, mapping, review, coverage = _inputs()
        review["tasks"][0]["claim_reviews"][0]["claim_index"] = 1

        with self.assertRaisesRegex(ValueError, "claim indices must be sequential"):
            summarize(packet, mapping, review, coverage)

    def test_rejects_invalid_claim_labels_and_empty_claims(self) -> None:
        packet, mapping, review, coverage = _inputs()
        review["tasks"][0]["claim_reviews"][0]["label"] = "maybe"

        with self.assertRaisesRegex(ValueError, "claim label"):
            summarize(packet, mapping, review, coverage)

        packet, mapping, review, coverage = _inputs()
        review["tasks"][0]["claim_reviews"][0]["claim"] = "  "
        with self.assertRaisesRegex(ValueError, "claim text must be nonempty"):
            summarize(packet, mapping, review, coverage)

    def test_requires_review_truncation_to_copy_packet(self) -> None:
        packet, mapping, review, coverage = _inputs()
        review["tasks"][0]["truncated"] = True

        with self.assertRaisesRegex(ValueError, "truncated flag must copy packet"):
            summarize(packet, mapping, review, coverage)

    def test_counts_copied_truncation_flags(self) -> None:
        packet, mapping, review, coverage = _inputs()
        task_id = "dev-base"
        packet_row = next(row for row in packet if row["task_id"] == task_id)
        packet_row["truncated"] = True
        review_row = next(row for row in review["tasks"] if row["task_id"] == task_id)
        review_row["truncated"] = True
        _refresh_packet_hash(packet, mapping)

        result = summarize(packet, mapping, review, coverage)

        self.assertEqual(result["cohorts"]["dev"]["arms"]["base"]["truncated"], 1)

    def test_rejects_invalid_coverage_indices_and_completeness(self) -> None:
        packet, mapping, review, coverage = _inputs()
        coverage[0]["matched_fact_indices"] = [2]

        with self.assertRaisesRegex(ValueError, "coverage indices must be distinct and in range"):
            summarize(packet, mapping, review, coverage)

        packet, mapping, review, coverage = _inputs()
        coverage[0]["complete_required_facts"] = True
        with self.assertRaisesRegex(ValueError, "coverage completeness is inconsistent"):
            summarize(packet, mapping, review, coverage)

    def test_counts_only_strict_readable_empty_response_as_abstention(self) -> None:
        packet, mapping, review, coverage = _inputs()
        unreadable_id = "test-prior"
        review_row = next(row for row in review["tasks"] if row["task_id"] == unreadable_id)
        review_row["readable"] = False
        fenced_id = "test-tuned"
        packet_row = next(row for row in packet if row["task_id"] == fenced_id)
        packet_row["response_text"] = '```json\n{"claims": []}\n```'
        _refresh_source_hash(packet, mapping, fenced_id)

        result = summarize(packet, mapping, review, coverage)

        arms = result["cohorts"]["test"]["arms"]
        self.assertEqual(arms["base"]["abstentions"], 1)
        self.assertEqual(arms["prior"]["abstentions"], 0)
        self.assertEqual(arms["prior"]["unreadable"], 1)
        self.assertEqual(arms["tuned"]["abstentions"], 0)

    def test_zero_review_labels_do_not_make_nonempty_response_an_abstention(self) -> None:
        packet, mapping, review, coverage = _inputs()
        task_id = "test-base"
        packet_row = next(row for row in packet if row["task_id"] == task_id)
        packet_row["response_text"] = '{"claims":[{"claim":"visible claim"}]}'
        _refresh_source_hash(packet, mapping, task_id)

        with self.assertRaisesRegex(ValueError, "every response claim must have one exact review"):
            summarize(packet, mapping, review, coverage)

    def test_requires_unambiguous_review_metadata(self) -> None:
        packet, mapping, review, coverage = _inputs()
        review["human_reviewers"] = False
        with self.assertRaisesRegex(ValueError, "human_reviewers must be integer zero"):
            summarize(packet, mapping, review, coverage)

        packet, mapping, review, coverage = _inputs()
        review["tasks"][0]["source_offers_eligible_facts"] = []
        with self.assertRaisesRegex(
            ValueError, "source_offers_eligible_facts must be boolean or null"
        ):
            summarize(packet, mapping, review, coverage)

        packet, mapping, review, coverage = _inputs()
        review["tasks"][0]["claim_reviews"][0]["reason"] = "  "
        with self.assertRaisesRegex(ValueError, "reason must be nonempty text"):
            summarize(packet, mapping, review, coverage)

        packet, mapping, review, coverage = _inputs()
        review["tasks"][0]["claim_reviews"][0]["quote_context_note"] = []
        with self.assertRaisesRegex(ValueError, "quote_context_note must be nonempty text"):
            summarize(packet, mapping, review, coverage)

    def test_malformed_response_requires_manual_alignment_acknowledgement(self) -> None:
        packet, mapping, review, coverage = _inputs()
        task_id = "dev-base"
        packet_row = next(row for row in packet if row["task_id"] == task_id)
        packet_row["response_text"] = "truncated model output"
        _refresh_source_hash(packet, mapping, task_id)

        with self.assertRaisesRegex(ValueError, "manual alignment acknowledgement"):
            summarize(packet, mapping, review, coverage)

        review_row = next(row for row in review["tasks"] if row["task_id"] == task_id)
        review_row["manual_alignment_acknowledged"] = True
        summarize(packet, mapping, review, coverage)

    def test_parseable_nonclaim_json_cannot_bypass_alignment_with_ack(self) -> None:
        packet, mapping, review, coverage = _inputs()
        task_id = "dev-base"
        packet_row = next(row for row in packet if row["task_id"] == task_id)
        packet_row["response_text"] = '{"other":[]}'
        _refresh_source_hash(packet, mapping, task_id)
        review_row = next(row for row in review["tasks"] if row["task_id"] == task_id)
        review_row["manual_alignment_acknowledged"] = True

        with self.assertRaisesRegex(ValueError, "parseable response claims cannot be aligned"):
            summarize(packet, mapping, review, coverage)

    def test_single_json_fence_is_allowed_for_claim_alignment_only(self) -> None:
        packet, mapping, review, coverage = _inputs()
        task_id = "test-base"
        packet_row = next(row for row in packet if row["task_id"] == task_id)
        packet_row["response_text"] = '```json\n{"claims": []}\n```'
        _refresh_source_hash(packet, mapping, task_id)

        result = summarize(packet, mapping, review, coverage)

        self.assertEqual(result["cohorts"]["test"]["arms"]["base"]["abstentions"], 0)

    def test_empty_claims_with_extra_json_key_is_not_an_abstention(self) -> None:
        packet, mapping, review, coverage = _inputs()
        task_id = "test-tuned"
        packet_row = next(row for row in packet if row["task_id"] == task_id)
        packet_row["response_text"] = '{"claims": [], "extra": true}'
        _refresh_source_hash(packet, mapping, task_id)

        result = summarize(packet, mapping, review, coverage)

        self.assertEqual(result["cohorts"]["test"]["arms"]["tuned"]["abstentions"], 0)

    def test_rejects_claim_text_that_does_not_align_to_response(self) -> None:
        packet, mapping, review, coverage = _inputs()
        review["tasks"][0]["claim_reviews"][0]["claim"] = "changed claim"

        with self.assertRaisesRegex(ValueError, "one exact review"):
            summarize(packet, mapping, review, coverage)

    def test_rejects_complete_coverage_for_unreadable_response(self) -> None:
        packet, mapping, review, coverage = _inputs()
        review_row = next(row for row in review["tasks"] if row["task_id"] == "dev-tuned")
        review_row["readable"] = False

        with self.assertRaisesRegex(
            ValueError, "unreadable response cannot have complete coverage"
        ):
            summarize(packet, mapping, review, coverage)

    def test_paired_arms_require_equal_expected_fact_counts(self) -> None:
        packet, mapping, review, coverage = _inputs()
        coverage[1]["expected_fact_count"] = 3

        with self.assertRaisesRegex(
            ValueError, "paired arms must have the same expected fact count"
        ):
            summarize(packet, mapping, review, coverage)

    def test_reports_nonempty_coverage_and_paired_tuned_prior_counts(self) -> None:
        packet, mapping, review, coverage = _inputs()

        result = summarize(packet, mapping, review, coverage)

        dev = result["cohorts"]["dev"]["arms"]
        empty = result["cohorts"]["test"]["arms"]["base"]
        self.assertEqual(dev["base"]["complete_required_facts"], 0)
        self.assertEqual(dev["tuned"]["complete_required_facts"], 1)
        self.assertEqual(
            dev["base"]["nonempty_coverage"],
            {"responses": 1, "matched_facts": 1, "expected_facts": 2},
        )
        self.assertEqual(
            empty["nonempty_coverage"], {"responses": 0, "matched_facts": 0, "expected_facts": 0}
        )
        dev_pair = result["paired_coverage"]["dev"]
        self.assertEqual(dev_pair["complete_case_outcomes"]["tuned_wins"], 1)
        self.assertEqual(result["paired_coverage"]["test"]["complete_case_outcomes"]["pairs"], 0)
        self.assertNotIn("overall", result["paired_coverage"])

        packet, mapping, review, coverage = _inputs()
        coverage[2]["matched_fact_indices"] = [0]
        coverage[2]["complete_required_facts"] = False
        partial_result = summarize(packet, mapping, review, coverage)
        partial = partial_result["paired_coverage"]["dev"]
        self.assertEqual(partial["complete_case_outcomes"]["neither_complete"], 1)
        self.assertEqual(partial["partial_fact_outcomes"]["ties"], 1)

        packet, mapping, review, coverage = _inputs()
        coverage[1]["matched_fact_indices"] = [0, 1]
        coverage[1]["complete_required_facts"] = True
        coverage[2]["matched_fact_indices"] = [0]
        coverage[2]["complete_required_facts"] = False
        loss_result = summarize(packet, mapping, review, coverage)
        self.assertEqual(
            loss_result["paired_coverage"]["dev"]["complete_case_outcomes"]["tuned_losses"], 1
        )

        packet, mapping, review, coverage = _inputs()
        coverage[1]["matched_fact_indices"] = [0, 1]
        coverage[1]["complete_required_facts"] = True
        both_result = summarize(packet, mapping, review, coverage)
        self.assertEqual(
            both_result["paired_coverage"]["dev"]["complete_case_outcomes"]["both_complete"], 1
        )


if __name__ == "__main__":
    unittest.main()
