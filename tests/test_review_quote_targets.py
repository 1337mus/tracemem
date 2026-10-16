from __future__ import annotations

import copy
import hashlib
import json
import unittest

from training.build_quote_data import build
from training.review_quote_targets import review_seed_targets


def _source() -> dict[str, object]:
    return {
        "seed": 23,
        "split": "train",
        "variants": 1,
        "value_banks": {
            "person": ["Ari"],
            "product": ["Wallet"],
            "object": ["record"],
            "place": ["home"],
            "date_from": ["2025-01-02"],
            "date_to": ["2025-01-03"],
        },
        "families": [
            {
                "family_id": "family-a",
                "tags": ["fact"],
                "messages": [
                    {
                        "ref": "u1",
                        "role": "user",
                        "session": 1,
                        "content_kind": "conversation",
                        "text": "I own the {object}.",
                    }
                ],
                "expected": [
                    {
                        "claim": "The user owns the {object}.",
                        "message_ref": "u1",
                        "quote": "I own the {object}.",
                        "valid_from": None,
                        "valid_to": None,
                    }
                ],
            }
        ],
    }


def _raw_source() -> bytes:
    return json.dumps(_source(), ensure_ascii=False, separators=(",", ":")).encode()


def _review(source_raw: bytes, families: list[dict[str, object]]) -> dict[str, object]:
    return {
        "version": 1,
        "source_sha256": hashlib.sha256(source_raw).hexdigest(),
        "annotation_status": "agent_reviewed_provisional",
        "provenance": "newly_authored_fiction",
        "families": families,
    }


def _canonical_hash(value: object) -> str:
    canonical = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _family_review(source: dict[str, object]) -> dict[str, object]:
    family = source["families"][0]  # type: ignore[index]
    return {
        "family_id": "family-a",
        "original_expected_sha256": _canonical_hash(family["expected"]),  # type: ignore[index]
        "disposition": "retain",
        "replacement_expected": None,
        "rationale": "The authored target matches the source statement.",
        "required_facts": [
            {
                "fact_id": "owns-record",
                "description": "The user owns the record.",
                "source_ref": "u1",
                "source_quote": "I own the",
                "target_claim_index": 0,
                "target_quote": "owns the",
            }
        ],
    }


class ReviewQuoteTargetsTests(unittest.TestCase):
    def test_rejects_review_bound_to_different_source_bytes(self) -> None:
        raw = _raw_source()
        review = _review(raw, [])
        review["source_sha256"] = "0" * 64

        with self.assertRaisesRegex(ValueError, "source_sha256"):
            review_seed_targets(raw, review)

    def test_rejects_non_integer_review_version(self) -> None:
        raw = _raw_source()
        review = _review(raw, [])
        review["version"] = True

        with self.assertRaisesRegex(ValueError, "version"):
            review_seed_targets(raw, review)

    def test_rejects_duplicate_json_keys_in_source(self) -> None:
        raw = b'{"seed":23,"seed":23,"split":"train"}'

        with self.assertRaisesRegex(ValueError, "valid UTF-8 JSON"):
            review_seed_targets(raw, _review(raw, []))

    def test_rejects_stale_original_expected_hash(self) -> None:
        source = _source()
        raw = json.dumps(source, separators=(",", ":")).encode()
        family = _family_review(source)
        family["original_expected_sha256"] = "0" * 64

        with self.assertRaisesRegex(ValueError, "original_expected_sha256"):
            review_seed_targets(raw, _review(raw, [family]))

    def test_does_not_count_citation_quote_as_target_claim_coverage(self) -> None:
        source = _source()
        raw = json.dumps(source, separators=(",", ":")).encode()
        family = _family_review(source)
        family["required_facts"][0]["target_quote"] = "I own the"  # type: ignore[index]

        with self.assertRaisesRegex(ValueError, "target_quote"):
            review_seed_targets(raw, _review(raw, [family]))

    def test_rejects_family_with_unexpected_fields(self) -> None:
        source = _source()
        raw = json.dumps(source, separators=(",", ":")).encode()
        family = _family_review(source)
        family["extra"] = True

        with self.assertRaisesRegex(ValueError, "family review"):
            review_seed_targets(raw, _review(raw, [family]))

    def test_rejects_missing_family_review(self) -> None:
        raw = _raw_source()

        with self.assertRaisesRegex(ValueError, "exactly once"):
            review_seed_targets(raw, _review(raw, []))

    def test_rejects_duplicate_family_review_ids(self) -> None:
        source = _source()
        second = copy.deepcopy(source["families"][0])  # type: ignore[index]
        second["family_id"] = "family-b"
        second["messages"][0]["ref"] = "u2"
        second["messages"][0]["text"] = "I own the product."
        second["expected"][0]["message_ref"] = "u2"
        second["expected"][0]["quote"] = "I own the product."
        source["families"].append(second)  # type: ignore[index]
        raw = json.dumps(source, separators=(",", ":")).encode()
        family = _family_review(source)

        with self.assertRaisesRegex(ValueError, "exactly once"):
            review_seed_targets(raw, _review(raw, [family, family]))

    def test_retain_requires_null_replacement_and_nonempty_original(self) -> None:
        source = _source()
        raw = json.dumps(source, separators=(",", ":")).encode()
        family = _family_review(source)
        family["replacement_expected"] = source["families"][0]["expected"]  # type: ignore[index]

        with self.assertRaisesRegex(ValueError, "retain"):
            review_seed_targets(raw, _review(raw, [family]))

    def test_rejects_expected_claim_from_wrong_speaker(self) -> None:
        source = _source()
        source["families"][0]["messages"][0]["role"] = "assistant"  # type: ignore[index]
        raw = json.dumps(source, separators=(",", ":")).encode()

        with self.assertRaisesRegex(ValueError, "user conversation"):
            review_seed_targets(raw, _review(raw, [_family_review(source)]))

    def test_rejects_expected_claim_from_nonconversation_content(self) -> None:
        source = _source()
        source["families"][0]["messages"][0]["content_kind"] = "quoted_document"  # type: ignore[index]
        raw = json.dumps(source, separators=(",", ":")).encode()

        with self.assertRaisesRegex(ValueError, "user conversation"):
            review_seed_targets(raw, _review(raw, [_family_review(source)]))

    def test_rejects_stale_source_span(self) -> None:
        source = _source()
        raw = json.dumps(source, separators=(",", ":")).encode()
        family = _family_review(source)
        family["required_facts"][0]["source_quote"] = "I own a"  # type: ignore[index]

        with self.assertRaisesRegex(ValueError, "source_quote"):
            review_seed_targets(raw, _review(raw, [family]))

    def test_rejects_non_unique_source_fact_span(self) -> None:
        source = _source()
        source["families"][0]["messages"][0]["text"] = "I own the {object}. I own a book."  # type: ignore[index]
        raw = json.dumps(source, separators=(",", ":")).encode()
        family = _family_review(source)
        family["required_facts"][0]["source_quote"] = "I"  # type: ignore[index]

        with self.assertRaisesRegex(ValueError, "source_quote"):
            review_seed_targets(raw, _review(raw, [family]))

    def test_rejects_fact_not_cited_by_selected_target(self) -> None:
        source = _source()
        source["families"][0]["expected"][0]["quote"] = "record"  # type: ignore[index]
        raw = json.dumps(source, separators=(",", ":")).encode()
        family = _family_review(source)

        with self.assertRaisesRegex(ValueError, "cite source_quote"):
            review_seed_targets(raw, _review(raw, [family]))

    def test_success_is_deterministic_validated_and_input_immutable(self) -> None:
        source = _source()
        raw = json.dumps(source, ensure_ascii=False, separators=(",", ":")).encode()
        review = _review(raw, [_family_review(source)])
        source_before = copy.deepcopy(source)
        review_before = copy.deepcopy(review)

        first = review_seed_targets(raw, review)
        second = review_seed_targets(raw, review)

        self.assertEqual(first, second)
        self.assertEqual(source, source_before)
        self.assertEqual(review, review_before)
        self.assertEqual(first["seeds"], source)
        self.assertEqual(len(build(first["seeds"], "train")["cases"]), 1)
        report = first["report"]
        self.assertEqual(report["family_count"], 1)
        self.assertEqual(report["row_count"], 1)
        self.assertEqual(report["claim_count"], 1)
        self.assertEqual(report["required_fact_count"], 1)
        self.assertEqual(report["retained_family_count"], 1)
        self.assertEqual(report["revised_family_count"], 0)
        self.assertEqual(report["abstaining_family_count"], 0)
        self.assertEqual(report["human_reviewers"], 0)
        self.assertIn("semantic entailment", report["limitation"])
        self.assertEqual(report["source_raw_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(report["corrected_seed_canonical_sha256"], _canonical_hash(source))

    def test_valid_revision_changes_only_expected_targets(self) -> None:
        source = _source()
        raw = json.dumps(source, separators=(",", ":")).encode()
        family = _family_review(source)
        replacement = copy.deepcopy(source["families"][0]["expected"])  # type: ignore[index]
        replacement[0]["claim"] = "User owns {object}."
        family["disposition"] = "revise"
        family["replacement_expected"] = replacement
        family["required_facts"][0]["target_quote"] = "owns"  # type: ignore[index]
        original_messages = copy.deepcopy(source["families"][0]["messages"])  # type: ignore[index]

        result = review_seed_targets(raw, _review(raw, [family]))

        self.assertEqual(result["seeds"]["families"][0]["expected"], replacement)
        self.assertEqual(result["seeds"]["families"][0]["messages"], original_messages)
        self.assertEqual(result["report"]["revised_family_count"], 1)
        self.assertEqual(result["report"]["retained_family_count"], 0)

    def test_valid_abstention_preserves_empty_target(self) -> None:
        source = _source()
        source["families"][0]["expected"] = []  # type: ignore[index]
        raw = json.dumps(source, separators=(",", ":")).encode()
        family = _family_review(source)
        family.update(
            {
                "original_expected_sha256": _canonical_hash([]),
                "disposition": "abstain",
                "replacement_expected": None,
                "required_facts": [],
            }
        )

        result = review_seed_targets(raw, _review(raw, [family]))

        self.assertEqual(result["seeds"]["families"][0]["expected"], [])
        self.assertEqual(result["report"]["abstaining_family_count"], 1)

    def test_rejects_revise_with_unchanged_or_empty_replacement(self) -> None:
        source = _source()
        raw = json.dumps(source, separators=(",", ":")).encode()
        family = _family_review(source)
        family["disposition"] = "revise"
        family["replacement_expected"] = copy.deepcopy(source["families"][0]["expected"])  # type: ignore[index]

        with self.assertRaisesRegex(ValueError, "differing non-empty"):
            review_seed_targets(raw, _review(raw, [family]))

    def test_abstain_requires_original_empty_target(self) -> None:
        source = _source()
        raw = json.dumps(source, separators=(",", ":")).encode()
        family = _family_review(source)
        family["disposition"] = "abstain"
        family["required_facts"] = []

        with self.assertRaisesRegex(ValueError, "abstain requires empty"):
            review_seed_targets(raw, _review(raw, [family]))

    def test_rejects_boolean_target_claim_index(self) -> None:
        source = _source()
        raw = json.dumps(source, separators=(",", ":")).encode()
        family = _family_review(source)
        family["required_facts"][0]["target_claim_index"] = True  # type: ignore[index]

        with self.assertRaisesRegex(ValueError, "target_claim_index"):
            review_seed_targets(raw, _review(raw, [family]))

    def test_rejects_duplicate_fact_ids_within_family(self) -> None:
        source = _source()
        raw = json.dumps(source, separators=(",", ":")).encode()
        family = _family_review(source)
        family["required_facts"].append(  # type: ignore[union-attr]
            {
                **family["required_facts"][0],  # type: ignore[index]
                "target_quote": "user owns",
            }
        )

        with self.assertRaisesRegex(ValueError, "duplicate fact_id"):
            review_seed_targets(raw, _review(raw, [family]))

    def test_rejects_nonempty_target_without_fact_annotations(self) -> None:
        source = _source()
        raw = json.dumps(source, separators=(",", ":")).encode()
        family = _family_review(source)
        family["required_facts"] = []

        with self.assertRaisesRegex(ValueError, "required_facts"):
            review_seed_targets(raw, _review(raw, [family]))


if __name__ == "__main__":
    unittest.main()
