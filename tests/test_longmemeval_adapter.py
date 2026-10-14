"""Contract tests for the fixture-only LongMemEval bytes adapter."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path


class LongMemEvalAdapterTests(unittest.TestCase):
    @staticmethod
    def _fixture_items() -> list[dict[str, object]]:
        fixture_path = (
            Path(__file__).resolve().parents[1] / "examples/synthetic/longmemeval-format.json"
        )
        return json.loads(fixture_path.read_text(encoding="utf-8"))

    @staticmethod
    def _payload(items: object) -> bytes:
        return json.dumps(items, ensure_ascii=False).encode("utf-8")

    def _parse_items(
        self, items: object, *, namespace: str = "study", variant: str = "s"
    ) -> tuple[object, ...]:
        from tracemem.adapters.longmemeval import parse_longmemeval_bytes

        return parse_longmemeval_bytes(self._payload(items), namespace=namespace, variant=variant)

    def _assert_schema_rejected(self, items: object) -> None:
        with self.assertRaisesRegex(
            ValueError, "LongMemEval input does not match the documented schema"
        ):
            self._parse_items(items)

    def test_fictional_fixture_preserves_order_and_separates_reference_metadata(self) -> None:
        from tracemem.adapters.longmemeval import (
            LongMemEvalQuestion,
            LongMemEvalSession,
            parse_longmemeval_bytes,
        )

        fixture_path = (
            Path(__file__).resolve().parents[1] / "examples/synthetic/longmemeval-format.json"
        )
        raw = fixture_path.read_bytes()
        before = raw

        questions = parse_longmemeval_bytes(raw, namespace="study-α", variant="s")

        self.assertIs(type(questions), tuple)
        self.assertEqual(len(questions), 2)
        first, abstention = questions
        self.assertIs(type(first), LongMemEvalQuestion)
        self.assertIs(type(first.sessions[0]), LongMemEvalSession)
        self.assertEqual(first.variant, "s")
        self.assertEqual(first.question_type, "multi-session")
        self.assertFalse(first.is_abstention)
        self.assertEqual(first.records, tuple(first.sessions[0].records))
        self.assertEqual(len(first.sessions), 2)
        self.assertEqual(first.sessions[1].records, ())
        self.assertEqual([record.message_id for record in first.records], ["0", "1", "2"])
        self.assertEqual(
            first.records[0].text,
            "The cobalt key is at café shelf 7.\nPlease keep this note exact.",
        )
        self.assertEqual(first.records[2].text, "The tiny label reads 東京 and 🧪.")
        self.assertEqual(first.records[2].role, "user")
        self.assertEqual(first.records[2].recorded_at, None)
        self.assertIsNone(first.records[2].source_locator)
        self.assertIsNone(first.records[2].line_number)
        self.assertEqual(first.records[2].content_kind, "conversation")
        self.assertEqual(first.answer_session_ids, ("shared-fictional-session",))
        self.assertEqual(first.answer_record_ids, (first.records[2].record_id,))
        self.assertEqual(
            first.records[0].source,
            "longmemeval:"
            + hashlib.sha256(
                json.dumps(["study-α", "s"], ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest(),
        )
        self.assertEqual(
            first.sessions[0].records[0].session_id,
            json.dumps(
                ["fictional-question-1", "shared-fictional-session"],
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )
        self.assertEqual(
            first.sessions[0].upstream_session_id, abstention.sessions[0].upstream_session_id
        )
        self.assertNotEqual(
            first.sessions[0].records[0].session_id, abstention.records[0].session_id
        )
        self.assertEqual(abstention.answer, "")
        self.assertTrue(abstention.is_abstention)
        self.assertEqual(abstention.answer_session_ids, ())
        self.assertEqual(abstention.answer_record_ids, ())
        with self.assertRaises(FrozenInstanceError):
            first.answer = "changed"  # type: ignore[misc]
        with self.assertRaises((AttributeError, TypeError)):
            first.extra = "changed"  # type: ignore[attr-defined]
        self.assertFalse(hasattr(first, "extra"))
        self.assertEqual(fixture_path.read_bytes(), before)

    def test_duplicate_json_keys_are_rejected_with_a_sanitized_error(self) -> None:
        from tracemem.adapters.longmemeval import parse_longmemeval_bytes

        fixture_path = (
            Path(__file__).resolve().parents[1] / "examples/synthetic/longmemeval-format.json"
        )
        raw = fixture_path.read_bytes()
        duplicate_key = raw.replace(b'"role": "user"', b'"role": "user", "role": "user"', 1)
        self.assertNotEqual(duplicate_key, raw)

        with self.assertRaisesRegex(ValueError, "LongMemEval input is not valid JSON") as failure:
            parse_longmemeval_bytes(duplicate_key, namespace="study", variant="s")

        self.assertNotIn("role", str(failure.exception))

    def test_non_finite_json_constants_are_rejected(self) -> None:
        from tracemem.adapters.longmemeval import parse_longmemeval_bytes

        fixture_path = (
            Path(__file__).resolve().parents[1] / "examples/synthetic/longmemeval-format.json"
        )
        raw = fixture_path.read_bytes()
        for constant in (b"NaN", b"Infinity", b"-Infinity"):
            with self.subTest(constant=constant):
                malformed = raw.replace(b'"has_answer": true', b'"has_answer": ' + constant)
                with self.assertRaisesRegex(ValueError, "LongMemEval input is not valid JSON"):
                    parse_longmemeval_bytes(malformed, namespace="study", variant="s")

    def test_invalid_utf8_is_rejected_with_a_sanitized_error(self) -> None:
        from tracemem.adapters.longmemeval import parse_longmemeval_bytes

        fixture_path = (
            Path(__file__).resolve().parents[1] / "examples/synthetic/longmemeval-format.json"
        )
        malformed = fixture_path.read_bytes() + b"\xff"

        with self.assertRaisesRegex(ValueError, "LongMemEval input is not valid UTF-8"):
            parse_longmemeval_bytes(malformed, namespace="study", variant="s")

    def test_escaped_lone_surrogates_are_rejected(self) -> None:
        from tracemem.adapters.longmemeval import parse_longmemeval_bytes

        fixture_path = (
            Path(__file__).resolve().parents[1] / "examples/synthetic/longmemeval-format.json"
        )
        raw = fixture_path.read_bytes()
        malformed = raw.replace("café".encode(), b"\\ud800", 1)
        self.assertNotEqual(malformed, raw)

        with self.assertRaisesRegex(ValueError, "LongMemEval input contains non-scalar Unicode"):
            parse_longmemeval_bytes(malformed, namespace="study", variant="s")

    def test_input_over_64_mib_is_rejected_before_json_parsing(self) -> None:
        from tracemem.adapters.longmemeval import parse_longmemeval_bytes

        oversized = b" " * (64 * 1024 * 1024 + 1)

        with self.assertRaisesRegex(ValueError, "LongMemEval input exceeds 64 MiB limit"):
            parse_longmemeval_bytes(oversized, namespace="study", variant="s")

    def test_raw_input_and_identity_arguments_are_validated(self) -> None:
        from tracemem.adapters.longmemeval import parse_longmemeval_bytes

        with self.assertRaisesRegex(ValueError, "LongMemEval input must be bytes"):
            parse_longmemeval_bytes("[]", namespace="study", variant="s")  # type: ignore[arg-type]
        raw = self._payload(self._fixture_items())
        for namespace in ("", " \t\n"):
            with self.subTest(namespace=namespace):
                with self.assertRaisesRegex(ValueError, "namespace must be a nonblank string"):
                    parse_longmemeval_bytes(raw, namespace=namespace, variant="s")
        for variant in ("", "longmemeval_s", "S"):
            with self.subTest(variant=variant):
                with self.assertRaisesRegex(ValueError, "variant must be one of s, m, oracle"):
                    parse_longmemeval_bytes(raw, namespace="study", variant=variant)
        with self.assertRaisesRegex(ValueError, "namespace must contain scalar Unicode"):
            parse_longmemeval_bytes(raw, namespace="\ud800", variant="s")

    def test_top_level_must_be_a_list_of_exact_question_objects(self) -> None:
        items = self._fixture_items()
        self._assert_schema_rejected({"items": items})
        self._assert_schema_rejected([None])

        missing = self._fixture_items()
        del missing[0]["answer"]
        self._assert_schema_rejected(missing)

        extra = self._fixture_items()
        extra[0]["unexpected"] = "not part of the pinned schema"
        self._assert_schema_rejected(extra)

    def test_question_types_and_question_ids_are_checked(self) -> None:
        invalid_type = self._fixture_items()
        invalid_type[0]["question_type"] = "new-category"
        self._assert_schema_rejected(invalid_type)

        blank_id = self._fixture_items()
        blank_id[0]["question_id"] = "  "
        self._assert_schema_rejected(blank_id)

        duplicate_id = self._fixture_items()
        duplicate_id[1]["question_id"] = duplicate_id[0]["question_id"]
        self._assert_schema_rejected(duplicate_id)

    def test_question_metadata_and_history_fields_require_documented_types(self) -> None:
        mutations = (
            lambda item: item.__setitem__("question", "  "),
            lambda item: item.__setitem__("question_date", 20320403),
            lambda item: item.__setitem__("haystack_session_ids", "not-a-list"),
            lambda item: item["haystack_session_ids"].__setitem__(0, 7),
            lambda item: item.__setitem__("haystack_dates", [None, "date"]),
            lambda item: item.__setitem__("haystack_sessions", "not-a-list"),
            lambda item: item.__setitem__("answer_session_ids", [None]),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                items = self._fixture_items()
                mutate(items[0])
                self._assert_schema_rejected(items)

    def test_integer_answer_is_normalized_to_decimal_string(self) -> None:
        for answer in (0, -7, 42):
            with self.subTest(answer=answer):
                numeric_items = self._fixture_items()
                numeric_items[0]["answer"] = answer
                string_items = self._fixture_items()
                string_items[0]["answer"] = str(answer)

                question = self._parse_items(numeric_items)[0]
                string_question = self._parse_items(string_items)[0]

                self.assertIs(type(question.answer), str)  # type: ignore[attr-defined]
                self.assertEqual(question.answer, str(answer))  # type: ignore[attr-defined]
                self.assertEqual(question.records, string_question.records)  # type: ignore[attr-defined]
                self.assertTrue(
                    all(str(answer) not in record.text for record in question.records)  # type: ignore[attr-defined]
                )

    def test_non_integer_non_string_answers_keep_the_schema_rejection(self) -> None:
        for answer in (False, True, 1.0, None, [], {}):
            with self.subTest(answer=answer):
                items = self._fixture_items()
                items[0]["answer"] = answer
                self._assert_schema_rejected(items)

    def test_string_answer_formatting_is_preserved(self) -> None:
        items = self._fixture_items()
        items[0]["answer"] = "  exact answer \n"

        question = self._parse_items(items)[0]

        self.assertEqual(question.answer, "  exact answer \n")  # type: ignore[attr-defined]

    def test_parallel_lists_and_unique_contained_session_labels_are_checked(self) -> None:
        mismatched = self._fixture_items()
        mismatched[0]["haystack_dates"] = []
        self._assert_schema_rejected(mismatched)

        duplicate_sessions = self._fixture_items()
        duplicate_sessions[0]["haystack_session_ids"] = [
            "same-session",
            "same-session",
        ]
        self._assert_schema_rejected(duplicate_sessions)

        unknown_answer_session = self._fixture_items()
        unknown_answer_session[0]["answer_session_ids"] = ["not-in-history"]
        self._assert_schema_rejected(unknown_answer_session)

        duplicate_answer_labels = self._fixture_items()
        duplicate_answer_labels[0]["answer_session_ids"] = [
            "shared-fictional-session",
            "shared-fictional-session",
        ]
        self._assert_schema_rejected(duplicate_answer_labels)

    def test_session_turn_roles_keys_and_flags_are_strictly_validated(self) -> None:
        bad_role = self._fixture_items()
        bad_role[0]["haystack_sessions"][0][0]["role"] = "tool"
        self._assert_schema_rejected(bad_role)

        extra_turn_key = self._fixture_items()
        extra_turn_key[0]["haystack_sessions"][0][0]["evidence"] = True
        self._assert_schema_rejected(extra_turn_key)

        non_boolean_flag = self._fixture_items()
        non_boolean_flag[0]["haystack_sessions"][0][2]["has_answer"] = 1
        self._assert_schema_rejected(non_boolean_flag)

        invalid_turn_content = self._fixture_items()
        invalid_turn_content[0]["haystack_sessions"][0][0]["content"] = None
        self._assert_schema_rejected(invalid_turn_content)

    def test_parser_failures_for_recursion_and_large_integers_are_sanitized(self) -> None:
        from tracemem.adapters.longmemeval import parse_longmemeval_bytes

        for malformed in (b"[" * 10000 + b"]" * 10000, b"[" + b"9" * 5000 + b"]"):
            with self.subTest(length=len(malformed)):
                with self.assertRaisesRegex(
                    ValueError, "LongMemEval input is not valid JSON"
                ) as failure:
                    parse_longmemeval_bytes(malformed, namespace="study", variant="s")
                self.assertNotIn("RecursionError", str(failure.exception))
                self.assertNotIn("5000", str(failure.exception))

    def test_empty_history_is_preserved_and_evidence_flags_are_not_inferred(self) -> None:
        items = self._fixture_items()
        items[0]["haystack_session_ids"] = []
        items[0]["haystack_dates"] = []
        items[0]["haystack_sessions"] = []
        items[0]["answer_session_ids"] = []
        items[1]["haystack_sessions"][0][0].pop("has_answer", None)

        questions = self._parse_items(items)

        self.assertEqual(questions[0].sessions, ())  # type: ignore[attr-defined]
        self.assertEqual(questions[0].records, ())  # type: ignore[attr-defined]
        self.assertEqual(questions[0].answer_record_ids, ())  # type: ignore[attr-defined]
        self.assertEqual(questions[1].answer_session_ids, ())  # type: ignore[attr-defined]
        self.assertEqual(questions[1].answer_record_ids, ())  # type: ignore[attr-defined]

    def test_oracle_sessions_preserve_supplied_nonchronological_order(self) -> None:
        items = self._fixture_items()
        items[0]["haystack_session_ids"].reverse()
        items[0]["haystack_dates"].reverse()
        items[0]["haystack_sessions"].reverse()

        question = self._parse_items(items, variant="oracle")[0]

        self.assertEqual(
            [session.upstream_session_id for session in question.sessions],  # type: ignore[attr-defined]
            ["empty-fictional-session", "shared-fictional-session"],
        )
        self.assertEqual(
            [session.date for session in question.sessions],  # type: ignore[attr-defined]
            ["2032-04-02", "2032-04-01"],
        )
        self.assertEqual(
            [record.message_id for record in question.records],  # type: ignore[attr-defined]
            ["0", "1", "2"],
        )

    def test_answer_session_labels_do_not_infer_turn_evidence_labels(self) -> None:
        items = self._fixture_items()
        items[0]["haystack_sessions"][0][2].pop("has_answer")

        question = self._parse_items(items)[0]

        self.assertEqual(question.answer_session_ids, ("shared-fictional-session",))  # type: ignore[attr-defined]
        self.assertEqual(question.answer_record_ids, ())  # type: ignore[attr-defined]

    def test_reference_metadata_changes_do_not_change_evidence_identity_or_revision(self) -> None:
        original = self._fixture_items()
        changed = self._fixture_items()
        changed[0]["question"] = "A rewritten question unrelated to this source transcript."
        changed[0]["answer"] = "A different expected answer."
        changed[0]["question_type"] = "knowledge-update"
        changed[0]["question_date"] = "2032-04-30"
        changed[0]["answer_session_ids"] = []
        changed[0]["haystack_sessions"][0][2].pop("has_answer")

        before = self._parse_items(original)[0].records  # type: ignore[attr-defined]
        after = self._parse_items(changed)[0].records  # type: ignore[attr-defined]

        self.assertEqual(
            [(record.record_id, record.revision) for record in before],
            [(record.record_id, record.revision) for record in after],
        )

    def test_session_date_change_does_not_become_timestamp_or_revision(self) -> None:
        original = self._fixture_items()
        changed = self._fixture_items()
        changed[0]["haystack_dates"][0] = "2032-04-10"

        before = self._parse_items(original)[0]  # type: ignore[attr-defined]
        after = self._parse_items(changed)[0]  # type: ignore[attr-defined]

        self.assertEqual(after.sessions[0].date, "2032-04-10")
        self.assertEqual(
            [(record.record_id, record.revision, record.recorded_at) for record in before.records],
            [(record.record_id, record.revision, record.recorded_at) for record in after.records],
        )

    def test_evidence_ids_are_scoped_by_question_namespace_and_variant(self) -> None:
        questions = self._parse_items(self._fixture_items(), namespace="study", variant="s")
        other_namespace = self._parse_items(
            self._fixture_items(), namespace="other-study", variant="s"
        )
        other_variant = self._parse_items(self._fixture_items(), namespace="study", variant="m")

        self.assertNotEqual(questions[0].records[0].record_id, questions[1].records[0].record_id)  # type: ignore[attr-defined]
        self.assertNotEqual(
            questions[0].records[0].record_id, other_namespace[0].records[0].record_id
        )  # type: ignore[attr-defined]
        self.assertNotEqual(
            questions[0].records[0].record_id, other_variant[0].records[0].record_id
        )  # type: ignore[attr-defined]

    def test_separate_question_stores_isolate_retrieval_and_resolve_unicode_citations(self) -> None:
        from tracemem.adapters.longmemeval import parse_longmemeval_bytes
        from tracemem.citations import fetch_citation
        from tracemem.retrieval import Retriever
        from tracemem.schema import EvidenceRecord
        from tracemem.storage import EvidenceStore

        questions = parse_longmemeval_bytes(
            (
                Path(__file__).resolve().parents[1] / "examples/synthetic/longmemeval-format.json"
            ).read_bytes(),
            namespace="study",
            variant="s",
        )
        first, second = questions
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "question-1.sqlite"
            second_path = Path(directory) / "question-2.sqlite"
            with (
                EvidenceStore(first_path) as first_store,
                EvidenceStore(second_path) as second_store,
            ):
                first_store.ingest(first.records)
                second_store.ingest(second.records)
                first_retriever = Retriever(first_store)
                second_retriever = Retriever(second_store)

                own_hit = first_retriever.search("cobalt key café shelf", mode="bm25", k=5)
                self.assertTrue(own_hit)
                other_query = "paper kite amber drawer"
                self.assertEqual(first_retriever.search(other_query, mode="bm25", k=5), [])
                self.assertTrue(second_retriever.search(other_query, mode="bm25", k=5))
                first_texts = [record.text for record in first_store.records()]
                self.assertNotIn(first.question, first_texts)
                self.assertNotIn(first.answer, first_texts)
                self.assertNotIn(
                    "A paper kite rests in the amber drawer.",
                    first_texts,
                )

                unicode_hits = first_retriever.search("tiny label 東京", mode="bm25", k=5)
                self.assertTrue(unicode_hits)
                self.assertEqual(unicode_hits[0].record.record_id, first.records[2].record_id)
                citation = unicode_hits[0].citation
                historical = unicode_hits[0].record
                later_revision = EvidenceRecord(
                    source=historical.source,
                    session_id=historical.session_id,
                    message_id=historical.message_id,
                    part_index=historical.part_index,
                    role=historical.role,
                    text=historical.text + " A later fictional note.",
                    recorded_at=historical.recorded_at,
                    source_locator=historical.source_locator,
                    line_number=historical.line_number,
                    content_kind=historical.content_kind,
                )
                first_store.ingest((later_revision,))
                self.assertEqual(first_retriever.search("velvet comet box", mode="bm25", k=5), [])

            with EvidenceStore(first_path) as reopened_store:
                resolved = fetch_citation(reopened_store, citation)
                self.assertEqual(resolved.quote, "The tiny label reads 東京 and 🧪.")
                self.assertEqual(resolved.record.recorded_at, None)
                self.assertEqual(resolved.record.revision, historical.revision)


if __name__ == "__main__":
    unittest.main()
