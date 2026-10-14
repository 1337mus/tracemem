import unittest
from collections.abc import Callable
from typing import cast

from tracemem.retrieval import SearchHit
from tracemem.schema import EvidenceRecord
from tracemem.token_budget import TokenPackedContext, pack_context_by_tokens


def _make_hit(
    citation: str,
    text: str,
    *,
    role: str = "user",
    recorded_at: str | None = None,
    content_kind: str = "conversation",
) -> SearchHit:
    return SearchHit(
        record=EvidenceRecord(
            source="token-packing-test",
            session_id="session",
            message_id=citation,
            part_index=0,
            role=role,
            text=text,
            recorded_at=recorded_at,
            content_kind=content_kind,
        ),
        score=1.0,
        citation=citation,
    )


def _test_counter_with_result(result: object) -> Callable[[str], int]:
    def count_prompt_tokens(context: str) -> int:
        return cast(int, result)

    return count_prompt_tokens


class TokenBudgetPackingTests(unittest.TestCase):
    def test_packs_whole_lines_at_exact_boundary_with_hit_identity_and_provenance(self) -> None:
        from tracemem import retrieval

        first = _make_hit("user@rev:0:1", "a compact fact")
        second = _make_hit("assistant@rev:0:2", "another useful fact", role="assistant")
        expected_lines = (
            "user@rev:0:1 [role=user recorded_at=unknown content_kind=conversation] a compact fact",
            "assistant@rev:0:2 [role=assistant recorded_at=unknown "
            "content_kind=conversation] another useful fact",
        )
        expected_text = "\n".join(expected_lines)
        measured_contexts: list[str] = []

        def test_whitespace_token_counter(context: str) -> int:
            measured_contexts.append(context)
            rendered_prompt = f"test-system {context} test-question"
            return len(rendered_prompt.split())

        exact_budget = test_whitespace_token_counter(expected_text)
        measured_contexts.clear()

        packed = pack_context_by_tokens(
            [first, second],
            exact_budget,
            test_whitespace_token_counter,
            counter_id="test:whitespace-map-v1",
        )

        self.assertEqual(packed.text, expected_text)
        self.assertEqual(len(packed.text.splitlines()), 2)
        self.assertEqual(len(packed.selected_hits), 2)
        self.assertIs(packed.selected_hits[0], first)
        self.assertIs(packed.selected_hits[1], second)
        self.assertEqual(packed.prompt_tokens, exact_budget)
        self.assertEqual(packed.max_prompt_tokens, exact_budget)
        self.assertEqual(packed.counter_id, "test:whitespace-map-v1")
        self.assertEqual(measured_contexts.count(""), 1)
        self.assertEqual(len(measured_contexts), 3)
        self.assertEqual(retrieval.pack_context([first, second], 10_000), expected_text)

    def test_counts_the_full_context_with_shared_prompt_overhead_and_separator(self) -> None:
        from tracemem import retrieval

        first = _make_hit("user@rev:0:3", "shared topic")
        second = _make_hit("user@rev:0:4", "shared answer")
        lines = (retrieval._context_line(first), retrieval._context_line(second))
        combined_context = "\n".join(lines)
        observed_contexts: list[str] = []
        test_token_map = {"<SYSTEM>": 4, "<QUESTION>": 3, "<NL>": 2}

        def test_nonadditive_token_map(context: str) -> int:
            observed_contexts.append(context)
            rendered_prompt = f"<SYSTEM> {context} <QUESTION>"
            prompt_tokens = rendered_prompt.replace("\n", " <NL> ").split()
            return sum(test_token_map.get(token, 1) for token in set(prompt_tokens))

        budget = test_nonadditive_token_map(combined_context)
        base_count = test_nonadditive_token_map("")
        isolated_sum = base_count + sum(
            test_nonadditive_token_map(line) - base_count for line in lines
        )
        self.assertGreater(isolated_sum, budget)
        observed_contexts.clear()

        packed = pack_context_by_tokens(
            [first, second],
            budget,
            test_nonadditive_token_map,
            counter_id="test:shared-token-map-v1",
        )

        self.assertEqual(packed.text, combined_context)
        self.assertEqual(packed.selected_hits, (first, second))
        self.assertEqual(packed.prompt_tokens, budget)
        self.assertIn(combined_context, observed_contexts)

    def test_accepts_a_candidate_when_the_exact_counter_is_nonmonotonic(self) -> None:
        from tracemem import retrieval

        first = _make_hit("user@rev:0:9", "first line")
        second = _make_hit("assistant@rev:0:10", "second line", role="assistant")
        first_line = retrieval._context_line(first)
        combined_context = "\n".join((first_line, retrieval._context_line(second)))
        measured_contexts: list[str] = []
        test_nonmonotonic_map = {"": 2, first_line: 8, combined_context: 3}

        def test_nonmonotonic_counter(context: str) -> int:
            measured_contexts.append(context)
            return test_nonmonotonic_map[context]

        packed = pack_context_by_tokens(
            [first, second],
            8,
            test_nonmonotonic_counter,
            counter_id="test:nonmonotonic-map-v1",
        )

        self.assertEqual(packed.text, combined_context)
        self.assertEqual(packed.selected_hits, (first, second))
        self.assertEqual(packed.prompt_tokens, 3)
        self.assertEqual(measured_contexts, ["", first_line, combined_context])

    def test_skips_an_oversized_ranked_hit_and_tries_later_hits(self) -> None:
        from tracemem import retrieval

        oversized = _make_hit("user@rev:0:5", "oversized evidence " * 30)
        small = _make_hit("assistant@rev:0:6", "small evidence", role="assistant")
        expected_text = retrieval._context_line(small)

        def test_rendered_character_counter(context: str) -> int:
            rendered_prompt = f"<SYSTEM>{context}<QUESTION>"
            return len(rendered_prompt)

        budget = test_rendered_character_counter(expected_text)
        packed = pack_context_by_tokens(
            [oversized, small],
            budget,
            test_rendered_character_counter,
            counter_id="test:rendered-character-map-v1",
        )

        self.assertEqual(packed.text, expected_text)
        self.assertEqual(len(packed.selected_hits), 1)
        self.assertIs(packed.selected_hits[0], small)
        self.assertEqual(packed.prompt_tokens, budget)

    def test_empty_prompt_over_budget_is_rejected_and_empty_hits_can_fit(self) -> None:
        observed_contexts: list[str] = []

        def test_prompt_overhead_counter(context: str) -> int:
            observed_contexts.append(context)
            rendered_prompt = f"<SYSTEM> fixed instructions {context} <QUESTION>"
            return len(rendered_prompt.split())

        empty_prompt_tokens = test_prompt_overhead_counter("")
        observed_contexts.clear()

        with self.assertRaisesRegex(ValueError, "empty prompt.*budget"):
            pack_context_by_tokens(
                [],
                empty_prompt_tokens - 1,
                test_prompt_overhead_counter,
                counter_id="test:prompt-overhead-map-v1",
            )

        valid_empty = pack_context_by_tokens(
            [],
            empty_prompt_tokens,
            test_prompt_overhead_counter,
            counter_id="test:prompt-overhead-map-v1",
        )
        self.assertEqual(valid_empty.text, "")
        self.assertEqual(valid_empty.selected_hits, ())
        self.assertEqual(valid_empty.prompt_tokens, empty_prompt_tokens)
        self.assertEqual(valid_empty.max_prompt_tokens, empty_prompt_tokens)
        self.assertEqual(valid_empty.counter_id, "test:prompt-overhead-map-v1")
        self.assertEqual(observed_contexts, ["", ""])

    def test_rejects_invalid_bounds_counts_and_counter_metadata(self) -> None:
        from tracemem import retrieval

        def test_zero_counter(context: str) -> int:
            return 0

        for invalid_budget in (True, -1, 1.5, "4"):
            with self.subTest(max_prompt_tokens=invalid_budget):
                with self.assertRaisesRegex(ValueError, "max_prompt_tokens"):
                    pack_context_by_tokens(
                        [],
                        cast(int, invalid_budget),
                        test_zero_counter,
                        counter_id="test:zero-map-v1",
                    )

        with self.assertRaisesRegex(TypeError, "callable"):
            pack_context_by_tokens(
                [], 0, cast(Callable[[str], int], None), counter_id="test:zero-map-v1"
            )
        with self.assertRaisesRegex(ValueError, "counter_id"):
            pack_context_by_tokens([], 0, test_zero_counter, counter_id="")
        with self.assertRaisesRegex(TypeError, "counter_id"):
            pack_context_by_tokens([], 0, test_zero_counter, counter_id=cast(str, None))

        for invalid_count in (True, -1, 1.5, "2"):
            with self.subTest(prompt_tokens=invalid_count):
                with self.assertRaisesRegex(ValueError, "token counter.*non-negative integer"):
                    pack_context_by_tokens(
                        [],
                        10,
                        _test_counter_with_result(invalid_count),
                        counter_id="test:invalid-map-v1",
                    )

        first = _make_hit("user@rev:0:7", "accepted first")
        second = _make_hit("user@rev:0:8", "invalid second")
        first_line = retrieval._context_line(first)

        def invalid_count_after_acceptance(context: str) -> int:
            if context in ("", first_line):
                return 0
            return True

        with self.assertRaisesRegex(ValueError, "token counter.*non-negative integer"):
            pack_context_by_tokens(
                [first, second],
                10,
                invalid_count_after_acceptance,
                counter_id="test:invalid-late-map-v1",
            )

        counter_error = RuntimeError("test counter failure")

        def test_raising_counter(context: str) -> int:
            raise counter_error

        with self.assertRaises(RuntimeError) as raised:
            pack_context_by_tokens([], 10, test_raising_counter, counter_id="test:raising-v1")
        self.assertIs(raised.exception, counter_error)

    def test_preserves_legacy_unicode_metadata_lines_without_mutating_or_truncating(self) -> None:
        from dataclasses import FrozenInstanceError

        from tracemem import retrieval

        first = _make_hit(
            "user@rev:0:11",
            "café 🪐 remains exact",
            role="user",
            recorded_at="2026-10-14T09:00:00Z\r\ncontinued",
            content_kind="mixed_untrusted",
        )
        second = _make_hit(
            "tool@rev:0:12",
            "status: ready",
            role="tool",
            recorded_at=None,
            content_kind="tool_output",
        )
        hits = [first, second]
        hits_before = tuple(hits)
        record_values_before = tuple(
            (hit.record.role, hit.record.recorded_at, hit.record.content_kind, hit.record.text)
            for hit in hits
        )
        expected_text = (
            "user@rev:0:11 [role=user recorded_at=2026-10-14T09:00:00Z  continued "
            "content_kind=mixed_untrusted] café 🪐 remains exact\n"
            "tool@rev:0:12 [role=tool recorded_at=unknown content_kind=tool_output] status: ready"
        )
        measured_contexts: list[str] = []

        def test_unicode_prompt_counter(context: str) -> int:
            measured_contexts.append(context)
            rendered_prompt = f"<SYSTEM>{context}<QUESTION>"
            return len(rendered_prompt)

        exact_budget = test_unicode_prompt_counter(expected_text)
        measured_contexts.clear()
        packed = pack_context_by_tokens(
            hits,
            exact_budget,
            test_unicode_prompt_counter,
            counter_id="test:unicode-character-map-v1",
        )

        self.assertEqual(packed.text, expected_text)
        self.assertEqual(retrieval.pack_context(hits, 10_000), expected_text)
        self.assertEqual(packed.selected_hits, (first, second))
        self.assertIs(packed.selected_hits[0], first)
        self.assertIs(packed.selected_hits[1], second)
        self.assertEqual(tuple(hits), hits_before)
        self.assertEqual(
            tuple(
                (hit.record.role, hit.record.recorded_at, hit.record.content_kind, hit.record.text)
                for hit in hits
            ),
            record_values_before,
        )
        self.assertEqual(measured_contexts, ["", retrieval._context_line(first), expected_text])

        invalid_citation = SearchHit(
            record=first.record, score=first.score, citation=cast(str, None)
        )
        with self.assertRaisesRegex(TypeError, "string citations and evidence metadata"):
            retrieval.pack_context([invalid_citation], 10_000)

        no_evidence = pack_context_by_tokens(
            hits,
            0,
            lambda context: 0 if not context else 1,
            counter_id="test:zero-budget-map-v1",
        )
        self.assertEqual(no_evidence.text, "")
        self.assertEqual(no_evidence.selected_hits, ())
        self.assertEqual(no_evidence.prompt_tokens, 0)
        self.assertNotIn("café", no_evidence.text)

        frozen_result = TokenPackedContext("", (), 0, 0, "test:frozen-v1")
        with self.assertRaises(FrozenInstanceError):
            frozen_result.text = "mutated"  # type: ignore[misc]
        self.assertFalse(hasattr(frozen_result, "__dict__"))
