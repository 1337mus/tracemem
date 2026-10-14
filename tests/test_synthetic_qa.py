"""Protocol tests for source-only synthetic QA request preparation."""

from __future__ import annotations

import hashlib
import json
import stat
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stderr
from copy import deepcopy
from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import patch

from experiments import prepare_synthetic_qa as qa


class SyntheticQAFixtureTests(unittest.TestCase):
    def test_preparation_returns_request_and_reference_bundles(self) -> None:
        try:
            from experiments.prepare_synthetic_qa import prepare_synthetic_qa
        except ModuleNotFoundError:
            prepare_synthetic_qa: Any = None

        self.assertTrue(callable(prepare_synthetic_qa), "the preparation entrypoint is missing")
        requests, references = prepare_synthetic_qa()
        self.assertEqual(requests["schema_version"], 1)
        self.assertEqual(references["schema_version"], 1)
        self.assertEqual(len(requests["requests"]), 72)
        self.assertEqual(sum(len(rows) for rows in references["questions"].values()), 8)

    def test_labeled_answer_sources_identify_the_asked_entity(self) -> None:
        fixture = Path(__file__).parents[1] / "examples" / "synthetic" / "qa-diagnostic-v1.json"
        envelope = json.loads(fixture.read_text(encoding="utf-8"))
        required_source_text = {
            "qa-correction-lumina": "Lumina's moonstone is teal",
            "qa-dated-corin": "Corin played the bassoon",
            "qa-negation-venn": "Venn did not keep the amber telescope",
            "qa-speaker-nemi": "Nemi said",
            "qa-conditional-sola": "Sola will move the lantern workshop",
            "qa-simple-fact-tavik": "Tavik's favorite tea is jasmine tea",
        }

        for case in envelope["cases"]:
            item = case["item"]
            labeled_text = [
                turn["content"]
                for session in item["haystack_sessions"]
                for turn in session
                if turn.get("has_answer") is True
            ]
            if item["question_id"] in required_source_text:
                self.assertEqual(len(labeled_text), 1, item["question_id"])
                self.assertIn(required_source_text[item["question_id"]], labeled_text[0])
            else:
                self.assertEqual(item["answer"], "UNKNOWN")
                self.assertEqual(labeled_text, [])

    def test_unhashable_split_values_are_rejected_as_invalid_fixture(self) -> None:
        fixture = Path(__file__).parents[1] / "examples" / "synthetic" / "qa-diagnostic-v1.json"
        envelope = json.loads(fixture.read_text(encoding="utf-8"))
        envelope["cases"][0]["split"] = []
        raw = json.dumps(envelope, ensure_ascii=False).encode("utf-8")

        with self.assertRaisesRegex(ValueError, "split or item is invalid"):
            qa._prepare_synthetic_qa_from_bytes(raw)

    def test_reference_sentinels_and_labels_never_enter_answering_messages(self) -> None:
        fixture = Path(__file__).parents[1] / "examples" / "synthetic" / "qa-diagnostic-v1.json"
        envelope = json.loads(fixture.read_text(encoding="utf-8"))
        sentinel = "REFERENCE_ONLY_SENTINEL_71c3"
        envelope["cases"][0]["item"]["answer"] = sentinel
        raw = json.dumps(envelope, ensure_ascii=False).encode("utf-8")

        requests, references = qa._prepare_synthetic_qa_from_bytes(raw)

        self.assertIn(sentinel, json.dumps(references, ensure_ascii=False))
        for row in requests["requests"]:
            self.assertEqual(
                [set(message) for message in row["messages"]], [{"role", "content"}] * 2
            )
            messages_json = json.dumps(row["messages"], ensure_ascii=False)
            self.assertNotIn(sentinel, messages_json)
            for reference_field in ("has_answer", "answer_record_ids", "is_abstention"):
                self.assertNotIn(reference_field, messages_json)

    def test_rankings_are_reused_and_citations_stay_with_their_question(self) -> None:
        fixture = Path(__file__).parents[1] / "examples" / "synthetic" / "qa-diagnostic-v1.json"
        envelope = json.loads(fixture.read_text(encoding="utf-8"))
        adapter_bytes = json.dumps(
            [case["item"] for case in envelope["cases"]], separators=(",", ":")
        ).encode("utf-8")
        parsed = qa.parse_longmemeval_bytes(adapter_bytes, namespace=qa._NAMESPACE, variant="s")
        citations_by_question = {
            question.question_id: {
                f"{record.record_id}@{record.revision}:0:{len(record.text)}"
                for record in question.records
            }
            for question in parsed
        }
        original_search = qa.Retriever.search
        calls: list[tuple[str, str]] = []

        def recording_search(retriever: Any, query: str, mode: str = "hybrid", k: int = 5) -> Any:
            calls.append((query, mode))
            return original_search(retriever, query, mode=mode, k=k)

        with patch.object(qa.Retriever, "search", recording_search):
            requests, _ = qa.prepare_synthetic_qa()

        self.assertEqual(len(calls), 24)
        self.assertEqual(Counter(mode for _, mode in calls), {"bm25": 8, "vector": 8, "hybrid": 8})
        self.assertEqual(len({(query, mode) for query, mode in calls}), 24)
        for row in requests["requests"]:
            context = row["messages"][1]["content"].partition("\nCONTEXT:\n")[2]
            actual_bytes = len(context.encode("utf-8"))
            self.assertEqual(row["packed_bytes"], actual_bytes)
            self.assertLessEqual(actual_bytes, row["byte_ceiling"])
            self.assertTrue(
                set(row["selected_citation_ids"]).issubset(
                    citations_by_question[row["question_id"]]
                )
            )
            if row["byte_ceiling"] == 0:
                self.assertEqual(context, "")
                self.assertEqual(row["selected_citation_ids"], [])

    def test_reference_bundle_matches_canonical_request_hash_and_splits(self) -> None:
        requests, references = qa.prepare_synthetic_qa()
        canonical = json.dumps(
            requests,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        reference_by_id = {
            row["question_id"]: (split, row)
            for split, rows in references["questions"].items()
            for row in rows
        }
        rows_by_id: dict[str, list[dict[str, Any]]] = {}
        for row in requests["requests"]:
            rows_by_id.setdefault(row["question_id"], []).append(row)

        self.assertEqual(references["request_bundle_sha256"], hashlib.sha256(canonical).hexdigest())
        self.assertEqual(references["fixture_sha256"], requests["fixture_sha256"])
        self.assertEqual(len(reference_by_id), 8)
        self.assertEqual(
            {split: len(rows) for split, rows in references["questions"].items()},
            {"dev": 4, "test": 4},
        )
        self.assertEqual(len(rows_by_id), 8)
        self.assertEqual(len({row["request_id"] for row in requests["requests"]}), 72)
        for question_id, rows in rows_by_id.items():
            split, reference = reference_by_id[question_id]
            self.assertEqual(len(rows), 9)
            self.assertTrue(all(row["split"] == split for row in rows))
            self.assertEqual(
                set(reference),
                {
                    "question_id",
                    "answer",
                    "question_type",
                    "is_abstention",
                    "authored_answer_record_ids",
                },
            )

    def test_invalid_budget_and_depth_values_are_rejected_before_fixture_io(self) -> None:
        invalid_budgets = [[], (True,), (0, 0), tuple(range(9)), (-1,), (16385,), (1.5,)]
        invalid_depths = [True, 0, 21, 2.5]
        with patch.object(Path, "read_bytes", side_effect=AssertionError("fixture read too early")):
            for budgets in invalid_budgets:
                with self.subTest(budgets=budgets), self.assertRaises(ValueError):
                    qa.prepare_synthetic_qa(budgets_bytes=budgets)  # type: ignore[arg-type]
            for top_k in invalid_depths:
                with self.subTest(top_k=top_k), self.assertRaises(ValueError):
                    qa.prepare_synthetic_qa(top_k=top_k)  # type: ignore[arg-type]

    def test_fixture_envelope_case_and_adapter_schemas_are_strict(self) -> None:
        fixture = Path(__file__).parents[1] / "examples" / "synthetic" / "qa-diagnostic-v1.json"
        original = json.loads(fixture.read_text(encoding="utf-8"))
        invalid_envelopes = []
        envelope = deepcopy(original)
        envelope["unexpected"] = True
        invalid_envelopes.append(envelope)
        envelope = deepcopy(original)
        envelope["cases"][0]["unexpected"] = True
        invalid_envelopes.append(envelope)
        envelope = deepcopy(original)
        envelope["cases"][0]["split"] = "train"
        invalid_envelopes.append(envelope)
        envelope = deepcopy(original)
        envelope["cases"][0]["item"]["unexpected"] = True
        invalid_envelopes.append(envelope)
        envelope = deepcopy(original)
        envelope["cases"].pop()
        invalid_envelopes.append(envelope)

        for invalid in invalid_envelopes:
            raw = json.dumps(invalid, ensure_ascii=False).encode("utf-8")
            with self.subTest(case_count=len(invalid["cases"])), self.assertRaises(ValueError):
                qa._prepare_synthetic_qa_from_bytes(raw)

    def test_cli_writes_fresh_private_files_and_preserves_existing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request_path = root / "requests.json"
            reference_path = root / "references.json"

            self.assertEqual(
                qa.main(["--requests", str(request_path), "--references", str(reference_path)]),
                0,
            )
            self.assertEqual(stat.S_IMODE(request_path.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(reference_path.stat().st_mode), 0o600)
            request_data = json.loads(request_path.read_text(encoding="utf-8"))
            reference_data = json.loads(reference_path.read_text(encoding="utf-8"))
            self.assertEqual(len(request_data["requests"]), 72)
            self.assertIn("request_bundle_sha256", reference_data)

            request_path.write_bytes(b"keep these bytes")
            reference_path.unlink()
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
                qa.main(["--requests", str(request_path), "--references", str(reference_path)])
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(request_path.read_bytes(), b"keep these bytes")
            self.assertFalse(reference_path.exists())

    def test_cli_refuses_the_same_request_and_reference_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "shared.json"
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
                qa.main(["--requests", str(destination), "--references", str(destination)])

            self.assertEqual(error.exception.code, 2)
            self.assertFalse(destination.exists())
