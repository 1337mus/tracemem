"""Focused controls for the authored-only QA contrast preparation."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments import prepare_qa_contrast, prepare_synthetic_qa
from tracemem.adapters.longmemeval import parse_longmemeval_bytes
from tracemem.retrieval import Retriever, SignedHashEmbedder
from tracemem.storage import EvidenceStore


class QAContrastTests(unittest.TestCase):
    def test_rejects_invalid_fixture_contracts_before_retrieval(self) -> None:
        fixture = Path("examples/synthetic/qa-contrast-dev-v1.json")
        base = json.loads(fixture.read_bytes())
        invalid = []
        duplicate = json.loads(json.dumps(base))
        duplicate["cases"][1]["item"]["question_id"] = duplicate["cases"][0]["item"]["question_id"]
        invalid.append(duplicate)
        non_dev = json.loads(json.dumps(base))
        non_dev["cases"][0]["split"] = "test"
        invalid.append(non_dev)
        unknown_label = json.loads(json.dumps(base))
        item = unknown_label["cases"][4]["item"]
        item["haystack_sessions"][0][0]["has_answer"] = True
        item["answer_session_ids"] = [item["haystack_session_ids"][0]]
        invalid.append(unknown_label)
        answerable_unlabeled = json.loads(json.dumps(base))
        item = answerable_unlabeled["cases"][0]["item"]
        item["answer_session_ids"] = []
        for session in item["haystack_sessions"]:
            for turn in session:
                turn.pop("has_answer", None)
        invalid.append(answerable_unlabeled)
        invalid_schema = json.loads(json.dumps(base))
        del invalid_schema["cases"][0]["item"]["question_date"]
        invalid.append(invalid_schema)
        for index, document in enumerate(invalid):
            with self.subTest(invalid_case=index), self.assertRaises(ValueError):
                prepare_qa_contrast.prepare_contrast(json.dumps(document).encode())

    def test_parser_bounds_input_before_json_and_rejects_bool_or_duplicate_schema(self) -> None:
        with patch.object(prepare_qa_contrast.json, "loads", side_effect=AssertionError):
            for raw in ("not bytes", b"x" * (1024 * 1024 + 1)):
                with self.subTest(raw_type=type(raw)), self.assertRaises(ValueError):
                    prepare_qa_contrast.prepare_contrast(raw)  # type: ignore[arg-type]

        raw = Path("examples/synthetic/qa-contrast-dev-v1.json").read_bytes()
        boolean_version = json.loads(raw)
        boolean_version["schema_version"] = True
        with self.assertRaises(ValueError):
            prepare_qa_contrast.prepare_contrast(json.dumps(boolean_version).encode())

        duplicated_key = raw.replace(
            b'"schema_version": 1',
            b'"schema_version": 1,\n  "schema_version": 1',
            1,
        )
        with self.assertRaises(ValueError):
            prepare_qa_contrast.prepare_contrast(duplicated_key)

    def test_catalog_enumerates_rank_ordered_subsets_without_reference_data(self) -> None:
        fixture = json.loads(Path("examples/synthetic/qa-contrast-dev-v1.json").read_bytes())
        sentinel = "REFERENCE_ONLY_SENTINEL_72da"
        fixture["cases"][0]["item"]["answer"] = sentinel
        catalog, references, audit = prepare_qa_contrast.prepare_contrast(
            json.dumps(fixture, ensure_ascii=False).encode()
        )
        items = {item["prompt_id"]: item["messages"] for item in catalog["items"]}
        self.assertEqual(len(items), len(catalog["items"]))
        self.assertLessEqual(len(items), 576)
        self.assertEqual(list(items), sorted(items))
        self.assertIn(sentinel, json.dumps(references, ensure_ascii=False))
        self.assertNotIn(sentinel, json.dumps(catalog, ensure_ascii=False))
        self.assertNotIn("answer_record_ids", json.dumps(catalog, ensure_ascii=False))
        self.assertNotIn("has_answer", json.dumps(catalog, ensure_ascii=False))

    def test_pack_manifests_obey_utf8_byte_ceilings_and_citation_membership(self) -> None:
        catalog, _, audit = prepare_qa_contrast.prepare_contrast(
            Path("examples/synthetic/qa-contrast-dev-v1.json").read_bytes()
        )
        items = {item["prompt_id"]: item["messages"] for item in catalog["items"]}
        self.assertEqual(catalog["byte_ceilings"], [0, 512, 1024, 2048])
        self.assertEqual(catalog["token_ceilings"], [128, 256, 512, 1024])
        self.assertNotIn("token_counts", json.dumps(catalog))
        for group in audit["groups"]:
            ranking = set(group["ranking_citation_ids"])
            packs = {pack["byte_ceiling"]: pack for pack in group["packs"]}
            self.assertEqual(set(packs), {0, 512, 1024, 2048})
            for ceiling, pack in packs.items():
                messages = items[pack["prompt_id"]]
                context = messages[1]["content"].partition("\nCONTEXT:\n")[2]
                self.assertEqual(pack["packed_bytes"], len(context.encode("utf-8")))
                self.assertLessEqual(pack["packed_bytes"], ceiling)
                self.assertTrue(set(pack["selected_citation_ids"]).issubset(ranking))
                if ceiling == 0:
                    self.assertEqual((context, pack["selected_citation_ids"]), ("", []))

    def test_subset_prompts_match_independent_rank_ordered_rendering(self) -> None:
        raw = Path("examples/synthetic/qa-contrast-dev-v1.json").read_bytes()
        catalog, _, audit = prepare_qa_contrast.prepare_contrast(raw)
        catalog_items = {item["prompt_id"]: item["messages"] for item in catalog["items"]}
        groups = {(row["question_id"], row["mode"]): row for row in audit["groups"]}
        fixture = json.loads(raw)
        items = [case["item"] for case in fixture["cases"]]
        adapted = json.dumps(items, ensure_ascii=False, separators=(",", ":")).encode()
        questions = parse_longmemeval_bytes(
            adapted, namespace=prepare_qa_contrast._NAMESPACE, variant="s"
        )
        for question in questions:
            with tempfile.TemporaryDirectory() as directory:
                with EvidenceStore(Path(directory) / "evidence.sqlite") as store:
                    store.ingest(question.records)
                    retriever = Retriever(store, SignedHashEmbedder(), rrf_k=60)
                    for mode in ("bm25", "vector", "hybrid"):
                        hits = retriever.search(question.question, mode=mode, k=5)
                        expected_ids = []
                        for mask in range(1 << len(hits)):
                            lines = []
                            for index, hit in enumerate(hits):
                                if mask & (1 << index):
                                    record = hit.record
                                    citation = (
                                        f"{record.record_id}@{record.revision}:0:{len(record.text)}"
                                    )
                                    lines.append(
                                        f"{citation} [role={record.role} recorded_at=unknown "
                                        f"content_kind={record.content_kind}] {record.text}"
                                    )
                            messages = [
                                {"role": "system", "content": prepare_synthetic_qa._SYSTEM_PROMPT},
                                {
                                    "role": "user",
                                    "content": (
                                        f"QUESTION_DATE: {question.question_date}\n"
                                        f"QUESTION: {question.question}\nCONTEXT:\n"
                                        + "\n".join(lines)
                                    ),
                                },
                            ]
                            prompt_id = hashlib.sha256(
                                json.dumps(
                                    messages,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                    ensure_ascii=False,
                                    allow_nan=False,
                                ).encode("utf-8")
                            ).hexdigest()
                            expected_ids.append(prompt_id)
                            self.assertEqual(catalog_items[prompt_id], messages)
                        group = groups[(question.question_id, mode)]
                        self.assertEqual(
                            group["ranking_citation_ids"],
                            [
                                f"{hit.record.record_id}@{hit.record.revision}:0:{len(hit.record.text)}"
                                for hit in hits
                            ],
                        )
                        self.assertEqual(group["subset_prompt_ids"], expected_ids)

    def test_pack_label_metrics_summaries_and_bindings_are_independently_correct(self) -> None:
        raw = Path("examples/synthetic/qa-contrast-dev-v1.json").read_bytes()
        first = prepare_qa_contrast.prepare_contrast(raw)
        second = prepare_qa_contrast.prepare_contrast(raw)
        self.assertEqual(first, second)
        catalog, references_bundle, audit = first
        references = {row["question_id"]: row for row in first[1]["questions"]}
        unknown_ids = {qid for qid, row in references.items() if row["is_abstention"]}
        self.assertEqual(len(unknown_ids), 2)
        self.assertEqual(len(references), 6)
        canonical_catalog = json.dumps(
            catalog, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
        catalog_sha256 = hashlib.sha256(canonical_catalog).hexdigest()
        self.assertEqual(references_bundle["catalog_sha256"], catalog_sha256)
        self.assertEqual(audit["catalog_sha256"], catalog_sha256)
        self.assertEqual(audit["top_k"], 5)
        self.assertEqual(audit["rrf_k"], 60)
        self.assertEqual(audit["embedder"], SignedHashEmbedder().metadata)
        answerable_count = len(references) - len(unknown_ids)
        by_question = {qid: [] for qid in references}
        for group in audit["groups"]:
            by_question[group["question_id"]].append(group)
            labels = set(references[group["question_id"]]["authored_answer_record_ids"])
            for pack in group["packs"]:
                selected = {citation.split("@", 1)[0] for citation in pack["selected_citation_ids"]}
                if group["question_id"] in unknown_ids:
                    self.assertEqual(
                        (
                            pack["explicit_label_count"],
                            pack["found_label_count"],
                            pack["label_recall"],
                        ),
                        (0, 0, None),
                    )
                    self.assertEqual(pack["exclusion_reason"], "abstention_has_no_positive_labels")
                else:
                    found = len(labels & selected)
                    self.assertEqual(pack["explicit_label_count"], len(labels))
                    self.assertEqual(pack["found_label_count"], found)
                    self.assertEqual(pack["label_recall"], found / len(labels))
                    self.assertIsNone(pack["exclusion_reason"])

        positive_packs = {
            qid: [pack for group in groups for pack in group["packs"] if pack["byte_ceiling"] > 0]
            for qid, groups in by_question.items()
        }
        ranking_count = sum(
            len({tuple(g["ranking_citation_ids"]) for g in rows}) > 1
            for rows in by_question.values()
        )
        evidence_count = sum(
            any(
                len(
                    {
                        frozenset(pack["selected_citation_ids"])
                        for pack in group["packs"]
                        if pack["byte_ceiling"] > 0
                    }
                )
                > 1
                for group in groups
            )
            for groups in by_question.values()
        )
        coverage_count = sum(
            len({p["label_recall"] for p in packs}) > 1
            for qid, packs in positive_packs.items()
            if qid not in unknown_ids
        )
        summary_counts = {
            "differing_ordered_mode_rankings": (ranking_count, 6),
            "differing_positive_byte_budget_evidence_within_mode": (evidence_count, 6),
            "differing_retained_label_coverage": (coverage_count, answerable_count),
        }
        self.assertEqual(
            audit["contrast_counts"],
            {
                key: {"count": count, "denominator": denominator}
                for key, (count, denominator) in summary_counts.items()
            },
        )

    def test_budget_evidence_contrast_is_within_mode_not_between_rankings(self) -> None:
        groups = [
            {
                "question_id": "fictional-case",
                "mode": mode,
                "ranking_citation_ids": ranking,
                "packs": [
                    {
                        "byte_ceiling": ceiling,
                        "selected_citation_ids": citations,
                        "label_recall": 1.0,
                    }
                    for ceiling, citations in zip((512, 1024, 2048), selections, strict=True)
                ],
            }
            for mode, ranking, selections in (
                ("bm25", ["a", "b"], [["a", "b"], ["a", "b"], ["a", "b"]]),
                ("vector", ["b", "a"], [["b", "a"], ["b", "a"], ["b", "a"]]),
            )
        ]
        references = [{"question_id": "fictional-case", "is_abstention": False}]
        summary = prepare_qa_contrast._contrast_counts(groups, references)
        self.assertEqual(summary["differing_positive_byte_budget_evidence_within_mode"]["count"], 0)
        groups[0]["packs"][1]["selected_citation_ids"] = ["a"]
        summary = prepare_qa_contrast._contrast_counts(groups, references)
        self.assertEqual(summary["differing_positive_byte_budget_evidence_within_mode"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
