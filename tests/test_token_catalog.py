"""Offline tests for the synthetic prompt-count catalog and receipt replay."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from experiments import prepare_synthetic_qa as qa
from experiments import token_catalog
from tracemem.adapters.longmemeval import parse_longmemeval_bytes
from tracemem.retrieval import Retriever, SignedHashEmbedder, _context_line
from tracemem.storage import EvidenceStore


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _utf8_byte_count_test_double(messages: list[dict[str, str]]) -> int:
    """Deterministic fixture counter only; canonical JSON byte lengths are not tokens."""
    return len(_canonical_bytes(messages))


def _expected_candidate_items() -> dict[str, list[dict[str, str]]]:
    raw = qa._FIXTURE_PATH.read_bytes()
    fixture_items, _ = qa._parse_fixture(raw)
    adapted = json.dumps(fixture_items, ensure_ascii=False, separators=(",", ":")).encode()
    questions = parse_longmemeval_bytes(adapted, namespace=qa._NAMESPACE, variant="s")
    expected: dict[str, list[dict[str, str]]] = {}
    for question in questions:
        with tempfile.TemporaryDirectory(prefix="tracemem-token-catalog-test-") as directory:
            with EvidenceStore(Path(directory) / "evidence.sqlite") as store:
                store.ingest(question.records)
                retriever = Retriever(store, SignedHashEmbedder(), rrf_k=60)
                for mode in qa._MODES:
                    hits = retriever.search(question.question, mode=mode, k=5)
                    for mask in range(1 << len(hits)):
                        context = "\n".join(
                            _context_line(hit)
                            for index, hit in enumerate(hits)
                            if mask & (1 << index)
                        )
                        messages = [
                            {"role": "system", "content": qa._SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": qa._USER_TEMPLATE.format(
                                    question_date=question.question_date,
                                    question=question.question,
                                    context=context,
                                ),
                            },
                        ]
                        prompt_id = hashlib.sha256(_canonical_bytes(messages)).hexdigest()
                        prior = expected.setdefault(prompt_id, messages)
                        if prior != messages:
                            raise AssertionError("prompt hash collision in test oracle")
    return expected


def _receipt_for_counts(
    catalog: dict[str, Any],
    counts_by_id: dict[str, int],
    *,
    counter_id: str = "test-only-fixed-count-fixture-not-a-tokenizer",
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "catalog_sha256": hashlib.sha256(_canonical_bytes(catalog)).hexdigest(),
        "counter_id": counter_id,
        "counts": [
            {"prompt_id": prompt_id, "prompt_tokens": counts_by_id[prompt_id]}
            for prompt_id in sorted(counts_by_id)
        ],
    }


class TokenCatalogTests(unittest.TestCase):
    def test_catalog_enumerates_deduplicated_rank_preserving_candidates(self) -> None:
        catalog: dict[str, Any] = token_catalog.build_catalog()

        self.assertEqual(
            set(catalog),
            {
                "schema_version",
                "kind",
                "provenance",
                "fixture_sha256",
                "prompt_sha256",
                "source_sha256",
                "items",
            },
        )
        self.assertEqual(catalog["schema_version"], 1)
        self.assertEqual(catalog["kind"], "tracemem-synthetic-prompt-count-catalog")
        self.assertEqual(catalog["provenance"], "authored-fictional-agent-only")
        self.assertEqual(
            catalog["prompt_sha256"],
            hashlib.sha256(
                _canonical_bytes({"system": qa._SYSTEM_PROMPT, "user": qa._USER_TEMPLATE})
            ).hexdigest(),
        )
        self.assertLessEqual(len(catalog["items"]), 768)
        self.assertEqual(catalog["items"], token_catalog.build_catalog()["items"])

        expected = _expected_candidate_items()
        self.assertEqual(len(expected), 28)
        actual_ids = [item["prompt_id"] for item in catalog["items"]]
        self.assertEqual(actual_ids, sorted(actual_ids))
        self.assertEqual(len(actual_ids), len(set(actual_ids)))
        self.assertEqual(set(actual_ids), set(expected))
        for item in catalog["items"]:
            self.assertEqual(set(item), {"prompt_id", "messages"})
            self.assertEqual(item["messages"], expected[item["prompt_id"]])
            self.assertEqual(
                item["prompt_id"], hashlib.sha256(_canonical_bytes(item["messages"])).hexdigest()
            )
            self.assertEqual(
                [set(message) for message in item["messages"]],
                [{"role", "content"}, {"role", "content"}],
            )

        self.assertEqual(
            catalog["fixture_sha256"],
            hashlib.sha256(qa._FIXTURE_PATH.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            catalog["source_sha256"],
            hashlib.sha256(Path(token_catalog.__file__).read_bytes()).hexdigest(),
        )

    def test_reference_only_sentinel_never_enters_catalog_items(self) -> None:
        fixture = json.loads(qa._FIXTURE_PATH.read_text(encoding="utf-8"))
        sentinel = "REFERENCE_ONLY_SENTINEL_9c7a6b"
        answerable_case = next(
            case for case in fixture["cases"] if not case["item"]["question_id"].endswith("_abs")
        )
        answerable_case["item"]["answer"] = sentinel
        raw = json.dumps(fixture, ensure_ascii=False).encode("utf-8")
        fixture_source = SimpleNamespace(read_bytes=lambda: raw)

        with patch.object(token_catalog, "_FIXTURE_PATH", fixture_source):
            catalog = token_catalog.build_catalog()

        self.assertIn(sentinel, json.dumps(fixture, ensure_ascii=False))
        self.assertNotIn(sentinel, json.dumps(catalog["items"], ensure_ascii=False))
        self.assertTrue(all(set(item) == {"prompt_id", "messages"} for item in catalog["items"]))

    def test_receipt_replay_uses_full_prompt_counts_and_skips_an_oversized_hit(self) -> None:
        catalog = token_catalog.build_catalog()
        _, groups = token_catalog._build_groups()
        target = next(
            group
            for group in groups
            if group.question_id == "qa-correction-lumina" and group.mode == "bm25"
        )
        self.assertEqual(len(target.hits), 2)

        counts_by_id = {
            item["prompt_id"]: _utf8_byte_count_test_double(item["messages"])
            for item in catalog["items"]
        }
        target_contexts = (
            "",
            _context_line(target.hits[0]),
            _context_line(target.hits[1]),
        )
        # Override only this edge case so the second rank fits after the first
        # rank is skipped; the byte-length helper above remains a test double.
        target_counts = (350, 800, 400)
        for context, prompt_tokens in zip(target_contexts, target_counts, strict=True):
            messages = [
                {"role": "system", "content": qa._SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": qa._USER_TEMPLATE.format(
                        question_date=target.question_date,
                        question=target.question,
                        context=context,
                    ),
                },
            ]
            prompt_id = hashlib.sha256(_canonical_bytes(messages)).hexdigest()
            self.assertIn(prompt_id, counts_by_id)
            counts_by_id[prompt_id] = prompt_tokens

        result = token_catalog.pack_from_counts(
            _receipt_for_counts(
                catalog,
                counts_by_id,
                counter_id="test-only-utf8-byte-length-with-skip-overrides-not-tokens",
            ),
            (500,),
        )
        packed = next(
            row
            for row in result["packs_by_split"]["dev"]
            if row["question_id"] == target.question_id and row["mode"] == target.mode
        )
        expected_context = _context_line(target.hits[1])
        self.assertEqual(packed["selected_citation_ids"], [target.hits[1].citation])
        self.assertEqual(packed["prompt_tokens"], 400)
        self.assertEqual(packed["max_prompt_tokens"], 500)
        self.assertEqual(
            packed["messages"][1]["content"],
            qa._USER_TEMPLATE.format(
                question_date=target.question_date,
                question=target.question,
                context=expected_context,
            ),
        )
        self.assertEqual(
            result["counter_id"],
            "test-only-utf8-byte-length-with-skip-overrides-not-tokens",
        )
        self.assertFalse(result["tokenizer_attested"])
        self.assertFalse(result["matched_token_qa"])
        self.assertEqual(len(result["packs_by_split"]["dev"]), 12)
        self.assertEqual(len(result["packs_by_split"]["test"]), 12)
        self.assertTrue(
            all(
                row["split"] == split
                for split, rows in result["packs_by_split"].items()
                for row in rows
            )
        )

    def test_receipt_validation_rejects_bad_bindings_before_packing(self) -> None:
        catalog = token_catalog.build_catalog()
        counts = {item["prompt_id"]: 0 for item in catalog["items"]}
        valid = _receipt_for_counts(catalog, counts)
        malformed: list[dict[str, Any]] = []

        receipt = deepcopy(valid)
        receipt.pop("counts")
        malformed.append(receipt)
        receipt = deepcopy(valid)
        receipt["schema_version"] = True
        malformed.append(receipt)
        receipt = deepcopy(valid)
        receipt["counter_id"] = "  "
        malformed.append(receipt)
        receipt = deepcopy(valid)
        receipt["catalog_sha256"] = "0" * 64
        malformed.append(receipt)
        receipt = deepcopy(valid)
        receipt["counts"].append({"prompt_id": "f" * 64, "prompt_tokens": 0})
        malformed.append(receipt)
        receipt = deepcopy(valid)
        receipt["counts"].pop()
        malformed.append(receipt)
        receipt = deepcopy(valid)
        receipt["counts"].append(deepcopy(receipt["counts"][0]))
        malformed.append(receipt)
        receipt = deepcopy(valid)
        receipt["counts"][0]["prompt_tokens"] = True
        malformed.append(receipt)
        receipt = deepcopy(valid)
        receipt["counts"][0]["prompt_tokens"] = -1
        malformed.append(receipt)
        receipt = deepcopy(valid)
        receipt["unexpected"] = "extra"
        malformed.append(receipt)

        with patch.object(
            token_catalog,
            "pack_context_by_tokens",
            side_effect=AssertionError("packing started before receipt validation"),
        ):
            for invalid in malformed:
                with self.subTest(receipt=invalid), self.assertRaises(ValueError):
                    token_catalog.pack_from_counts(invalid, (100,))

    def test_malformed_budgets_are_rejected_before_catalog_build(self) -> None:
        invalid_budgets = [
            [],
            (),
            (True,),
            (-1,),
            (2049,),
            (1.5,),
            (0, 0),
            ([],),
            tuple(range(9)),
        ]
        with patch.object(
            token_catalog,
            "_build_catalog_state",
            side_effect=AssertionError("catalog built before budget validation"),
        ):
            for budgets in invalid_budgets:
                with self.subTest(budgets=budgets), self.assertRaises(ValueError):
                    token_catalog.pack_from_counts({}, budgets)  # type: ignore[arg-type]

    def test_zero_budget_rejects_positive_empty_prompt_overhead(self) -> None:
        catalog = token_catalog.build_catalog()
        counts = {item["prompt_id"]: 1 for item in catalog["items"]}
        receipt = _receipt_for_counts(
            catalog, counts, counter_id="test-only-constant-one-not-tokenizer"
        )

        with self.assertRaisesRegex(ValueError, "empty prompt uses 1 tokens"):
            token_catalog.pack_from_counts(receipt, (0,))
