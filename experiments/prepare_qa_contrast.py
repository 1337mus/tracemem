"""Prepare a development-only, source-only contrast catalog for authored QA cases."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from experiments.prepare_synthetic_qa import _SYSTEM_PROMPT, _USER_TEMPLATE
from tracemem.adapters.longmemeval import LongMemEvalQuestion, parse_longmemeval_bytes
from tracemem.retrieval import (
    Retriever,
    SignedHashEmbedder,
    _context_line,
    pack_context_with_manifest,
)
from tracemem.storage import EvidenceStore

_PROVENANCE = "authored-fictional-agent-only"
_NAMESPACE = "tracemem-authored-qa-contrast-dev-v1"
_MODES = ("bm25", "vector", "hybrid")
_BYTE_CEILINGS = (0, 512, 1024, 2048)
_TOKEN_CEILINGS = (128, 256, 512, 1024)
_TOP_K = 5
_RRF_K = 60
_MAX_FIXTURE_BYTES = 1024 * 1024
_MAX_CATALOG_ITEMS = 576


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _messages(question: LongMemEvalQuestion, context: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _USER_TEMPLATE.format(
                question_date=question.question_date,
                question=question.question,
                context=context,
            ),
        },
    ]


def _contrast_counts(
    groups: list[dict[str, Any]], references: list[dict[str, Any]]
) -> dict[str, dict[str, int]]:
    by_question: dict[str, list[dict[str, Any]]] = {}
    for group in groups:
        by_question.setdefault(group["question_id"], []).append(group)
    labels_by_id = {row["question_id"]: row for row in references}
    differing_rankings = differing_evidence = differing_coverage = 0
    for question_id, rows in by_question.items():
        rankings = {tuple(row["ranking_citation_ids"]) for row in rows}
        evidence_changes_by_mode = any(
            len(
                {
                    frozenset(pack["selected_citation_ids"])
                    for pack in row["packs"]
                    if pack["byte_ceiling"] > 0
                }
            )
            > 1
            for row in rows
        )
        differing_rankings += len(rankings) > 1
        differing_evidence += evidence_changes_by_mode
        if not labels_by_id[question_id]["is_abstention"]:
            coverage = {
                pack["label_recall"]
                for row in rows
                for pack in row["packs"]
                if pack["byte_ceiling"] > 0
            }
            differing_coverage += len(coverage) > 1
    answerable = sum(not row["is_abstention"] for row in references)
    return {
        "differing_ordered_mode_rankings": {"count": differing_rankings, "denominator": 6},
        "differing_positive_byte_budget_evidence_within_mode": {
            "count": differing_evidence,
            "denominator": 6,
        },
        "differing_retained_label_coverage": {
            "count": differing_coverage,
            "denominator": answerable,
        },
    }


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("fixture contains duplicate object keys")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _load_fixture(raw: bytes) -> tuple[list[dict[str, Any]], tuple[LongMemEvalQuestion, ...]]:
    if type(raw) is not bytes:
        raise ValueError("fixture must be bytes")
    if len(raw) > _MAX_FIXTURE_BYTES:
        raise ValueError("fixture exceeds the 1 MiB input limit")
    try:
        envelope = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise ValueError("fixture is not valid JSON") from None
    if (
        type(envelope) is not dict
        or envelope.keys() != {"schema_version", "provenance", "cases"}
        or type(envelope["schema_version"]) is not int
        or envelope["schema_version"] != 1
        or envelope["provenance"] != _PROVENANCE
        or type(envelope["cases"]) is not list
        or len(envelope["cases"]) != 6
    ):
        raise ValueError("fixture envelope must contain exactly six authored development cases")
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for case in envelope["cases"]:
        if type(case) is not dict or case.keys() != {"split", "item"} or case["split"] != "dev":
            raise ValueError("all cases must be development items with exact fields")
        item = case["item"]
        if type(item) is not dict:
            raise ValueError("case item must be an object")
        question_id = item.get("question_id")
        if type(question_id) is not str or not question_id or question_id in seen:
            raise ValueError("question ids must be unique nonblank strings")
        seen.add(question_id)
        items.append(item)
    try:
        adapter_raw = json.dumps(items, ensure_ascii=False, separators=(",", ":")).encode()
        questions = parse_longmemeval_bytes(adapter_raw, namespace=_NAMESPACE, variant="s")
    except (TypeError, ValueError):
        raise ValueError("fixture items do not match the LongMemEval schema") from None
    if len(questions) != 6:
        raise ValueError("fixture must contain six valid questions")
    unknown = [question for question in questions if question.is_abstention]
    answerable = [question for question in questions if not question.is_abstention]
    if len(unknown) != 2 or len(answerable) != 4:
        raise ValueError("fixture must contain exactly two unknown and four answerable cases")
    item_by_id = {item["question_id"]: item for item in items}
    for question in questions:
        item = item_by_id[question.question_id]
        has_unknown_reference = item["answer"] == "UNKNOWN"
        if question.is_abstention:
            if not has_unknown_reference or question.answer_record_ids:
                raise ValueError("unknown cases require UNKNOWN and no true answer labels")
        elif has_unknown_reference or not question.answer_record_ids:
            raise ValueError("answerable cases require a non-UNKNOWN reference and true labels")
    return items, questions


def prepare_contrast(raw: bytes) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate the frozen development fixture and return catalog, references, and audit."""
    items, questions = _load_fixture(raw)
    if any(len(question.question) > 4096 for question in questions):
        raise ValueError("question exceeds Retriever's 4096-character limit")

    split_by_id = {item["question_id"]: "dev" for item in items}
    fixture_sha256 = _sha256(raw)
    embedder = SignedHashEmbedder()
    catalog_items: dict[str, list[dict[str, str]]] = {}
    references: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []

    for question in questions:
        references.append(
            {
                "question_id": question.question_id,
                "answer": question.answer,
                "question_type": question.question_type,
                "is_abstention": question.is_abstention,
                "authored_answer_record_ids": list(question.answer_record_ids),
            }
        )
        with tempfile.TemporaryDirectory(prefix="tracemem-qa-contrast-") as directory:
            with EvidenceStore(Path(directory) / "evidence.sqlite") as store:
                store.ingest(question.records)
                retriever = Retriever(store, embedder, rrf_k=_RRF_K)
                for mode in _MODES:
                    hits = retriever.search(question.question, mode=mode, k=_TOP_K)
                    ranking_ids = [hit.citation for hit in hits]
                    subset_prompt_ids: list[str] = []
                    for mask in range(1 << len(hits)):
                        context = "\n".join(
                            _context_line(hit)
                            for index, hit in enumerate(hits)
                            if mask & (1 << index)
                        )
                        messages = _messages(question, context)
                        prompt_id = _sha256(_canonical_bytes(messages))
                        previous = catalog_items.setdefault(prompt_id, messages)
                        if previous != messages:
                            raise ValueError("canonical prompt hash collision")
                        subset_prompt_ids.append(prompt_id)

                    relevant = set(question.answer_record_ids)
                    packs: list[dict[str, Any]] = []
                    for ceiling in _BYTE_CEILINGS:
                        packed = pack_context_with_manifest(hits, ceiling)
                        messages = _messages(question, packed.text)
                        prompt_id = _sha256(_canonical_bytes(messages))
                        previous = catalog_items.setdefault(prompt_id, messages)
                        if previous != messages:
                            raise ValueError("canonical prompt hash collision")
                        if question.is_abstention:
                            explicit_count = found_count = 0
                            label_recall = None
                            exclusion_reason = "abstention_has_no_positive_labels"
                        else:
                            explicit_count = len(relevant)
                            found_count = sum(
                                hit.record.record_id in relevant for hit in packed.selected_hits
                            )
                            label_recall = found_count / explicit_count
                            exclusion_reason = None
                        packs.append(
                            {
                                "byte_ceiling": ceiling,
                                "packed_bytes": len(packed.text.encode("utf-8")),
                                "prompt_id": prompt_id,
                                "selected_citation_ids": [
                                    hit.citation for hit in packed.selected_hits
                                ],
                                "explicit_label_count": explicit_count,
                                "found_label_count": found_count,
                                "label_recall": label_recall,
                                "exclusion_reason": exclusion_reason,
                            }
                        )

                    if question.is_abstention:
                        label_metrics = {"recall_at_5": None, "first_relevant_rank": None}
                    else:
                        ranks = [
                            index
                            for index, hit in enumerate(hits, start=1)
                            if hit.record.record_id in relevant
                        ]
                        label_metrics = {
                            "recall_at_5": len(ranks) / len(relevant),
                            "first_relevant_rank": min(ranks) if ranks else None,
                        }
                    groups.append(
                        {
                            "question_id": question.question_id,
                            "split": split_by_id[question.question_id],
                            "mode": mode,
                            "ranking_citation_ids": ranking_ids,
                            "answer_record_ids": list(question.answer_record_ids),
                            "subset_prompt_ids": subset_prompt_ids,
                            "packs": packs,
                            "label_metrics": label_metrics,
                        }
                    )

    if len(catalog_items) > _MAX_CATALOG_ITEMS:
        raise ValueError("authored QA contrast catalog exceeds its fixed candidate bound")
    prompt_sha256 = _sha256(_canonical_bytes({"system": _SYSTEM_PROMPT, "user": _USER_TEMPLATE}))
    catalog = {
        "schema_version": 1,
        "kind": "tracemem-authored-qa-contrast-catalog",
        "provenance": _PROVENANCE,
        "fixture_sha256": fixture_sha256,
        "prompt_sha256": prompt_sha256,
        "source_sha256": _sha256(Path(__file__).read_bytes()),
        "modes": list(_MODES),
        "top_k": _TOP_K,
        "rrf_k": _RRF_K,
        "byte_ceilings": list(_BYTE_CEILINGS),
        "token_ceilings": list(_TOKEN_CEILINGS),
        "token_ceilings_are_configuration_only": True,
        "items": [
            {"prompt_id": prompt_id, "messages": catalog_items[prompt_id]}
            for prompt_id in sorted(catalog_items)
        ],
    }
    reference_bundle = {
        "schema_version": 1,
        "kind": "tracemem-authored-qa-contrast-references",
        "provenance": _PROVENANCE,
        "fixture_sha256": fixture_sha256,
        "catalog_sha256": _sha256(_canonical_bytes(catalog)),
        "questions": references,
    }
    audit = {
        "schema_version": 1,
        "kind": "tracemem-authored-qa-contrast-audit",
        "fixture_sha256": fixture_sha256,
        "catalog_sha256": _sha256(_canonical_bytes(catalog)),
        "question_splits": split_by_id,
        "embedder": embedder.metadata,
        "top_k": _TOP_K,
        "rrf_k": _RRF_K,
        "contrast_counts": _contrast_counts(groups, references),
        "groups": groups,
    }
    return catalog, reference_bundle, audit
