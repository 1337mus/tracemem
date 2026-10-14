"""Prepare and replay an offline catalog of synthetic full-prompt counts."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.prepare_synthetic_qa import (
    _FIXTURE_PATH,
    _MODES,
    _NAMESPACE,
    _SYSTEM_PROMPT,
    _USER_TEMPLATE,
    _parse_fixture,
)
from tracemem.adapters.longmemeval import parse_longmemeval_bytes
from tracemem.retrieval import Retriever, SearchHit, SignedHashEmbedder, _context_line
from tracemem.storage import EvidenceStore
from tracemem.token_budget import pack_context_by_tokens

_TOP_K = 5
_RRF_K = 60
_MAX_PROMPTS = 768
_KIND = "tracemem-synthetic-prompt-count-catalog"
_PROVENANCE = "authored-fictional-agent-only"


@dataclass(frozen=True, slots=True)
class _Group:
    question_id: str
    split: str
    mode: str
    question_date: str
    question: str
    hits: tuple[SearchHit, ...]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _messages_for_context(group: _Group, context: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _USER_TEMPLATE.format(
                question_date=group.question_date,
                question=group.question,
                context=context,
            ),
        },
    ]


def _prompt_id(messages: list[dict[str, str]]) -> str:
    return _sha256(_canonical_bytes(messages))


def _build_groups() -> tuple[bytes, list[_Group]]:
    fixture = _FIXTURE_PATH.read_bytes()
    items, split_by_id = _parse_fixture(fixture)
    adapter_bytes = json.dumps(items, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    questions = parse_longmemeval_bytes(adapter_bytes, namespace=_NAMESPACE, variant="s")
    if any(len(question.question) > 4096 for question in questions):
        raise ValueError("question exceeds Retriever's 4096-character limit")

    groups: list[_Group] = []
    for question in questions:
        with tempfile.TemporaryDirectory(prefix="tracemem-token-catalog-") as directory:
            with EvidenceStore(Path(directory) / "evidence.sqlite") as store:
                store.ingest(question.records)
                retriever = Retriever(store, SignedHashEmbedder(), rrf_k=_RRF_K)
                for mode in _MODES:
                    hits = retriever.search(question.question, mode=mode, k=_TOP_K)
                    groups.append(
                        _Group(
                            question_id=question.question_id,
                            split=split_by_id[question.question_id],
                            mode=mode,
                            question_date=question.question_date,
                            question=question.question,
                            hits=tuple(hits),
                        )
                    )
    return fixture, groups


def _build_catalog_state() -> tuple[dict[str, Any], list[_Group]]:
    fixture, groups = _build_groups()
    items_by_id: dict[str, list[dict[str, str]]] = {}
    for group in groups:
        for mask in range(1 << len(group.hits)):
            context = "\n".join(
                _context_line(hit) for index, hit in enumerate(group.hits) if mask & (1 << index)
            )
            messages = _messages_for_context(group, context)
            prompt_id = _prompt_id(messages)
            previous = items_by_id.setdefault(prompt_id, messages)
            if previous != messages:
                raise ValueError("canonical prompt hash collision")

    if len(items_by_id) > _MAX_PROMPTS:
        raise ValueError("synthetic prompt catalog exceeds its fixed candidate bound")
    catalog = {
        "schema_version": 1,
        "kind": _KIND,
        "provenance": _PROVENANCE,
        "fixture_sha256": _sha256(fixture),
        "prompt_sha256": _sha256(
            _canonical_bytes({"system": _SYSTEM_PROMPT, "user": _USER_TEMPLATE})
        ),
        "source_sha256": _sha256(Path(__file__).read_bytes()),
        "items": [
            {"prompt_id": prompt_id, "messages": items_by_id[prompt_id]}
            for prompt_id in sorted(items_by_id)
        ],
    }
    return catalog, groups


def build_catalog() -> dict[str, Any]:
    """Build the fixed fictional prompt catalog without counting or uploading it."""
    return _build_catalog_state()[0]


def _validate_budgets(budgets: tuple[int, ...]) -> None:
    if (
        type(budgets) is not tuple
        or not 1 <= len(budgets) <= 8
        or any(type(budget) is not int or not 0 <= budget <= 2048 for budget in budgets)
        or len(set(budgets)) != len(budgets)
    ):
        raise ValueError("budgets must be a distinct tuple of 1 to 8 integers from 0 to 2048")


def _validate_receipt(
    receipt: dict[str, Any], catalog: dict[str, Any]
) -> tuple[str, dict[str, int], str]:
    expected_keys = {"schema_version", "catalog_sha256", "counter_id", "counts"}
    if type(receipt) is not dict or set(receipt) != expected_keys:
        raise ValueError("count receipt must match the exact schema")
    if type(receipt["schema_version"]) is not int or receipt["schema_version"] != 1:
        raise ValueError("count receipt schema_version must be integer 1")
    if type(receipt["counter_id"]) is not str or not receipt["counter_id"].strip():
        raise ValueError("count receipt counter_id must be a nonblank string")

    catalog_sha256 = _sha256(_canonical_bytes(catalog))
    if type(receipt["catalog_sha256"]) is not str or receipt["catalog_sha256"] != catalog_sha256:
        raise ValueError("count receipt catalog_sha256 does not match this catalog")
    if type(receipt["counts"]) is not list:
        raise ValueError("count receipt counts must be a list")

    known_ids = {item["prompt_id"] for item in catalog["items"]}
    counts_by_id: dict[str, int] = {}
    for item in receipt["counts"]:
        if type(item) is not dict or set(item) != {"prompt_id", "prompt_tokens"}:
            raise ValueError("each receipt count must contain exactly prompt_id and prompt_tokens")
        prompt_id = item["prompt_id"]
        prompt_tokens = item["prompt_tokens"]
        if type(prompt_id) is not str or prompt_id not in known_ids:
            raise ValueError("count receipt contains an unknown prompt_id")
        if type(prompt_tokens) is not int or prompt_tokens < 0:
            raise ValueError("prompt_tokens must be a non-negative integer")
        if prompt_id in counts_by_id:
            raise ValueError("count receipt contains a duplicate prompt_id")
        counts_by_id[prompt_id] = prompt_tokens
    if counts_by_id.keys() != known_ids:
        raise ValueError("count receipt is missing catalog prompt ids")
    return receipt["counter_id"], counts_by_id, catalog_sha256


def _count_lookup(group: _Group, counts_by_id: dict[str, int]) -> Callable[[str], int]:
    def count_prompt_tokens(context: str) -> int:
        prompt_id = _prompt_id(_messages_for_context(group, context))
        try:
            return counts_by_id[prompt_id]
        except KeyError:
            raise ValueError("catalog does not contain a proposed packed prompt") from None

    return count_prompt_tokens


def pack_from_counts(receipt: dict[str, Any], budgets: tuple[int, ...]) -> dict[str, Any]:
    """Replay caller-supplied full-prompt counts without making an inference call."""
    _validate_budgets(budgets)
    catalog, groups = _build_catalog_state()
    counter_id, counts_by_id, catalog_sha256 = _validate_receipt(receipt, catalog)

    packs_by_split: dict[str, list[dict[str, Any]]] = {"dev": [], "test": []}
    for group in groups:
        for budget in budgets:
            count_prompt_tokens = _count_lookup(group, counts_by_id)
            packed = pack_context_by_tokens(
                group.hits,
                budget,
                count_prompt_tokens,
                counter_id=counter_id,
            )
            packs_by_split[group.split].append(
                {
                    "question_id": group.question_id,
                    "split": group.split,
                    "mode": group.mode,
                    "max_prompt_tokens": packed.max_prompt_tokens,
                    "prompt_tokens": packed.prompt_tokens,
                    "messages": _messages_for_context(group, packed.text),
                    "selected_citation_ids": [hit.citation for hit in packed.selected_hits],
                }
            )

    return {
        "schema_version": 1,
        "kind": "tracemem-synthetic-token-count-replay",
        "provenance": _PROVENANCE,
        "catalog_sha256": catalog_sha256,
        "counter_id": counter_id,
        "count_origin": "caller-supplied; not independently verified",
        "tokenizer_attested": False,
        "matched_token_qa": False,
        "packs_by_split": packs_by_split,
    }
