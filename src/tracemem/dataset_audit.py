"""Deterministic, lexical-only readiness diagnostics for claim datasets."""

from __future__ import annotations

import math
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

_STANDARD_SPLITS = ("train", "dev", "test", "diagnostic")
_LIMITATIONS = [
    "Pairwise candidate enumeration is O(n²) in case count.",
    "Lexical overlap cannot prove entity or user isolation.",
    "Lexical overlap cannot prove template isolation.",
    "This diagnostic does not establish legal permission.",
    "This diagnostic does not assess annotation quality or sample adequacy.",
    "No automatic Gate A approval is granted.",
]
_COMPARISON = {
    "normalization": (
        "Per-message text is Unicode NFKC normalized, casefolded, and has all Unicode "
        "whitespace collapsed to single spaces."
    ),
    "exact_duplicate": (
        "Compare every case pair's ordered sequence of literal message roles and "
        "normalized per-message text, preserving message and role boundaries."
    ),
}
_NEAR_DUPLICATE_COMPARISON: dict[str, object] = {
    "method": "token-set Jaccard similarity",
    "tokenization": (
        "Split each normalized message on whitespace and union tokens across the "
        "conversation; word order, repetitions, roles, and message boundaries are ignored."
    ),
    "scope": "Cross-split only; exact-duplicate pairs are reported separately.",
    "inclusive": True,
}


def audit_cases(
    cases: Sequence[Mapping[str, Any]], *, near_duplicate_threshold: float = 0.8
) -> dict[str, Any]:
    """Report split readiness and cross-split lexical overlap without exposing message text."""
    threshold = _validate_threshold(near_duplicate_threshold)
    if isinstance(cases, (str, bytes)) or not isinstance(cases, Sequence):
        raise TypeError("cases must be a sequence of mappings")
    if not cases:
        raise ValueError("cases must be a non-empty sequence")

    case_ids: set[str] = set()
    scenario_splits: dict[str, set[str]] = defaultdict(set)
    scenario_case_ids: dict[str, list[str]] = defaultdict(list)
    split_case_counts: Counter[str] = Counter()
    scenarios_by_split: dict[str, set[str]] = defaultdict(set)
    no_memory_by_split: Counter[str] = Counter()
    conversations: list[tuple[str, str, tuple[tuple[str, str], ...], frozenset[str]]] = []

    for case in cases:
        if not isinstance(case, Mapping):
            raise TypeError("each case must be a mapping")
        case_id = _required_text(case.get("case_id"), "case_id")
        if case_id in case_ids:
            raise ValueError(f"duplicate case_id: {case_id}")
        case_ids.add(case_id)
        scenario_id = _required_text(case.get("scenario_id"), "scenario_id")
        split = _required_text(case.get("split"), "split")
        messages = case.get("messages")
        if type(messages) is not list or not messages:
            raise ValueError("case messages must be a non-empty list")
        expected = case.get("expected")
        if type(expected) is not list:
            raise ValueError("case expected claims must be a list")

        exact_messages: list[tuple[str, str]] = []
        conversation_tokens: set[str] = set()
        message_ids: set[tuple[str, str]] = set()
        for message in messages:
            if not isinstance(message, Mapping):
                raise TypeError("each message must be a mapping")
            session_id = _required_text(message.get("session_id"), "session_id")
            message_id = _required_text(message.get("message_id"), "message_id")
            identity = (session_id, message_id)
            if identity in message_ids:
                raise ValueError("case contains duplicate session/message IDs")
            message_ids.add(identity)
            role = _required_text(message.get("role"), "message role")
            text = _required_text(message.get("text"), "message text")
            normalized = _normalize_text(text)
            exact_messages.append((role, normalized))
            conversation_tokens.update(normalized.split())

        conversations.append(
            (case_id, split, tuple(exact_messages), frozenset(conversation_tokens))
        )
        split_case_counts[split] += 1
        scenarios_by_split[split].add(scenario_id)
        no_memory_by_split[split] += not expected
        scenario_splits[scenario_id].add(split)
        scenario_case_ids[scenario_id].append(case_id)

    split_names = sorted(set(_STANDARD_SPLITS) | set(split_case_counts))
    all_scenarios = set(scenario_splits)
    scenario_leaks = [
        {
            "scenario_id": scenario_id,
            "splits": sorted(scenario_splits[scenario_id]),
            "case_ids": sorted(scenario_case_ids[scenario_id]),
        }
        for scenario_id in sorted(scenario_splits)
        if len(scenario_splits[scenario_id]) > 1
    ]

    exact_pairs: list[dict[str, Any]] = []
    near_pairs: list[dict[str, Any]] = []
    for index, left in enumerate(conversations):
        for right in conversations[index + 1 :]:
            left_id, left_split, left_messages, left_tokens = left
            right_id, right_split, right_messages, right_tokens = right
            pair_ids = sorted((left_id, right_id))
            pair_splits = sorted({left_split, right_split})
            if left_messages == right_messages:
                exact_pairs.append({"case_ids": pair_ids, "splits": pair_splits})
                continue
            if left_split == right_split:
                continue
            union = left_tokens | right_tokens
            similarity = len(left_tokens & right_tokens) / len(union) if union else 1.0
            if similarity >= threshold:
                near_pairs.append(
                    {"case_ids": pair_ids, "splits": pair_splits, "jaccard": similarity}
                )

    exact_pairs.sort(key=lambda pair: pair["case_ids"])
    near_pairs.sort(key=lambda pair: pair["case_ids"])
    return {
        "case_count": len(cases),
        "split_case_counts": {split: split_case_counts[split] for split in split_names},
        "unique_scenario_count": len(all_scenarios),
        "unique_scenario_counts_by_split": {
            split: len(scenarios_by_split[split]) for split in split_names
        },
        "no_memory_case_count": sum(no_memory_by_split.values()),
        "no_memory_case_counts_by_split": {
            split: no_memory_by_split[split] for split in split_names
        },
        "diagnostic_case_count": split_case_counts["diagnostic"],
        "held_out_split_names": ["test"],
        "held_out_case_count": split_case_counts["test"],
        "scenarios_across_splits": scenario_leaks,
        "exact_duplicate_pairs": exact_pairs,
        "near_duplicate_pairs": near_pairs,
        "comparison": {
            **_COMPARISON,
            "near_duplicate": {**_NEAR_DUPLICATE_COMPARISON, "threshold": threshold},
        },
        "no_memory_definition": "A case whose expected claim list is empty.",
        "limitations": list(_LIMITATIONS),
        "automatic_gate_a_approval": False,
    }


def _validate_threshold(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("near_duplicate_threshold must be a finite number in (0, 1]")
    try:
        threshold = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError("near_duplicate_threshold must be a finite number in (0, 1]") from exc
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("near_duplicate_threshold must be a finite number in (0, 1]")
    return threshold


def _required_text(value: object, field: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field} must be a string")
    if not value.strip():
        raise ValueError(f"{field} must be non-empty")
    return value


def _normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


__all__ = ["audit_cases"]
