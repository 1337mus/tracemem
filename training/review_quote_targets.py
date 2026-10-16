"""Validate reviewed corrections to authored quote-training targets."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, cast

from training.build_quote_data import build
from training.evaluate_quote_sft import _json_object

JsonMap = dict[str, Any]
_REVIEW_KEYS = {"version", "source_sha256", "annotation_status", "provenance", "families"}
_FAMILY_KEYS = {
    "family_id",
    "original_expected_sha256",
    "disposition",
    "replacement_expected",
    "rationale",
    "required_facts",
}
_FACT_KEYS = {
    "fact_id",
    "description",
    "source_ref",
    "source_quote",
    "target_claim_index",
    "target_quote",
}


def review_seed_targets(source_raw: bytes, review: object) -> dict[str, Any]:
    """Validate fact annotations and return a corrected train seed payload."""
    if type(source_raw) is not bytes:
        raise ValueError("source_raw must be bytes")
    if type(review) is not dict or set(review) != _REVIEW_KEYS:
        raise ValueError("review root has missing or unexpected fields")
    review_map = cast(JsonMap, review)
    if type(review_map["version"]) is not int or review_map["version"] != 1:
        raise ValueError("review version must be integer 1")
    if (
        type(review_map["annotation_status"]) is not str
        or review_map["annotation_status"] != "agent_reviewed_provisional"
    ):
        raise ValueError("review annotation_status is invalid")
    if (
        type(review_map["provenance"]) is not str
        or review_map["provenance"] != "newly_authored_fiction"
    ):
        raise ValueError("review provenance is invalid")

    source_raw_sha256 = hashlib.sha256(source_raw).hexdigest()
    if (
        type(review_map["source_sha256"]) is not str
        or review_map["source_sha256"] != source_raw_sha256
    ):
        raise ValueError("review source_sha256 does not match source bytes")

    source = _json_object(source_raw, "source seeds")
    # Validate both the reviewed input and the eventual output through the existing builder.
    build(source, "train")
    source_families = cast(list[JsonMap], source["families"])
    review_families = review_map["families"]
    if type(review_families) is not list:
        raise ValueError("review families must be a list")

    source_by_id = {family["family_id"]: family for family in source_families}
    review_by_id: dict[str, JsonMap] = {}
    for raw_entry in review_families:
        if type(raw_entry) is not dict or set(raw_entry) != _FAMILY_KEYS:
            raise ValueError("family review has missing or unexpected fields")
        entry = cast(JsonMap, raw_entry)
        family_id = _required_text(entry["family_id"], "family_id")
        if family_id in review_by_id:
            raise ValueError("family IDs must appear exactly once")
        review_by_id[family_id] = entry
    if set(review_by_id) != set(source_by_id) or len(review_by_id) != len(source_by_id):
        raise ValueError("family IDs must appear exactly once")

    corrected = copy.deepcopy(source)
    corrected_by_id = {family["family_id"]: family for family in corrected["families"]}
    disposition_counts = {"retain": 0, "revise": 0, "abstain": 0}
    required_fact_count = 0

    for family_id, entry in review_by_id.items():
        original = source_by_id[family_id]
        original_expected = original["expected"]
        expected_hash = _canonical_sha256(original_expected)
        if (
            type(entry["original_expected_sha256"]) is not str
            or entry["original_expected_sha256"] != expected_hash
        ):
            raise ValueError(f"family {family_id} original_expected_sha256 is stale")
        _required_text(entry["rationale"], "rationale")

        disposition = entry["disposition"]
        replacement = entry["replacement_expected"]
        if type(disposition) is not str or disposition not in disposition_counts:
            raise ValueError("disposition must be retain, revise, or abstain")
        if replacement is not None and type(replacement) is not list:
            raise ValueError("replacement_expected must be a list or null")
        if disposition == "retain":
            if not original_expected or replacement is not None:
                raise ValueError("retain requires non-empty original targets and null replacement")
            effective_expected = original_expected
        elif disposition == "revise":
            if type(replacement) is not list or not replacement or replacement == original_expected:
                raise ValueError("revise requires a differing non-empty replacement_expected")
            effective_expected = cast(list[JsonMap], replacement)
            corrected_by_id[family_id]["expected"] = copy.deepcopy(effective_expected)
        else:
            if original_expected or replacement is not None:
                raise ValueError("abstain requires empty original targets and null replacement")
            effective_expected = original_expected
        disposition_counts[disposition] += 1

        facts = entry["required_facts"]
        if type(facts) is not list:
            raise ValueError("required_facts must be a list")
        if bool(effective_expected) != bool(facts):
            raise ValueError("required_facts must be non-empty exactly when targets are non-empty")
        required_fact_count += len(facts)
        _validate_required_facts(original, effective_expected, facts, family_id)

    corrected_build = build(corrected, "train")
    expanded_claim_count = sum(len(case["expected"]) for case in corrected_build["cases"])
    seed_claim_count = sum(len(family["expected"]) for family in corrected["families"])
    report = {
        "source_raw_sha256": source_raw_sha256,
        "corrected_seed_canonical_sha256": _canonical_sha256(corrected),
        "family_count": len(corrected["families"]),
        "row_count": len(corrected_build["cases"]),
        "claim_count": expanded_claim_count,
        "seed_claim_count": seed_claim_count,
        "required_fact_count": required_fact_count,
        "revised_family_count": disposition_counts["revise"],
        "retained_family_count": disposition_counts["retain"],
        "abstaining_family_count": disposition_counts["abstain"],
        "human_reviewers": 0,
        "limitation": (
            "This checks literal annotated coverage and provenance only; it does not establish "
            "semantic entailment or exhaustive recall."
        ),
    }
    return {"seeds": corrected, "report": report}


def _validate_required_facts(
    source_family: JsonMap,
    expected: list[JsonMap],
    facts: object,
    family_id: str,
) -> None:
    if type(facts) is not list:
        raise ValueError("required_facts must be a list")
    messages = source_family["messages"]
    message_by_ref = {message["ref"]: message for message in messages}
    fact_ids: set[str] = set()
    covered_claims: set[int] = set()

    for raw_fact in facts:
        if type(raw_fact) is not dict or set(raw_fact) != _FACT_KEYS:
            raise ValueError("required fact has missing or unexpected fields")
        fact = cast(JsonMap, raw_fact)
        fact_id = _required_text(fact["fact_id"], "fact_id")
        if fact_id in fact_ids:
            raise ValueError(f"family {family_id} has duplicate fact_id")
        fact_ids.add(fact_id)
        _required_text(fact["description"], "description")
        source_ref = _required_text(fact["source_ref"], "source_ref")
        source_quote = _required_text(fact["source_quote"], "source_quote")
        source_message = message_by_ref.get(source_ref)
        if (
            source_message is None
            or source_message["role"] != "user"
            or source_message["content_kind"] != "conversation"
            or source_message["text"].count(source_quote) != 1
        ):
            raise ValueError("source_quote must be a unique exact user conversation span")

        target_index = fact["target_claim_index"]
        if type(target_index) is not int or target_index < 0 or target_index >= len(expected):
            raise ValueError("target_claim_index must be an in-range integer")
        target_quote = _required_text(fact["target_quote"], "target_quote")
        target = expected[target_index]
        if (
            type(target) is not dict
            or set(target) != {"claim", "message_ref", "quote", "valid_from", "valid_to"}
            or type(target.get("claim")) is not str
            or type(target.get("message_ref")) is not str
            or type(target.get("quote")) is not str
        ):
            raise ValueError("replacement expected claim has missing or invalid fields")
        claim = target["claim"]
        if claim.count(target_quote) != 1:
            raise ValueError("target_quote must occur exactly once in target claim")
        if target["message_ref"] != source_ref or source_quote not in target["quote"]:
            raise ValueError("selected target must cite source_ref and cite source_quote")
        covered_claims.add(target_index)

    if covered_claims != set(range(len(expected))):
        raise ValueError("every non-empty target claim needs at least one required fact")


def _required_text(value: object, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _canonical_sha256(value: object) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
