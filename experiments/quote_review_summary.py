"""Local deterministic tallies for blinded quote-v4 semantic reviews."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

ARMS = ("base", "prior", "tuned")
COHORTS = ("dev", "test")
LABELS = ("supported", "unsupported", "uncertain")
PACKET_FIELDS = {"task_id", "messages", "raw_text", "response_text", "truncated", "support_policy"}
CURATOR_FIELDS = {"task_id", "cohort", "case_id", "arm", "source_sha256"}


def _json_bytes(value: Any, compact: bool = False) -> bytes:
    if compact:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    else:
        serialized = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)
    return (serialized + ("" if compact else "\n")).encode("utf-8")


def _rows_by_id(value: Any, name: str) -> dict[str, dict[str, Any]]:
    if type(value) is not list or any(type(row) is not dict for row in value):
        raise ValueError(f"{name} rows must be a list of objects")
    ids = [row.get("task_id") for row in value]
    if any(type(task_id) is not str or not task_id for task_id in ids):
        raise ValueError(f"{name} rows must contain nonempty task ids")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{name} task ids must be unique")
    return dict(zip(ids, value, strict=True))


def _loads_strict(text: str) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"invalid JSON constant: {value}")

    return json.loads(text, object_pairs_hook=unique, parse_constant=reject_constant)


def _response_claims(text: str) -> tuple[bool, list[str] | None]:
    candidate = text
    fence = re.fullmatch(r"\s*```json\s*\n(.*?)\n```\s*", text, flags=re.DOTALL)
    if fence:
        candidate = fence.group(1)
    try:
        parsed = _loads_strict(candidate)
    except (json.JSONDecodeError, ValueError):
        return False, None
    values = parsed.get("claims") if type(parsed) is dict else None
    if type(values) is not list:
        return True, None
    claims: list[str] = []
    for item in values:
        if type(item) is str:
            claim = item
        elif type(item) is dict:
            claim_value = item.get("claim")
            if type(claim_value) is not str:
                return True, None
            claim = claim_value
        else:
            return True, None
        if not claim.strip():
            return True, None
        claims.append(claim)
    return True, claims


def _strict_empty_response(text: str) -> bool:
    try:
        parsed = _loads_strict(text)
    except (json.JSONDecodeError, ValueError):
        return False
    return (
        type(parsed) is dict
        and set(parsed) == {"claims"}
        and type(parsed["claims"]) is list
        and not parsed["claims"]
    )


def summarize(packet: Any, curator_map: Any, blind_review: Any, coverage: Any) -> dict[str, Any]:
    """Validate synthetic review inputs and return counts by cohort and arm."""
    if type(packet) is not list or type(curator_map) is not dict:
        raise ValueError("packet must be a list and curator map an object")
    try:
        packet_hash = hashlib.sha256(_json_bytes(packet)).hexdigest()
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ValueError("packet cannot be serialized for packet hash validation") from error
    if curator_map.get("packet_sha256") != packet_hash:
        raise ValueError("curator packet hash does not match packet bytes")

    packet_rows = _rows_by_id(packet, "packet")
    mapping = _rows_by_id(curator_map.get("tasks"), "curator")
    if set(packet_rows) != set(mapping):
        raise ValueError("packet and curator task ids must be a bijection")
    for task in packet_rows.values():
        if (
            set(task) != PACKET_FIELDS
            or any(
                type(task[key]) is not str
                for key in ("raw_text", "response_text", "support_policy")
            )
            or type(task["truncated"]) is not bool
        ):
            raise ValueError("packet task fields do not match the builder contract")
    case_arms: dict[tuple[str, str], list[str]] = {}
    cohorts: set[str] = set()
    for task_id, meta in mapping.items():
        if set(meta) != CURATOR_FIELDS:
            raise ValueError("curator task fields do not match the builder contract")
        cohort, case_id, arm = meta["cohort"], meta["case_id"], meta["arm"]
        if cohort not in COHORTS or type(case_id) is not str or not case_id or arm not in ARMS:
            raise ValueError("curator cohort, case id, or arm is invalid")
        cohorts.add(cohort)
        case_arms.setdefault((cohort, case_id), []).append(arm)
        task = packet_rows[task_id]
        try:
            hashes = {
                "messages": hashlib.sha256(_json_bytes(task["messages"], True)).hexdigest(),
                "raw_text": hashlib.sha256(task["raw_text"].encode()).hexdigest(),
                "response_text": hashlib.sha256(task["response_text"].encode()).hexdigest(),
            }
        except (TypeError, ValueError, UnicodeEncodeError) as error:
            raise ValueError(
                "packet source cannot be serialized for source hash validation"
            ) from error
        if meta["source_sha256"] != hashes:
            raise ValueError(f"source hash does not match packet task {task_id}")
    if cohorts != set(COHORTS):
        raise ValueError("curator map must include dev and test cohorts")
    if any(len(arms) != len(ARMS) or set(arms) != set(ARMS) for arms in case_arms.values()):
        raise ValueError("each case must map once to every arm")
    if (
        type(blind_review) is not dict
        or type(blind_review.get("human_reviewers")) is not int
        or blind_review["human_reviewers"] != 0
    ):
        raise ValueError("human_reviewers must be integer zero")
    reviews = _rows_by_id(blind_review.get("tasks"), "review")
    if set(reviews) != set(mapping):
        raise ValueError("review task ids must uniquely match curator task ids")
    for task_id, row in reviews.items():
        packet_row = packet_rows[task_id]
        if type(row.get("readable")) is not bool or type(row.get("truncated")) is not bool:
            raise ValueError("review readable and truncated fields must be booleans")
        if row["truncated"] != packet_row["truncated"]:
            raise ValueError(f"review truncated flag must copy packet task {task_id}")
        if (
            "source_offers_eligible_facts" in row
            and row["source_offers_eligible_facts"] is not None
            and type(row["source_offers_eligible_facts"]) is not bool
        ):
            raise ValueError("source_offers_eligible_facts must be boolean or null")
        if "quote_context_note" in row and (
            type(row["quote_context_note"]) is not str or not row["quote_context_note"].strip()
        ):
            raise ValueError("quote_context_note must be nonempty text")
        if (
            "manual_alignment_acknowledged" in row
            and type(row["manual_alignment_acknowledged"]) is not bool
        ):
            raise ValueError("manual_alignment_acknowledged must be boolean")
        claims = row.get("claim_reviews")
        if type(claims) is not list:
            raise ValueError("claim_reviews must be a list")
        for index, claim in enumerate(claims):
            if (
                type(claim) is not dict
                or type(claim.get("claim_index")) is not int
                or claim["claim_index"] != index
            ):
                raise ValueError("claim indices must be sequential starting at zero")
            if type(claim.get("claim")) is not str or not claim["claim"].strip():
                raise ValueError("claim text must be nonempty")
            if claim.get("label") not in LABELS:
                raise ValueError("claim label is invalid")
            if type(claim.get("reason")) is not str or not claim["reason"].strip():
                raise ValueError("reason must be nonempty text")
            if "quote_context_note" in claim and (
                type(claim["quote_context_note"]) is not str
                or not claim["quote_context_note"].strip()
            ):
                raise ValueError("quote_context_note must be nonempty text")
        parseable, response_claims = _response_claims(packet_row["response_text"])
        if not parseable:
            if row.get("manual_alignment_acknowledged") is not True:
                raise ValueError(f"manual alignment acknowledgement required for task {task_id}")
        elif response_claims is None:
            raise ValueError(f"parseable response claims cannot be aligned for task {task_id}")
        elif len(claims) != len(response_claims) or any(
            claim["claim"] != response_claims[index] for index, claim in enumerate(claims)
        ):
            raise ValueError(f"every response claim must have one exact review for task {task_id}")

    cov = _rows_by_id(coverage, "coverage")
    if set(cov) != set(mapping):
        raise ValueError("coverage task ids must uniquely match curator task ids")
    for task_id, row in cov.items():
        expected, matched, complete = (
            row.get("expected_fact_count"),
            row.get("matched_fact_indices"),
            row.get("complete_required_facts"),
        )
        if (
            type(expected) is not int
            or expected < 0
            or type(complete) is not bool
            or type(matched) is not list
            or any(type(index) is not int for index in matched)
        ):
            raise ValueError("coverage counts and indices have invalid types")
        if len(matched) != len(set(matched)) or any(
            index < 0 or index >= expected for index in matched
        ):
            raise ValueError("coverage indices must be distinct and in range")
        if complete != (expected > 0 and len(matched) == expected):
            raise ValueError("coverage completeness is inconsistent with matched facts")
        if complete and not reviews[task_id]["readable"]:
            raise ValueError("unreadable response cannot have complete coverage")
    result: dict[str, Any] = {"cohorts": {}}
    paired: dict[str, dict[str, Any]] = {}
    for cohort in COHORTS:
        result["cohorts"][cohort] = {"arms": {}}
        for arm in ARMS:
            ids = [
                task_id
                for task_id, meta in mapping.items()
                if meta["cohort"] == cohort and meta["arm"] == arm
            ]
            labels = {label: 0 for label in LABELS}
            adverse = complete = nonempty = matched_sum = expected_sum = abstentions = (
                unreadable
            ) = truncated = 0
            for task_id in ids:
                row, coverage_row = reviews[task_id], cov[task_id]
                task_claims = row["claim_reviews"]
                for item in task_claims:
                    labels[item["label"]] += 1
                adverse += any(
                    item["label"] in ("unsupported", "uncertain") for item in task_claims
                )
                expected = coverage_row["expected_fact_count"]
                complete += coverage_row["complete_required_facts"]
                if expected:
                    nonempty += 1
                    matched_sum += len(coverage_row["matched_fact_indices"])
                    expected_sum += expected
                abstentions += (
                    row["readable"]
                    and expected == 0
                    and _strict_empty_response(packet_rows[task_id]["response_text"])
                )
                unreadable += not row["readable"]
                truncated += row["truncated"]
            result["cohorts"][cohort]["arms"][arm] = {
                "responses": len(ids),
                "adverse_responses": adverse,
                "label_counts": labels,
                "complete_required_facts": complete,
                "nonempty_coverage": {
                    "responses": nonempty,
                    "matched_facts": matched_sum,
                    "expected_facts": expected_sum,
                },
                "abstentions": abstentions,
                "unreadable": unreadable,
                "truncated": truncated,
            }

        outcomes = {
            key: 0
            for key in ("pairs", "tuned_wins", "tuned_losses", "both_complete", "neither_complete")
        }
        partial = {key: 0 for key in ("pairs", "tuned_wins", "prior_wins", "ties")}
        cases: dict[tuple[str, str], dict[str, dict[str, Any]]] = {
            (meta["cohort"], meta["case_id"]): {} for meta in mapping.values()
        }
        for task_id, meta in mapping.items():
            cases[(meta["cohort"], meta["case_id"])][meta["arm"]] = cov[task_id]
        for (case_cohort, _), arms in cases.items():
            if case_cohort != cohort:
                continue
            expected_counts = {row["expected_fact_count"] for row in arms.values()}
            if len(expected_counts) != 1:
                raise ValueError("paired arms must have the same expected fact count")
            if not next(iter(expected_counts)):
                continue
            outcomes["pairs"] += 1
            tuned, prior = arms["tuned"], arms["prior"]
            tc, pc = tuned["complete_required_facts"], prior["complete_required_facts"]
            if tc and pc:
                outcomes["both_complete"] += 1
            elif tc:
                outcomes["tuned_wins"] += 1
            elif pc:
                outcomes["tuned_losses"] += 1
            else:
                outcomes["neither_complete"] += 1
                partial["pairs"] += 1
                tm, pm = len(tuned["matched_fact_indices"]), len(prior["matched_fact_indices"])
                partial["tuned_wins" if tm > pm else "prior_wins" if pm > tm else "ties"] += 1
        paired[cohort] = {"complete_case_outcomes": outcomes, "partial_fact_outcomes": partial}
    result["paired_coverage"] = paired
    return result
