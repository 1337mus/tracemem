"""Replay pinned fictional QA contrast token counts through shared context packing."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from experiments import modal_qa_contrast_token_count as token_runner
from experiments import prepare_qa_contrast
from tracemem.retrieval import Retriever, SignedHashEmbedder
from tracemem.storage import EvidenceStore
from tracemem.token_budget import pack_context_by_tokens

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE = _ROOT / "examples" / "synthetic" / "qa-contrast-dev-v1.json"
_DEFAULT_RECEIPT = _ROOT / "examples" / "synthetic" / "qwen3-4b-qa-contrast-token-count-remote.json"
_FIXTURE_SHA256 = "b4fb6ea6242c294574ca9ed15549f5b524a77d3154b4c87b9831d2f357beefed"
_RECEIPT_SHA256 = "a48defa04fad82435a979980a3d327232b83260adb7eafe42aa165bef35400bd"
_MAX_RECEIPT_BYTES = 1024 * 1024
_MAX_BUDGETS = 16
_MAX_BUDGET = 1_000_000
_DEFAULT_BUDGETS = (128, 256, 512, 1024)
_MODES = ("bm25", "vector", "hybrid")
_TOP_K = 5
_RRF_K = 60


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _validated_receipt(receipt_raw: bytes) -> tuple[dict[str, Any], dict[str, int], str]:
    if type(receipt_raw) is not bytes:
        raise ValueError("receipt must be bytes")
    if len(receipt_raw) > _MAX_RECEIPT_BYTES:
        raise ValueError("receipt exceeds the 1 MiB input limit")
    receipt_sha256 = _sha256(receipt_raw)
    if receipt_sha256 != _RECEIPT_SHA256:
        raise ValueError("receipt does not match the exact committed fictional receipt")

    try:
        envelope = json.loads(receipt_raw)
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise ValueError("receipt is not valid JSON") from None
    if type(envelope) is not dict or envelope.keys() != {
        "schema_version",
        "payload_sha256",
        "runner_sha256",
        "client_elapsed_seconds",
        "result",
    }:
        raise ValueError("receipt envelope fields are invalid")
    if type(envelope["schema_version"]) is not int or envelope["schema_version"] != 1:
        raise ValueError("receipt schema_version must be integer 1")

    payload = token_runner.build_payload()
    payload_sha256 = _sha256(_canonical_bytes(payload))
    runner_sha256 = _sha256(Path(token_runner.__file__).read_bytes())
    if (
        envelope["payload_sha256"] != payload_sha256
        or payload_sha256 != token_runner._EXPECTED_PAYLOAD_SHA256
        or envelope["runner_sha256"] != runner_sha256
    ):
        raise ValueError("receipt payload or runner provenance does not match current source")
    result = token_runner._validate_remote_result(envelope["result"], payload, runner_sha256)
    counts = {row["prompt_id"]: row["prompt_tokens"] for row in result["count_receipt"]["counts"]}
    return result, counts, receipt_sha256


def _validate_budgets(budgets: tuple[int, ...]) -> None:
    if type(budgets) is not tuple or not budgets or len(budgets) > _MAX_BUDGETS:
        raise ValueError("budgets must be a nonempty tuple of at most 16 values")
    if any(type(value) is not int or not 0 <= value <= _MAX_BUDGET for value in budgets):
        raise ValueError("budgets must be non-negative integers no greater than 1,000,000")
    if len(set(budgets)) != len(budgets):
        raise ValueError("budgets must not contain duplicates")


def _make_exact_counter(
    question: Any, catalog_messages: dict[str, Any], token_counts: dict[str, int]
) -> Any:
    def count_prompt_tokens(context: str) -> int:
        messages = prepare_qa_contrast._messages(question, context)
        prompt_id = _sha256(_canonical_bytes(messages))
        catalog_entry = catalog_messages.get(prompt_id)
        if catalog_entry is None:
            raise ValueError("exact prompt is missing from the frozen count catalog")
        if catalog_entry != messages:
            raise ValueError("catalog prompt messages differ from reconstructed prompt")
        if prompt_id not in token_counts:
            raise ValueError("exact prompt is missing from the token count receipt")
        return token_counts[prompt_id]

    return count_prompt_tokens


def _make_cell(
    question: Any,
    mode: str,
    ceiling: int,
    status: str,
    empty_cost: int,
    full_cost: int,
    used_tokens: int | None,
    selected_hits: Sequence[Any],
    label_ids: set[str],
    found_count: int,
    prompt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    unknown = question.is_abstention
    excluded = unknown or status != "feasible"
    scoring = {
        "explicit_label_count": 0 if unknown else len(label_ids),
        "found_label_count": 0 if unknown else None if excluded else found_count,
        "label_recall": None if excluded else found_count / len(label_ids),
        "exclusion_reason": (
            "abstention_has_no_positive_labels"
            if unknown
            else "infeasible_empty_prompt"
            if excluded
            else None
        ),
    }
    cell = {
        "cell_id": f"{question.question_id}:{mode}:{ceiling}",
        "question_id": question.question_id,
        "split": "dev",
        "mode": mode,
        "ceiling": ceiling,
        "status": status,
        "empty_prompt_tokens": empty_cost,
        "full_prompt_tokens": full_cost,
        "used_tokens": used_tokens,
        "selected_citation_ids": [hit.citation for hit in selected_hits],
        "selected_record_ids": [hit.record.record_id for hit in selected_hits],
        "scoring": scoring,
    }
    if prompt is not None:
        cell["final_prompt"] = prompt
    return cell


def replay_from_counts(
    receipt_raw: bytes, budgets: tuple[int, ...] = _DEFAULT_BUDGETS
) -> dict[str, Any]:
    """Replay the fictional receipt offline; never loads or calls a tokenizer."""
    _validate_budgets(budgets)
    remote_result, token_counts, receipt_sha256 = _validated_receipt(receipt_raw)
    fixture_raw = _FIXTURE.read_bytes()
    if _sha256(fixture_raw) != _FIXTURE_SHA256:
        raise ValueError("QA contrast development fixture differs from the reviewed fixture")

    catalog, _, audit = prepare_qa_contrast.prepare_contrast(fixture_raw)
    _, questions = prepare_qa_contrast._load_fixture(fixture_raw)
    payload = token_runner.build_payload()
    catalog_sha256 = _sha256(_canonical_bytes(catalog))
    if catalog_sha256 != payload["catalog_sha256"]:
        raise ValueError("reconstructed catalog differs from the count payload")
    catalog_messages = {row["prompt_id"]: row["messages"] for row in payload["items"]}
    if token_counts.keys() != catalog_messages.keys():
        raise ValueError("receipt counts do not cover the reconstructed prompt catalog")

    audit_groups = {(row["question_id"], row["mode"]): row for row in audit["groups"]}
    cells: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    embedder = SignedHashEmbedder()

    for question in questions:
        with _temporary_store() as store:
            store.ingest(question.records)
            retriever = Retriever(store, embedder, rrf_k=_RRF_K)
            for mode in _MODES:
                hits = retriever.search(question.question, mode=mode, k=_TOP_K)
                ranking_ids = [hit.citation for hit in hits]
                audited = audit_groups[(question.question_id, mode)]
                if ranking_ids != audited["ranking_citation_ids"]:
                    raise ValueError(
                        "reconstructed ordered citations differ from preparation audit"
                    )

                count_prompt_tokens = _make_exact_counter(question, catalog_messages, token_counts)
                empty_cost = count_prompt_tokens("")
                full_context = "\n".join(prepare_qa_contrast._context_line(hit) for hit in hits)
                full_cost = count_prompt_tokens(full_context)
                group_cell_ids: list[str] = []
                for ceiling in budgets:
                    key = f"{question.question_id}:{mode}:{ceiling}"
                    group_cell_ids.append(key)
                    label_ids = set(question.answer_record_ids)
                    if empty_cost > ceiling:
                        cell = _make_cell(
                            question,
                            mode,
                            ceiling,
                            "infeasible_empty_prompt",
                            empty_cost,
                            full_cost,
                            None,
                            (),
                            label_ids,
                            0,
                        )
                    else:
                        packed = pack_context_by_tokens(
                            hits,
                            ceiling,
                            count_prompt_tokens,
                            counter_id=remote_result["count_receipt"]["counter_id"],
                        )
                        messages = prepare_qa_contrast._messages(question, packed.text)
                        prompt_id = _sha256(_canonical_bytes(messages))
                        if catalog_messages.get(prompt_id) != messages:
                            raise ValueError("packed prompt differs from the frozen catalog")
                        found_count = sum(
                            hit.record.record_id in label_ids for hit in packed.selected_hits
                        )
                        cell = _make_cell(
                            question,
                            mode,
                            ceiling,
                            "feasible",
                            empty_cost,
                            full_cost,
                            packed.prompt_tokens,
                            packed.selected_hits,
                            label_ids,
                            found_count,
                            {"prompt_id": prompt_id, "messages": messages},
                        )
                    cells.append(cell)
                groups.append(
                    {
                        "question_id": question.question_id,
                        "split": "dev",
                        "mode": mode,
                        "ranking_citation_ids": ranking_ids,
                        "cell_ids": group_cell_ids,
                    }
                )

    aggregates: list[dict[str, Any]] = []
    for mode in _MODES:
        for ceiling in budgets:
            cohort = [cell for cell in cells if cell["mode"] == mode and cell["ceiling"] == ceiling]
            unknown_exclusions = sum(
                cell["scoring"]["exclusion_reason"] == "abstention_has_no_positive_labels"
                for cell in cohort
            )
            infeasible_exclusions = sum(
                cell["status"] == "infeasible_empty_prompt" for cell in cohort
            )
            answerable = [
                cell
                for cell in cohort
                if cell["status"] == "feasible" and cell["scoring"]["explicit_label_count"] > 0
            ]
            denominator = len(answerable)
            aggregates.append(
                {
                    "mode": mode,
                    "ceiling": ceiling,
                    "macro_label_recall": (
                        sum(cell["scoring"]["label_recall"] for cell in answerable) / denominator
                        if denominator
                        else None
                    ),
                    "macro_denominator": denominator,
                    "feasible_answerable_cells": denominator,
                    "unknown_exclusion_count": unknown_exclusions,
                    "infeasible_exclusion_count": infeasible_exclusions,
                    "infeasible_answerable_exclusion_count": sum(
                        cell["status"] == "infeasible_empty_prompt"
                        and cell["scoring"]["explicit_label_count"] > 0
                        for cell in cohort
                    ),
                }
            )

    provenance = remote_result["tokenizer_provenance"].copy()
    provenance.pop("elapsed_seconds", None)
    runner_sha256 = _sha256(Path(token_runner.__file__).read_bytes())
    return {
        "schema_version": 1,
        "kind": "offline-qa-contrast-token-replay",
        "budgets": list(budgets),
        "groups": groups,
        "cells": cells,
        "aggregates": aggregates,
        "provenance": {
            "fixture_sha256": _sha256(fixture_raw),
            "catalog_sha256": catalog_sha256,
            "receipt_sha256": receipt_sha256,
            "payload_sha256": _sha256(_canonical_bytes(payload)),
            "runner_sha256": runner_sha256,
            "preparation_source_sha256": catalog["source_sha256"],
            "replay_source_sha256": _sha256(Path(__file__).read_bytes()),
            "counter_id": remote_result["count_receipt"]["counter_id"],
            "tokenizer_provenance": provenance,
            "embedder": embedder.metadata,
            "retrieval": {"modes": list(_MODES), "top_k": _TOP_K, "rrf_k": _RRF_K},
            "provenance_class": "authored-fictional-agent-only",
            "dev_only": True,
            "authored_agent_only": True,
            "provisional": True,
            "held_out_test_data": False,
            "measures_qa_accuracy": False,
            "retained_labels_establish_evidence_sufficiency": False,
        },
    }


@contextmanager
def _temporary_store() -> Iterator[EvidenceStore]:
    with tempfile.TemporaryDirectory(prefix="tracemem-qa-contrast-replay-") as directory:
        with EvidenceStore(Path(directory) / "evidence.sqlite") as store:
            yield store


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="replay-qa-contrast-tokens")
    parser.add_argument("--receipt", type=Path, default=_DEFAULT_RECEIPT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = replay_from_counts(args.receipt.read_bytes())
    except (OSError, TypeError, ValueError) as error:
        parser = build_parser()
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
