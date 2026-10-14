"""Bounded, local-only LongMemEval turn-retrieval diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sqlite3
import stat
import tempfile
from pathlib import Path
from typing import Any

from tracemem.adapters.longmemeval import LongMemEvalQuestion, parse_longmemeval_bytes
from tracemem.adapters.longmemeval_file import scan_longmemeval_file
from tracemem.retrieval import Retriever, SignedHashEmbedder
from tracemem.schema import EVIDENCE_SCHEMA_VERSION
from tracemem.storage import EvidenceStore

_MODES = ("bm25", "vector", "hybrid")
_DEFAULT_CUTOFFS = (5, 10, 50)
_MAX_QUERY_CHARS = 4096
_MAX_INPUT_BYTES = 64 * 1024 * 1024
_DEFAULT_RRF_K = 60


def _validate_cutoffs(cutoffs: tuple[int, ...]) -> None:
    if type(cutoffs) is not tuple or not cutoffs:
        raise ValueError("cutoffs must be a nonempty tuple of unique integers from 1 to 100")
    if any(type(cutoff) is not int or not 1 <= cutoff <= 100 for cutoff in cutoffs):
        raise ValueError("cutoffs must be a nonempty tuple of unique integers from 1 to 100")
    if len(set(cutoffs)) != len(cutoffs):
        raise ValueError("cutoffs must be a nonempty tuple of unique integers from 1 to 100")


def _default_embedding_report() -> dict[str, Any]:
    embedder = SignedHashEmbedder()
    metadata = embedder.metadata
    fingerprint_payload = json.dumps(
        metadata, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return {
        "metadata": metadata,
        "fingerprint": hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest(),
    }


def _question_retrieval(
    question: LongMemEvalQuestion, cutoff_list: tuple[int, ...]
) -> tuple[dict[str, Any], dict[str, Any], int]:
    max_cutoff = max(cutoff_list)
    with tempfile.TemporaryDirectory(prefix="tracemem-longmemeval-") as directory:
        database_path = Path(directory) / "evidence.sqlite"
        with EvidenceStore(database_path) as store:
            store.ingest(question.records)
            retriever = Retriever(store, rrf_k=_DEFAULT_RRF_K)
            mode_hits = {
                mode: retriever.search(question.question, mode=mode, k=max_cutoff)
                for mode in _MODES
            }

            retrieved: dict[str, list[dict[str, str]]] = {}
            for mode, hits in mode_hits.items():
                retrieved[mode] = [
                    {
                        "record_id": hit.record.record_id,
                        "revision": hit.record.revision,
                        "citation": hit.citation,
                    }
                    for hit in hits
                ]

            labels = set(question.answer_record_ids)
            label_count = len(question.answer_record_ids)
            cells_by_mode: dict[str, list[dict[str, Any]]] = {}
            for mode in _MODES:
                cells: list[dict[str, Any]] = []
                for cutoff in cutoff_list:
                    prefix = retrieved[mode][:cutoff]
                    returned_ids = {hit["record_id"] for hit in prefix}
                    found_count = len(labels.intersection(returned_ids))
                    exclusion_reason: str | None = None
                    if question.is_abstention:
                        exclusion_reason = "abstention"
                    elif label_count == 0:
                        exclusion_reason = "missing_turn_labels"

                    if exclusion_reason is None:
                        label_recall: float | None = found_count / label_count
                        all_found: bool | None = found_count == label_count
                    else:
                        label_recall = None
                        all_found = None

                    cells.append(
                        {
                            "cutoff": cutoff,
                            "returned": prefix,
                            "explicit_label_count": label_count,
                            "found_label_count": found_count,
                            "label_recall": label_recall,
                            "all_labeled_turns_found": all_found,
                            "exclusion_reason": exclusion_reason,
                        }
                    )
                cells_by_mode[mode] = cells

            embedding_report = {
                "metadata": retriever.embedder.metadata,
                "fingerprint": retriever.embedder_fingerprint,
            }
            return (
                {"question_id": question.question_id, "retrieval": cells_by_mode},
                embedding_report,
                retriever.rrf_k,
            )


def _evaluate_questions(
    questions: tuple[LongMemEvalQuestion, ...],
    *,
    input_sha256: str,
    variant: str,
    cutoffs: tuple[int, ...],
    schema_version: int = 1,
    input_bytes: int | None = None,
    selection: dict[str, int | str] | None = None,
) -> dict[str, Any]:
    question_results: list[dict[str, Any]] = []
    embedding_report = _default_embedding_report()
    rrf_k = _DEFAULT_RRF_K
    for question in questions:
        result, embedding_report, rrf_k = _question_retrieval(question, cutoffs)
        question_results.append(result)

    question_count = len(questions)
    abstention_count = sum(question.is_abstention for question in questions)
    unlabeled_count = sum(
        not question.is_abstention and not question.answer_record_ids for question in questions
    )
    scored_count = question_count - abstention_count - unlabeled_count
    counts = {
        "question_count": question_count,
        "abstention_count": abstention_count,
        "unlabeled_count": unlabeled_count,
        "scored_count": scored_count,
    }

    aggregates: dict[str, list[dict[str, Any]]] = {}
    for mode in _MODES:
        mode_aggregates: list[dict[str, Any]] = []
        for cutoff in cutoffs:
            cells = [
                next(
                    cell for cell in question_result["retrieval"][mode] if cell["cutoff"] == cutoff
                )
                for question_result in question_results
            ]
            scored_cells = [cell for cell in cells if cell["label_recall"] is not None]
            if scored_cells:
                macro_recall: float | None = sum(
                    cell["label_recall"] for cell in scored_cells
                ) / len(scored_cells)
                all_found_fraction: float | None = sum(
                    cell["all_labeled_turns_found"] is True for cell in scored_cells
                ) / len(scored_cells)
            else:
                macro_recall = None
                all_found_fraction = None
            mode_aggregates.append(
                {
                    "cutoff": cutoff,
                    "question_count": question_count,
                    "abstention_count": abstention_count,
                    "unlabeled_count": unlabeled_count,
                    "scored_question_count": scored_count,
                    "nonempty_retrieval_question_count": sum(
                        bool(cell["returned"]) for cell in cells
                    ),
                    "macro_label_recall": macro_recall,
                    "all_labels_found_fraction": all_found_fraction,
                }
            )
        aggregates[mode] = mode_aggregates

    report: dict[str, Any] = {
        "schema_version": schema_version,
        "input_sha256": input_sha256,
        "variant": variant,
        "cutoffs": list(cutoffs),
        "modes": list(_MODES),
        "environment": {
            "python": platform.python_version(),
            "sqlite": sqlite3.sqlite_version,
        },
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "embedding": embedding_report,
        "rrf_k": rrf_k,
        "counts": counts,
        "questions": question_results,
        "aggregates": aggregates,
    }
    if input_bytes is not None:
        report["input_bytes"] = input_bytes
    if selection is not None:
        report["selection"] = selection
    return report


def evaluate_retrieval_bytes(
    raw: bytes,
    *,
    namespace: str,
    variant: str,
    cutoffs: tuple[int, ...] = _DEFAULT_CUTOFFS,
) -> dict[str, Any]:
    """Evaluate explicit turn-label retrieval using isolated local evidence stores."""
    _validate_cutoffs(cutoffs)
    questions = parse_longmemeval_bytes(raw, namespace=namespace, variant=variant)
    if any(len(question.question) > _MAX_QUERY_CHARS for question in questions):
        raise ValueError("query exceeds Retriever's 4096-character limit")
    return _evaluate_questions(
        questions,
        input_sha256=hashlib.sha256(raw).hexdigest(),
        variant=variant,
        cutoffs=cutoffs,
    )


def evaluate_retrieval_file(
    path: str | os.PathLike[str],
    *,
    namespace: str,
    variant: str,
    question_ids: tuple[str, ...],
    cutoffs: tuple[int, ...] = _DEFAULT_CUTOFFS,
) -> dict[str, Any]:
    """Evaluate selected questions after the complete file passes validation."""
    _validate_cutoffs(cutoffs)
    if type(question_ids) is not tuple or not question_ids:
        raise ValueError("question_ids must be a nonempty tuple")

    scan = scan_longmemeval_file(
        path,
        namespace=namespace,
        variant=variant,
        question_ids=question_ids,
    )
    questions = scan.selected_questions
    if any(len(question.question) > _MAX_QUERY_CHARS for question in questions):
        raise ValueError("query exceeds Retriever's 4096-character limit")

    return _evaluate_questions(
        questions,
        input_sha256=scan.input_sha256,
        variant=variant,
        cutoffs=cutoffs,
        schema_version=2,
        input_bytes=scan.input_bytes,
        selection={
            "kind": "explicit_question_ids",
            "source_question_count": scan.question_count,
            "evaluated_question_count": len(questions),
        },
    )


def _read_input(path: str) -> bytes:
    flags = os.O_RDONLY | os.O_NONBLOCK
    flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise ValueError("cannot open input file") from None
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("input must be a regular file")
        if metadata.st_size > _MAX_INPUT_BYTES:
            raise ValueError("input exceeds the 64 MiB limit")
        with os.fdopen(descriptor, "rb") as input_stream:
            descriptor = -1
            raw = input_stream.read(_MAX_INPUT_BYTES + 1)
    except OSError:
        raise ValueError("cannot read input file") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(raw) > _MAX_INPUT_BYTES:
        raise ValueError("input exceeds the 64 MiB limit")
    return raw


def _write_output(path: str, serialized: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, 0o600)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as output_stream:
            descriptor = None
            output_stream.write(serialized)
    except FileExistsError:
        raise ValueError("output path already exists") from None
    except OSError:
        raise ValueError("cannot write output file") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def main(argv: list[str] | None = None) -> int:
    """Run the local JSON report CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="local LongMemEval-shaped JSON file")
    parser.add_argument("--namespace", required=True, help="stable identity namespace")
    parser.add_argument("--variant", required=True, choices=("s", "m", "oracle"))
    parser.add_argument("--output", required=True, help="new private JSON report path")
    parser.add_argument(
        "--question-id",
        action="append",
        dest="question_ids",
        help="explicit question ID to evaluate; repeat to select up to five",
    )
    arguments = parser.parse_args(argv)

    try:
        if arguments.question_ids is None:
            raw = _read_input(arguments.input)
            report = evaluate_retrieval_bytes(
                raw, namespace=arguments.namespace, variant=arguments.variant
            )
        else:
            report = evaluate_retrieval_file(
                arguments.input,
                namespace=arguments.namespace,
                variant=arguments.variant,
                question_ids=tuple(arguments.question_ids),
            )
        serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        _write_output(arguments.output, serialized)
    except ValueError as error:
        parser.error(str(error))
    except Exception:
        parser.error("evaluation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
