"""Replay a pinned MiniLM receipt against the fictional cross-session fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path
from typing import Any

from experiments import evaluate_longmemeval_retrieval as retrieval
from experiments import modal_baseline
from tracemem.remote_vectors import PrecomputedEmbedder

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE_PATH = _ROOT / "examples/synthetic/cross-session-test-v1.json"
_DEFAULT_RECEIPT_PATH = _ROOT / "examples/synthetic/minilm-cross-session-remote.json"
_FIXTURE_SHA256 = "dc7e301a9acb0e0d26293c2d8fa80bf8ebe838ecd543a7917a64bcd8f3cdfedf"
_EXPECTED_PAYLOAD_SHA256 = "95158af854e806eb14c19ee4f3635446c912d5a1efddd7933f7e639f5f43aada"
# Historical identity of the runner that produced this receipt; do not hash today's source here.
_EXPECTED_RUNNER_SHA256 = "043b2c34e8d226c5dd77fb06a6b254f36886edb0217d841218570da19f89942e"
_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
_DIMENSIONS = 384
_VECTOR_COUNT = 78
_MAX_RECEIPT_BYTES = 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}")
_RECEIPT_KEYS = {
    "client_call_wall_seconds_including_image_and_startup",
    "gpu_compute_seconds",
    "gpu_name",
    "input_sha256",
    "mode",
    "modal_app_id",
    "modal_function_call_id",
    "model",
    "model_load_seconds",
    "package_versions",
    "payload_sha256",
    "peak_gpu_memory_bytes",
    "prompt_version",
    "python_version",
    "remote_function_elapsed_seconds",
    "runner_sha256",
    "schema_version",
    "torch_cuda_version",
    "vector_count",
    "vectors",
}
_CUTOFFS = (1, 3, 5)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _is_sha256(value: Any) -> bool:
    return type(value) is str and _SHA256.fullmatch(value) is not None


def _fixture_payload() -> tuple[bytes, set[str]]:
    try:
        fixture_raw = _FIXTURE_PATH.read_bytes()
    except OSError:
        raise ValueError("cannot read the pinned cross-session fixture") from None
    if _sha256(fixture_raw) != _FIXTURE_SHA256:
        raise ValueError("cross-session fixture differs from the reviewed input")

    payload = modal_baseline.build_payload("embeddings", fixture="cross-session-test-v1")
    payload_hash = modal_baseline.payload_sha256(payload)
    if payload_hash != _EXPECTED_PAYLOAD_SHA256:
        raise ValueError("cross-session embedding payload differs from the reviewed payload")
    items = payload.get("items")
    if type(items) is not list or len(items) != _VECTOR_COUNT:
        raise ValueError("cross-session embedding payload must contain exactly 78 items")
    hashes = {item.get("text_sha256") for item in items if type(item) is dict}
    if len(hashes) != _VECTOR_COUNT or any(not _is_sha256(value) for value in hashes):
        raise ValueError("cross-session embedding payload text hashes are invalid")
    return fixture_raw, hashes


def _validated_receipt(
    receipt_raw: bytes, expected_hashes: set[str]
) -> tuple[dict[str, Any], PrecomputedEmbedder]:
    if type(receipt_raw) is not bytes or not receipt_raw:
        raise ValueError("receipt must be nonempty bytes")
    if len(receipt_raw) > _MAX_RECEIPT_BYTES:
        raise ValueError("receipt exceeds the 1 MiB input limit")
    try:
        receipt = json.loads(
            receipt_raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except ValueError as error:
        if str(error) == "duplicate JSON key" or str(error).startswith("non-finite JSON number"):
            raise ValueError(str(error)) from None
        raise ValueError("receipt is not valid unique-key UTF-8 JSON") from None
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        raise ValueError("receipt is not valid unique-key UTF-8 JSON") from None
    if type(receipt) is not dict or set(receipt) != _RECEIPT_KEYS:
        raise ValueError("receipt fields do not match the pinned embedding schema")
    if type(receipt["schema_version"]) is not int or receipt["schema_version"] != 1:
        raise ValueError("receipt schema_version must be integer 1")
    if receipt["mode"] != "embeddings":
        raise ValueError("receipt mode must be embeddings")
    if receipt["prompt_version"] != "v1":
        raise ValueError("receipt prompt_version must be v1")
    if receipt["input_sha256"] != _FIXTURE_SHA256:
        raise ValueError("receipt input hash does not match the pinned fixture")
    if receipt["payload_sha256"] != _EXPECTED_PAYLOAD_SHA256:
        raise ValueError("receipt payload hash does not match the reviewed payload")
    if receipt["runner_sha256"] != _EXPECTED_RUNNER_SHA256:
        raise ValueError("receipt runner SHA-256 does not match the reviewed runner")

    model = receipt["model"]
    if (
        type(model) is not dict
        or set(model) != {"dimensions", "model_id", "revision"}
        or model.get("model_id") != _MODEL_ID
        or model.get("revision") != _MODEL_REVISION
        or type(model.get("dimensions")) is not int
        or model["dimensions"] != _DIMENSIONS
    ):
        raise ValueError("receipt model identity, revision, or dimensions are invalid")
    if type(receipt["vector_count"]) is not int or receipt["vector_count"] != _VECTOR_COUNT:
        raise ValueError("receipt vector_count must be integer 78")
    vectors = receipt["vectors"]
    if type(vectors) is not dict or set(vectors) != expected_hashes:
        raise ValueError("receipt vectors must provide exact payload coverage")

    versions = receipt["package_versions"]
    if (
        type(versions) is not dict
        or not versions
        or any(
            type(name) is not str or type(version) is not str for name, version in versions.items()
        )
    ):
        raise ValueError("receipt package_versions must map strings to strings")
    string_fields = (
        "gpu_name",
        "python_version",
        "torch_cuda_version",
        "modal_app_id",
        "modal_function_call_id",
    )
    for field in string_fields:
        if type(receipt[field]) is not str or not receipt[field] or len(receipt[field]) > 256:
            raise ValueError(f"receipt {field} is invalid")
    if not receipt["python_version"].startswith("3.12."):
        raise ValueError("receipt python_version must be Python 3.12")
    for field in (
        "client_call_wall_seconds_including_image_and_startup",
        "gpu_compute_seconds",
        "model_load_seconds",
        "remote_function_elapsed_seconds",
    ):
        value = receipt[field]
        if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
            raise ValueError(f"receipt {field} must be a finite non-negative number")
    if type(receipt["peak_gpu_memory_bytes"]) is not int or receipt["peak_gpu_memory_bytes"] < 0:
        raise ValueError("receipt peak_gpu_memory_bytes must be a non-negative integer")

    try:
        embedder = PrecomputedEmbedder(
            model_id=model["model_id"],
            revision=model["revision"],
            dimensions=model["dimensions"],
            vectors=vectors,
        )
    except (TypeError, ValueError):
        raise ValueError("receipt vectors must be finite 384-dimensional numeric arrays") from None
    return receipt, embedder


def replay_receipt(receipt_raw: bytes) -> dict[str, Any]:
    """Replay an exact fixture receipt without importing a model or contacting a service."""
    fixture_raw, expected_hashes = _fixture_payload()
    receipt, embedder = _validated_receipt(receipt_raw, expected_hashes)
    baseline = retrieval.evaluate_retrieval_bytes(
        fixture_raw,
        namespace="cross-session-test-v1",
        variant="s",
        cutoffs=_CUTOFFS,
    )
    minilm = retrieval.evaluate_retrieval_bytes(
        fixture_raw,
        namespace="cross-session-test-v1",
        variant="s",
        cutoffs=_CUTOFFS,
        embedder=embedder,
    )
    bm25_identical = (
        baseline["counts"] == minilm["counts"]
        and baseline["aggregates"]["bm25"] == minilm["aggregates"]["bm25"]
        and [
            (question["question_id"], question["retrieval"]["bm25"])
            for question in baseline["questions"]
        ]
        == [
            (question["question_id"], question["retrieval"]["bm25"])
            for question in minilm["questions"]
        ]
    )
    if not bm25_identical:
        raise ValueError("MiniLM replay changed the frozen BM25 results")

    return {
        "schema_version": 1,
        "mode": "cross-session-minilm-retrieval",
        "input_sha256": _FIXTURE_SHA256,
        "payload_sha256": _EXPECTED_PAYLOAD_SHA256,
        "raw_receipt_sha256": _sha256(receipt_raw),
        "runner_sha256": _EXPECTED_RUNNER_SHA256,
        "model": receipt["model"],
        "vector_count": receipt["vector_count"],
        "annotation": {
            "status": "agent_authored_provisional",
            "human_labels": 0,
            "assigned_split": "test",
            "held_out_quality_established": False,
            "qa_accuracy_measured": False,
            "unknowns_excluded_from_recall": True,
        },
        "remote_measurement": {
            "gpu_name": receipt["gpu_name"],
            "python_version": receipt["python_version"],
            "torch_cuda_version": receipt["torch_cuda_version"],
            "package_versions": receipt["package_versions"],
            "gpu_compute_seconds": receipt["gpu_compute_seconds"],
            "model_load_seconds": receipt["model_load_seconds"],
            "remote_function_elapsed_seconds": receipt["remote_function_elapsed_seconds"],
            "client_call_wall_seconds_including_image_and_startup": receipt[
                "client_call_wall_seconds_including_image_and_startup"
            ],
            "peak_gpu_memory_bytes": receipt["peak_gpu_memory_bytes"],
        },
        "invariants": {"bm25_identical": bm25_identical},
        "baseline": baseline,
        "minilm": minilm,
    }


def _read_receipt(path: str) -> bytes:
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise ValueError("cannot open receipt file") from None
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("receipt must be a regular file")
        if metadata.st_size > _MAX_RECEIPT_BYTES:
            raise ValueError("receipt exceeds the 1 MiB input limit")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            raw = stream.read(_MAX_RECEIPT_BYTES + 1)
    except OSError:
        raise ValueError("cannot read receipt file") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(raw) > _MAX_RECEIPT_BYTES:
        raise ValueError("receipt exceeds the 1 MiB input limit")
    return raw


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--receipt", default=str(_DEFAULT_RECEIPT_PATH), help="local pinned MiniLM receipt"
    )
    parser.add_argument("--output", required=True, help="new private JSON report path")
    arguments = parser.parse_args(argv)
    try:
        report = replay_receipt(_read_receipt(arguments.receipt))
        serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        retrieval._write_output(arguments.output, serialized)
    except ValueError as error:
        parser.error(str(error))
    except Exception:
        parser.error("replay failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
