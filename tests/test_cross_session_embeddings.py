"""Offline replay tests for the pinned fictional MiniLM retrieval measurement."""

from __future__ import annotations

import hashlib
import json
import math
import stat
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from experiments import evaluate_cross_session_embeddings as replay
from experiments import modal_baseline

_RECEIPT_PATH = Path("examples/synthetic/minilm-cross-session-remote.json")
_BASELINE_PATH = Path("docs/cross-session-test-retrieval.json")


def _test_only_receipt() -> dict[str, object]:
    """Build deterministic test vectors; these are never used as model measurements."""
    payload = modal_baseline.build_payload("embeddings", fixture="cross-session-test-v1")
    model = payload["model"]
    assert isinstance(model, dict)
    hashes = [item["text_sha256"] for item in payload["items"]]
    return {
        "schema_version": 1,
        "mode": "embeddings",
        "prompt_version": "v1",
        "input_sha256": payload["input_sha256"],
        "payload_sha256": modal_baseline.payload_sha256(payload),
        "runner_sha256": replay._EXPECTED_RUNNER_SHA256,
        "model": model,
        "vector_count": len(hashes),
        "vectors": {key: [1.0] + [0.0] * 383 for key in hashes},
        "package_versions": {"torch": "test-only"},
        "gpu_name": "test-only",
        "python_version": "3.12.0",
        "torch_cuda_version": "test-only",
        "gpu_compute_seconds": 0.0,
        "model_load_seconds": 0.0,
        "remote_function_elapsed_seconds": 0.0,
        "client_call_wall_seconds_including_image_and_startup": 0.0,
        "peak_gpu_memory_bytes": 0,
        "modal_app_id": "test-only",
        "modal_function_call_id": "test-only",
    }


def _encode(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class CrossSessionEmbeddingReplayTests(unittest.TestCase):
    def test_actual_receipt_replays_offline_and_preserves_frozen_bm25(self) -> None:
        raw = _RECEIPT_PATH.read_bytes()

        report = replay.replay_receipt(raw)

        baseline = json.loads(_BASELINE_PATH.read_bytes())
        self.assertEqual(report["input_sha256"], baseline["input_sha256"])
        self.assertEqual(report["payload_sha256"], replay._EXPECTED_PAYLOAD_SHA256)
        self.assertEqual(report["runner_sha256"], replay._EXPECTED_RUNNER_SHA256)
        self.assertEqual(report["raw_receipt_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(report["vector_count"], 78)
        self.assertEqual(report["baseline"]["questions"], baseline["questions"])
        self.assertEqual(
            report["annotation"],
            {
                "status": "agent_authored_provisional",
                "human_labels": 0,
                "assigned_split": "test",
                "held_out_quality_established": False,
                "qa_accuracy_measured": False,
                "unknowns_excluded_from_recall": True,
            },
        )
        self.assertEqual(report["baseline"]["aggregates"]["bm25"], baseline["aggregates"]["bm25"])
        self.assertEqual(report["minilm"]["aggregates"]["bm25"], baseline["aggregates"]["bm25"])
        self.assertEqual(report["invariants"]["bm25_identical"], True)
        self.assertEqual(
            report["minilm"]["embedding"]["metadata"]["model_id"],
            "sentence-transformers/all-MiniLM-L6-v2",
        )

    def test_test_only_vectors_cover_fixture_and_drive_replay(self) -> None:
        report = replay.replay_receipt(_encode(_test_only_receipt()))

        self.assertEqual(report["vector_count"], 78)
        self.assertEqual(report["invariants"]["bm25_identical"], True)
        self.assertEqual(
            report["baseline"]["aggregates"]["bm25"],
            report["minilm"]["aggregates"]["bm25"],
        )

    def test_receipt_rejects_wrong_hashes_runner_schema_and_vector_coverage(self) -> None:
        mutations = (
            ("payload_sha256", "0" * 64, "payload hash"),
            ("input_sha256", "0" * 64, "input hash"),
            ("runner_sha256", "0" * 64, "runner SHA-256"),
            ("schema_version", True, "schema_version"),
            ("vector_count", True, "vector_count"),
        )
        for key, value, message in mutations:
            with self.subTest(key=key):
                receipt = _test_only_receipt()
                receipt[key] = value
                with self.assertRaisesRegex(ValueError, message):
                    replay.replay_receipt(_encode(receipt))

        receipt = _test_only_receipt()
        vectors = dict(receipt["vectors"])
        vectors.pop(next(iter(vectors)))
        receipt["vectors"] = vectors
        with self.assertRaisesRegex(ValueError, "exact payload coverage"):
            replay.replay_receipt(_encode(receipt))

    def test_receipt_rejects_wrong_model_revision_dimensions_and_vector_lengths(self) -> None:
        mutations = (
            ("model_id", "synthetic/test-only", "model identity"),
            ("revision", "0" * 40, "model identity"),
            ("dimensions", 383, "model identity"),
        )
        for field, value, message in mutations:
            with self.subTest(field=field):
                receipt = _test_only_receipt()
                model = dict(receipt["model"])
                model[field] = value
                receipt["model"] = model
                with self.assertRaisesRegex(ValueError, message):
                    replay.replay_receipt(_encode(receipt))

        receipt = _test_only_receipt()
        vectors = dict(receipt["vectors"])
        first_key = next(iter(vectors))
        vectors[first_key] = [1.0] * 383
        receipt["vectors"] = vectors
        with self.assertRaisesRegex(ValueError, "finite 384-dimensional"):
            replay.replay_receipt(_encode(receipt))

        receipt = _test_only_receipt()
        vectors = dict(receipt["vectors"])
        extra_key = "f" * 64
        while extra_key in vectors:
            extra_key = "e" + extra_key[1:]
        vectors[extra_key] = [1.0] + [0.0] * 383
        receipt["vectors"] = vectors
        with self.assertRaisesRegex(ValueError, "exact payload coverage"):
            replay.replay_receipt(_encode(receipt))

    def test_receipt_rejects_duplicate_keys_nonfinite_vectors_and_oversize(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            replay.replay_receipt(b'{"mode":"embeddings","mode":"extraction"}')

        receipt = _test_only_receipt()
        vectors = dict(receipt["vectors"])
        first_key = next(iter(vectors))
        vectors[first_key] = [math.inf] + [0.0] * 383
        receipt["vectors"] = vectors
        nonfinite = json.dumps(receipt, separators=(",", ":"), allow_nan=True).encode()
        nonfinite = nonfinite.replace(b"Infinity", b"1e999")
        with self.assertRaisesRegex(ValueError, "finite 384-dimensional"):
            replay.replay_receipt(nonfinite)

        with self.assertRaisesRegex(ValueError, "1 MiB input limit"):
            replay.replay_receipt(b" " * (replay._MAX_RECEIPT_BYTES + 1))

    def test_cli_is_offline_writes_exclusive_private_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "replay.json"
            stdout = StringIO()
            stderr = StringIO()
            with (
                patch.object(
                    modal_baseline,
                    "_build_modal_function",
                    side_effect=AssertionError("remote launch"),
                ),
                patch.object(
                    modal_baseline, "_remote_compute", side_effect=AssertionError("remote compute")
                ),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = replay.main(["--receipt", str(_RECEIPT_PATH), "--output", str(output)])

            self.assertEqual(exit_code, 0)
            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(stderr.getvalue(), "")
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
            self.assertEqual(
                json.loads(output.read_bytes()),
                replay.replay_receipt(_RECEIPT_PATH.read_bytes()),
            )

            output.write_text("preserve", encoding="utf-8")
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                replay.main(["--receipt", str(_RECEIPT_PATH), "--output", str(output)])
            self.assertEqual(output.read_text(encoding="utf-8"), "preserve")


if __name__ == "__main__":
    unittest.main()
