"""Offline checks for the QA interpretation answer analyzer.

Every receipt built here is synthetic; no model ran.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import math
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from experiments import analyze_qa_interpretation_answers as _analysis
from experiments import prepare_qa_interpretation_controls as _preparer
from experiments import qa_interpretation_answer_receipt as _receipt

_EXPECTED_RUNNER_SHA256 = "9ee4f7fbb440d166eff415c978239ca71da69a6e607fe63f98cc47fcf9a8568c"
_EOS_TOKEN_ID = 151645


def _fake_receipt(
    *,
    answer_overrides: dict[str, str] | None = None,
    stop_overrides: dict[str, str] | None = None,
    input_token_overrides: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build a complete synthetic receipt solely for tests; no model ran."""
    answer_overrides = answer_overrides or {}
    stop_overrides = stop_overrides or {}
    input_token_overrides = input_token_overrides or {}
    payload, references, _audit = _preparer.prepare_controls()
    request = _receipt.build_request()
    cells = {cell["prompt_id"]: cell for cell in references["cells"]}
    different_axis_cell = next(
        cell
        for cell in references["cells"]
        if cell["expected_prompt_answer"] != cell["full_source_expected_answer"]
    )
    outputs = []
    for item in request["items"]:
        prompt_id = item["prompt_id"]
        cell = cells[prompt_id]
        answer = cell["expected_prompt_answer"]
        if prompt_id == different_axis_cell["prompt_id"]:
            answer = cell["full_source_expected_answer"]
        answer = answer_overrides.get(prompt_id, answer)
        stop_reason = stop_overrides.get(prompt_id, "eos_token")
        generated_ids = (
            list(range(128)) if stop_reason == "max_new_tokens" else [101, _EOS_TOKEN_ID]
        )
        input_tokens = input_token_overrides.get(prompt_id, 3)
        outputs.append(
            {
                "prompt_id": prompt_id,
                "raw_output": answer,
                "input_tokens": input_tokens,
                "input_token_ids": list(range(input_tokens)),
                "generated_tokens": len(generated_ids),
                "generated_token_ids": generated_ids,
                "stop_reason": stop_reason,
                "hit_max_new_tokens": stop_reason == "max_new_tokens",
                "truncated": stop_reason == "max_new_tokens",
            }
        )
    model = request["model"]
    tokenizer = request["tokenizer"]
    return {
        "schema_version": 1,
        "request_sha256": _receipt.request_sha256(request),
        "runner_sha256": _EXPECTED_RUNNER_SHA256,
        "client_elapsed_seconds": 0.125,
        "result": {
            "schema_version": 1,
            "request_sha256": _receipt.request_sha256(request),
            "runner_sha256": _EXPECTED_RUNNER_SHA256,
            "model": {
                "model_id": model["model_id"],
                "revision": model["revision"],
                "generation_protocol_sha256": _receipt.request_sha256(model),
            },
            "tokenizer": {
                "model_id": model["model_id"],
                "revision": model["revision"],
                "template_sha256": tokenizer["template_sha256"],
                "backend_tokenizer_sha256": tokenizer["backend_tokenizer_sha256"],
                "use_fast": True,
                "trust_remote_code": False,
                "package_versions": copy.deepcopy(tokenizer["package_versions"]),
                "python_version": "3.12.14",
                "template_options": copy.deepcopy(tokenizer["template_options"]),
                "eos_token_id": _EOS_TOKEN_ID,
            },
            "runtime": {
                "python_version": "3.12.14",
                "torch_version": "2.7.1",
                "accelerate_version": "1.6.0",
                "cuda_version": "12.8",
                "gpu_name": "Tesla T4",
                "gpu_capability": "7.5",
                "model_load_seconds": 1.0,
                "gpu_compute_seconds": 2.0,
                "elapsed_seconds": 3.0,
                "gpu_peak_memory_bytes": 1024,
                "request_count": 36,
                "modal_call_id": "fc-fake-no-model-ran",
            },
            "outputs": outputs,
        },
    }


class QAInterpretationAnswerAnalysisTests(unittest.TestCase):
    def test_fake_receipt_scores_axes_separately_and_reports_dev_baselines(self) -> None:
        receipt = _fake_receipt()
        analysis = _analysis.analyze_receipt(receipt)

        self.assertEqual(analysis["scope"]["split"], "dev")
        self.assertEqual(analysis["scope"]["test"], [])
        self.assertEqual(analysis["scope"]["human_label_count"], 0)
        self.assertEqual(analysis["scope"]["scenario_groups"], 6)
        self.assertEqual(analysis["scope"]["questions"], 12)
        self.assertEqual(analysis["scope"]["cells"], 36)
        self.assertFalse(analysis["scope"]["cells_are_independent_trials"])
        self.assertEqual(analysis["metrics"]["prompt"]["denominator"], 36)
        self.assertEqual(analysis["metrics"]["prompt"]["numerator"], 35)
        self.assertNotEqual(
            analysis["metrics"]["prompt"]["numerator"],
            analysis["metrics"]["full_source"]["numerator"],
        )
        self.assertEqual(
            analysis["baseline"]["metrics"]["prompt"],
            {"numerator": 26, "denominator": 36, "rate": 26 / 36},
        )
        self.assertEqual(
            analysis["baseline"]["metrics"]["full_source"],
            {"numerator": 6, "denominator": 36, "rate": 6 / 36},
        )
        self.assertEqual(len(analysis["per_cell"]), 36)
        self.assertEqual(
            analysis["provenance"]["receipt_sha256"],
            hashlib.sha256(_receipt._canonical_bytes(receipt)).hexdigest(),
        )
        self.assertEqual(
            analysis["provenance"]["result_sha256"],
            hashlib.sha256(_receipt._canonical_bytes(receipt["result"])).hexdigest(),
        )
        changed = copy.deepcopy(receipt)
        changed["result"]["outputs"][0]["raw_output"] += " changed"
        changed_analysis = _analysis.analyze_receipt(changed)
        self.assertNotEqual(
            analysis["provenance"]["receipt_sha256"],
            changed_analysis["provenance"]["receipt_sha256"],
        )

    def test_cell_identity_normalization_abstention_stop_and_token_counts(self) -> None:
        _, references, _ = _preparer.prepare_controls()
        cells = references["cells"]
        known_cells = [cell for cell in cells if cell["expected_prompt_answer"] != "UNKNOWN"]
        unknown_context = next(
            cell
            for cell in cells
            if cell["expected_prompt_answer"] == "UNKNOWN"
            and cell["full_source_expected_answer"] != "UNKNOWN"
        )
        empty_cell = next(cell for cell in cells if cell["variant_id"] == "empty")
        cap_cell = next(
            cell
            for cell in cells
            if cell["variant_id"] == "support_omitted"
            and cell["prompt_id"] != unknown_context["prompt_id"]
        )
        answers = {
            known_cells[0]["prompt_id"]: " \t"
            + known_cells[0]["expected_prompt_answer"].swapcase()
            + "  ",
            known_cells[1]["prompt_id"]: known_cells[1]["expected_prompt_answer"] + ".",
            unknown_context["prompt_id"]: " \n UnKnOwN\t",
            empty_cell["prompt_id"]: " \t\n ",
            cap_cell["prompt_id"]: "UNKNOWN because context is missing",
        }
        receipt = _fake_receipt(
            answer_overrides=answers,
            stop_overrides={cap_cell["prompt_id"]: "max_new_tokens"},
            input_token_overrides={cap_cell["prompt_id"]: 9},
        )
        report = _analysis.analyze_receipt(receipt)
        rows = report["per_cell"]
        by_prompt = {row["prompt_id"]: row for row in rows}

        self.assertEqual([row["cell_id"] for row in rows], [cell["cell_id"] for cell in cells])
        self.assertEqual(set(by_prompt), {cell["prompt_id"] for cell in cells})
        self.assertTrue(by_prompt[known_cells[0]["prompt_id"]]["matches"]["prompt"])
        self.assertFalse(by_prompt[known_cells[1]["prompt_id"]]["matches"]["prompt"])
        self.assertTrue(by_prompt[unknown_context["prompt_id"]]["exact_unknown_abstention"])
        self.assertFalse(by_prompt[cap_cell["prompt_id"]]["exact_unknown_abstention"])
        self.assertTrue(by_prompt[empty_cell["prompt_id"]]["empty_output"])
        self.assertEqual(report["counts"]["empty_outputs"], 1)
        self.assertEqual(report["counts"]["stop_counts"]["max_new_tokens"], 1)
        self.assertEqual(report["counts"]["cap_count"], 1)
        self.assertEqual(report["counts"]["unique_prompt_input_tokens"], 36 * 3 + 6)
        self.assertEqual(report["counts"]["unique_prompt_generated_tokens"], 35 * 2 + 128)
        self.assertEqual(by_prompt[cap_cell["prompt_id"]]["generated_tokens"], 128)
        self.assertEqual(
            report["by_variant"]["support_omitted"]["unique_prompt_input_tokens"], 12 * 3 + 6
        )

    def test_wrapper_and_complete_receipt_contract_reject_malformed_values(self) -> None:
        cases = []
        extra = _fake_receipt()
        extra["unrecognized"] = True
        cases.append(extra)
        wrong_request = _fake_receipt()
        wrong_request["request_sha256"] = "0" * 64
        cases.append(wrong_request)
        wrong_runner = _fake_receipt()
        wrong_runner["runner_sha256"] = "0" * 64
        cases.append(wrong_runner)
        partial = _fake_receipt()
        partial["result"]["outputs"].pop()
        cases.append(partial)
        duplicate_output = _fake_receipt()
        duplicate_output["result"]["outputs"][1]["prompt_id"] = duplicate_output["result"][
            "outputs"
        ][0]["prompt_id"]
        cases.append(duplicate_output)
        arbitrary_answer_key = _fake_receipt()
        arbitrary_answer_key["result"]["outputs"][0]["answer_key"] = "foreign"
        cases.append(arbitrary_answer_key)
        for receipt in cases:
            with self.subTest(receipt=receipt):
                with self.assertRaises(ValueError):
                    _analysis.analyze_receipt(receipt)
        for elapsed in (True, -0.1, math.nan, math.inf, -math.inf, 10**400):
            with self.subTest(elapsed=elapsed):
                receipt = _fake_receipt()
                receipt["client_elapsed_seconds"] = elapsed
                with self.assertRaises(ValueError):
                    _analysis.analyze_receipt(receipt)

    def test_strict_reader_rejects_duplicate_keys_nonfinite_and_oversize(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            fake = _fake_receipt()
            path.write_text(json.dumps(fake), encoding="utf-8")
            self.assertEqual(_analysis.read_receipt(path), fake)
            for raw in (b'{"value":1,"value":2}', b'{"value":NaN}'):
                path.write_bytes(raw)
                with self.subTest(raw=raw):
                    with self.assertRaises(ValueError):
                        _analysis.read_receipt(path)
            path.write_bytes(b" " * (_analysis._MAX_RECEIPT_BYTES + 1))
            with self.assertRaises(ValueError):
                _analysis.read_receipt(path)

    def test_reader_rejects_fifo_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fifo = Path(directory) / "receipt.fifo"
            os.mkfifo(fifo)
            script = """
import sys
from pathlib import Path
from experiments.analyze_qa_interpretation_answers import read_receipt

try:
    read_receipt(Path(sys.argv[1]))
except ValueError:
    raise SystemExit(0)
raise SystemExit(1)
"""
            try:
                result = subprocess.run(
                    [sys.executable, "-c", script, str(fifo)],
                    capture_output=True,
                    text=True,
                    timeout=1,
                    cwd=Path(__file__).resolve().parents[1],
                )
            except subprocess.TimeoutExpired:
                self.fail("receipt reader blocked while opening a FIFO")
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_cli_writes_exclusively_and_preserves_receipt_and_existing_paths(self) -> None:
        receipt = _fake_receipt()
        receipt_before = json.dumps(receipt, sort_keys=True).encode()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt_path = root / "fake-receipt.json"
            receipt_path.write_bytes(receipt_before)
            existing = root / "existing-analysis.json"
            existing.write_text("preserve me", encoding="utf-8")
            symlink_target = root / "symlink-target.json"
            symlink_target.write_text("also preserve me", encoding="utf-8")
            existing_symlink = root / "existing-analysis-link.json"
            existing_symlink.symlink_to(symlink_target)
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(
                    _analysis.main(["--receipt", str(receipt_path), "--output", str(existing)]),
                    2,
                )
            self.assertEqual(existing.read_text(encoding="utf-8"), "preserve me")
            self.assertEqual(receipt_path.read_bytes(), receipt_before)
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(
                    _analysis.main(
                        ["--receipt", str(receipt_path), "--output", str(existing_symlink)]
                    ),
                    2,
                )
            self.assertTrue(existing_symlink.is_symlink())
            self.assertEqual(symlink_target.read_text(encoding="utf-8"), "also preserve me")

            output = root / "new-analysis.json"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    _analysis.main(["--receipt", str(receipt_path), "--output", str(output)]),
                    0,
                )
            output_bytes = output.read_bytes()
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
            self.assertEqual(json.loads(output_bytes)["scope"]["cells"], 36)
            self.assertEqual(receipt_path.read_bytes(), receipt_before)
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(
                    _analysis.main(["--receipt", str(receipt_path), "--output", str(output)]),
                    2,
                )
            self.assertEqual(output.read_bytes(), output_bytes)

    def test_empty_metric_denominator_is_explicitly_null(self) -> None:
        self.assertEqual(_analysis._metric(0, 0), {"numerator": 0, "denominator": 0, "rate": None})


if __name__ == "__main__":
    unittest.main()
