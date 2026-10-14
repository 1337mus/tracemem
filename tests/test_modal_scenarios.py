"""Tests for the fixture-only scenario inference runner and receipt evaluator."""

from __future__ import annotations

import dis
import hashlib
import json
import math
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from experiments import evaluate_scenarios, modal_scenarios, scenario_payload
from tracemem.challenge import load_challenge
from tracemem.prompts import render_explicit_prompt


def _fixture() -> tuple[list[dict[str, object]], dict[str, object], str]:
    source_cases, _ = load_challenge()
    source = next(case for case in source_cases if case["expected"])
    cases: list[dict[str, object]] = []
    case_metadata: dict[str, object] = {}
    for index in range(48):
        case = deepcopy(source)
        case_id = f"scenario-{index:02d}"
        case["case_id"] = case_id
        case["scenario_id"] = f"scenario-group-{index:02d}"
        case["split"] = "dev" if index < 24 else "test"
        cases.append(case)
        case_metadata[case_id] = {
            "scenario_group": f"group-{index:02d}",
            "template_group": f"template-{index:02d}",
            "entity_ids": [f"entity-{index:02d}"],
        }
    manifest: dict[str, object] = {
        "version": 1,
        "license": "MIT",
        "annotation_status": "agent_authored",
        "case_metadata": case_metadata,
    }
    _, _, fixture_hash = scenario_payload.load_scenarios()
    return cases, manifest, fixture_hash


def _loader_patch():
    cases, manifest, digest = _fixture()
    return patch.object(scenario_payload, "load_scenarios", return_value=(cases, manifest, digest))


def _valid_receipt(model_key: str = "qwen3-4b") -> dict[str, object]:
    with _loader_patch():
        payload = scenario_payload.build_payload(model_key)
    eos_ids = [32007, 32001, 32000] if model_key == "phi-3.5-mini" else [42, 43]
    cases, _, _ = _fixture()
    outputs = []
    for item, case in zip(payload["items"], cases, strict=True):
        raw_output = json.dumps({"claims": case["expected"]}, ensure_ascii=False)
        outputs.append(
            {
                "case_id": item["case_id"],
                "raw_output": raw_output,
                "input_tokens": 200,
                "generated_tokens": 20,
                "hit_max_new_tokens": False,
                "output_characters": len(raw_output),
                "finish_reason": "eos",
                "generated_last_token_id": eos_ids[-1],
            }
        )
    return {
        "schema_version": 1,
        "mode": "scenarios",
        "input_sha256": payload["input_sha256"],
        "payload_sha256": scenario_payload.payload_sha256(payload),
        "prompt_source_sha256": payload["prompt_source_sha256"],
        "prompt_sha256": payload["prompt_sha256"],
        "prompt_version": payload["prompt_version"],
        "model": payload["model"],
        "package_versions": {"torch": "2.7.1", "transformers": "4.51.3"},
        "gpu_compute_seconds": 12.5,
        "model_load_seconds": 4.0,
        "remote_function_elapsed_seconds": 20.0,
        "peak_gpu_memory_bytes": 3_000_000,
        "python_version": "3.12.0",
        "torch_cuda_version": "12.6",
        "gpu_name": "Tesla T4",
        "modal_function_call_id": "fc-test",
        "modal_app_id": "ap-test",
        "client_call_wall_seconds_including_image_and_startup": 30.0,
        "runner_sha256": "b" * 64,
        "generation_eos_token_ids": eos_ids,
        "generation_pad_token_id": eos_ids[-1],
        "case_count": len(outputs),
        "generated_outputs": outputs,
    }


class ScenarioPayloadTests(unittest.TestCase):
    def test_payload_contains_only_rendered_prompts_and_pinned_inference_settings(self) -> None:
        cases, _, fixture_hash = _fixture()
        with _loader_patch():
            payload = scenario_payload.build_payload("qwen3-4b")

        self.assertEqual(
            set(payload),
            {
                "schema_version",
                "input_sha256",
                "prompt_source_sha256",
                "prompt_sha256",
                "prompt_version",
                "model",
                "items",
            },
        )
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["input_sha256"], fixture_hash)
        self.assertEqual(payload["prompt_version"], "explicit-v2")
        self.assertEqual(len(payload["items"]), 48)
        self.assertEqual(sum(case["split"] == "dev" for case in cases), 24)
        for item, case in zip(payload["items"], cases, strict=True):
            self.assertEqual(set(item), {"case_id", "prompt"})
            self.assertEqual(
                item, {"case_id": case["case_id"], "prompt": render_explicit_prompt(case)}
            )
        serialized = json.dumps(payload)
        for prohibited in ('"expected"', '"case_metadata"', '"scenario_group"', '"tags"'):
            self.assertNotIn(prohibited, serialized)
        self.assertEqual(payload["model"]["revision"], "1cfa9a7208912126459214e8b04321603b3df60c")
        self.assertEqual(payload["model"]["precision"], "float16")
        self.assertFalse(payload["model"]["do_sample"])
        self.assertEqual(payload["model"]["max_new_tokens"], 1024)
        self.assertEqual(payload["model"]["max_prompt_tokens"], 4096)

    def test_both_candidate_revisions_and_prompt_hash_are_pinned(self) -> None:
        with _loader_patch():
            qwen = scenario_payload.build_payload("qwen3-4b")
            phi = scenario_payload.build_payload("phi-3.5-mini")

        self.assertEqual(phi["model"]["revision"], "2fe192450127e6a83f7441aef6e3ca586c338b77")
        self.assertEqual(phi["model"]["model_id"], "microsoft/Phi-3.5-mini-instruct")
        self.assertEqual(phi["model"]["trust_remote_code"], False)
        self.assertEqual(phi["prompt_sha256"], qwen["prompt_sha256"])
        self.assertNotEqual(
            scenario_payload.payload_sha256(phi), scenario_payload.payload_sha256(qwen)
        )

    def test_payload_rejects_wrong_case_count_and_oversized_prompt(self) -> None:
        cases, manifest, digest = _fixture()
        with patch.object(
            scenario_payload, "load_scenarios", return_value=(cases[:47], manifest, digest)
        ):
            with self.assertRaisesRegex(ValueError, "48"):
                scenario_payload.build_payload("qwen3-4b")
        oversized = deepcopy(cases)
        oversized[0]["messages"][0]["text"] = "x" * 110_000
        with patch.object(
            scenario_payload, "load_scenarios", return_value=(oversized, manifest, digest)
        ):
            with self.assertRaisesRegex(ValueError, "payload limit"):
                scenario_payload.build_payload("qwen3-4b")

    def test_payload_hash_is_canonical_and_nonfinite_data_is_rejected(self) -> None:
        payload = {"b": 2, "a": 1}
        self.assertEqual(
            scenario_payload.payload_sha256(payload),
            hashlib.sha256(b'{"a":1,"b":2}').hexdigest(),
        )
        with self.assertRaises(ValueError):
            scenario_payload.payload_sha256({"x": math.nan})


class ScenarioRunnerTests(unittest.TestCase):
    def test_resource_envelope_is_single_use_source_free_and_bounded(self) -> None:
        options = modal_scenarios.modal_function_options()
        self.assertEqual(options["gpu"], "T4")
        self.assertEqual(options["cpu"], (2, 2))
        self.assertEqual(options["memory"], (16_384, 16_384))
        self.assertEqual(options["max_containers"], 1)
        self.assertEqual(options["min_containers"], 0)
        self.assertEqual(options["buffer_containers"], 0)
        self.assertEqual(options["retries"], 0)
        self.assertEqual(options["startup_timeout"], 300)
        self.assertEqual(options["timeout"], 1200)
        self.assertTrue(options["single_use_containers"])
        self.assertTrue(options["serialized"])
        self.assertFalse(options["include_source"])

    def test_prepare_is_default_and_wrong_hash_cannot_launch(self) -> None:
        stdout, stderr = StringIO(), StringIO()
        with _loader_patch(), redirect_stdout(stdout):
            self.assertEqual(modal_scenarios.main(["--model", "qwen3-4b"]), 0)
        prepared = json.loads(stdout.getvalue())
        self.assertFalse(prepared["remote_started"])
        self.assertEqual(prepared["item_count"], 48)
        with _loader_patch(), redirect_stderr(stderr):
            code = modal_scenarios.main(
                ["--model", "qwen3-4b", "--launch", "--approve-payload-sha256", "0" * 64]
            )
        self.assertEqual(code, 2)
        self.assertIn("exact prepared payload", stderr.getvalue())

    def test_fresh_workspace_rejection_prevents_function_creation(self) -> None:
        with _loader_patch():
            payload = scenario_payload.build_payload("qwen3-4b")
        digest = scenario_payload.payload_sha256(payload)
        stderr = StringIO()
        with (
            _loader_patch(),
            patch.object(modal_scenarios, "verify_launch_preconditions"),
            patch.object(
                modal_scenarios,
                "verify_personal_workspace",
                side_effect=RuntimeError("personal workspace mismatch"),
            ),
            patch.object(modal_scenarios, "_build_modal_function") as build_function,
            redirect_stderr(stderr),
        ):
            code = modal_scenarios.main(
                ["--model", "qwen3-4b", "--launch", "--approve-payload-sha256", digest]
            )
        self.assertEqual(code, 2)
        self.assertIn("personal workspace mismatch", stderr.getvalue())
        build_function.assert_not_called()

    def test_modal_sdk_serializes_remote_body_by_value_without_tracemem_imports(self) -> None:
        try:
            from modal._utils.function_utils import FunctionSourceInfo
            from modal._vendor import cloudpickle
        except ModuleNotFoundError as error:
            self.skipTest(f"optional Modal SDK serialization check unavailable: {error.name}")

        function = modal_scenarios._remote_compute
        original_module = function.__module__
        function.__module__ = "__main__"
        try:
            source_info = FunctionSourceInfo(function, serialized=True)
            serialized = source_info.serialized_function()
            restored = cloudpickle.loads(serialized)
        finally:
            function.__module__ = original_module

        self.assertTrue(source_info.is_serialized())
        self.assertGreater(len(serialized), 1_000)
        imports = [
            instruction.argval
            for instruction in dis.get_instructions(restored)
            if instruction.opname == "IMPORT_NAME"
        ]
        self.assertNotIn("tracemem", imports)
        self.assertNotIn("experiments.scenario_payload", imports)


class ScenarioReceiptTests(unittest.TestCase):
    def evaluate(self, receipt: dict[str, object]) -> dict[str, object]:
        data = json.dumps(receipt, ensure_ascii=False, allow_nan=False).encode()
        with _loader_patch():
            return evaluate_scenarios.evaluate_receipt(data)

    def test_receipt_scores_dev_test_raw_and_anchored_and_retains_outputs(self) -> None:
        receipt = _valid_receipt()
        report = self.evaluate(receipt)
        self.assertEqual(
            report["raw_outputs"]["scenario-00"], receipt["generated_outputs"][0]["raw_output"]
        )
        self.assertEqual(report["metrics"]["dev"]["raw"]["exact_match"]["f1"], 1.0)
        self.assertEqual(
            report["metrics"]["test"]["unique_quote_anchored"]["exact_match"]["f1"], 1.0
        )
        self.assertEqual(report["metrics"]["dev"]["raw"]["parse_failures"], 0)
        self.assertEqual(report["provenance"]["modal_function_call_id"], "fc-test")

    def test_phi_receipt_retains_all_pinned_eos_ids(self) -> None:
        report = self.evaluate(_valid_receipt("phi-3.5-mini"))
        self.assertEqual(report["provenance"]["generation_eos_token_ids"], [32007, 32001, 32000])

    def test_eos_at_token_limit_differs_from_length_and_keeps_limit_flag(self) -> None:
        receipt = _valid_receipt()
        eos_row, length_row = receipt["generated_outputs"][:2]
        for row in (eos_row, length_row):
            row["generated_tokens"] = 1024
            row["hit_max_new_tokens"] = True
        eos_row["finish_reason"] = "eos"
        eos_row["generated_last_token_id"] = 42
        length_row["finish_reason"] = "length"
        length_row["generated_last_token_id"] = 999

        report = self.evaluate(receipt)
        self.assertEqual(report["token_totals"]["outputs_at_generation_limit"], 2)
        self.assertEqual(report["raw_generated_outputs"][0]["finish_reason"], "eos")
        self.assertTrue(report["raw_generated_outputs"][0]["hit_max_new_tokens"])
        self.assertEqual(report["raw_generated_outputs"][1]["finish_reason"], "length")

    def test_unique_quote_repair_improves_anchored_score_and_parse_failure_is_retained(
        self,
    ) -> None:
        receipt = _valid_receipt()
        first = receipt["generated_outputs"][0]
        prediction = json.loads(first["raw_output"])
        span = prediction["claims"][0]["evidence"][0]
        span["char_start"] += 1
        span["char_end"] += 1
        first["raw_output"] = json.dumps(prediction, ensure_ascii=False)
        first["output_characters"] = len(first["raw_output"])
        receipt["generated_outputs"][24]["raw_output"] = "{invalid"
        receipt["generated_outputs"][24]["output_characters"] = len("{invalid")

        report = self.evaluate(receipt)
        self.assertLess(report["metrics"]["dev"]["raw"]["exact_match"]["f1"], 1.0)
        self.assertEqual(
            report["metrics"]["dev"]["unique_quote_anchored"]["exact_match"]["f1"], 1.0
        )
        self.assertEqual(report["metrics"]["test"]["raw"]["parse_failures"], 1)
        self.assertEqual(report["raw_outputs"]["scenario-24"], "{invalid")

    def test_receipt_rejects_bad_hash_coverage_provenance_and_nonfinite_metrics(self) -> None:
        original = _valid_receipt()
        for name, mutate, expected in (
            ("hash", lambda value: value.__setitem__("payload_sha256", "c" * 64), "payload hash"),
            ("revision", lambda value: value["model"].__setitem__("revision", "d" * 40), "model"),
            (
                "duplicate",
                lambda value: value["generated_outputs"].__setitem__(
                    47, value["generated_outputs"][0]
                ),
                "duplicate",
            ),
            (
                "token limit",
                lambda value: value["generated_outputs"][0].__setitem__("input_tokens", 4097),
                "token",
            ),
            (
                "finite",
                lambda value: value.__setitem__("gpu_compute_seconds", float("inf")),
                "JSON",
            ),
        ):
            with self.subTest(name=name):
                receipt = deepcopy(original)
                mutate(receipt)
                raw = (
                    json.dumps(receipt, allow_nan=False).encode()
                    if name != "finite"
                    else b'{"bad":NaN}'
                )
                with _loader_patch(), self.assertRaisesRegex(ValueError, expected):
                    evaluate_scenarios.evaluate_receipt(raw)

    def test_evaluation_output_is_exclusive_and_does_not_overwrite_receipt(self) -> None:
        receipt = _valid_receipt()
        receipt_bytes = json.dumps(receipt, ensure_ascii=False).encode()
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "receipt.json", Path(directory) / "report.json"
            source.write_bytes(receipt_bytes)
            with _loader_patch():
                code = modal_scenarios.main(["--evaluate", str(source), "--output", str(output)])
            self.assertEqual(code, 0)
            self.assertEqual(source.read_bytes(), receipt_bytes)
            before = output.read_bytes()
            with redirect_stderr(StringIO()):
                self.assertEqual(
                    modal_scenarios.main(["--evaluate", str(source), "--output", str(output)]),
                    2,
                )
            self.assertEqual(output.read_bytes(), before)
            hardlink = Path(directory) / "receipt-hardlink.json"
            hardlink.hardlink_to(source)
            with redirect_stderr(StringIO()):
                self.assertEqual(
                    modal_scenarios.main(["--evaluate", str(source), "--output", str(hardlink)]),
                    2,
                )
            self.assertEqual(source.read_bytes(), receipt_bytes)
            self.assertEqual(hardlink.read_bytes(), receipt_bytes)


if __name__ == "__main__":
    unittest.main()
