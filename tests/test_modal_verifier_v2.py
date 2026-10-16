"""Offline contract tests for verifier v2 continuation training."""

from __future__ import annotations

import builtins
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from experiments import modal_verifier_v2, verifier_v2_remote
from training.verifier_data import render_training_rows

_ROOT = Path(__file__).resolve().parents[1]
_CORPUS_KEY = "examples/synthetic/verifier-v2-train-dev.json"
_PAIRED_KEY = "examples/synthetic/verifier-pairs-v1.json"


def _fixture_corpus() -> dict[str, object]:
    cases: list[dict[str, object]] = []
    for index in range(120):
        split = "train" if index < 96 else "dev"
        suffix = f"{index:03d}"
        source_text = (
            f"For fixture {suffix}, the operator says the east ramp is closed until noon, "
            "but I have not checked."
        )
        candidates = []
        for candidate_index, (claim, support) in enumerate(
            (
                (f"The operator says the east ramp closes for fixture {suffix}.", "supported"),
                (f"The east ramp is confirmed closed for fixture {suffix}.", "unsupported"),
            )
        ):
            candidates.append(
                {
                    "candidate_id": f"candidate-{suffix}-{candidate_index}",
                    "claim": claim,
                    "message_id": f"message-{suffix}-user",
                    "quote": source_text,
                    "valid_from": None,
                    "valid_to": None,
                    "support": support,
                    "eligibility": "eligible",
                    "support_reason": "The source wording determines attribution.",
                    "eligibility_reason": "This is a practical update about the user's trip.",
                }
            )
        cases.append(
            {
                "case_id": f"case-{suffix}",
                "split": split,
                "scenario_group": f"scenario-{suffix}",
                "template_group": f"template-{suffix}",
                "entity_ids": [f"entity-{suffix}"],
                "phenomenon": "attribution",
                "messages": [
                    {
                        "session_id": f"session-{suffix}",
                        "message_id": f"message-{suffix}-assistant",
                        "role": "assistant",
                        "text": f"What does the update say in fixture {suffix}?",
                        "recorded_at": "2026-01-01T10:00:00Z",
                        "content_kind": "conversation",
                    },
                    {
                        "session_id": f"session-{suffix}",
                        "message_id": f"message-{suffix}-user",
                        "role": "user",
                        "text": source_text,
                        "recorded_at": "2026-01-01T10:01:00Z",
                        "content_kind": "conversation",
                    },
                ],
                "candidates": candidates,
            }
        )
    return {
        "version": 1,
        "license": "MIT",
        "annotation_status": "agent_authored",
        "provenance": "newly_authored_fiction",
        "scope_policy": "personal-memory-research-v1",
        "cases": cases,
    }


@contextmanager
def _fixed_sources() -> Iterator[tuple[dict[str, object], str, dict[str, object]]]:
    corpus = _fixture_corpus()
    corpus_bytes = json.dumps(corpus, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(corpus_bytes).hexdigest()
    paired = json.loads(modal_verifier_v2._PAIRED_DATASET_PATH.read_text(encoding="utf-8"))
    test_case = next(case for case in paired["cases"] if case["split"] == "test")
    candidate_message_id = test_case["candidates"][0]["message_id"]
    sentinel = "TEST_SENTINEL_ONLY_IN_OLD_TEST_ROWS"
    for message in test_case["messages"]:
        if message["message_id"] == candidate_message_id:
            message["text"] += " " + sentinel
    for candidate in test_case["candidates"]:
        if candidate["message_id"] == candidate_message_id:
            candidate["quote"] += " " + sentinel
    paired_bytes = json.dumps(paired, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    paired_digest = hashlib.sha256(paired_bytes).hexdigest()
    with tempfile.TemporaryDirectory() as directory:
        corpus_path = Path(directory) / "verifier-v2-train-dev.json"
        corpus_path.write_bytes(corpus_bytes)
        paired_path = Path(directory) / "verifier-pairs-v1.json"
        paired_path.write_bytes(paired_bytes)
        hashes = dict(modal_verifier_v2._SOURCE_HASHES)
        hashes[_CORPUS_KEY] = digest
        hashes[_PAIRED_KEY] = paired_digest
        with (
            patch.object(modal_verifier_v2, "_DATASET_PATH", corpus_path),
            patch.object(modal_verifier_v2, "_PAIRED_DATASET_PATH", paired_path),
            patch.object(modal_verifier_v2, "_SOURCE_HASHES", hashes),
        ):
            yield hashes, digest, corpus


def _remote_validation_fixture(
    root: Path, *, relative_path: str = "runs/prior/adapter", symlink_prior: bool = False
) -> tuple[object, dict[str, object], Path, Path, Path]:
    with _fixed_sources():
        payload = modal_verifier_v2.build_payload()
        remote_function = modal_verifier_v2._make_remote_function(
            None, payload["runner_sha256"], payload
        )

    prior_mount = root / "prior-volume"
    prior_mount.mkdir()
    prior_root = prior_mount / relative_path
    output_mount = root / "output-volume"
    output_mount.mkdir()
    config_bytes = b"fixture adapter config"
    weights_bytes = b"fixture adapter weights"
    if symlink_prior:
        target = root / "outside-prior-volume"
        target.mkdir()
        prior_root.parent.mkdir(parents=True, exist_ok=True)
        prior_root.symlink_to(target, target_is_directory=True)
    else:
        prior_root.mkdir(parents=True)
        (prior_root / "adapter_config.json").write_bytes(config_bytes)
        (prior_root / "adapter_model.safetensors").write_bytes(weights_bytes)

    adapter = dict(payload["prior_adapter"])
    adapter.update(
        volume_mount=str(prior_mount),
        relative_path=relative_path,
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        safetensors_sha256=hashlib.sha256(weights_bytes).hexdigest(),
    )
    payload["prior_adapter"] = adapter
    digest = modal_verifier_v2.payload_sha256(payload)
    remote_globals = remote_function.__globals__
    remote_globals["_EXPECTED_PAYLOAD_SHA256"] = digest
    remote_globals["_EXPECTED_PRIOR_ADAPTER"] = adapter
    remote_globals["_EXPECTED_PRIOR_ADAPTER_ROOT"] = str(prior_root)
    remote_globals["_EXPECTED_OUTPUT_MOUNT"] = str(output_mount)
    request = {
        "payload": payload,
        "payload_sha256": digest,
        "runner_sha256": payload["runner_sha256"],
    }
    return remote_function, request, prior_root, prior_mount, output_mount


class ModalVerifierV2Tests(unittest.TestCase):
    def test_parity_differences_collect_every_field_in_item_order(self) -> None:
        items = [
            {"item_id": "case-2::candidate-0", "cohort": "transfer_regression"},
            {"item_id": "case-1::candidate-0", "cohort": "transfer_regression"},
            {"item_id": "case-3::candidate-0", "cohort": "new_dev"},
        ]
        expected_baseline = [
            {
                "item_id": "case-1::candidate-0",
                "finish_reason": "eos",
                "output_token_ids": [1, 2],
                "raw_output": "supported",
            },
            {
                "item_id": "case-2::candidate-0",
                "finish_reason": "eos",
                "output_token_ids": [3, 2],
                "raw_output": "unsupported",
            },
        ]
        actual_baseline = [
            {
                "item_id": "case-2::candidate-0",
                "finish_reason": "length",
                "output_token_ids": [3, 2],
                "raw_output": "unsupported",
            },
            {
                "item_id": "case-1::candidate-0",
                "finish_reason": "eos",
                "output_token_ids": [1, 7, 2],
                "raw_output": "changed",
            },
        ]
        expected_prior = [
            {
                "item_id": "case-1::candidate-0",
                "finish_reason": "eos",
                "output_token_ids": [4, 2],
                "raw_output": "supported",
            },
            {
                "item_id": "case-2::candidate-0",
                "finish_reason": "eos",
                "output_token_ids": [5, 2],
                "raw_output": "unsupported",
            },
        ]
        actual_prior = [
            {
                "item_id": "case-2::candidate-0",
                "finish_reason": "eos",
                "output_token_ids": [5, 2],
                "raw_output": "unsupported",
            },
            {
                "item_id": "case-1::candidate-0",
                "finish_reason": "length",
                "output_token_ids": [4, 2],
                "raw_output": "supported",
            },
        ]

        differences = verifier_v2_remote._collect_parity_differences(
            items,
            actual_baseline,
            actual_prior,
            {
                "baseline_outputs": expected_baseline,
                "tuned_outputs": expected_prior,
            },
        )

        self.assertEqual(
            differences,
            [
                {
                    "item_id": "case-2::candidate-0",
                    "arm": "baseline_outputs",
                    "field": "finish_reason",
                    "expected": "eos",
                    "actual": "length",
                },
                {
                    "item_id": "case-1::candidate-0",
                    "arm": "baseline_outputs",
                    "field": "output_token_ids",
                    "expected": [1, 2],
                    "actual": [1, 7, 2],
                },
                {
                    "item_id": "case-1::candidate-0",
                    "arm": "baseline_outputs",
                    "field": "raw_output",
                    "expected": "supported",
                    "actual": "changed",
                },
                {
                    "item_id": "case-1::candidate-0",
                    "arm": "prior_outputs",
                    "field": "finish_reason",
                    "expected": "eos",
                    "actual": "length",
                },
            ],
        )

    def test_run_once_preserves_remote_protocol_failure_receipt(self) -> None:
        from experiments import sft_launch

        items = [
            {"item_id": f"case-{index:03d}::candidate-0", "cohort": "transfer_regression"}
            for index in range(108)
        ]
        output = {
            "item_id": "",
            "raw_output": "captured",
            "output_token_ids": [1, 2],
        }
        remote_receipt = {
            "schema_version": 1,
            "status": "protocol_failed",
            "mode": "continue_train",
            "split": "dev",
            "payload_sha256": "a" * 64,
            "runner_sha256": "b" * 64,
            "source_sha256": {"helper": "c" * 64},
            "model": {"model_id": "fixture-model"},
            "settings": {"seed": 17},
            "prior_adapter_artifact_sha256": {"adapter_model.safetensors": "d" * 64},
            "versions": {"torch": "fixture-version"},
            "tokenizer": {"chat_template_sha256": "e" * 64},
            "modal_function_call_id": "fc-fixture",
            "baseline_outputs": [dict(output, item_id=item["item_id"]) for item in items],
            "prior_outputs": [dict(output, item_id=item["item_id"]) for item in items],
            "parity_differences": [
                {
                    "item_id": items[3]["item_id"],
                    "arm": "baseline_outputs",
                    "field": "raw_output",
                    "expected": "historical",
                    "actual": "captured",
                }
            ],
            "training_performed": False,
            "training": {"row_count": 0, "microsteps": 0, "optimizer_steps": 0},
            "timings": {"baseline_seconds": 12.5, "prior_seconds": 13.5},
            "hardware": {"gpu_name": "fixture-gpu"},
        }

        class FakeApp:
            app_id = "ap-fixture"

            @contextmanager
            def run(self, *, detach: bool) -> Iterator[None]:
                self.detach = detach
                yield

        class FakeFunction:
            def remote(self, _request: dict[str, object]) -> dict[str, object]:
                return remote_receipt

        payload = {"items": items}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_path = root / "receipt.json"
            lock_path = root / "lock"
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            with (
                patch.object(
                    modal_verifier_v2,
                    "_validate_current_payload",
                    return_value=(payload, "a" * 64, "b" * 64),
                ),
                patch.object(
                    modal_verifier_v2,
                    "_build_modal_function",
                    return_value=(FakeApp(), FakeFunction()),
                ),
                patch.object(sft_launch, "_acquire_launch_lock", return_value=lock_fd),
                patch.object(sft_launch, "verify_personal_workspace"),
                patch.object(sft_launch, "verify_modal_inventory", return_value=[]),
            ):
                receipt = modal_verifier_v2.run_once(
                    payload,
                    approved_payload_sha256="a" * 64,
                    approved_runner_sha256="b" * 64,
                    output_path=output_path,
                )

            written = json.loads(output_path.read_text(encoding="utf-8"))

        for key, value in remote_receipt.items():
            self.assertEqual(receipt[key], value)
            self.assertEqual(written[key], value)
        self.assertEqual(len(receipt["baseline_outputs"]), 108)
        self.assertEqual(len(receipt["prior_outputs"]), 108)
        self.assertEqual(receipt["status"], "protocol_failed")
        self.assertFalse(receipt["training_performed"])
        self.assertEqual(receipt["training"]["optimizer_steps"], 0)
        self.assertEqual(receipt["modal_app_id"], "ap-fixture")

    def test_main_exits_nonzero_for_remote_protocol_failure_receipt(self) -> None:
        runner_hash = hashlib.sha256(Path(modal_verifier_v2.__file__).read_bytes()).hexdigest()
        payload = {"runner_sha256": runner_hash}
        digest = modal_verifier_v2.payload_sha256(payload)
        failure_receipt = {"status": "protocol_failed", "parity_differences": [{}]}
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(modal_verifier_v2, "build_payload", return_value=payload),
                patch.object(modal_verifier_v2, "run_once", return_value=failure_receipt),
                redirect_stdout(output),
            ):
                exit_code = modal_verifier_v2.main(
                    [
                        "launch",
                        "--approve-payload-sha256",
                        digest,
                        "--approve-runner-sha256",
                        runner_hash,
                        "--output",
                        str(Path(directory) / "receipt.json"),
                    ]
                )

        self.assertNotEqual(exit_code, 0)
        self.assertIn('"status": "protocol_failed"', output.getvalue())

    def test_remote_function_serializes_below_modal_size_limit(self) -> None:
        from modal._serialization import serialize

        with _fixed_sources():
            payload = modal_verifier_v2.build_payload()
            remote_function = modal_verifier_v2._make_remote_function(
                None, payload["runner_sha256"], payload
            )
        serialized = serialize(remote_function)
        self.assertLess(len(serialized), 60 * 1024)
        helper = remote_function.__globals__["_collect_parity_differences"]
        self.assertEqual(helper.__module__, "__main__")
        self.assertIsNot(helper, verifier_v2_remote._collect_parity_differences)

    def test_remote_rejects_prior_adapter_alias_and_symlink_before_ml_imports(self) -> None:
        original_import = builtins.__import__

        def reject_ml_imports(name: str, *args: object, **kwargs: object) -> object:
            if name.split(".", maxsplit=1)[0] in {"modal", "torch", "peft", "transformers"}:
                raise AssertionError(f"ML_IMPORT_SENTINEL: {name}")
            return original_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixtures = (
                {"relative_path": "../outside-prior-volume/adapter"},
                {"symlink_prior": True},
            )
            for index, options in enumerate(fixtures):
                case_root = root / str(index)
                case_root.mkdir()
                remote_function, request, _prior_root, _prior_mount, _output_mount = (
                    _remote_validation_fixture(case_root, **options)
                )
                with patch("builtins.__import__", side_effect=reject_ml_imports):
                    with self.assertRaisesRegex(ValueError, "prior adapter directory is invalid"):
                        remote_function(request)

    def test_remote_rejects_existing_or_symlinked_output_run_before_ml_imports(self) -> None:
        original_import = builtins.__import__

        def reject_ml_imports(name: str, *args: object, **kwargs: object) -> object:
            if name.split(".", maxsplit=1)[0] in {"modal", "torch", "peft", "transformers"}:
                raise AssertionError(f"ML_IMPORT_SENTINEL: {name}")
            return original_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, use_symlink in enumerate((False, True)):
                case_root = root / str(index)
                case_root.mkdir()
                remote_function, request, _prior_root, _prior_mount, output_mount = (
                    _remote_validation_fixture(case_root)
                )
                runs_root = output_mount / "runs"
                runs_root.mkdir()
                run_path = runs_root / request["payload_sha256"]
                if use_symlink:
                    run_path.symlink_to(case_root / "missing-run")
                else:
                    run_path.mkdir()
                with patch("builtins.__import__", side_effect=reject_ml_imports):
                    with self.assertRaisesRegex(
                        ValueError, "mounted volumes or prior artifacts are unsafe"
                    ) as raised:
                        remote_function(request)
                self.assertIsInstance(raised.exception.__cause__, FileExistsError)

    def test_launch_lock_and_inventory_failures_write_local_failure_receipts(self) -> None:
        from experiments import sft_launch

        with _fixed_sources():
            payload = modal_verifier_v2.build_payload()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = (
                (
                    "lock",
                    RuntimeError("another SFT launcher is already in progress"),
                    "another SFT launcher is already in progress",
                ),
                (
                    "inventory",
                    RuntimeError("fresh Modal app inventory check failed"),
                    "fresh Modal app inventory check failed",
                ),
            )
            for failure, exception, expected_message in cases:
                with self.subTest(failure=failure):
                    output = root / f"{failure}.json"
                    lock_patch = (
                        patch.object(
                            sft_launch,
                            "_acquire_launch_lock",
                            side_effect=exception,
                        )
                        if failure == "lock"
                        else patch.object(sft_launch, "_LAUNCH_LOCK", root / "launch.lock")
                    )
                    with (
                        lock_patch,
                        patch.object(sft_launch, "verify_personal_workspace") as workspace,
                        patch.object(
                            sft_launch, "verify_modal_inventory", side_effect=exception
                        ) as inventory,
                        patch.object(
                            modal_verifier_v2,
                            "_build_modal_function",
                            side_effect=AssertionError("failed preflight reached Modal build"),
                        ) as build,
                    ):
                        with _fixed_sources():
                            with self.assertRaisesRegex(RuntimeError, expected_message):
                                modal_verifier_v2.run_once(
                                    payload,
                                    approved_payload_sha256=modal_verifier_v2.payload_sha256(
                                        payload
                                    ),
                                    approved_runner_sha256=payload["runner_sha256"],
                                    output_path=output,
                                )
                    receipt = json.loads(output.read_text(encoding="utf-8"))
                    self.assertEqual(receipt["status"], "launch_failed")
                    self.assertFalse(receipt["remote_started"])
                    self.assertEqual(receipt["error"]["type"], "RuntimeError")
                    self.assertEqual(receipt["error"]["message"], expected_message)
                    build.assert_not_called()
                    if failure == "lock":
                        workspace.assert_not_called()
                        inventory.assert_not_called()
                    else:
                        workspace.assert_called_once_with()
                        inventory.assert_called_once_with()

    def test_payload_has_only_approved_train_rows_and_three_fixed_eval_cohorts(self) -> None:
        with _fixed_sources() as (source_hashes, _digest, corpus):
            payload = modal_verifier_v2.build_payload()

        self.assertEqual(payload["mode"], "continue_train")
        self.assertEqual(payload["counts"]["train_cases"], 96)
        self.assertEqual(payload["counts"]["train_rows"], 192)
        self.assertEqual(payload["counts"]["new_dev_cases"], 24)
        self.assertEqual(payload["counts"]["new_dev_candidates"], 48)
        self.assertEqual(payload["counts"]["transfer_regression_candidates"], 48)
        self.assertEqual(payload["counts"]["original_paired_dev_candidates"], 12)
        self.assertEqual(payload["counts"]["evaluation_candidates"], 108)
        self.assertEqual(len(payload["train_rows"]), 192)
        self.assertEqual(len(payload["items"]), 108)
        self.assertEqual(
            {item["cohort"] for item in payload["items"]},
            {"new_dev", "transfer_regression", "original_paired_dev"},
        )
        self.assertEqual(
            {
                item["cohort"]: sum(row["cohort"] == item["cohort"] for row in payload["items"])
                for item in payload["items"]
            },
            {"new_dev": 48, "transfer_regression": 48, "original_paired_dev": 12},
        )
        self.assertEqual(payload["source_sha256"], source_hashes)
        self.assertEqual(payload["train_rows"], render_training_rows(corpus, "train"))
        self.assertTrue(all(set(row) == {"prompt", "completion"} for row in payload["train_rows"]))
        self.assertNotIn(
            "TEST_SENTINEL_ONLY_IN_OLD_TEST_ROWS",
            json.dumps(payload, ensure_ascii=False),
        )
        self.assertTrue(any("fixture 096" in item["prompt"] for item in payload["items"]))
        self.assertFalse(
            any(
                "fixture 096" in row["prompt"] or "fixture 096" in row["completion"]
                for row in payload["train_rows"]
            )
        )
        item_ids = {item["item_id"] for item in payload["items"]}
        self.assertFalse(any("test" in item_id.lower() for item_id in item_ids))
        self.assertFalse(any("sealed" in item_id.lower() for item_id in item_ids))

    def test_payload_metadata_is_deep_copied(self) -> None:
        with _fixed_sources():
            first = modal_verifier_v2.build_payload()
            first["model"]["quantization"]["type"] = "changed"
            first["settings"]["seed"] = 999
            first["prior_adapter"]["config_sha256"] = "0" * 64
            second = modal_verifier_v2.build_payload()

        self.assertEqual(second["model"]["quantization"]["type"], "nf4")
        self.assertEqual(second["settings"]["seed"], 17)
        self.assertNotEqual(second["prior_adapter"]["config_sha256"], "0" * 64)

    def test_changed_corpus_hash_fails_before_payload_rendering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus_path = Path(directory) / "corpus.json"
            corpus_path.write_text("{}", encoding="utf-8")
            hashes = dict(modal_verifier_v2._SOURCE_HASHES)
            hashes[_CORPUS_KEY] = "0" * 64
            with (
                patch.object(modal_verifier_v2, "_DATASET_PATH", corpus_path),
                patch.object(modal_verifier_v2, "_SOURCE_HASHES", hashes),
            ):
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    modal_verifier_v2.build_payload()

    def test_modal_envelope_uses_one_source_free_a10g_call_without_retries(self) -> None:
        options = modal_verifier_v2.modal_function_options()
        self.assertEqual(options["gpu"], "A10G")
        self.assertEqual(options["timeout"], 1800)
        self.assertEqual(options["retries"], 0)
        self.assertEqual(options["max_containers"], 1)
        self.assertEqual(options["min_containers"], 0)
        self.assertTrue(options["single_use_containers"])
        self.assertTrue(options["serialized"])
        self.assertFalse(options["include_source"])

    def test_modal_builder_mounts_read_only_prior_and_separate_output_volumes(self) -> None:
        captured: dict[str, object] = {"created_volumes": []}

        class FakeImage:
            def pip_install(self, *_packages: str) -> FakeImage:
                return self

        class FakeVolume:
            def __init__(self, name: str, read_only: bool = False) -> None:
                self.name = name
                self.read_only = read_only

            def with_mount_options(self, *, read_only: bool) -> FakeVolume:
                return FakeVolume(self.name, read_only=read_only)

        class FakeApp:
            def __init__(self, _name: str) -> None:
                pass

            def function(self, *, image: object, **options: object):
                captured["function_options"] = options
                return lambda function: function

        class FakeImageFactory:
            @staticmethod
            def debian_slim(**_kwargs: object) -> FakeImage:
                return FakeImage()

        class FakeVolumeFactory:
            @staticmethod
            def from_name(name: str, *, create_if_missing: bool) -> FakeVolume:
                volume = FakeVolume(name)
                captured["created_volumes"].append((name, create_if_missing, volume))
                return volume

        modal_module = ModuleType("modal")
        modal_module.Image = FakeImageFactory  # type: ignore[attr-defined]
        modal_module.App = FakeApp  # type: ignore[attr-defined]
        modal_module.Volume = FakeVolumeFactory  # type: ignore[attr-defined]
        with _fixed_sources():
            payload = modal_verifier_v2.build_payload()
            with patch.dict(sys.modules, {"modal": modal_module}):
                modal_verifier_v2._build_modal_function(payload)

        created_volumes = captured["created_volumes"]
        prior_record, output_record = created_volumes
        self.assertEqual(
            [(name, create_if_missing) for name, create_if_missing, _ in created_volumes],
            [
                (modal_verifier_v2._PRIOR_ADAPTER["volume_name"], False),
                (modal_verifier_v2._OUTPUT_VOLUME_NAME, True),
            ],
        )
        options = captured["function_options"]
        volumes = options["volumes"]
        prior = volumes[modal_verifier_v2._PRIOR_ADAPTER["volume_mount"]]
        output = volumes[modal_verifier_v2._OUTPUT_VOLUME_MOUNT]
        self.assertIsNot(prior, output)
        self.assertTrue(prior.read_only)
        self.assertFalse(output.read_only)
        self.assertEqual(prior.name, prior_record[0])
        self.assertEqual(output.name, output_record[0])

    def test_remote_rejects_changed_config_and_prompt_before_ml_imports(self) -> None:
        with _fixed_sources():
            base_payload = modal_verifier_v2.build_payload()
            remote_function = modal_verifier_v2._make_remote_function(
                None, base_payload["runner_sha256"]
            )
        original_import = builtins.__import__

        def reject_ml_imports(name: str, *args: object, **kwargs: object) -> object:
            if name.split(".", maxsplit=1)[0] in {"modal", "torch", "peft", "transformers"}:
                raise AssertionError(f"ML_IMPORT_SENTINEL: {name}")
            return original_import(name, *args, **kwargs)

        for mutate, expected_message, adjust_allowlist in (
            (lambda payload: payload["settings"].update(seed=99), "model, settings", True),
            (
                lambda payload: payload["items"][0].update(prompt="changed"),
                "payload SHA-256",
                False,
            ),
        ):
            payload = json.loads(json.dumps(base_payload))
            mutate(payload)
            digest = modal_verifier_v2.payload_sha256(payload)
            remote_function.__globals__["_EXPECTED_PAYLOAD_SHA256"] = (
                modal_verifier_v2.payload_sha256(base_payload)
            )
            if adjust_allowlist:
                remote_function.__globals__["_EXPECTED_PAYLOAD_SHA256"] = digest
            request = {
                "payload": payload,
                "payload_sha256": digest,
                "runner_sha256": payload["runner_sha256"],
            }
            with patch("builtins.__import__", side_effect=reject_ml_imports):
                with self.assertRaises(ValueError) as raised:
                    remote_function(request)
            self.assertNotIn("ML_IMPORT_SENTINEL", str(raised.exception))
            self.assertIn(expected_message, str(raised.exception))

    def test_run_once_rejects_stale_approval_before_receipt_or_workspace_checks(self) -> None:
        from experiments import sft_launch

        with _fixed_sources():
            payload = modal_verifier_v2.build_payload()
        changed = json.loads(json.dumps(payload))
        changed["items"][0]["prompt"] += " changed after approval"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not-created" / "receipt.json"
            with (
                patch.object(
                    sft_launch,
                    "_reserve_output",
                    side_effect=AssertionError("invalid approval reached receipt reservation"),
                ),
                patch.object(
                    sft_launch,
                    "verify_personal_workspace",
                    side_effect=AssertionError("invalid approval reached workspace check"),
                ),
                patch.object(
                    sft_launch,
                    "verify_modal_inventory",
                    side_effect=AssertionError("invalid approval reached inventory check"),
                ),
                patch.object(
                    modal_verifier_v2,
                    "_build_modal_function",
                    side_effect=AssertionError("invalid approval reached Modal construction"),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "current fixed verifier v2 payload"):
                    modal_verifier_v2.run_once(
                        changed,
                        approved_payload_sha256=modal_verifier_v2.payload_sha256(changed),
                        approved_runner_sha256=payload["runner_sha256"],
                        output_path=output,
                    )
            self.assertFalse(output.parent.exists())


if __name__ == "__main__":
    unittest.main()
