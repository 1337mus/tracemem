"""Offline contract checks for the bounded Modal SFT launcher."""

from __future__ import annotations

import builtins
import contextlib
import hashlib
import importlib
import io
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class ModalSFTPayloadTests(unittest.TestCase):
    def test_cli_sends_only_exactly_approved_job_to_lazy_launch_helper(self) -> None:
        runner = importlib.import_module("experiments.modal_sft")
        job = runner.build_payload()
        receipt_path = Path("/tmp/tracemem-sft-cli-receipt.json")
        calls: list[dict[str, object]] = []
        helper = types.ModuleType("experiments.sft_launch")

        def record_run_once(
            supplied_job: dict[str, object],
            *,
            approved_job_sha256: str,
            output_path: Path,
        ) -> dict[str, object]:
            calls.append(
                {
                    "job": supplied_job,
                    "approved_job_sha256": approved_job_sha256,
                    "output_path": output_path,
                }
            )
            return {}

        helper.run_once = record_run_once  # type: ignore[attr-defined]
        package = importlib.import_module("experiments")
        stdout = io.StringIO()
        with (
            patch.dict(sys.modules, {"experiments.sft_launch": helper}),
            patch.object(package, "sft_launch", helper, create=True),
        ):
            with contextlib.redirect_stdout(stdout):
                exit_code = runner.main(
                    [
                        "--launch",
                        "--mode",
                        "preflight",
                        "--approve-job-sha256",
                        job["job_sha256"],
                        "--output",
                        str(receipt_path),
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["job"], job)
        self.assertEqual(calls[0]["approved_job_sha256"], job["job_sha256"])
        self.assertEqual(calls[0]["output_path"], receipt_path)
        self.assertIs(json.loads(stdout.getvalue())["remote_started"], True)

    def test_launch_without_exact_approval_and_output_is_refused_before_helper_import(self) -> None:
        runner = importlib.import_module("experiments.modal_sft")
        original_import = builtins.__import__
        stderr = io.StringIO()

        def reject_launch_helper(name: str, *args: object, **kwargs: object) -> object:
            if name == "experiments.sft_launch":
                raise AssertionError("launch helper imported before required approvals")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=reject_launch_helper):
            with contextlib.redirect_stderr(stderr):
                exit_code = runner.main(["--launch", "--mode", "preflight"])

        self.assertEqual(exit_code, 2)
        self.assertIn("requires an exact job approval", stderr.getvalue())

    def test_launch_refuses_stale_job_hash_before_importing_helper(self) -> None:
        runner = importlib.import_module("experiments.modal_sft")
        original_import = builtins.__import__
        stderr = io.StringIO()

        def reject_launch_helper(name: str, *args: object, **kwargs: object) -> object:
            if name == "experiments.sft_launch":
                raise AssertionError("stale approval reached launch helper")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=reject_launch_helper):
            with contextlib.redirect_stderr(stderr):
                exit_code = runner.main(
                    [
                        "--launch",
                        "--mode",
                        "preflight",
                        "--approve-job-sha256",
                        "0" * 64,
                        "--output",
                        "/tmp/tracemem-stale-sft-receipt.json",
                    ]
                )

        self.assertEqual(exit_code, 2)
        self.assertIn("exact current job approval is required", stderr.getvalue())

    def test_offline_cli_prints_summary_without_importing_remote_dependencies(self) -> None:
        runner = importlib.import_module("experiments.modal_sft")
        original_import = builtins.__import__
        stdout = io.StringIO()
        stderr = io.StringIO()

        def reject_remote_imports(name: str, *args: object, **kwargs: object) -> object:
            if name == "modal" or name == "torch" or name.startswith("transformers"):
                raise AssertionError(f"offline preparation imported {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=reject_remote_imports):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exit_code = runner.main(["--mode", "preflight"])

        summary = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(summary["train_row_count"], 128)
        self.assertEqual(summary["dev_prompt_count"], 8)
        self.assertIs(summary["remote_started"], False)

    def test_build_summary_exposes_only_fingerprints_and_row_counts(self) -> None:
        runner = importlib.import_module("experiments.modal_sft")

        summary = runner.build_summary(runner.build_payload())

        self.assertEqual(
            set(summary),
            {
                "schema_version",
                "mode",
                "train_row_count",
                "dev_prompt_count",
                "job_sha256",
                "runner_sha256",
                "source_sha256",
                "module_sha256",
                "remote_started",
            },
        )
        self.assertEqual(summary["train_row_count"], 128)
        self.assertEqual(summary["dev_prompt_count"], 8)
        self.assertIs(summary["remote_started"], False)

    def test_build_payload_is_offline_and_contains_only_train_rows_and_dev_prompts(self) -> None:
        runner = importlib.import_module("experiments.modal_sft")
        original_import = builtins.__import__

        def reject_remote_ml_imports(name: str, *args: object, **kwargs: object) -> object:
            if name == "modal" or name == "torch" or name.startswith("transformers"):
                raise AssertionError(f"offline payload construction imported {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=reject_remote_ml_imports):
            job = runner.build_payload()

        request = job["request"]
        self.assertEqual(set(request), {"mode", "train_rows", "dev_prompts", "remote_run_dir"})
        self.assertEqual(request["mode"], "preflight")
        self.assertIsNone(request["remote_run_dir"])
        self.assertEqual(len(request["train_rows"]), 128)
        self.assertEqual(len({row["case_id"] for row in request["train_rows"]}), 128)
        for row in request["train_rows"]:
            self.assertEqual(
                set(row), {"case_id", "split", "scenario_group", "template_group", "messages"}
            )
            self.assertEqual(row["split"], "train")
            self.assertEqual(
                [message["role"] for message in row["messages"]], ["user", "assistant"]
            )

        self.assertEqual(len(request["dev_prompts"]), 8)
        self.assertEqual(len({row["case_id"] for row in request["dev_prompts"]}), 8)
        self.assertFalse(
            {row["case_id"] for row in request["train_rows"]}
            & {row["case_id"] for row in request["dev_prompts"]}
        )
        for row in request["dev_prompts"]:
            self.assertEqual(set(row), {"case_id", "prompt"})
            self.assertIsInstance(row["prompt"], str)

    def test_job_hash_binds_data_module_resource_and_dependency_fingerprints(self) -> None:
        runner = importlib.import_module("experiments.modal_sft")
        job = runner.build_payload()

        self.assertEqual(
            job["source_sha256"],
            {
                "train": hashlib.sha256(runner._TRAIN_PATH.read_bytes()).hexdigest(),
                "dev": hashlib.sha256(runner._DEV_PATH.read_bytes()).hexdigest(),
            },
        )
        expected_module_names = {
            "training/prepare_sft.py",
            "training/train_sft.py",
            "training/remote_sft.py",
            "experiments/sft_launch.py",
            "experiments/scenario_payload.py",
            "src/tracemem/challenge.py",
            "src/tracemem/prompts.py",
        }
        self.assertEqual(set(job["module_sha256"]), expected_module_names)
        for relative_path, digest in job["module_sha256"].items():
            source = runner._ROOT / relative_path
            self.assertEqual(digest, hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertEqual(
            job["runner_sha256"],
            hashlib.sha256(Path(runner.__file__).read_bytes()).hexdigest(),
        )
        unsigned = {key: value for key, value in job.items() if key != "job_sha256"}
        canonical = json.dumps(
            unsigned,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        self.assertEqual(job["job_sha256"], hashlib.sha256(canonical).hexdigest())
        self.assertEqual(job["resource_options"], runner.modal_function_options("preflight"))
        self.assertEqual(job["dependency_pins"], runner.dependency_pins("preflight"))


if __name__ == "__main__":
    unittest.main()
