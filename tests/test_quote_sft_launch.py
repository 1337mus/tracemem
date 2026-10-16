from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock, patch


def _payload(split: str, family_count: int, variants_per_family: int) -> dict[str, object]:
    cases: list[dict[str, object]] = []
    for family in range(family_count):
        for variant in range(variants_per_family):
            case_id = f"{split}-case-{family:02d}-{variant:02d}"
            message_id = f"{split}-message-{family:02d}-{variant:02d}"
            text = f"Fictional source line for {case_id}."
            cases.append(
                {
                    "case_id": case_id,
                    "split": split,
                    "scenario_group": f"{split}-scenario-{family:02d}",
                    "template_group": f"{split}-template-{family:02d}",
                    "entity_ids": [f"{split}-entity-{family:02d}-{variant:02d}"],
                    "tags": ["synthetic"],
                    "messages": [
                        {
                            "session_id": f"{split}-session-{family:02d}-{variant:02d}",
                            "message_id": message_id,
                            "role": "user",
                            "text": text,
                            "recorded_at": "2026-01-02T10:00:00Z",
                            "content_kind": "conversation",
                        }
                    ],
                    "expected": [
                        {
                            "claim": f"The user mentioned {case_id}.",
                            "message_id": message_id,
                            "quote": text,
                            "valid_from": None,
                            "valid_to": None,
                        }
                    ],
                }
            )
    return {
        "version": 2,
        "license": "MIT",
        "annotation_status": "agent_authored",
        "provenance": "newly_authored_fiction",
        "cases": cases,
    }


def _build_from_payloads(
    train_payload: dict[str, object], dev_payload: dict[str, object]
) -> dict[str, object]:
    from experiments import modal_sft

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        train_path = root / "train.json"
        dev_path = root / "dev.json"
        train_path.write_text(json.dumps(train_payload), encoding="utf-8")
        dev_path.write_text(json.dumps(dev_payload), encoding="utf-8")
        with (
            patch.object(modal_sft, "_QUOTE_TRAIN_PATH", train_path, create=True),
            patch.object(modal_sft, "_QUOTE_DEV_PATH", dev_path, create=True),
        ):
            return modal_sft.build_payload(profile="quote-v2")


def _quote_v3_train_payload() -> dict[str, object]:
    cases: list[dict[str, object]] = []
    source = _payload("train", 72, 12)["cases"]
    for family in range(72):
        variants = 12 if family < 48 else 8
        for variant in range(variants):
            case = deepcopy(source[family * 12 + variant])
            kind = "REPLAY" if family < 48 else "TRAIN"
            family_number = family if family < 48 else family - 48
            case["case_id"] = f"V3T-{kind}-{family_number:02d}-V{variant + 1:02d}"
            case["scenario_group"] = f"v3-{kind.lower()}-family-{family_number:02d}"
            case["template_group"] = f"v3-{kind.lower()}-template-{family_number:02d}"
            cases.append(case)
    return {**_payload("train", 1, 1), "cases": cases}


def _prefix_quote_cases(payload: dict[str, object], prefix: str) -> dict[str, object]:
    result = deepcopy(payload)
    for index, case in enumerate(result["cases"]):
        case["case_id"] = f"{prefix}-case-{index:02d}"
        case["scenario_group"] = f"{prefix}-scenario-{index:02d}"
        case["template_group"] = f"{prefix}-template-{index:02d}"
        case["entity_ids"] = [f"{prefix}-entity-{index:02d}"]
        message = case["messages"][0]
        message["session_id"] = f"{prefix}-session-{index:02d}"
        message["message_id"] = f"{prefix}-message-{index:02d}"
        message["text"] = f"Fictional source line for {prefix}-{index:02d}."
        for claim in case["expected"]:
            claim["message_id"] = message["message_id"]
            claim["quote"] = message["text"]
    return result


def _build_quote_v3_from_payloads(
    train: dict[str, object],
    regression: dict[str, object],
    dev: dict[str, object],
    test: dict[str, object],
    *,
    mode: str = "preflight",
    run_id: str | None = None,
) -> dict[str, object]:
    from experiments import modal_sft

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        paths = {
            "_QUOTE_V3_TRAIN_PATH": root / "train.json",
            "_QUOTE_V3_REGRESSION_PATH": root / "regression.json",
            "_QUOTE_V3_DEV_PATH": root / "dev.json",
            "_QUOTE_V3_TEST_PATH": root / "test.json",
        }
        payloads = {"train": train, "regression": regression, "dev": dev, "test": test}
        raw_by_name: dict[str, bytes] = {}
        for name, path in paths.items():
            key = name.removeprefix("_QUOTE_V3_").removesuffix("_PATH").lower()
            raw_by_name[key] = json.dumps(payloads[key]).encode("utf-8")
            path.write_bytes(raw_by_name[key])
        with unittest.mock.patch.multiple(modal_sft, create=True, **paths):
            job = modal_sft.build_payload(mode, run_id=run_id, profile="quote-v3")
    return job


class QuoteSftPayloadTests(unittest.TestCase):
    def test_modal_cli_accepts_quote_v3_profile(self) -> None:
        from experiments import modal_sft

        args = modal_sft.build_parser().parse_args(["--profile", "quote-v3"])

        self.assertEqual(args.profile, "quote-v3")

    def test_quote_v3_jobs_copy_the_fixed_prior_adapter_manifest(self) -> None:
        from training import train_sft

        sources = (
            _quote_v3_train_payload(),
            _prefix_quote_cases(_payload("dev", 24, 1), "regression"),
            _prefix_quote_cases(_payload("dev", 24, 1), "v3-dev"),
            _prefix_quote_cases(_payload("test", 48, 1), "v3-test"),
        )
        pinned = deepcopy(train_sft._QUOTE_V3_PRIOR_ADAPTER)
        try:
            first_job = _build_quote_v3_from_payloads(*sources)
            first_job["request"]["prior_adapter"]["sha256"]["adapter_config.json"] = "changed"

            second_job = _build_quote_v3_from_payloads(*sources)

            self.assertEqual(second_job["request"]["prior_adapter"], pinned)
        finally:
            train_sft._QUOTE_V3_PRIOR_ADAPTER.clear()
            train_sft._QUOTE_V3_PRIOR_ADAPTER.update(pinned)

    def test_quote_v3_payload_is_prompt_only_and_binds_all_fixed_sources(self) -> None:
        train = _quote_v3_train_payload()
        regression = _prefix_quote_cases(_payload("dev", 24, 1), "regression")
        dev = _prefix_quote_cases(_payload("dev", 24, 1), "v3-dev")
        test = _prefix_quote_cases(_payload("test", 48, 1), "v3-test")

        job = _build_quote_v3_from_payloads(train, regression, dev, test)

        request = job["request"]
        self.assertEqual(request["profile"], "quote-v3")
        self.assertEqual(len(request["train_rows"]), 768)
        self.assertEqual(len(request["dev_prompts"]), 96)
        self.assertEqual(
            set(request),
            {
                "mode",
                "profile",
                "train_rows",
                "dev_prompts",
                "eval_cohorts",
                "prior_adapter",
                "remote_run_dir",
            },
        )
        self.assertEqual(
            {
                prefix: sum(row["case_id"].startswith(prefix) for row in request["train_rows"])
                for prefix in ("V3T-REPLAY-", "V3T-TRAIN-")
            },
            {"V3T-REPLAY-": 576, "V3T-TRAIN-": 192},
        )
        self.assertEqual(
            [len(request["eval_cohorts"][name]) for name in ("regression", "dev", "test")],
            [24, 24, 48],
        )
        self.assertEqual(
            request["prior_adapter"],
            {
                "path": "/mnt/tracemem-sft-quote-v2/adapters/quote-v2-20261015-01/adapter",
                "sha256": {
                    "adapter_model.safetensors": (
                        "3b30cbf98a685792734d8c16ffd7f31e000707f8f2b72a74e5ca9bc364214093"
                    ),
                    "adapter_config.json": (
                        "d6cb445350cae1da7b9508d1cf4cc9d20f9b638983e7c27a886127db0b28125a"
                    ),
                },
            },
        )
        self.assertEqual(
            [row["case_id"] for row in request["dev_prompts"]],
            [
                *request["eval_cohorts"]["regression"],
                *request["eval_cohorts"]["dev"],
                *request["eval_cohorts"]["test"],
            ],
        )
        self.assertTrue(all(set(row) == {"case_id", "prompt"} for row in request["dev_prompts"]))
        self.assertEqual(
            job["source_sha256"],
            {
                name: hashlib.sha256(json.dumps(payload).encode("utf-8")).hexdigest()
                for name, payload in (
                    ("train", train),
                    ("regression", regression),
                    ("dev", dev),
                    ("test", test),
                )
            },
        )
        self.assertEqual(
            job["module_sha256"]["training/build_quote_data.py"],
            hashlib.sha256(Path("training/build_quote_data.py").read_bytes()).hexdigest(),
        )
        from experiments import modal_sft

        summary = modal_sft.build_summary(job)
        self.assertEqual(summary["eval_cohort_counts"], {"regression": 24, "dev": 24, "test": 48})
        self.assertEqual(summary["prior_adapter"], request["prior_adapter"])

    def test_quote_v2_payload_is_fixed_profile_and_pins_only_train_dev_sources(self) -> None:
        from experiments import modal_sft

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_path = root / "train.json"
            dev_path = root / "dev.json"
            train_bytes = json.dumps(_payload("train", 48, 16)).encode("utf-8")
            dev_bytes = json.dumps(_payload("dev", 24, 1)).encode("utf-8")
            train_path.write_bytes(train_bytes)
            dev_path.write_bytes(dev_bytes)
            with (
                patch.object(modal_sft, "_QUOTE_TRAIN_PATH", train_path, create=True),
                patch.object(modal_sft, "_QUOTE_DEV_PATH", dev_path, create=True),
            ):
                job = modal_sft.build_payload(profile="quote-v2")
                from experiments import sft_launch

                self.assertEqual(sft_launch._validate_approved_job(job, job["job_sha256"]), job)

        request = job["request"]
        self.assertEqual(
            set(request), {"mode", "profile", "train_rows", "dev_prompts", "remote_run_dir"}
        )
        self.assertEqual(request["profile"], "quote-v2")
        self.assertEqual(len(request["train_rows"]), 768)
        self.assertEqual(len(request["dev_prompts"]), 24)
        target = json.loads(request["train_rows"][0]["messages"][1]["content"])
        self.assertEqual(set(target), {"claims"})
        self.assertEqual(
            set(target["claims"][0]), {"claim", "message_id", "quote", "valid_from", "valid_to"}
        )
        self.assertEqual(
            [row["case_id"] for row in request["dev_prompts"]],
            [case["case_id"] for case in _payload("dev", 24, 1)["cases"]],
        )
        from training.train_sft import select_training_rows

        source_rows = [
            {"case_id": f"train-case-{family:02d}-{variant:02d}"}
            for family in range(48)
            for variant in range(16)
        ]
        self.assertEqual(
            [row["case_id"] for row in request["train_rows"]],
            [row["case_id"] for row in select_training_rows(source_rows, max_rows=768, seed=17)],
        )
        self.assertEqual(
            job["source_sha256"],
            {
                "train": hashlib.sha256(train_bytes).hexdigest(),
                "dev": hashlib.sha256(dev_bytes).hexdigest(),
            },
        )
        expected_modules = {
            "training/prepare_sft.py",
            "training/train_sft.py",
            "training/remote_sft.py",
            "experiments/sft_launch.py",
            "experiments/scenario_payload.py",
            "src/tracemem/challenge.py",
            "src/tracemem/prompts.py",
            "training/quote_data.py",
            "training/build_quote_data.py",
        }
        self.assertEqual(set(job["module_sha256"]), expected_modules)
        for relative_path, digest in job["module_sha256"].items():
            self.assertEqual(
                digest,
                hashlib.sha256((modal_sft._ROOT / relative_path).read_bytes()).hexdigest(),
            )

    def test_quote_v2_run_directory_is_bound_to_its_profile_volume(self) -> None:
        from experiments import modal_sft, sft_launch

        job = modal_sft.build_payload("train", run_id="quote-run")
        job["request"]["profile"] = "quote-v2"
        job["request"]["remote_run_dir"] = "/mnt/tracemem-sft-quote-v2/adapters/quote-run"
        job["job_sha256"] = sft_launch.canonical_job_sha256(job)

        self.assertEqual(sft_launch._run_id_from_job(job), "quote-run")

    def test_quote_v3_run_directory_is_bound_to_its_profile_volume(self) -> None:
        from experiments import sft_launch

        sources = (
            _quote_v3_train_payload(),
            _prefix_quote_cases(_payload("dev", 24, 1), "regression"),
            _prefix_quote_cases(_payload("dev", 24, 1), "v3-dev"),
            _prefix_quote_cases(_payload("test", 48, 1), "v3-test"),
        )
        job = _build_quote_v3_from_payloads(*sources, mode="train", run_id="quote-v3-run")

        self.assertEqual(sft_launch._run_id_from_job(job), "quote-v3-run")
        job["request"]["remote_run_dir"] = "/mnt/tracemem-sft-quote-v2/adapters/quote-v3-run"
        with self.assertRaisesRegex(ValueError, "fixed remote run path"):
            sft_launch._run_id_from_job(job)
        job["request"]["profile"] = "quote-v2"
        job["request"]["remote_run_dir"] = "/mnt/tracemem-sft-quote-v3/adapters/quote-v3-run"
        with self.assertRaisesRegex(ValueError, "fixed remote run path"):
            sft_launch._run_id_from_job(job)

    def test_quote_v3_approved_job_is_rebuilt_from_its_fixed_sources(self) -> None:
        from experiments import modal_sft, sft_launch

        payloads = {
            "train": _quote_v3_train_payload(),
            "regression": _prefix_quote_cases(_payload("dev", 24, 1), "regression"),
            "dev": _prefix_quote_cases(_payload("dev", 24, 1), "v3-dev"),
            "test": _prefix_quote_cases(_payload("test", 48, 1), "v3-test"),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_paths = {
                "_QUOTE_V3_TRAIN_PATH": root / "train.json",
                "_QUOTE_V3_REGRESSION_PATH": root / "regression.json",
                "_QUOTE_V3_DEV_PATH": root / "dev.json",
                "_QUOTE_V3_TEST_PATH": root / "test.json",
            }
            for key, path in source_paths.items():
                name = key.removeprefix("_QUOTE_V3_").removesuffix("_PATH").lower()
                path.write_text(json.dumps(payloads[name]), encoding="utf-8")
            with patch.multiple(modal_sft, create=True, **source_paths):
                job = modal_sft.build_payload("train", run_id="quote-v3-run", profile="quote-v3")
                rebuilt = sft_launch._validate_approved_job(job, job["job_sha256"])

        self.assertEqual(rebuilt, job)

    def test_quote_v2_launcher_uses_its_own_volume_app_and_two_worker_files(self) -> None:
        from experiments import modal_sft, sft_launch

        job = modal_sft.build_payload("train", run_id="quote-run")
        job["request"]["profile"] = "quote-v2"
        job["request"]["remote_run_dir"] = "/mnt/tracemem-sft-quote-v2/adapters/quote-run"

        image = MagicMock()
        image.pip_install.return_value = image
        image.add_local_file.return_value = image
        app = MagicMock()
        app.function.return_value = lambda function: function
        volume = object()
        modal = MagicMock()
        modal.Image.debian_slim.return_value = image
        modal.Volume.from_name.return_value = volume
        modal.App.return_value = app

        with patch.dict("sys.modules", {"modal": modal}):
            built_app, _function = sft_launch._build_modal_function(job)

        self.assertIs(built_app, app)
        modal.Volume.from_name.assert_called_once_with(
            "tracemem-sft-quote-v2", create_if_missing=True
        )
        modal.App.assert_called_once_with("tracemem-sft-quote-v2")
        self.assertEqual(image.add_local_file.call_count, 2)
        self.assertEqual(
            {call.args[0] for call in image.add_local_file.call_args_list},
            {
                modal_sft._ROOT / "training/train_sft.py",
                modal_sft._ROOT / "training/remote_sft.py",
            },
        )
        self.assertEqual(
            {call.kwargs["remote_path"] for call in image.add_local_file.call_args_list},
            {"/root/training/train_sft.py", "/root/training/remote_sft.py"},
        )
        self.assertEqual(
            app.function.call_args.kwargs["volumes"],
            {"/mnt/tracemem-sft-quote-v2": volume},
        )

    def test_quote_v3_mounts_prior_read_only_and_commits_only_output_after_failure(self) -> None:
        from experiments import sft_launch

        sources = (
            _quote_v3_train_payload(),
            _prefix_quote_cases(_payload("dev", 24, 1), "regression"),
            _prefix_quote_cases(_payload("dev", 24, 1), "v3-dev"),
            _prefix_quote_cases(_payload("test", 48, 1), "v3-test"),
        )
        job = _build_quote_v3_from_payloads(*sources, mode="train", run_id="quote-v3-run")

        with tempfile.TemporaryDirectory() as temporary:
            training_dir = Path(temporary) / "training"
            training_dir.mkdir()
            train_source = (
                "def run_remote_request(request):\n    raise RuntimeError('training failed')\n"
            )
            remote_source = "REMOTE_SOURCE_MARKER = True\n"
            (training_dir / "train_sft.py").write_text(train_source, encoding="utf-8")
            (training_dir / "remote_sft.py").write_text(remote_source, encoding="utf-8")
            job["module_sha256"]["training/train_sft.py"] = hashlib.sha256(
                train_source.encode("utf-8")
            ).hexdigest()
            job["module_sha256"]["training/remote_sft.py"] = hashlib.sha256(
                remote_source.encode("utf-8")
            ).hexdigest()
            job["job_sha256"] = sft_launch.canonical_job_sha256(job)

            class Volume:
                def __init__(self, *, mounted_from: Volume | None = None) -> None:
                    self.commits = 0
                    self.mount_options: dict[str, bool] = {}
                    self.mounted_from = mounted_from

                def commit(self) -> None:
                    self.commits += 1

                def with_mount_options(self, *, read_only: bool) -> Volume:
                    mounted = Volume(mounted_from=self)
                    mounted.mount_options["read_only"] = read_only
                    mounted_volumes.append(mounted)
                    return mounted

            mounted_volumes: list[Volume] = []
            output_volume, prior_volume = Volume(), Volume()
            image = MagicMock()
            image.pip_install.return_value = image
            image.add_local_file.return_value = image
            app = MagicMock()
            app.function.return_value = lambda function: function
            modal = MagicMock()
            modal.Image.debian_slim.return_value = image
            modal.Volume.from_name.side_effect = [output_volume, prior_volume]
            modal.App.return_value = app
            original_make_remote_entry = sft_launch._make_remote_entry

            def make_remote_entry(job_value: dict[str, object], *, volume: object):
                return original_make_remote_entry(
                    job_value, volume=volume, training_dir=training_dir
                )

            with (
                patch.dict("sys.modules", {"modal": modal}),
                patch.object(sft_launch, "_make_remote_entry", side_effect=make_remote_entry),
            ):
                _built_app, remote_entry = sft_launch._build_modal_function(job)
            self.assertEqual(len(mounted_volumes), 1)
            mounted_prior_volume = mounted_volumes[0]

            modal.App.assert_called_once_with("tracemem-sft-quote-v3")
            self.assertEqual(
                [(call.args, call.kwargs) for call in modal.Volume.from_name.call_args_list],
                [
                    (("tracemem-sft-quote-v3",), {"create_if_missing": True}),
                    (("tracemem-sft-quote-v2",), {"create_if_missing": False}),
                ],
            )
            self.assertIsNot(mounted_prior_volume, prior_volume)
            self.assertIs(mounted_prior_volume.mounted_from, prior_volume)
            self.assertEqual(mounted_prior_volume.mount_options, {"read_only": True})
            self.assertEqual(
                app.function.call_args.kwargs["volumes"],
                {
                    "/mnt/tracemem-sft-quote-v3": output_volume,
                    "/mnt/tracemem-sft-quote-v2": mounted_prior_volume,
                },
            )
            self.assertEqual(image.add_local_file.call_count, 2)
            self.assertIs(remote_entry.__globals__["_REMOTE_VOLUME"], output_volume)
            with patch.dict(os.environ, {"TRACEMEM_SFT_REMOTE": "1"}):
                with self.assertRaisesRegex(RuntimeError, "training failed"):
                    remote_entry(job["request"], job["job_sha256"])

            self.assertEqual(output_volume.commits, 1)
            self.assertEqual(prior_volume.commits, 0)
            self.assertEqual(mounted_prior_volume.commits, 0)

    def test_quote_v2_accepts_shared_assistant_filler_in_distinct_conversations(self) -> None:
        train_payload = _payload("train", 48, 16)
        for case in train_payload["cases"]:
            message_number = len(case["messages"]) + 1
            case["messages"].append(
                {
                    "session_id": f"{case['case_id']}-session",
                    "message_id": f"{case['case_id']}-assistant-{message_number}",
                    "role": "assistant",
                    "text": "Thanks for sharing.",
                    "recorded_at": "2026-01-02T10:01:00Z",
                    "content_kind": "conversation",
                }
            )

        job = _build_from_payloads(train_payload, _payload("dev", 24, 1))

        self.assertEqual(len(job["request"]["train_rows"]), 768)

    def test_quote_v2_rejects_duplicate_full_train_conversations_with_new_ids(self) -> None:
        train_payload = _payload("train", 48, 16)
        first, duplicate = train_payload["cases"][:2]
        duplicate["messages"][0]["text"] = first["messages"][0]["text"]
        duplicate["expected"][0]["quote"] = first["messages"][0]["text"]

        with self.assertRaisesRegex(ValueError, "duplicate conversation"):
            _build_from_payloads(train_payload, _payload("dev", 24, 1))

    def test_quote_v2_rejects_duplicate_full_conversation_across_train_and_dev(self) -> None:
        train_payload = _payload("train", 48, 16)
        dev_payload = _payload("dev", 24, 1)
        train_case, dev_case = train_payload["cases"][0], dev_payload["cases"][0]
        dev_case["messages"][0]["text"] = train_case["messages"][0]["text"]
        dev_case["expected"][0]["quote"] = train_case["messages"][0]["text"]

        with self.assertRaisesRegex(ValueError, "duplicate conversation"):
            _build_from_payloads(train_payload, dev_payload)

    def test_quote_v2_retains_global_message_id_and_group_split_guards(self) -> None:
        for guarded_field in ("scenario_group", "template_group", "entity_ids"):
            with self.subTest(guarded_field=guarded_field):
                train_payload = _payload("train", 48, 16)
                dev_payload = _payload("dev", 24, 1)
                train_case, dev_case = train_payload["cases"][0], dev_payload["cases"][0]
                dev_case[guarded_field] = train_case[guarded_field]
                with self.assertRaisesRegex(ValueError, "appears in both train and dev"):
                    _build_from_payloads(train_payload, dev_payload)

        train_payload = _payload("train", 48, 16)
        dev_payload = _payload("dev", 24, 1)
        dev_payload["cases"][0]["messages"][0]["message_id"] = train_payload["cases"][0][
            "messages"
        ][0]["message_id"]
        dev_payload["cases"][0]["expected"][0]["message_id"] = dev_payload["cases"][0]["messages"][
            0
        ]["message_id"]
        with self.assertRaisesRegex(ValueError, "duplicate global message identity"):
            _build_from_payloads(train_payload, dev_payload)


if __name__ == "__main__":
    unittest.main()
