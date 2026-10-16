from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


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


def _quote_v4_train_payload() -> dict[str, object]:
    source_cases = _payload("train", 72, 12)["cases"]
    cases: list[dict[str, object]] = []
    for family in range(72):
        kind = "REPLAY" if family < 48 else "TRAIN"
        family_number = family if family < 48 else family - 48
        count = 12 if family < 48 else 8
        for variant in range(count):
            case = copy.deepcopy(source_cases[family * 12 + variant])
            case["case_id"] = f"V4T-V3T-{kind}-{family_number:02d}-V{variant + 1:02d}"
            case["scenario_group"] = f"v4-{kind.lower()}-family-{family_number:02d}"
            case["template_group"] = f"v4-{kind.lower()}-template-{family_number:02d}"
            cases.append(case)
    return {**_payload("train", 1, 1), "cases": cases}


def _prefixed_eval_payload(split: str, count: int) -> dict[str, object]:
    payload = _payload(split, count, 1)
    for index, case in enumerate(payload["cases"]):
        prefix = f"v4-{split}"
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
    return payload


def _build_quote_v4(
    *,
    mode: str = "preflight",
    run_id: str | None = None,
    train_payload: dict[str, object] | None = None,
    dev_payload: dict[str, object] | None = None,
    test_payload: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, str]]:
    from experiments import modal_sft

    payloads = {
        "train": train_payload or _quote_v4_train_payload(),
        "dev": dev_payload or _prefixed_eval_payload("dev", 12),
        "test": test_payload or _prefixed_eval_payload("test", 24),
    }
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        paths = {name: root / f"{name}.json" for name in payloads}
        for name, payload in payloads.items():
            paths[name].write_text(json.dumps(payload), encoding="utf-8")
        source_sha256 = {
            name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()
        }
        constants = {
            "_QUOTE_V4_TRAIN_PATH": paths["train"],
            "_QUOTE_V4_DEV_PATH": paths["dev"],
            "_QUOTE_V4_TEST_PATH": paths["test"],
        }
        with patch.multiple(modal_sft, create=True, **constants):
            job = modal_sft.build_payload(mode, run_id=run_id, profile="quote-v4")
        return job, source_sha256


class QuoteV4LaunchTests(unittest.TestCase):
    def test_quote_v4_payload_binds_reviewed_rows_cohorts_and_prior_manifest(self) -> None:
        from experiments import modal_sft
        from training import train_sft

        job, source_sha256 = _build_quote_v4()
        request = job["request"]

        self.assertEqual(request["profile"], "quote-v4")
        self.assertEqual(len(request["train_rows"]), 768)
        self.assertEqual(len(request["dev_prompts"]), 36)
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
                for prefix in ("V4T-V3T-REPLAY-", "V4T-V3T-TRAIN-")
            },
            {"V4T-V3T-REPLAY-": 576, "V4T-V3T-TRAIN-": 192},
        )
        self.assertEqual([len(request["eval_cohorts"][name]) for name in ("dev", "test")], [12, 24])
        self.assertEqual(
            [row["case_id"] for row in request["dev_prompts"]],
            [*request["eval_cohorts"]["dev"], *request["eval_cohorts"]["test"]],
        )
        self.assertEqual(request["prior_adapter"], train_sft._QUOTE_V4_PRIOR_ADAPTER)
        summary = modal_sft.build_summary(job)
        self.assertEqual(summary["eval_cohort_counts"], {"dev": 12, "test": 24})
        self.assertEqual(summary["prior_adapter"], train_sft._QUOTE_V4_PRIOR_ADAPTER)
        self.assertEqual(
            job["source_sha256"],
            source_sha256,
        )
        self.assertEqual(
            set(job["module_sha256"]),
            {
                "training/prepare_sft.py",
                "training/train_sft.py",
                "training/remote_sft.py",
                "experiments/sft_launch.py",
                "experiments/scenario_payload.py",
                "src/tracemem/challenge.py",
                "src/tracemem/prompts.py",
                "training/quote_data.py",
                "training/build_quote_data.py",
                "training/build_quote_v3.py",
                "training/build_reviewed_quote_train.py",
                "training/review_quote_targets.py",
            },
        )
        for relative_path, digest in job["module_sha256"].items():
            self.assertEqual(
                digest,
                hashlib.sha256((modal_sft._ROOT / relative_path).read_bytes()).hexdigest(),
            )
        self.assertEqual(job["resource_options"], modal_sft.modal_function_options("preflight"))

    def test_quote_v4_job_copies_the_fixed_prior_manifest(self) -> None:
        from training import train_sft

        pinned = copy.deepcopy(train_sft._QUOTE_V4_PRIOR_ADAPTER)
        first, _ = _build_quote_v4()
        first["request"]["prior_adapter"]["sha256"]["adapter_config.json"] = "changed"
        second, _ = _build_quote_v4()

        self.assertEqual(second["request"]["prior_adapter"], pinned)

    def test_quote_v4_profile_fixes_training_and_generation_limits(self) -> None:
        from training.train_sft import resolve_sft_profile

        profile = resolve_sft_profile("quote-v4")

        self.assertEqual(
            (
                profile.train_rows,
                profile.train_steps,
                profile.dev_prompts,
                profile.dev_max_new_tokens,
                profile.train_context_tokens,
                profile.dev_context_tokens,
            ),
            (768, 192, 36, 1024, 2048, 4096),
        )

    def test_quote_v4_source_guards_reject_bad_train_ids_and_split_leakage(self) -> None:
        bad_train = _quote_v4_train_payload()
        bad_train["cases"][0]["case_id"] = "V3T-REPLAY-00-V01"
        with self.assertRaisesRegex(ValueError, "V4T-V3T-REPLAY or V4T-V3T-TRAIN"):
            _build_quote_v4(train_payload=bad_train)

        leaked_dev = _prefixed_eval_payload("dev", 12)
        leaked_dev["cases"][0]["scenario_group"] = _quote_v4_train_payload()["cases"][0][
            "scenario_group"
        ]
        with self.assertRaisesRegex(ValueError, "appears in both train and dev"):
            _build_quote_v4(dev_payload=leaked_dev)

    def test_remote_preflight_requires_ordered_disjoint_v4_cohorts(self) -> None:
        from training.train_sft import _QUOTE_V4_PRIOR_ADAPTER, run_remote_request

        class FakeTokenizer:
            eos_token_id = 2
            pad_token_id = 2

            def apply_chat_template(self, *_args: object, **_kwargs: object) -> list[int]:
                return [11, 12]

            def encode(self, *_args: object, **_kwargs: object) -> list[int]:
                return [21, 22]

        train_rows = [
            {
                "case_id": f"V4T-V3T-REPLAY-{index:04d}",
                "split": "train",
                "messages": [
                    {"role": "user", "content": f"source prompt {index}"},
                    {"role": "assistant", "content": '{"claims":[]}'},
                ],
            }
            for index in range(768)
        ]
        cohorts = {
            "dev": [f"dev-{index}" for index in range(12)],
            "test": [f"test-{index}" for index in range(24)],
        }
        request = {
            "mode": "preflight",
            "profile": "quote-v4",
            "train_rows": train_rows,
            "dev_prompts": [
                {"case_id": case_id, "prompt": "synthetic prompt"}
                for case_id in (*cohorts["dev"], *cohorts["test"])
            ],
            "eval_cohorts": cohorts,
            "prior_adapter": copy.deepcopy(_QUOTE_V4_PRIOR_ADAPTER),
            "remote_run_dir": None,
        }
        with (
            patch.dict(os.environ, {"TRACEMEM_SFT_REMOTE": "1"}),
            patch("training.train_sft._check_dependency_versions", return_value={}),
            patch("training.train_sft._load_pinned_tokenizer", return_value=FakeTokenizer()),
            patch("training.train_sft._tokenizer_provenance", return_value={}),
        ):
            report = run_remote_request(request)

        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["limits"]["max_train_steps"], 192)
        self.assertEqual(report["limits"]["dev_max_new_tokens"], 1024)
        self.assertEqual(report["dev"]["prompt_count"], 36)

        request["dev_prompts"].reverse()
        with (
            patch.dict(os.environ, {"TRACEMEM_SFT_REMOTE": "1"}),
            patch("training.train_sft._check_dependency_versions", return_value={}),
            patch("training.train_sft._load_pinned_tokenizer", return_value=FakeTokenizer()),
            patch("training.train_sft._tokenizer_provenance", return_value={}),
        ):
            with self.assertRaisesRegex(ValueError, "follow eval_cohorts order"):
                run_remote_request(request)

    def test_v4_run_path_and_modal_volumes_are_isolated(self) -> None:
        from experiments import modal_sft, sft_launch

        job, _ = _build_quote_v4(mode="train", run_id="v4-run")
        self.assertEqual(sft_launch._run_id_from_job(job), "v4-run")
        job["request"]["remote_run_dir"] = "/mnt/tracemem-sft-quote-v3/adapters/v4-run"
        with self.assertRaisesRegex(ValueError, "fixed remote run path"):
            sft_launch._run_id_from_job(job)

        captured: dict[str, object] = {"volumes": {}}

        class Volume:
            def __init__(self, name: str) -> None:
                self.name = name
                self.read_only = False

            def with_mount_options(self, *, read_only: bool) -> Volume:
                result = Volume(self.name)
                result.read_only = read_only
                return result

        class Image:
            @staticmethod
            def debian_slim(*, python_version: str) -> Image:
                return Image()

            def pip_install(self, *_packages: str) -> Image:
                return self

            def add_local_file(self, *_args: object, **_kwargs: object) -> Image:
                captured.setdefault("uploaded_files", []).append(
                    (str(_args[0]), _kwargs["remote_path"], _kwargs["copy"])
                )
                return self

        class App:
            def __init__(self, name: str) -> None:
                captured["app_name"] = name

            def function(self, *, image: Image, **options: object):
                captured["options"] = options
                return lambda function: function

        def from_name(name: str, *, create_if_missing: bool) -> Volume:
            captured.setdefault("created_volumes", []).append((name, create_if_missing))
            return Volume(name)

        fake_modal = SimpleNamespace(
            Image=Image,
            App=App,
            Volume=SimpleNamespace(from_name=from_name),
        )
        with patch.dict("sys.modules", {"modal": fake_modal}):
            sft_launch._build_modal_function(job)

        self.assertEqual(captured["app_name"], "tracemem-sft-quote-v4")
        self.assertEqual(
            captured["created_volumes"],
            [
                ("tracemem-sft-quote-v4", True),
                ("tracemem-sft-quote-v3", False),
            ],
        )
        options = captured["options"]
        self.assertEqual(
            set(options["volumes"]),
            {"/mnt/tracemem-sft-quote-v4", "/mnt/tracemem-sft-quote-v3"},
        )
        self.assertFalse(options["volumes"]["/mnt/tracemem-sft-quote-v4"].read_only)
        self.assertTrue(options["volumes"]["/mnt/tracemem-sft-quote-v3"].read_only)
        self.assertIs(options["include_source"], False)
        self.assertEqual(
            [
                (Path(local).relative_to(modal_sft._ROOT).as_posix(), remote, copy_flag)
                for local, remote, copy_flag in captured["uploaded_files"]
            ],
            [
                ("training/train_sft.py", "/root/training/train_sft.py", True),
                ("training/remote_sft.py", "/root/training/remote_sft.py", True),
            ],
        )

    def test_v4_approval_rebuild_binds_current_synthetic_sources(self) -> None:
        from experiments import modal_sft, sft_launch

        payloads = {
            "train": _quote_v4_train_payload(),
            "dev": _prefixed_eval_payload("dev", 12),
            "test": _prefixed_eval_payload("test", 24),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {name: root / f"{name}.json" for name in payloads}
            for name, payload in payloads.items():
                paths[name].write_text(json.dumps(payload), encoding="utf-8")
            constants = {
                "_QUOTE_V4_TRAIN_PATH": paths["train"],
                "_QUOTE_V4_DEV_PATH": paths["dev"],
                "_QUOTE_V4_TEST_PATH": paths["test"],
            }
            with patch.multiple(modal_sft, create=True, **constants):
                job = modal_sft.build_payload(profile="quote-v4")
                rebuilt = sft_launch._validate_approved_job(job, job["job_sha256"])
                tampered = copy.deepcopy(job)
                tampered["request"]["dev_prompts"][0]["prompt"] += " changed"
                with self.assertRaisesRegex(ValueError, "exact approved job SHA-256"):
                    sft_launch._validate_approved_job(tampered, job["job_sha256"])

        self.assertEqual(rebuilt, job)


if __name__ == "__main__":
    unittest.main()
