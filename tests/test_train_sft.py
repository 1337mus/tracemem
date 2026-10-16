"""Focused tests for SFT token masking and the remote training boundary."""

from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from training.train_sft import (
    _tokenizer_provenance,
    collate_examples,
    encode_sft_example,
    preflight_train_rows,
    resolve_sft_profile,
    run_remote_request,
    select_training_rows,
)


class FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 2

    def __init__(self) -> None:
        self.template_calls: list[tuple[list[dict[str, str]], dict[str, object]]] = []
        self.encode_calls: list[tuple[str, bool]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> list[int]:
        self.template_calls.append(
            (
                messages,
                {
                    "tokenize": tokenize,
                    "add_generation_prompt": add_generation_prompt,
                    "enable_thinking": enable_thinking,
                },
            )
        )
        return [11, 12]

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        self.encode_calls.append((text, add_special_tokens))
        return [21, 22]


def _row() -> dict[str, object]:
    return {
        "case_id": "case-1",
        "split": "train",
        "messages": [
            {"role": "user", "content": "user source prompt"},
            {"role": "assistant", "content": '{"claims":[]}'},
        ],
    }


class LengthTokenizer(FakeTokenizer):
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> list[int]:
        return [11] * len(messages[0]["content"])

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        return [21] * len(text)


class TrainSftTests(unittest.TestCase):
    def test_quote_v3_profile_uses_fixed_training_and_eval_limits(self) -> None:
        profile = resolve_sft_profile("quote-v3")

        self.assertEqual(
            (
                profile.train_rows,
                profile.train_steps,
                profile.dev_prompts,
                profile.dev_max_new_tokens,
                profile.train_context_tokens,
                profile.dev_context_tokens,
            ),
            (768, 192, 96, 512, 2048, 4096),
        )

    def test_encode_masks_user_prefix_and_supervises_completion_eos(self) -> None:
        tokenizer = FakeTokenizer()

        encoded = encode_sft_example(_row(), tokenizer)

        self.assertEqual(encoded["input_ids"], [11, 12, 21, 22, 2])
        self.assertEqual(encoded["labels"], [-100, -100, 21, 22, 2])
        self.assertEqual(encoded["attention_mask"], [1, 1, 1, 1, 1])
        self.assertEqual(encoded["prefix_tokens"], 2)
        self.assertEqual(encoded["supervised_tokens"], 3)
        self.assertEqual(
            tokenizer.template_calls[0][0],
            [{"role": "user", "content": "user source prompt"}],
        )
        self.assertEqual(
            tokenizer.template_calls[0][1],
            {"tokenize": True, "add_generation_prompt": True, "enable_thinking": False},
        )
        self.assertEqual(tokenizer.encode_calls, [('{"claims":[]}', False)])

    def test_remote_request_rejects_local_execution(self) -> None:
        with patch.dict(os.environ, {"TRACEMEM_SFT_REMOTE": "0"}):
            with self.assertRaisesRegex(RuntimeError, "remote runtime"):
                run_remote_request({"mode": "preflight"})

    def test_remote_preflight_preserves_preselected_training_row_order(self) -> None:
        rows = [{**_row(), "case_id": f"case-{index}"} for index in range(128)]
        dev_prompts = [{"case_id": f"dev-{index}", "prompt": "source prompt"} for index in range(8)]
        request = {
            "mode": "preflight",
            "train_rows": rows,
            "dev_prompts": dev_prompts,
            "remote_run_dir": None,
        }
        with (
            patch.dict(os.environ, {"TRACEMEM_SFT_REMOTE": "1"}),
            patch("training.train_sft._check_dependency_versions", return_value={}),
            patch("training.train_sft._load_pinned_tokenizer", return_value=FakeTokenizer()),
            patch("training.train_sft._tokenizer_provenance", return_value={}),
        ):
            report = run_remote_request(request)

        self.assertEqual(
            [row["case_id"] for row in report["train"]["rows"]],
            [row["case_id"] for row in rows],
        )

    def test_remote_preflight_requires_null_run_directory(self) -> None:
        request = {
            "mode": "preflight",
            "train_rows": [{**_row(), "case_id": f"case-{index}"} for index in range(128)],
            "dev_prompts": [
                {"case_id": f"dev-{index}", "prompt": "source prompt"} for index in range(8)
            ],
            "remote_run_dir": "/remote/run",
        }
        with (
            patch.dict(os.environ, {"TRACEMEM_SFT_REMOTE": "1"}),
            patch("training.train_sft._check_dependency_versions", return_value={}),
            patch("training.train_sft._load_pinned_tokenizer", return_value=FakeTokenizer()),
        ):
            with self.assertRaisesRegex(ValueError, "remote_run_dir must be null"):
                run_remote_request(request)

    def test_remote_preflight_rejects_unknown_chat_template(self) -> None:
        request = {
            "mode": "preflight",
            "train_rows": [{**_row(), "case_id": f"case-{index}"} for index in range(128)],
            "dev_prompts": [
                {"case_id": f"dev-{index}", "prompt": "source prompt"} for index in range(8)
            ],
            "remote_run_dir": None,
        }
        with (
            patch.dict(os.environ, {"TRACEMEM_SFT_REMOTE": "1"}),
            patch("training.train_sft._check_dependency_versions", return_value={}),
            patch("training.train_sft._load_pinned_tokenizer", return_value=FakeTokenizer()),
        ):
            with self.assertRaisesRegex(RuntimeError, "chat template"):
                run_remote_request(request)

    def test_tokenizer_provenance_records_template_and_backend_hashes(self) -> None:
        tokenizer = FakeTokenizer()
        tokenizer.chat_template = "pinned test template"
        backend_json = '{"model":"test"}'
        tokenizer.backend_tokenizer = SimpleNamespace(to_str=lambda: backend_json)
        template_hash = hashlib.sha256(tokenizer.chat_template.encode("utf-8")).hexdigest()
        backend_hash = hashlib.sha256(backend_json.encode("utf-8")).hexdigest()
        versions = {"transformers": "4.51.3", "tokenizers": "0.21.1", "jinja2": "3.1.6"}

        with patch("training.train_sft.EXPECTED_CHAT_TEMPLATE_SHA256", template_hash):
            provenance = _tokenizer_provenance(tokenizer, versions)

        self.assertEqual(provenance["chat_template_sha256"], template_hash)
        self.assertEqual(provenance["backend_tokenizer_sha256"], backend_hash)
        self.assertEqual(provenance["dependency_versions"], versions)

    def test_remote_training_rejects_a_reused_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as existing_dir:
            request = {
                "mode": "train",
                "train_rows": [{**_row(), "case_id": f"case-{index}"} for index in range(128)],
                "dev_prompts": [
                    {"case_id": f"dev-{index}", "prompt": "source prompt"} for index in range(8)
                ],
                "remote_run_dir": existing_dir,
            }
            with patch.dict(os.environ, {"TRACEMEM_SFT_REMOTE": "1"}):
                with self.assertRaisesRegex(ValueError, "must be fresh"):
                    run_remote_request(request)

    def test_remote_train_dispatches_to_lazily_loaded_training_module(self) -> None:
        request = {
            "mode": "train",
            "train_rows": [{**_row(), "case_id": f"case-{index}"} for index in range(128)],
            "dev_prompts": [
                {"case_id": f"dev-{index}", "prompt": "source prompt"} for index in range(8)
            ],
            "remote_run_dir": "/remote/fresh-run",
        }
        remote_module = unittest.mock.Mock()
        remote_module.run_training.return_value = {"mode": "train", "status": "completed"}
        with (
            patch.dict(os.environ, {"TRACEMEM_SFT_REMOTE": "1"}),
            patch(
                "training.train_sft._load_remote_sft",
                create=True,
                return_value=remote_module,
            ),
        ):
            result = run_remote_request(request)

        self.assertEqual(result, {"mode": "train", "status": "completed"})
        remote_module.run_training.assert_called_once()
        self.assertEqual(remote_module.run_training.call_args.args[0], request)

    def test_preflight_rejects_dev_targets(self) -> None:
        dev_prompts = [{"case_id": f"dev-{index}", "prompt": "source prompt"} for index in range(8)]
        dev_prompts[0]["expected"] = []

        with self.assertRaisesRegex(ValueError, "only case_id and prompt"):
            preflight_train_rows([_row()], FakeTokenizer(), dev_prompts=dev_prompts)

    def test_preflight_requires_exactly_128_preselected_training_rows(self) -> None:
        rows = [{**_row(), "case_id": f"case-{index}"} for index in range(127)]
        dev_prompts = [{"case_id": f"dev-{index}", "prompt": "source prompt"} for index in range(8)]

        with self.assertRaisesRegex(ValueError, "exactly 128 rows"):
            preflight_train_rows(
                rows, FakeTokenizer(), dev_prompts=dev_prompts, selected_in_order=True
            )

    def test_preflight_rejects_non_train_rows_in_preselected_input(self) -> None:
        rows = [{**_row(), "case_id": f"case-{index}"} for index in range(128)]
        rows[0]["split"] = "dev"
        dev_prompts = [{"case_id": f"dev-{index}", "prompt": "source prompt"} for index in range(8)]

        with self.assertRaisesRegex(ValueError, "split='train'"):
            preflight_train_rows(
                rows, FakeTokenizer(), dev_prompts=dev_prompts, selected_in_order=True
            )

    def test_preflight_reports_all_training_and_dev_lengths_before_rejection(self) -> None:
        rows = [
            {
                "case_id": "case-short",
                "split": "train",
                "messages": [
                    {"role": "user", "content": "p"},
                    {"role": "assistant", "content": "t"},
                ],
            },
            {
                "case_id": "case-long",
                "split": "train",
                "messages": [
                    {"role": "user", "content": "p" * 21},
                    {"role": "assistant", "content": "t"},
                ],
            },
        ]
        dev_prompts = [
            {"case_id": f"dev-{index}", "prompt": "d" if index < 7 else "d" * 3600}
            for index in range(8)
        ]

        report = preflight_train_rows(
            rows, LengthTokenizer(), dev_prompts=dev_prompts, max_context_tokens=20, seed=17
        )

        self.assertEqual(report["status"], "rejected_oversized")
        train_rows = {item["case_id"]: item for item in report["train"]["rows"]}
        self.assertEqual(train_rows["case-short"]["total_tokens"], 3)
        self.assertTrue(train_rows["case-short"]["prefix_masked"])
        self.assertTrue(train_rows["case-short"]["eos_supervised"])
        self.assertGreater(train_rows["case-long"]["total_tokens"], 20)
        self.assertFalse(train_rows["case-long"]["within_context_limit"])
        self.assertEqual(len(report["dev"]["rows"]), 8)
        dev_rows = {item["case_id"]: item for item in report["dev"]["rows"]}
        self.assertEqual(dev_rows["dev-7"]["max_new_tokens"], 1024)
        self.assertGreater(dev_rows["dev-7"]["required_context_tokens"], 4096)
        self.assertFalse(dev_rows["dev-7"]["within_context_limit"])

    def test_training_selection_is_seeded_shuffled_and_bounded(self) -> None:
        rows = [{"case_id": f"case-{index}"} for index in range(200)]

        selected = select_training_rows(rows, max_rows=128, seed=17)

        self.assertEqual(len(selected), 128)
        self.assertEqual(selected, select_training_rows(rows, max_rows=128, seed=17))
        self.assertNotEqual(selected, rows[:128])
        self.assertEqual(
            {row["case_id"] for row in selected},
            {row["case_id"] for row in select_training_rows(rows, max_rows=128, seed=17)},
        )

    def test_encode_rejects_oversized_sequence_without_truncating(self) -> None:
        tokenizer = FakeTokenizer()

        with self.assertRaisesRegex(ValueError, "over the 4-token limit"):
            encode_sft_example(_row(), tokenizer, max_context_tokens=4)

    def test_collation_masks_padding_even_when_pad_equals_eos(self) -> None:
        examples = [
            {"input_ids": [1, 2, 2], "attention_mask": [1, 1, 1], "labels": [-100, 2, 2]},
            {"input_ids": [3, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
        ]

        batch = collate_examples(examples, pad_token_id=2)

        self.assertEqual(batch["input_ids"], [[1, 2, 2], [3, 2, 2]])
        self.assertEqual(batch["attention_mask"], [[1, 1, 1], [1, 1, 0]])
        self.assertEqual(batch["labels"], [[-100, 2, 2], [-100, 2, -100]])

    def test_quote_v2_preflight_uses_its_fixed_row_and_prompt_counts(self) -> None:
        rows = [{**_row(), "case_id": f"case-{index}"} for index in range(768)]
        dev_prompts = [
            {"case_id": f"dev-{index}", "prompt": "source prompt"} for index in range(24)
        ]

        report = preflight_train_rows(
            rows,
            FakeTokenizer(),
            dev_prompts=dev_prompts,
            selected_in_order=True,
            profile="quote-v2",
        )

        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["limits"]["max_train_rows"], 768)
        self.assertEqual(report["limits"]["max_train_steps"], 192)
        self.assertEqual(report["train"]["selected_rows"], 768)
        self.assertEqual(report["dev"]["prompt_count"], 24)
        self.assertEqual(report["limits"]["dev_max_new_tokens"], 512)

    def test_remote_quote_v2_preflight_accepts_only_its_fixed_profile_field(self) -> None:
        request = {
            "mode": "preflight",
            "profile": "quote-v2",
            "train_rows": [{**_row(), "case_id": f"case-{index}"} for index in range(768)],
            "dev_prompts": [
                {"case_id": f"dev-{index}", "prompt": "source prompt"} for index in range(24)
            ],
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
        self.assertEqual(report["limits"]["profile"], "quote-v2")
        self.assertEqual(report["limits"]["max_train_rows"], 768)
        self.assertEqual(report["limits"]["max_train_steps"], 192)
        self.assertEqual(report["dev"]["prompt_count"], 24)

    def test_remote_quote_v3_preflight_accepts_bound_eval_cohorts_and_prior_adapter(self) -> None:
        cohorts = {
            "dev": [f"dev-{index}" for index in range(24)],
            "regression": [f"regression-{index}" for index in range(24)],
            "test": [f"test-{index}" for index in range(48)],
        }
        request = {
            "mode": "preflight",
            "profile": "quote-v3",
            "train_rows": [{**_row(), "case_id": f"case-{index}"} for index in range(768)],
            "dev_prompts": [
                {"case_id": case_id, "prompt": "source prompt"}
                for case_id in (*cohorts["regression"], *cohorts["dev"], *cohorts["test"])
            ],
            "eval_cohorts": cohorts,
            "prior_adapter": {
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
        self.assertEqual(report["limits"]["profile"], "quote-v3")
        self.assertEqual(report["limits"]["max_train_rows"], 768)
        self.assertEqual(report["limits"]["max_train_steps"], 192)
        self.assertEqual(report["dev"]["prompt_count"], 96)


if __name__ == "__main__":
    unittest.main()
