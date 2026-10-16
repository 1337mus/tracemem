"""Tests for the prompt-only quote-dev baseline preparation and runner."""

from __future__ import annotations

import builtins
import dis
import hashlib
import json
import tempfile
import types
import unittest
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from experiments import modal_quote_baseline
from experiments import quote_baseline_payload as quote_payload
from experiments.quote_baseline_payload import build_payload


def _remote_request(payload: dict[str, object] | None = None) -> dict[str, object]:
    ready = build_payload() if payload is None else payload
    return {
        "payload": ready,
        "payload_sha256": quote_payload.payload_sha256(ready),
        "runner_sha256": ready["runner_source_sha256"],
    }


def _fake_runtime(
    *, long_prompt: bool = False
) -> tuple[dict[str, types.ModuleType], dict[str, object]]:
    state: dict[str, object] = {"generate_calls": 0, "tokenize_calls": 0}

    class FakeTensor:
        ndim = 2

        def __init__(self, tokens: int) -> None:
            self.shape = (1, tokens)

        def to(self, device: object) -> FakeTensor:
            return self

        def detach(self) -> FakeTensor:
            return self

        def cpu(self) -> FakeTensor:
            return self

        def tolist(self) -> list[list[int]]:
            return [[11] * self.shape[-1]]

    class FakeToken:
        def item(self) -> int:
            return 7

    class FakeGeneratedIds:
        shape = (1,)

        def detach(self) -> FakeGeneratedIds:
            return self

        def cpu(self) -> FakeGeneratedIds:
            return self

        def tolist(self) -> list[int]:
            return [7]

        def __getitem__(self, index: int) -> FakeToken:
            return FakeToken()

    class FakeGenerated:
        def __getitem__(self, index: tuple[int, slice]) -> FakeGeneratedIds:
            return FakeGeneratedIds()

    class FakeEvent:
        def __init__(self, enable_timing: bool) -> None:
            pass

        def record(self) -> None:
            pass

        def elapsed_time(self, end: FakeEvent) -> float:
            return 1.0

    class FakeTokenizer:
        chat_template = "fake chat template"
        is_fast = True

        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> FakeTokenizer:
            return cls()

        def apply_chat_template(
            self, messages: list[dict[str, str]], **kwargs: object
        ) -> FakeTensor:
            state["tokenize_calls"] = int(state["tokenize_calls"]) + 1
            text = messages[0]["content"]
            return FakeTensor(4097 if long_prompt and "TOKEN-LIMIT-SENTINEL" in text else 1)

        def decode(self, token_ids: FakeGeneratedIds, *, skip_special_tokens: bool) -> str:
            return "```malformed output```"

    class FakeModel:
        device = "cuda:0"
        generation_config = types.SimpleNamespace(eos_token_id=7, pad_token_id=None)

        def eval(self) -> FakeModel:
            return self

        def generate(self, *args: object, **kwargs: object) -> FakeGenerated:
            state["generate_calls"] = int(state["generate_calls"]) + 1
            return FakeGenerated()

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> FakeModel:
            return FakeModel()

    torch = types.ModuleType("torch")
    torch.Tensor = FakeTensor
    torch.bfloat16 = object()
    torch.inference_mode = nullcontext
    torch.manual_seed = lambda seed: None
    torch.version = types.SimpleNamespace(cuda="fake-cuda")
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        manual_seed_all=lambda seed: None,
        current_device=lambda: 0,
        get_device_name=lambda device: "fake-A10G",
        reset_peak_memory_stats=lambda device: None,
        Event=FakeEvent,
        synchronize=lambda: None,
        max_memory_allocated=lambda device: 1234,
    )
    modal = types.ModuleType("modal")
    modal.current_function_call_id = lambda: "fc-fake"
    transformers = types.ModuleType("transformers")
    transformers.AutoModelForCausalLM = FakeAutoModel
    transformers.AutoTokenizer = FakeTokenizer
    return {"torch": torch, "modal": modal, "transformers": transformers}, state


class QuoteBaselinePayloadTests(unittest.TestCase):
    def test_payload_has_two_named_prompt_variants_for_each_pinned_dev_case(self) -> None:
        payload = build_payload("qwen3-4b")

        self.assertEqual(
            payload["input_sha256"],
            "b9efbc113e24112786475333868f781baae3af58d43bee7bdc338f229f651cc0",
        )
        self.assertEqual(len(payload["items"]), 24)
        self.assertEqual(len({item["case_id"] for item in payload["items"]}), 12)
        self.assertEqual(
            {item["prompt_version"] for item in payload["items"]},
            {"quote-v2", "quote-coverage-v1"},
        )

    def test_payload_pins_models_settings_source_and_item_hashes(self) -> None:
        four = build_payload("qwen3-4b")
        eight = build_payload("qwen3-8b")

        self.assertEqual(four["model"]["revision"], "1cfa9a7208912126459214e8b04321603b3df60c")
        self.assertEqual(eight["model"]["revision"], "b968826d9c46dd6066d109eabc6255188de91218")
        for payload in (four, eight):
            self.assertEqual(payload["model"]["license"], "apache-2.0")
            self.assertFalse(payload["model"]["gated"])
            self.assertEqual(
                payload["settings"],
                {
                    "precision": "bfloat16",
                    "max_new_tokens": 1024,
                    "max_prompt_tokens": 4096,
                    "do_sample": False,
                    "num_beams": 1,
                    "attn_implementation": "eager",
                    "trust_remote_code": False,
                    "use_safetensors": True,
                    "enable_thinking": False,
                },
            )
            self.assertEqual(
                payload["prompt_source_sha256"],
                quote_payload._source_sha256(Path(quote_payload.__file__)),
            )
            self.assertEqual(
                payload["runner_source_sha256"],
                quote_payload._source_sha256(
                    quote_payload._ROOT / "experiments" / "modal_quote_baseline.py"
                ),
            )
            self.assertEqual(
                payload["items_sha256"], quote_payload._canonical_sha256(payload["items"])
            )
            canonical = json.dumps(
                payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
            self.assertEqual(
                quote_payload.payload_sha256(payload), hashlib.sha256(canonical).hexdigest()
            )
            self.assertTrue(
                all(
                    set(item) == {"item_id", "case_id", "prompt_version", "prompt"}
                    for item in payload["items"]
                )
            )
            self.assertEqual(
                set(payload),
                {
                    "schema_version",
                    "dataset",
                    "input_sha256",
                    "prompt_source_sha256",
                    "items_sha256",
                    "runner_source_sha256",
                    "model",
                    "settings",
                    "items",
                },
            )

    def test_only_dev_split_and_two_named_models_are_supported(self) -> None:
        for model_key in ("qwen3-2b", "qwen3-14b"):
            with self.subTest(model_key=model_key), self.assertRaisesRegex(ValueError, "model"):
                build_payload(model_key)
        with self.assertRaisesRegex(ValueError, "dev split"):
            build_payload(split="test")

    def test_prompt_text_depends_on_messages_not_targets_or_case_metadata(self) -> None:
        original = build_payload()
        cases, digest = quote_payload._load_dev_cases()
        changed = deepcopy(cases)
        for case in changed:
            case["expected"] = [{"claim": "TARGET-LEAK-SENTINEL"}]
            case["split"] = "train"
            case["scenario_group"] = "changed-scenario"
            case["template_group"] = "changed-template"
            case["entity_ids"] = ["changed-entity"]
            case["tags"] = ["changed-tag"]
        with patch.object(quote_payload, "_load_dev_cases", return_value=(changed, digest)):
            mutated = build_payload()

        self.assertEqual(
            [(item["prompt_version"], item["prompt"]) for item in original["items"]],
            [(item["prompt_version"], item["prompt"]) for item in mutated["items"]],
        )
        self.assertNotIn("TARGET-LEAK-SENTINEL", str(mutated))

    def test_prompt_variants_use_identical_serialized_message_records(self) -> None:
        payload = build_payload()
        grouped: dict[str, dict[str, str]] = {}
        for item in payload["items"]:
            grouped.setdefault(item["case_id"], {})[item["prompt_version"]] = item["prompt"]
        for variants in grouped.values():
            quote_v2_messages = variants["quote-v2"].split("\nMessages:", maxsplit=1)[1]
            coverage_messages = variants["quote-coverage-v1"].split("\nMessages:", maxsplit=1)[1]
            self.assertEqual(quote_v2_messages, coverage_messages)

    def test_coverage_prompt_covers_claim_scope_without_scenario_examples(self) -> None:
        payload = build_payload()
        prompts = [
            item["prompt"]
            for item in payload["items"]
            if item["prompt_version"] == "quote-coverage-v1"
        ]
        instruction = prompts[0].split("\nMessages:", maxsplit=1)[0]
        for phrase in (
            "atomic claims",
            "every user-authored conversation message",
            "nested reports",
            "uncertainty",
            "past events",
            "Do not infer an actor, date",
            "unique exact quote",
            "never from recorded_at",
            "fictional material",
            "embedded instructions",
        ):
            self.assertIn(phrase, instruction)
        self.assertNotIn("Example:", instruction)
        self.assertNotIn("For example", instruction)

    def test_fixture_drift_is_rejected_before_dataset_validation(self) -> None:
        raw = quote_payload._FIXTURE.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            drifted = Path(directory) / "dev.json"
            drifted.write_bytes(raw + b"\n")
            with (
                patch.object(quote_payload, "_FIXTURE", drifted),
                patch.object(quote_payload, "validate_dataset") as validate,
                self.assertRaisesRegex(ValueError, "fixture SHA-256"),
            ):
                build_payload()
            validate.assert_not_called()


class QuoteBaselineRunnerTests(unittest.TestCase):
    def test_resource_envelope_is_single_use_source_free_and_bounded(self) -> None:
        options = modal_quote_baseline.modal_function_options()
        self.assertEqual(options["gpu"], "A10G")
        self.assertEqual(options["cpu"], 2)
        self.assertEqual(options["memory"], 32768)
        self.assertEqual(options["max_containers"], 1)
        self.assertEqual(options["min_containers"], 0)
        self.assertEqual(options["retries"], 0)
        self.assertEqual(options["startup_timeout"], 600)
        self.assertEqual(options["timeout"], 2400)
        self.assertTrue(options["single_use_containers"])
        self.assertTrue(options["serialized"])
        self.assertFalse(options["include_source"])

    def test_prepare_is_default_and_does_not_import_modal_or_model_runtime(self) -> None:
        stdout = StringIO()
        real_import = builtins.__import__

        def guarded_import(name: str, *args: object, **kwargs: object) -> object:
            if name.split(".", maxsplit=1)[0] in {"modal", "torch", "transformers"}:
                raise AssertionError(f"unexpected runtime import: {name}")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import), redirect_stdout(stdout):
            self.assertEqual(modal_quote_baseline.main([]), 0)

        prepared = json.loads(stdout.getvalue())
        self.assertFalse(prepared["remote_started"])
        self.assertEqual(prepared["item_count"], 24)
        self.assertEqual(prepared["input_sha256"], quote_payload._FIXTURE_SHA256)
        self.assertEqual(len(prepared["payload_sha256"]), 64)
        self.assertEqual(len(prepared["runner_sha256"]), 64)

    def test_launch_requires_exact_payload_and_runner_hashes_before_reserving_output(self) -> None:
        payload = build_payload()
        payload_hash = quote_payload.payload_sha256(payload)
        runner_hash = payload["runner_source_sha256"]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "receipt.json"
            for approved_payload, approved_runner in (
                ("0" * 64, runner_hash),
                (payload_hash, "0" * 64),
            ):
                stderr = StringIO()
                with (
                    patch.object(modal_quote_baseline, "verify_personal_workspace") as workspace,
                    patch.object(modal_quote_baseline, "_build_modal_function") as build_function,
                    redirect_stderr(stderr),
                ):
                    code = modal_quote_baseline.main(
                        [
                            "--launch",
                            "--approve-payload-sha256",
                            approved_payload,
                            "--approve-runner-sha256",
                            approved_runner,
                            "--output",
                            str(output),
                        ]
                    )
                self.assertEqual(code, 2)
                self.assertFalse(output.exists())
                workspace.assert_not_called()
                build_function.assert_not_called()
                self.assertIn("exact", stderr.getvalue())

    def test_output_path_is_exclusive_and_reserved_before_workspace_check(self) -> None:
        payload = build_payload()
        args = [
            "--launch",
            "--approve-payload-sha256",
            quote_payload.payload_sha256(payload),
            "--approve-runner-sha256",
            payload["runner_source_sha256"],
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "receipt.json"
            existing = Path(directory) / "existing.json"
            original_bytes = b"leave this file untouched"
            existing.write_bytes(original_bytes)
            with redirect_stderr(StringIO()):
                self.assertEqual(modal_quote_baseline.main([*args, "--output", str(existing)]), 2)
            self.assertEqual(existing.read_bytes(), original_bytes)

            stderr = StringIO()

            def fail_workspace() -> None:
                self.assertTrue(output.exists())
                raise RuntimeError("workspace check stopped before invocation")

            with (
                patch.object(
                    modal_quote_baseline, "verify_personal_workspace", side_effect=fail_workspace
                ),
                patch.object(modal_quote_baseline, "_build_modal_function") as build_function,
                redirect_stderr(stderr),
            ):
                self.assertEqual(modal_quote_baseline.main([*args, "--output", str(output)]), 2)
            failure = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(failure["status"], "launch_failed")
            self.assertFalse(failure["remote_started"])
            build_function.assert_not_called()
            first_bytes = output.read_bytes()
            with redirect_stderr(StringIO()):
                self.assertEqual(modal_quote_baseline.main([*args, "--output", str(output)]), 2)
            self.assertEqual(output.read_bytes(), first_bytes)

    def test_remote_rejects_bad_request_before_modal_torch_or_transformers_import(self) -> None:
        real_import = builtins.__import__

        def guarded_import(name: str, *args: object, **kwargs: object) -> object:
            if name.split(".", maxsplit=1)[0] in {"modal", "torch", "transformers"}:
                raise AssertionError(f"runtime import happened before request validation: {name}")
            return real_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=guarded_import),
            patch("socket.create_connection") as network,
            self.assertRaisesRegex(ValueError, "request keys|approval hashes|payload hash"),
        ):
            modal_quote_baseline._remote_compute(
                {"payload": {}, "payload_sha256": "bad", "runner_sha256": "bad"}
            )
        network.assert_not_called()

    def test_remote_body_has_no_local_package_imports(self) -> None:
        imported = {
            instruction.argval
            for instruction in dis.get_instructions(modal_quote_baseline._remote_compute)
            if instruction.opname == "IMPORT_NAME"
        }
        self.assertNotIn("tracemem", imported)
        self.assertNotIn("experiments", imported)

    def test_remote_mirrors_fixture_source_model_settings_runner_and_item_guards(self) -> None:
        original = build_payload()
        mutations = (
            ("fixture", lambda payload: payload.__setitem__("input_sha256", "0" * 64)),
            (
                "prompt source",
                lambda payload: payload.__setitem__("prompt_source_sha256", "0" * 64),
            ),
            ("model", lambda payload: payload["model"].__setitem__("revision", "0" * 40)),
            ("settings", lambda payload: payload["settings"].__setitem__("precision", "float16")),
            ("items hash", lambda payload: payload.__setitem__("items_sha256", "0" * 64)),
            (
                "item version",
                lambda payload: payload["items"][0].__setitem__("prompt_version", "quote-v9"),
            ),
        )
        real_import = builtins.__import__

        def guarded_import(name: str, *args: object, **kwargs: object) -> object:
            if name.split(".", maxsplit=1)[0] in {"modal", "torch", "transformers"}:
                raise AssertionError(f"runtime import happened before request validation: {name}")
            return real_import(name, *args, **kwargs)

        for label, mutate in mutations:
            with self.subTest(label=label):
                changed = deepcopy(original)
                mutate(changed)
                if label == "item version":
                    changed["items"][0]["item_id"] = f"{changed['items'][0]['case_id']}::quote-v9"
                    changed["items_sha256"] = quote_payload._canonical_sha256(changed["items"])
                request = _remote_request(changed)
                with (
                    patch("builtins.__import__", side_effect=guarded_import),
                    patch("socket.create_connection") as network,
                    self.assertRaises(ValueError),
                ):
                    modal_quote_baseline._remote_compute(request)
                network.assert_not_called()

        mismatch = _remote_request(original)
        mismatch["runner_sha256"] = "0" * 64
        with (
            patch("builtins.__import__", side_effect=guarded_import),
            self.assertRaisesRegex(ValueError, "runner hash"),
        ):
            modal_quote_baseline._remote_compute(mismatch)

    def test_tokenized_prompt_limit_fails_before_any_generation(self) -> None:
        payload = build_payload()
        payload["items"][0]["prompt"] += " TOKEN-LIMIT-SENTINEL"
        payload["items_sha256"] = quote_payload._canonical_sha256(payload["items"])
        request = _remote_request(payload)
        modules, state = _fake_runtime(long_prompt=True)
        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            root = name.split(".", maxsplit=1)[0]
            if root in modules:
                return modules[root]
            return real_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=fake_import),
            self.assertRaisesRegex(ValueError, "4096-token limit"),
        ):
            modal_quote_baseline._remote_compute(request)
        self.assertEqual(state["generate_calls"], 0)
        self.assertEqual(state["tokenize_calls"], 1)

    def test_remote_receipt_preserves_raw_text_and_generation_flags(self) -> None:
        modules, state = _fake_runtime()
        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            root = name.split(".", maxsplit=1)[0]
            if root in modules:
                return modules[root]
            return real_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=fake_import),
            patch("socket.create_connection") as network,
        ):
            receipt = modal_quote_baseline._remote_compute(_remote_request())

        network.assert_not_called()
        self.assertEqual(state["generate_calls"], 24)
        self.assertEqual(receipt["mode"], "quote-baseline-prompt-only")
        self.assertEqual(receipt["item_count"], 24)
        self.assertEqual(receipt["modal_function_call_id"], "fc-fake")
        self.assertEqual(receipt["runner_sha256"], _remote_request()["runner_sha256"])
        self.assertEqual(
            receipt["tokenizer"],
            {
                "class": "FakeTokenizer",
                "backend": "fast",
                "chat_template_sha256": hashlib.sha256(b"fake chat template").hexdigest(),
            },
        )
        row = receipt["generated_outputs"][0]
        self.assertEqual(row["raw_output"], "```malformed output```")
        self.assertEqual(row["input_token_ids"], [11])
        self.assertEqual(row["output_token_ids"], [7])
        self.assertEqual(row["input_tokens"], 1)
        self.assertEqual(row["output_tokens"], 1)
        self.assertEqual(row["last_token_id"], 7)
        self.assertTrue(row["ended_with_eos"])
        self.assertFalse(row["hit_max_new_tokens"])
        self.assertEqual(row["finish_reason"], "eos")

    def test_validator_rejects_source_model_settings_and_item_hash_drift(self) -> None:
        original = build_payload()
        mutations = (
            ("prompt source", "prompt_source_sha256", "0" * 64),
            ("runner source", "runner_source_sha256", "0" * 64),
            ("model revision", "model", {**original["model"], "revision": "0" * 40}),
            ("generation settings", "settings", {**original["settings"], "do_sample": True}),
            ("item hash", "items_sha256", "0" * 64),
        )
        for label, field, value in mutations:
            with self.subTest(label=label):
                changed = deepcopy(original)
                changed[field] = value
                with self.assertRaises(ValueError):
                    quote_payload.validate_payload(changed)

    def test_registry_model_metadata_drift_is_rejected(self) -> None:
        original = build_payload()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            registry = quote_payload._ROOT / "configs" / "model-candidates.json"
            data = json.loads(registry.read_text(encoding="utf-8"))
            candidate = next(row for row in data["candidates"] if row["id"] == "Qwen/Qwen3-4B")
            candidate["gated"] = True
            (root / "configs" / "model-candidates.json").write_text(
                json.dumps(data), encoding="utf-8"
            )
            original_hash = quote_payload._source_sha256

            def source_hash(path: Path) -> str:
                if path.name == "modal_quote_baseline.py":
                    return original["runner_source_sha256"]
                return original_hash(path)

            with (
                patch.object(quote_payload, "_ROOT", root),
                patch.object(quote_payload, "_source_sha256", side_effect=source_hash),
                self.assertRaisesRegex(ValueError, "candidate metadata"),
            ):
                quote_payload.validate_payload(original)

    def test_validator_rejects_duplicate_missing_extra_and_unknown_variants(self) -> None:
        original = build_payload()
        changes = (
            (
                "duplicate id",
                lambda items: items[1].__setitem__("item_id", items[0]["item_id"]),
                "IDs",
            ),
            (
                "duplicate variant",
                lambda items: (
                    items[1].__setitem__("prompt_version", items[0]["prompt_version"]),
                    items[1].__setitem__(
                        "item_id", f"{items[1]['case_id']}::{items[1]['prompt_version']}"
                    ),
                ),
                "duplicate prompt variant",
            ),
            (
                "unknown version",
                lambda items: (
                    items[0].__setitem__("prompt_version", "quote-v9"),
                    items[0].__setitem__("item_id", f"{items[0]['case_id']}::quote-v9"),
                ),
                "not allowlisted",
            ),
            (
                "unknown case",
                lambda items: items[0].__setitem__("case_id", "foreign-case"),
                "unknown case",
            ),
        )
        for label, mutate, message in changes:
            with self.subTest(label=label):
                changed = deepcopy(original)
                mutate(changed["items"])
                changed["items_sha256"] = quote_payload._canonical_sha256(changed["items"])
                with self.assertRaisesRegex(ValueError, message):
                    quote_payload.validate_payload(changed)

        for label, items in (
            ("missing", original["items"][:-1]),
            ("extra", [*original["items"], deepcopy(original["items"][0])]),
        ):
            with self.subTest(label=label):
                changed = deepcopy(original)
                changed["items"] = items
                changed["items_sha256"] = quote_payload._canonical_sha256(items)
                with self.assertRaisesRegex(ValueError, "exactly 24"):
                    quote_payload.validate_payload(changed)

    def test_oversized_prompt_is_rejected_before_payload_is_returned(self) -> None:
        cases, digest = quote_payload._load_dev_cases()
        cases = deepcopy(cases)
        cases[0]["messages"][0]["text"] = "x" * 100_001
        with patch.object(quote_payload, "_load_dev_cases", return_value=(cases, digest)):
            with self.assertRaisesRegex(ValueError, "payload limit"):
                build_payload()


if __name__ == "__main__":
    unittest.main()
