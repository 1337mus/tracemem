"""Tests for remote-only QLoRA lifecycle helpers without loading a model."""

from __future__ import annotations

import hashlib
import sys
import tempfile
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from training.remote_sft import (
    _generate_outputs,
    _hash_artifacts,
    _require_adapter_mutation,
    _reserve_run_directory,
)


class FakeTensor:
    def __init__(self, values: list[object]) -> None:
        self.values = values

    def __getitem__(self, key: tuple[int, slice]) -> FakeTensor:
        row, token_slice = key
        return FakeTensor(self.values[row][token_slice])  # type: ignore[index]

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def tolist(self) -> list[object]:
        return self.values


class FakeTorch:
    long = "long"

    def tensor(self, values: list[list[int]], **_: object) -> FakeTensor:
        return FakeTensor(values)

    def ones_like(self, tensor: FakeTensor) -> FakeTensor:
        return FakeTensor([[1] * len(tensor.values[0])])  # type: ignore[arg-type]

    def inference_mode(self) -> object:
        return nullcontext()


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def decode(self, token_ids: list[object], *, skip_special_tokens: bool) -> str:
        return '{"ok":true}' if skip_special_tokens else '{"ok":true}<|im_end|>'


class FakeModel:
    def __init__(self) -> None:
        self.generation_kwargs: dict[str, object] = {}

    def eval(self) -> None:
        return None

    def generate(self, **kwargs: object) -> FakeTensor:
        self.generation_kwargs = kwargs
        return FakeTensor([[11, 12, 21, 22, 2]])


class RemoteSftTests(unittest.TestCase):
    def test_run_directory_is_reserved_exclusively(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            target = Path(temporary_dir) / "run-1"

            self.assertEqual(_reserve_run_directory(target), target)
            self.assertTrue(target.is_dir())
            with self.assertRaises(FileExistsError):
                _reserve_run_directory(target)

    def test_adapter_mutation_guard_rejects_an_unchanged_digest(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "did not change"):
            _require_adapter_mutation("same-digest", "same-digest")

        _require_adapter_mutation("before", "after")

    def test_generation_is_greedy_and_returns_unmodified_raw_text(self) -> None:
        model = FakeModel()
        outputs = _generate_outputs(
            model,
            FakeTokenizer(),
            [{"case_id": "dev-1", "input_ids": [11, 12]}],
            FakeTorch(),
            "cuda:0",
            1024,
        )

        self.assertEqual(outputs[0]["prompt_token_ids"], [11, 12])
        self.assertEqual(outputs[0]["completion_token_ids"], [21, 22, 2])
        self.assertEqual(outputs[0]["raw_text"], '{"ok":true}<|im_end|>')
        self.assertEqual(outputs[0]["response_text"], '{"ok":true}')
        self.assertEqual(outputs[0]["max_new_tokens"], 1024)
        self.assertEqual(model.generation_kwargs["max_new_tokens"], 1024)
        self.assertFalse(model.generation_kwargs["do_sample"])
        self.assertEqual(model.generation_kwargs["pad_token_id"], 0)

    def test_training_prepares_before_baseline_and_attaches_lora_after_it(self) -> None:
        from training import remote_sft

        events: list[tuple[str, str]] = []

        class FakeLoss:
            def detach(self) -> FakeLoss:
                return self

            def float(self) -> FakeLoss:
                return self

            def cpu(self) -> FakeLoss:
                return self

            def __float__(self) -> float:
                return 0.5

            def __truediv__(self, _: int) -> FakeLoss:
                return self

            def backward(self) -> None:
                return None

        class FakeParameter:
            requires_grad = True

            def numel(self) -> int:
                return 2

        class FakeModel:
            def __init__(self, label: str) -> None:
                self.label = label
                self.config = types.SimpleNamespace(use_cache=True)
                self.is_gradient_checkpointing = True

            def get_input_embeddings(self) -> object:
                return types.SimpleNamespace(weight=types.SimpleNamespace(device="cuda:0"))

            def eval(self) -> None:
                return None

            def train(self) -> None:
                return None

            def num_parameters(self) -> int:
                return 100

            def named_parameters(self) -> list[tuple[str, FakeParameter]]:
                return [("adapter.lora_A.weight", FakeParameter())]

            def __call__(self, **_: object) -> object:
                return types.SimpleNamespace(loss=FakeLoss())

            def save_pretrained(self, directory: Path, **_: object) -> None:
                (directory / "adapter.safetensors").write_bytes(b"adapter")

        loaded_model = FakeModel("loaded")
        prepared_model = FakeModel("prepared")
        adapter_model = FakeModel("adapter")

        class FakeCuda:
            def is_available(self) -> bool:
                return True

            def is_bf16_supported(self) -> bool:
                return True

            def manual_seed_all(self, _: int) -> None:
                return None

            def reset_peak_memory_stats(self) -> None:
                return None

            def synchronize(self) -> None:
                return None

            def get_device_name(self) -> str:
                return "fake-cuda"

            def max_memory_allocated(self) -> int:
                return 1

            def max_memory_reserved(self) -> int:
                return 2

        class FakeOptimizer:
            def __init__(self, *_: object, **__: object) -> None:
                return None

            def zero_grad(self, **_: object) -> None:
                return None

            def step(self) -> None:
                return None

        class FakeTokenizer:
            eos_token_id = 2
            pad_token_id = 0
            padding_side = "right"

            def save_pretrained(self, directory: Path) -> None:
                (directory / "tokenizer.json").write_text("{}", encoding="utf-8")

        def prepare_model(model: FakeModel, **_: object) -> FakeModel:
            events.append(("prepare", model.label))
            return prepared_model

        def attach_lora(model: FakeModel, _: object) -> FakeModel:
            events.append(("attach_lora", model.label))
            return adapter_model

        def generate_outputs(
            model: FakeModel,
            *_: object,
            **__: object,
        ) -> list[dict[str, str]]:
            events.append(("generate", model.label))
            return [{"model": model.label}]

        torch_module = types.ModuleType("torch")
        torch_module.long = "long"
        torch_module.bfloat16 = "bfloat16"
        torch_module.tensor = lambda *_args, **_kwargs: object()
        torch_module.manual_seed = lambda _: None
        torch_module.cuda = FakeCuda()
        torch_module.optim = types.SimpleNamespace(AdamW=FakeOptimizer)
        torch_module.inference_mode = nullcontext

        peft_module = types.ModuleType("peft")
        peft_module.LoraConfig = lambda **kwargs: kwargs
        peft_module.get_peft_model = attach_lora
        peft_module.prepare_model_for_kbit_training = prepare_model

        class FakeAutoModel:
            @staticmethod
            def from_pretrained(*_: object, **__: object) -> FakeModel:
                return loaded_model

        transformers_module = types.ModuleType("transformers")
        transformers_module.AutoModelForCausalLM = FakeAutoModel
        transformers_module.BitsAndBytesConfig = lambda **kwargs: kwargs

        tokenizer = FakeTokenizer()
        support = types.SimpleNamespace(
            MODEL_ID="fake/model",
            MODEL_REVISION="pinned-revision",
            _check_dependency_versions=lambda _: {"torch": "fake"},
            _TRAIN_DEPENDENCIES={},
            _load_pinned_tokenizer=lambda: tokenizer,
            _tokenizer_provenance=lambda *_: {"revision": "pinned"},
            preflight_train_rows=lambda *_args, **_kwargs: {"status": "ready"},
            encode_sft_example=lambda *_: {
                "input_ids": [1, 2],
                "attention_mask": [1, 1],
                "labels": [-100, 2],
            },
            _validated_dev_prompts=lambda prompts: [
                (prompt["case_id"], prompt["prompt"]) for prompt in prompts
            ],
            _encode_user_prompt=lambda *_: [3, 4],
            _DEV_MAX_NEW_TOKENS=1024,
            SEED=17,
            _LORA_TARGETS=("q_proj",),
            MAX_TRAIN_STEPS=1,
            GRADIENT_ACCUMULATION_STEPS=1,
            LEARNING_RATE=0.001,
            MICROBATCH_SIZE=1,
        )
        request = {
            "train_rows": [{"case_id": "train-1"}],
            "dev_prompts": [
                {"case_id": f"dev-{index}", "prompt": f"prompt-{index}"} for index in range(8)
            ],
        }

        with tempfile.TemporaryDirectory() as temporary_dir:
            request["remote_run_dir"] = str(Path(temporary_dir) / "run-1")
            with (
                patch.dict(
                    sys.modules,
                    {
                        "torch": torch_module,
                        "peft": peft_module,
                        "transformers": transformers_module,
                    },
                ),
                patch.object(remote_sft, "_generate_outputs", side_effect=generate_outputs),
                patch.object(remote_sft, "_adapter_state_hash", side_effect=["before", "after"]),
            ):
                result = remote_sft.run_training(request, support)

        self.assertEqual(result["status"], "completed")
        self.assertLess(events.index(("prepare", "loaded")), events.index(("generate", "prepared")))
        self.assertLess(
            events.index(("generate", "prepared")), events.index(("attach_lora", "prepared"))
        )
        self.assertLess(
            events.index(("attach_lora", "prepared")), events.index(("generate", "adapter"))
        )

    def test_artifact_hashes_are_sha256_and_base_weights_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            adapter = root / "adapter" / "adapter.safetensors"
            adapter.parent.mkdir()
            adapter.write_bytes(b"adapter-only")

            self.assertEqual(
                _hash_artifacts(root),
                {"adapter/adapter.safetensors": hashlib.sha256(b"adapter-only").hexdigest()},
            )

            (root / "model.safetensors").write_bytes(b"base")
            with self.assertRaisesRegex(RuntimeError, "base model weights"):
                _hash_artifacts(root)


if __name__ == "__main__":
    unittest.main()
