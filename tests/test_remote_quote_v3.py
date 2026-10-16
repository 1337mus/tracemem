"""Fake-runtime tests for quote-v3 prior-adapter comparison."""

from __future__ import annotations

import hashlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class _FakeParameter:
    def __init__(self, trainable: bool) -> None:
        self.requires_grad = trainable

    def numel(self) -> int:
        return 2


class _FakeLoss:
    def detach(self) -> _FakeLoss:
        return self

    def float(self) -> _FakeLoss:
        return self

    def cpu(self) -> _FakeLoss:
        return self

    def __float__(self) -> float:
        return 0.5

    def __truediv__(self, _: int) -> _FakeLoss:
        return self

    def backward(self) -> None:
        return None


def _write_fake_prior_adapter(prior_path: Path) -> dict[str, object]:
    prior_path.mkdir()
    file_contents = {
        "adapter_model.safetensors": b"fake prior adapter weights",
        "adapter_config.json": b'{"fake":true}',
    }
    for name, contents in file_contents.items():
        (prior_path / name).write_bytes(contents)
    return {
        "path": str(prior_path),
        "sha256": {
            name: hashlib.sha256(contents).hexdigest() for name, contents in file_contents.items()
        },
    }


def _run_quote_training(
    remote_run_dir: Path,
    metadata: dict[str, object],
    *,
    profile_name: str = "quote-v3",
    dev_max_new_tokens: int = 512,
    tracking: dict[str, object] | None = None,
    tamper_during_tuned_generation: bool = False,
    leave_prior_lora_after_unload: bool = False,
) -> dict[str, object]:
    from training import remote_sft

    events: list[tuple[str, str]] = []
    generation_calls: list[tuple[str, list[dict[str, object]], int]] = []
    if tracking is not None:
        tracking["events"] = events
        tracking["generation_calls"] = generation_calls
    optimizer_parameters: list[_FakeParameter] = []
    rng = {"cpu": "initial-cpu", "cuda": ["initial-cuda"]}
    prior_parameter = _FakeParameter(False)
    base_parameter = _FakeParameter(False)
    fresh_parameter = _FakeParameter(True)

    class FakeModel:
        def __init__(
            self,
            label: str,
            parameters: list[tuple[str, _FakeParameter]],
        ) -> None:
            self.label = label
            self._parameters = parameters
            self.config = SimpleNamespace(use_cache=True)
            self.is_gradient_checkpointing = True

        def get_input_embeddings(self) -> object:
            return SimpleNamespace(weight=SimpleNamespace(device="cuda:0"))

        def eval(self) -> None:
            return None

        def train(self) -> None:
            return None

        def num_parameters(self) -> int:
            return 100

        def named_parameters(self) -> list[tuple[str, _FakeParameter]]:
            return self._parameters

        def __call__(self, **_: object) -> object:
            return SimpleNamespace(loss=_FakeLoss())

        def save_pretrained(self, directory: Path, **_: object) -> None:
            (directory / "adapter.safetensors").write_bytes(b"fresh-adapter")

    loaded_model = FakeModel("loaded", [("base.weight", base_parameter)])
    prepared_model = FakeModel("prepared", [("base.weight", base_parameter)])
    unloaded_parameters = [("base.weight", base_parameter)]
    if leave_prior_lora_after_unload:
        unloaded_parameters.append(("lora_A.weight", prior_parameter))
    unloaded_model = FakeModel("unloaded", unloaded_parameters)
    fresh_model = FakeModel(
        "fresh",
        [("base.weight", base_parameter), ("lora_A.weight", fresh_parameter)],
    )

    class FakePriorModel:
        label = "prior"

        def unload(self) -> FakeModel:
            events.append(("unload", "without-merge"))
            return unloaded_model

        def merge_and_unload(self) -> FakeModel:
            events.append(("merge", "called"))
            return prepared_model

        def named_parameters(self) -> list[tuple[str, _FakeParameter]]:
            return [
                ("base.weight", base_parameter),
                ("lora_A.weight", prior_parameter),
            ]

    class FakePeftModel:
        @staticmethod
        def from_pretrained(
            model: FakeModel,
            path: str,
            *,
            is_trainable: bool,
        ) -> FakePriorModel:
            events.append(("load-prior", model.label))
            events.append(("prior-path", path))
            events.append(("prior-trainable", str(is_trainable)))
            rng["cpu"] = "prior-load-consumed"
            rng["cuda"] = ["prior-load-consumed"]
            return FakePriorModel()

    class FakeCuda:
        def is_available(self) -> bool:
            return True

        def is_bf16_supported(self) -> bool:
            return True

        def manual_seed_all(self, seed: int) -> None:
            rng["cuda"] = [f"seed-{seed}"]

        def reset_peak_memory_stats(self) -> None:
            return None

        def synchronize(self) -> None:
            return None

        def get_rng_state_all(self) -> list[str]:
            return list(rng["cuda"])

        def set_rng_state_all(self, states: list[str]) -> None:
            rng["cuda"] = list(states)

        def get_device_name(self) -> str:
            return "fake-cuda"

        def max_memory_allocated(self) -> int:
            return 1

        def max_memory_reserved(self) -> int:
            return 2

    class FakeOptimizer:
        def __init__(self, parameters: list[_FakeParameter], **_: object) -> None:
            optimizer_parameters.extend(parameters)

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
        events.append(("fresh-lora", model.label))
        events.append(
            (
                "fresh-input-has-lora",
                str(any("lora_" in name for name, _ in model.named_parameters())),
            )
        )
        events.append(("fresh-rng", f"{rng['cpu']}|{rng['cuda'][0]}"))
        return fresh_model

    def generate_outputs(
        model: FakeModel | FakePriorModel,
        _tokenizer: object,
        prompts: list[dict[str, object]],
        _torch: object,
        _device: object,
        max_new_tokens: int,
    ) -> list[dict[str, str]]:
        events.append(("generate", model.label))
        generation_calls.append((model.label, prompts, max_new_tokens))
        if model.label == "prepared":
            rng["cpu"] = "base-ready"
            rng["cuda"] = ["base-ready-cuda"]
        elif model.label == "prior":
            rng["cpu"] = "prior-inference-consumed"
            rng["cuda"] = ["prior-inference-consumed-cuda"]
        elif model.label == "fresh" and tamper_during_tuned_generation:
            prior_path = Path(str(metadata["path"]))
            (prior_path / "adapter_config.json").write_text("tampered", encoding="utf-8")
        return [{"case_id": str(prompt["case_id"]), "model": model.label} for prompt in prompts]

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(*_: object, **__: object) -> FakeModel:
            return loaded_model

    torch_module = types.ModuleType("torch")
    torch_module.long = "long"
    torch_module.bfloat16 = "bfloat16"
    torch_module.tensor = lambda *_args, **_kwargs: object()
    torch_module.manual_seed = lambda seed: rng.update(cpu=f"seed-{seed}")
    torch_module.get_rng_state = lambda: rng["cpu"]
    torch_module.set_rng_state = lambda state: rng.update(cpu=state)
    torch_module.equal = lambda left, right: left == right
    torch_module.cuda = FakeCuda()
    torch_module.optim = SimpleNamespace(AdamW=FakeOptimizer)

    peft_module = types.ModuleType("peft")
    peft_module.LoraConfig = lambda **kwargs: kwargs
    peft_module.PeftModel = FakePeftModel
    peft_module.get_peft_model = attach_lora
    peft_module.prepare_model_for_kbit_training = prepare_model

    transformers_module = types.ModuleType("transformers")
    transformers_module.AutoModelForCausalLM = FakeAutoModel
    transformers_module.BitsAndBytesConfig = lambda **kwargs: kwargs

    profile = SimpleNamespace(
        name=profile_name,
        train_context_tokens=64,
        dev_max_new_tokens=dev_max_new_tokens,
        train_steps=1,
    )
    tokenizer = FakeTokenizer()
    support = SimpleNamespace(
        MODEL_ID="fake/model",
        MODEL_REVISION="pinned-revision",
        resolve_sft_profile=lambda _: profile,
        _check_dependency_versions=lambda _: {"peft": "0.15.2"},
        _TRAIN_DEPENDENCIES={},
        _load_pinned_tokenizer=lambda: tokenizer,
        _tokenizer_provenance=lambda *_: {"revision": "pinned"},
        _QUOTE_V3_PRIOR_ADAPTER=metadata,
        _QUOTE_V4_PRIOR_ADAPTER=metadata,
        preflight_train_rows=lambda *_args, **_kwargs: {"status": "ready"},
        encode_sft_example=lambda *_args, **_kwargs: {
            "input_ids": [1, 2],
            "attention_mask": [1, 1],
            "labels": [-100, 2],
        },
        _validated_dev_prompts=lambda prompts, *_: [
            (prompt["case_id"], prompt["prompt"]) for prompt in prompts
        ],
        _encode_user_prompt=lambda *_: [3, 4],
        SEED=17,
        _LORA_TARGETS=("q_proj",),
        GRADIENT_ACCUMULATION_STEPS=1,
        LEARNING_RATE=0.001,
        MICROBATCH_SIZE=1,
    )
    request = {
        "profile": profile_name,
        "train_rows": [{"case_id": "train-1"}],
        "dev_prompts": [{"case_id": "dev-1", "prompt": "prompt-1"}],
        "remote_run_dir": str(remote_run_dir),
    }
    with (
        patch.dict(
            sys.modules,
            {"torch": torch_module, "peft": peft_module, "transformers": transformers_module},
        ),
        patch.object(remote_sft, "_generate_outputs", side_effect=generate_outputs),
        patch.object(remote_sft, "_adapter_state_hash", side_effect=["before", "after"]),
    ):
        result = remote_sft.run_training(request, support)
    return {
        "result": result,
        "events": events,
        "generation_calls": generation_calls,
        "optimizer_parameters": optimizer_parameters,
        "prior_parameter": prior_parameter,
        "fresh_parameter": fresh_parameter,
    }


class RemoteQuoteV3Tests(unittest.TestCase):
    def test_prior_adapter_is_compared_then_unloaded_before_fresh_lora(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            prior_path = Path(temporary_dir) / "prior"
            metadata = _write_fake_prior_adapter(prior_path)
            run = _run_quote_training(Path(temporary_dir) / "run", metadata)

        result = run["result"]
        events = run["events"]
        generation_calls = run["generation_calls"]
        self.assertEqual(result["status"], "completed")
        self.assertLess(
            events.index(("generate", "prepared")), events.index(("load-prior", "prepared"))
        )
        self.assertIn(("load-prior", "prepared"), events)
        self.assertLess(
            events.index(("load-prior", "prepared")), events.index(("generate", "prior"))
        )
        self.assertLess(
            events.index(("generate", "prior")), events.index(("unload", "without-merge"))
        )
        self.assertLess(
            events.index(("unload", "without-merge")), events.index(("fresh-lora", "unloaded"))
        )
        self.assertNotIn(("merge", "called"), events)
        self.assertIn(("fresh-input-has-lora", "False"), events)
        self.assertEqual(generation_calls[0][1:], generation_calls[1][1:])
        self.assertEqual(generation_calls[1][1:], generation_calls[2][1:])
        self.assertEqual(generation_calls[1][0], "prior")
        self.assertEqual(generation_calls[1][2], 512)
        self.assertIn(("prior-path", metadata["path"]), events)
        self.assertIn(("prior-trainable", "False"), events)
        self.assertFalse(result["generation"]["do_sample"])
        self.assertEqual(
            events[events.index(("fresh-lora", "unloaded")) + 2],
            ("fresh-rng", "base-ready|base-ready-cuda"),
        )
        self.assertNotIn(run["prior_parameter"], run["optimizer_parameters"])
        self.assertIn(run["fresh_parameter"], run["optimizer_parameters"])
        self.assertEqual(
            result["generation"]["prior_outputs"], [{"case_id": "dev-1", "model": "prior"}]
        )
        self.assertGreaterEqual(result["generation"]["prior_seconds"], 0)
        self.assertEqual(result["training"]["adapter_hash_before_training"], "before")
        self.assertEqual(result["training"]["adapter_hash_after_training"], "after")
        self.assertEqual(
            result["comparison"],
            {
                "prior_adapter": metadata,
                "verified_before": True,
                "verified_after": True,
                "unloaded_without_merge": True,
                "rng_restored": True,
                "no_prior_lora_parameters": True,
            },
        )

    def test_prior_adapter_digest_mismatch_aborts_before_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            metadata = _write_fake_prior_adapter(Path(temporary_dir) / "prior")
            metadata["sha256"]["adapter_config.json"] = "0" * 64  # type: ignore[index]
            tracking: dict[str, object] = {}

            with self.assertRaisesRegex(RuntimeError, "hash mismatch for adapter_config.json"):
                _run_quote_training(Path(temporary_dir) / "run", metadata, tracking=tracking)

        self.assertEqual(tracking["generation_calls"], [])
        self.assertFalse(any(event[0] == "load-prior" for event in tracking["events"]))

    def test_prior_adapter_tampering_after_training_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            metadata = _write_fake_prior_adapter(Path(temporary_dir) / "prior")
            tracking: dict[str, object] = {}

            with self.assertRaisesRegex(RuntimeError, "hash mismatch for adapter_config.json"):
                _run_quote_training(
                    Path(temporary_dir) / "run",
                    metadata,
                    tracking=tracking,
                    tamper_during_tuned_generation=True,
                )

        self.assertEqual(
            [call[0] for call in tracking["generation_calls"]],
            ["prepared", "prior", "fresh"],
        )

    def test_residual_prior_lora_parameters_abort_before_fresh_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            metadata = _write_fake_prior_adapter(Path(temporary_dir) / "prior")
            tracking: dict[str, object] = {}

            with self.assertRaisesRegex(RuntimeError, "parameters remained after unload"):
                _run_quote_training(
                    Path(temporary_dir) / "run",
                    metadata,
                    tracking=tracking,
                    leave_prior_lora_after_unload=True,
                )

        self.assertIn(("unload", "without-merge"), tracking["events"])
        self.assertNotIn(("fresh-lora", "unloaded"), tracking["events"])


if __name__ == "__main__":
    unittest.main()
