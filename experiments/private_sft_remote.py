"""Source-free remote worker for the bounded private verifier SFT run."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_EXPECTED_PAYLOAD_SHA256 = ""
_EXPECTED_RUNNER_SHA256 = ""
_APPROVED_PAYLOAD_SHA256 = ""
_APPROVED_RUNNER_SHA256 = ""
_EXPECTED_PROVENANCE: dict[str, Any] = {}
_EXPECTED_MODEL: dict[str, Any] = {}
_EXPECTED_SETTINGS: dict[str, Any] = {}
_EXPECTED_PRIOR_ADAPTER: dict[str, Any] = {}
_EXPECTED_OUTPUT_VOLUME: dict[str, str] = {}
_VALIDATE_PAYLOAD_SHAPE: Callable[..., dict[str, Any]]
_COMPLETION_LABELS: Callable[..., list[int]]
_ACCUMULATION_WINDOW_SIZE: Callable[..., int]
_TOKEN_IDS_SHA256: Callable[..., str]
_VERIFY_ADAPTER_MOUNT: Callable[[str, str, dict[str, str]], dict[str, str]]
_WRITE_PRIVATE_JSON: Callable[[Any, dict[str, Any]], None]
_REMOTE_OUTPUT_VOLUME: Any
_REMOTE_COMPUTE: Callable[[dict[str, Any]], dict[str, Any]]


def _completion_labels(prompt_ids: list[int], completion_ids: list[int], eos_id: int) -> list[int]:
    """Mask prompt tokens while supervising every completion token and EOS."""
    if (
        not prompt_ids
        or not completion_ids
        or type(eos_id) is not int
        or any(type(value) is not int or value < 0 for value in [*prompt_ids, *completion_ids])
    ):
        raise ValueError("private SFT token sequence is invalid")
    return [-100] * len(prompt_ids) + completion_ids + [eos_id]


def _accumulation_window_size(position: int, total: int, steps: int) -> int:
    """Return the actual divisor for this full or final partial accumulation window."""
    if (
        any(type(value) is not int for value in (position, total, steps))
        or total < 1
        or steps < 1
        or not 0 <= position < total
    ):
        raise ValueError("private SFT accumulation window is invalid")
    window_start = (position // steps) * steps
    return min(steps, total - window_start)


def _token_ids_sha256(token_ids: list[int]) -> str:
    """Hash token IDs using the same canonical JSON form as the receipt contract."""
    import hashlib
    import json

    if type(token_ids) is not list or any(
        type(value) is not int or value < 0 for value in token_ids
    ):
        raise ValueError("private SFT token IDs are invalid")
    encoded = json.dumps(token_ids, separators=(",", ":"), allow_nan=False).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _write_private_json_exclusive(path: Any, value: dict[str, Any]) -> None:
    """Write a canonical private JSON artifact once, with mode 0600."""
    import json
    import os

    data = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_payload_shape(
    payload: object,
    expected_provenance: dict[str, Any],
    expected_model: dict[str, Any],
    expected_settings: dict[str, Any],
    expected_prior_adapter: dict[str, Any],
    expected_output_volume: dict[str, str],
) -> dict[str, Any]:
    """Validate the fixed private SFT envelope and every text row before ML imports."""
    import json
    import re

    try:
        if type(payload) is not dict or set(payload) != {
            "schema_version",
            "mode",
            "provenance",
            "source_sha256",
            "model",
            "settings",
            "prior_adapter",
            "output_volume",
            "train_rows",
            "dev_prompts",
        }:
            raise ValueError
        if (
            type(payload["schema_version"]) is not int
            or payload["schema_version"] != 1
            or payload["mode"] != "private_sft"
            or payload["provenance"] != expected_provenance
            or payload["model"] != expected_model
            or payload["settings"] != expected_settings
            or payload["prior_adapter"] != expected_prior_adapter
            or payload["output_volume"] != expected_output_volume
        ):
            raise ValueError
        source_hashes = payload["source_sha256"]
        if type(source_hashes) is not dict or set(source_hashes) != {
            "private_dataset",
            "source_records",
            "synthetic_replay_dataset",
        }:
            raise ValueError
        if any(
            type(source_hashes[name]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", source_hashes[name]) is None
            for name in ("private_dataset", "source_records")
        ):
            raise ValueError
        replay_hash = source_hashes["synthetic_replay_dataset"]
        if replay_hash is not None and (
            type(replay_hash) is not str or re.fullmatch(r"[0-9a-f]{64}", replay_hash) is None
        ):
            raise ValueError
        train_rows, dev_prompts = payload["train_rows"], payload["dev_prompts"]
        if (
            type(train_rows) is not list
            or not 1 <= len(train_rows) <= 256
            or type(dev_prompts) is not list
            or not 1 <= len(dev_prompts) <= 32
        ):
            raise ValueError
        seen_prompts: set[str] = set()
        for row in train_rows:
            if type(row) is not dict or set(row) != {"prompt", "completion"}:
                raise ValueError
            prompt, completion = row["prompt"], row["completion"]
            if (
                type(prompt) is not str
                or not prompt.strip()
                or len(prompt) > 12_000
                or "\x00" in prompt
                or prompt in seen_prompts
                or type(completion) is not str
                or not completion.strip()
                or len(completion) > 4_000
                or "\x00" in completion
            ):
                raise ValueError
            target = json.loads(completion)
            if (
                type(target) is not dict
                or set(target) != {"support", "eligibility"}
                or target["support"] not in {"supported", "unsupported", "uncertain"}
                or target["eligibility"] not in {"eligible", "ineligible", "uncertain"}
                or json.dumps(target, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                != completion
            ):
                raise ValueError
            seen_prompts.add(prompt)
        for prompt in dev_prompts:
            if (
                type(prompt) is not str
                or not prompt.strip()
                or len(prompt) > 12_000
                or "\x00" in prompt
                or prompt in seen_prompts
            ):
                raise ValueError
            seen_prompts.add(prompt)
    except (KeyError, TypeError, ValueError, UnicodeError):
        raise ValueError("private SFT payload is invalid") from None
    return payload


def _remote_compute(request: dict[str, Any]) -> dict[str, Any]:
    """Check exact approvals before importing any model runtime."""
    import hashlib
    import importlib.metadata as metadata
    import json
    import os
    import platform
    import random
    import re
    import time
    from pathlib import Path
    from stat import S_ISDIR, S_ISREG
    from typing import cast

    stage = "approval"
    base_outputs: list[dict[str, Any]] | None = None
    prior_outputs: list[dict[str, Any]] | None = None
    trained_outputs: list[dict[str, Any]] | None = None
    run_root: Path | None = None

    try:
        if type(request) is not dict or set(request) != {
            "payload",
            "payload_sha256",
            "runner_sha256",
        }:
            raise ValueError("private SFT request is invalid")
        payload = request["payload"]
        approved_payload = request["payload_sha256"]
        approved_runner = request["runner_sha256"]
        if any(
            type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in (approved_payload, approved_runner)
        ):
            raise ValueError("private SFT approval is invalid")
        try:
            canonical = json.dumps(
                payload,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError, UnicodeError):
            raise ValueError("private SFT payload is invalid") from None
        if len(canonical) > 4_500_000:
            raise ValueError("private SFT payload is invalid")
        digest = hashlib.sha256(canonical).hexdigest()
        if digest != approved_payload or digest != _EXPECTED_PAYLOAD_SHA256:
            raise ValueError("private SFT payload approval mismatch")
        if approved_runner != _APPROVED_RUNNER_SHA256 or approved_runner != _EXPECTED_RUNNER_SHA256:
            raise ValueError("private SFT runner approval mismatch")
        payload = _VALIDATE_PAYLOAD_SHAPE(  # noqa: F821
            payload,
            _EXPECTED_PROVENANCE,
            _EXPECTED_MODEL,
            _EXPECTED_SETTINGS,
            _EXPECTED_PRIOR_ADAPTER,
            _EXPECTED_OUTPUT_VOLUME,
        )

        stage = "prior_mount_check"
        # Verify the reviewed prior adapter before importing torch or Transformers.
        prior = payload["prior_adapter"]
        prior_hashes = prior["artifact_sha256"]
        if type(prior_hashes) is not dict or set(prior_hashes) != {
            "adapter_config.json",
            "adapter_model.safetensors",
        }:
            raise ValueError("private SFT prior manifest is invalid")
        prior_actual = _VERIFY_ADAPTER_MOUNT(  # noqa: F821
            prior["volume_mount"], prior["relative_path"], prior_hashes
        )
        canonical_prior_mount = Path(prior["volume_mount"]).resolve(strict=True)
        prior_root = canonical_prior_mount / prior["relative_path"]

        # Reserve the payload-specific destination exclusively; never overwrite a run.
        stage = "output_mount_check"
        output_spec = payload["output_volume"]
        output_mount = Path(output_spec["mount"])
        canonical_output_mount = output_mount.resolve(strict=True)
        if not S_ISDIR(canonical_output_mount.lstat().st_mode):
            raise ValueError("private SFT output mount is invalid")
        runs_root = canonical_output_mount / "runs"
        try:
            runs_root.mkdir(mode=0o700)
        except FileExistsError:
            if (
                not S_ISDIR(runs_root.lstat().st_mode)
                or runs_root.resolve(strict=True) != runs_root
            ):
                raise ValueError("private SFT output directory is invalid") from None
        run_root = runs_root / digest
        run_root.mkdir(mode=0o700, exist_ok=False)
        adapter_path = run_root / "adapter"
        adapter_path.mkdir(mode=0o700, exist_ok=False)
        artifact_relative_path = f"runs/{digest}/adapter"

        # Heavy runtime imports happen only after exact approval and mount checks.
        stage = "runtime_setup"
        import modal as modal_sdk
        import torch  # type: ignore[import-not-found]
        from huggingface_hub.utils import disable_progress_bars  # type: ignore[import-not-found]
        from peft import (  # type: ignore[import-not-found]
            PeftModel,
            prepare_model_for_kbit_training,
        )
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        from transformers.utils import (  # type: ignore[import-not-found]
            logging as transformers_logging,
        )

        transformers_logging.set_verbosity_error()
        disable_progress_bars()
        package_names = (
            "torch",
            "transformers",
            "tokenizers",
            "accelerate",
            "peft",
            "bitsandbytes",
            "safetensors",
        )
        versions = {"modal": getattr(modal_sdk, "__version__", "injected runtime")}
        versions.update({name: metadata.version(name) for name in package_names})
        if tuple(versions[name] for name in package_names) != (
            "2.7.1",
            "4.51.3",
            "0.21.1",
            "1.6.0",
            "0.15.2",
            "0.45.5",
            "0.5.3",
        ):
            raise RuntimeError("private SFT dependencies differ from the pinned image")
        if platform.python_version_tuple()[:2] != ("3", "12"):
            raise RuntimeError("private SFT requires Python 3.12")
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("private SFT requires A10G CUDA with BF16 support")

        started = time.perf_counter()
        model_spec = payload["model"]
        settings = payload["settings"]
        torch.manual_seed(settings["seed"])
        torch.cuda.manual_seed_all(settings["seed"])
        device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device)
        torch.cuda.reset_peak_memory_stats(device)
        model_started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(
            model_spec["model_id"],
            revision=model_spec["revision"],
            trust_remote_code=False,
        )
        template, eos_id = tokenizer.chat_template, tokenizer.eos_token_id
        if type(template) is not str or not template or type(eos_id) is not int or eos_id < 0:
            raise ValueError("private SFT tokenizer metadata is invalid")
        pad_id = tokenizer.pad_token_id if type(tokenizer.pad_token_id) is int else eos_id
        quantization = BitsAndBytesConfig(
            load_in_4bit=model_spec["quantization"]["load_in_4bit"],
            bnb_4bit_quant_type=model_spec["quantization"]["type"],
            bnb_4bit_use_double_quant=model_spec["quantization"]["double_quant"],
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            model_spec["model_id"],
            revision=model_spec["revision"],
            trust_remote_code=False,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
            attn_implementation=settings["attn_implementation"],
            quantization_config=quantization,
            device_map="cuda",
            local_files_only=False,
        )
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=settings["gradient_checkpointing"],
        )
        base_model.config.use_cache = True
        base_model.eval()
        model_load_seconds = time.perf_counter() - model_started
        input_device = base_model.get_input_embeddings().weight.device

        def encode_prompt(prompt: str) -> list[int]:
            encoded = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=settings["enable_thinking"],
                return_tensors="pt",
            )
            if not isinstance(encoded, torch.Tensor) or encoded.ndim != 2 or encoded.shape[0] != 1:
                raise ValueError("private SFT chat template returned invalid token IDs")
            ids = encoded[0].detach().cpu().tolist()
            if (
                not ids
                or len(ids) > settings["max_input_tokens"]
                or any(type(token_id) is not int or token_id < 0 for token_id in ids)
            ):
                raise ValueError("private SFT prompt tokenization is invalid")
            return cast(list[int], ids)

        dev_prompts = payload["dev_prompts"]
        prepared = [
            (f"dev_{index:03d}", encode_prompt(prompt)) for index, prompt in enumerate(dev_prompts)
        ]

        def generate(active_model: Any) -> list[dict[str, Any]]:
            active_model.eval()
            outputs: list[dict[str, Any]] = []
            for item_id, input_ids in prepared:
                input_tensor = torch.tensor([input_ids], dtype=torch.long, device=input_device)
                with torch.inference_mode():
                    generated = active_model.generate(
                        input_ids=input_tensor,
                        attention_mask=torch.ones_like(input_tensor),
                        do_sample=settings["do_sample"],
                        num_beams=1,
                        max_new_tokens=settings["max_new_tokens"],
                        eos_token_id=eos_id,
                        pad_token_id=pad_id,
                        use_cache=True,
                    )
                if (
                    not isinstance(generated, torch.Tensor)
                    or generated.ndim != 2
                    or generated.shape[0] != 1
                ):
                    raise ValueError("private SFT generation returned invalid token IDs")
                output_ids = generated[0, len(input_ids) :].detach().cpu().tolist()
                if any(type(token_id) is not int or token_id < 0 for token_id in output_ids):
                    raise ValueError("private SFT output token IDs are invalid")
                if not output_ids:
                    finish_reason = "empty"
                elif output_ids[-1] == eos_id:
                    finish_reason = "eos"
                elif len(output_ids) >= settings["max_new_tokens"]:
                    finish_reason = "length"
                else:
                    finish_reason = "unknown"
                outputs.append(
                    {
                        "item_id": item_id,
                        "raw_output": tokenizer.decode(output_ids, skip_special_tokens=True),
                        "finish_reason": finish_reason,
                        "input_tokens": len(input_ids),
                        "output_tokens": len(output_ids),
                        "input_token_ids_sha256": _TOKEN_IDS_SHA256(input_ids),  # noqa: F821
                        "output_token_ids_sha256": _TOKEN_IDS_SHA256(output_ids),  # noqa: F821
                        "input_token_ids": input_ids,
                        "output_token_ids": output_ids,
                    }
                )
            torch.cuda.synchronize()
            return outputs

        stage = "base_eval"
        base_started = time.perf_counter()
        base_outputs = generate(base_model)
        _WRITE_PRIVATE_JSON(run_root / "base_outputs.json", {"outputs": base_outputs})  # noqa: F821
        _REMOTE_OUTPUT_VOLUME.commit()  # noqa: F821
        base_seconds = time.perf_counter() - base_started
        stage = "prior_eval"
        prior_started = time.perf_counter()
        prior_model = PeftModel.from_pretrained(
            base_model,
            str(prior_root),
            is_trainable=False,
            local_files_only=True,
        )
        if any(parameter.requires_grad for parameter in prior_model.parameters()):
            raise RuntimeError("private SFT prior adapter unexpectedly has trainable parameters")
        prior_outputs = generate(prior_model)
        _WRITE_PRIVATE_JSON(run_root / "prior_outputs.json", {"outputs": prior_outputs})  # noqa: F821
        _REMOTE_OUTPUT_VOLUME.commit()  # noqa: F821
        prior_seconds = time.perf_counter() - prior_started
        base_model = prior_model.unload()
        del prior_model
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        stage = "training"
        base_model.config.use_cache = False
        if not base_model.is_gradient_checkpointing:
            base_model.gradient_checkpointing_enable()
        tuned_model = PeftModel.from_pretrained(
            base_model,
            str(prior_root),
            is_trainable=True,
            local_files_only=True,
        )
        trainable = [parameter for parameter in tuned_model.parameters() if parameter.requires_grad]
        if not trainable or any(
            "lora_" not in name
            for name, parameter in tuned_model.named_parameters()
            if parameter.requires_grad
        ):
            raise RuntimeError("private SFT may train only prior LoRA adapter parameters")
        input_device = tuned_model.get_input_embeddings().weight.device
        train_examples: list[tuple[list[int], list[int]]] = []
        for row in payload["train_rows"]:
            prompt_ids = encode_prompt(row["prompt"])
            completion_ids = tokenizer.encode(row["completion"], add_special_tokens=False)
            labels = _COMPLETION_LABELS(prompt_ids, completion_ids, eos_id)  # noqa: F821
            sequence = prompt_ids + completion_ids + [eos_id]
            if len(sequence) > settings["max_input_tokens"]:
                raise ValueError("private SFT training sequence exceeds 2048 tokens")
            train_examples.append((sequence, labels))

        def adapter_state_sha256(active_model: Any) -> str:
            state = hashlib.sha256()
            for name, parameter in sorted(active_model.named_parameters()):
                if "lora_" not in name:
                    continue
                tensor = parameter.detach().contiguous()
                state.update(name.encode("utf-8"))
                state.update(str(tuple(tensor.shape)).encode("ascii"))
                state.update(str(tensor.dtype).encode("ascii"))
                state.update(tensor.view(torch.uint8).cpu().numpy().tobytes())
            return state.hexdigest()

        initial_adapter_hash = adapter_state_sha256(tuned_model)
        optimizer = torch.optim.AdamW(trainable, lr=settings["learning_rate"])
        optimizer.zero_grad(set_to_none=True)
        microstep_losses: list[float] = []
        optimizer_step_losses: list[float] = []
        training_started = time.perf_counter()
        tuned_model.train()
        indices = list(range(len(train_examples)))
        random.Random(settings["seed"]).shuffle(indices)
        pending_losses: list[float] = []
        for position, index in enumerate(indices):
            sequence, labels = train_examples[index]
            ids = torch.tensor([sequence], dtype=torch.long, device=input_device)
            label_tensor = torch.tensor([labels], dtype=torch.long, device=input_device)
            loss = tuned_model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                labels=label_tensor,
            ).loss
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("private SFT training produced a non-finite loss")
            loss_value = float(loss.detach().cpu().item())
            microstep_losses.append(loss_value)
            pending_losses.append(loss_value)
            accumulation_size = _ACCUMULATION_WINDOW_SIZE(  # noqa: F821
                position, len(indices), settings["gradient_accumulation_steps"]
            )
            (loss / accumulation_size).backward()
            if (position + 1) % settings["gradient_accumulation_steps"] == 0 or position + 1 == len(
                indices
            ):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step_losses.append(sum(pending_losses) / len(pending_losses))
                pending_losses = []
        torch.cuda.synchronize()
        training_seconds = time.perf_counter() - training_started
        final_adapter_hash = adapter_state_sha256(tuned_model)
        if initial_adapter_hash == final_adapter_hash:
            raise RuntimeError("private SFT adapter did not change during training")

        prior_after: dict[str, str] = {}
        for filename in prior_hashes:
            prior_after[filename] = hashlib.sha256((prior_root / filename).read_bytes()).hexdigest()
        if prior_after != prior_actual:
            raise RuntimeError("private SFT read-only prior adapter changed during training")

        tuned_model.save_pretrained(str(adapter_path), safe_serialization=True)
        artifact_hashes: dict[str, str] = {}
        for filename in sorted(os.listdir(adapter_path)):
            artifact_path = adapter_path / filename
            if filename.endswith((".bin", ".pt", ".pth")) or not S_ISREG(
                artifact_path.lstat().st_mode
            ):
                raise RuntimeError("private SFT adapter emitted an unsafe artifact")
            artifact_hashes[filename] = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if not {"adapter_config.json", "adapter_model.safetensors"} <= set(artifact_hashes) or set(
            artifact_hashes
        ) - {"README.md", "adapter_config.json", "adapter_model.safetensors"}:
            raise RuntimeError("private SFT adapter artifact set is invalid")
        artifact_manifest = {
            "volume_name": output_spec["name"],
            "volume_mount": output_spec["mount"],
            "relative_path": artifact_relative_path,
            "artifact_sha256": artifact_hashes,
        }
        trainable_parameters = sum(parameter.numel() for parameter in trainable)
        training = {
            "row_count": len(train_examples),
            "microsteps": len(microstep_losses),
            "optimizer_steps": len(optimizer_step_losses),
            "input_tokens": sum(len(sequence) for sequence, _ in train_examples),
            "supervised_tokens": sum(
                sum(label != -100 for label in labels) for _, labels in train_examples
            ),
            "mean_loss": sum(microstep_losses) / len(microstep_losses),
            "optimizer_step_losses": optimizer_step_losses,
            "trainable_parameters": trainable_parameters,
            "adapter_hash_before_training": initial_adapter_hash,
            "adapter_hash_after_training": final_adapter_hash,
            "adapter_changed": True,
            "seconds": round(training_seconds, 6),
        }

        del optimizer, trainable, train_examples, tuned_model, base_model
        gc.collect()
        torch.cuda.empty_cache()
        trained_base = AutoModelForCausalLM.from_pretrained(
            model_spec["model_id"],
            revision=model_spec["revision"],
            trust_remote_code=False,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
            attn_implementation=settings["attn_implementation"],
            quantization_config=quantization,
            device_map="cuda",
            local_files_only=False,
        )
        trained_base = prepare_model_for_kbit_training(
            trained_base,
            use_gradient_checkpointing=settings["gradient_checkpointing"],
        )
        trained_base.config.use_cache = True
        trained_model = PeftModel.from_pretrained(
            trained_base,
            str(adapter_path),
            is_trainable=False,
            local_files_only=True,
        )
        if any(parameter.requires_grad for parameter in trained_model.parameters()):
            raise RuntimeError("private SFT trained adapter unexpectedly has trainable parameters")
        input_device = trained_model.get_input_embeddings().weight.device
        stage = "trained_eval"
        trained_started = time.perf_counter()
        trained_outputs = generate(trained_model)
        trained_seconds = time.perf_counter() - trained_started
        torch.cuda.synchronize()
        receipt = {
            "schema_version": 1,
            "status": "success",
            "mode": "private_sft",
            "training_performed": True,
            "payload_sha256": approved_payload,
            "runner_sha256": approved_runner,
            "source_sha256": payload["source_sha256"],
            "provenance": payload["provenance"],
            "model": model_spec,
            "settings": settings,
            "prior_adapter": prior,
            "artifact_manifest": artifact_manifest,
            "training": training,
            "versions": {**versions, "python": platform.python_version()},
            "tokenizer": {
                "class": type(tokenizer).__name__,
                "is_fast": bool(tokenizer.is_fast),
                "chat_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
                "eos_token_id": eos_id,
                "pad_token_id": pad_id,
            },
            "modal_function_call_id": modal_sdk.current_function_call_id(),
            "base_outputs": base_outputs,
            "prior_outputs": prior_outputs,
            "trained_outputs": trained_outputs,
            "timings": {
                "model_load_seconds": round(model_load_seconds, 6),
                "base_seconds": round(base_seconds, 6),
                "prior_seconds": round(prior_seconds, 6),
                "trained_seconds": round(trained_seconds, 6),
                "total_seconds": round(time.perf_counter() - started, 6),
            },
            "hardware": {
                "gpu_name": gpu_name,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            },
        }
        stage = "write_success_receipt"
        _WRITE_PRIVATE_JSON(run_root / "receipt.json", receipt)  # noqa: F821
        _REMOTE_OUTPUT_VOLUME.commit()  # noqa: F821
        return receipt
    except Exception as error:
        failure = {
            "schema_version": 1,
            "status": "remote_failed",
            "mode": "private_sft",
            "training_performed": None,
            "payload_sha256": _EXPECTED_PAYLOAD_SHA256,
            "runner_sha256": _APPROVED_RUNNER_SHA256,
            "stage": stage,
            "error": {"type": type(error).__name__, "message": "private SFT failed"},
            "base_outputs": base_outputs,
            "prior_outputs": prior_outputs,
            "trained_outputs": trained_outputs,
        }
        if run_root is not None:
            _WRITE_PRIVATE_JSON(run_root / "receipt.json", failure)  # noqa: F821
        return failure
    finally:
        _REMOTE_OUTPUT_VOLUME.commit()  # noqa: F821


def _remote_safe_entry(request: dict[str, Any]) -> dict[str, Any]:
    """Return only fixed metadata and a generic failure, never private details."""
    try:
        return _REMOTE_COMPUTE(request)  # noqa: F821
    except Exception as error:
        return {
            "schema_version": 1,
            "status": "remote_failed",
            "mode": "private_sft",
            "payload_sha256": _APPROVED_PAYLOAD_SHA256,
            "runner_sha256": _APPROVED_RUNNER_SHA256,
            "training_performed": None,
            "training": None,
            "artifact_manifest": None,
            "stage": "remote_entry",
            "base_outputs": None,
            "prior_outputs": None,
            "trained_outputs": None,
            "error": {"type": type(error).__name__, "message": "private SFT failed"},
        }
