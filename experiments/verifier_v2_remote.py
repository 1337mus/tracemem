"""Source-free remote implementation for the bounded verifier v2 run."""

from __future__ import annotations

from typing import Any

_EXPECTED_COUNTS: dict[str, int] = {}
_EXPECTED_COHORT_COUNTS: dict[str, int] = {}
_EXPECTED_MODEL: dict[str, Any] = {}
_EXPECTED_OUTPUT_MOUNT = ""
_EXPECTED_PAYLOAD_SHA256 = ""
_EXPECTED_PRIOR_ADAPTER: dict[str, Any] = {}
_EXPECTED_PRIOR_ADAPTER_ROOT = ""
_EXPECTED_PRIOR_REFERENCE_OUTPUTS_SHA256 = ""
_EXPECTED_PRIOR_REFERENCE_SHA256 = ""
_EXPECTED_RUNNER_SHA256 = ""
_EXPECTED_SETTINGS: dict[str, Any] = {}
_EXPECTED_SOURCE_HASHES: dict[str, str] = {}
_REMOTE_OUTPUT_VOLUME: Any = None


def _collect_parity_differences(
    items: list[dict[str, Any]],
    baseline_outputs: list[dict[str, Any]],
    prior_outputs: list[dict[str, Any]],
    prior_reference_outputs: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Return all transfer output field differences in payload item order."""
    reference_by_arm = {
        "baseline_outputs": {
            row["item_id"]: row for row in prior_reference_outputs["baseline_outputs"]
        },
        "prior_outputs": {row["item_id"]: row for row in prior_reference_outputs["tuned_outputs"]},
    }
    actual_by_arm = {
        "baseline_outputs": {row["item_id"]: row for row in baseline_outputs},
        "prior_outputs": {row["item_id"]: row for row in prior_outputs},
    }
    differences: list[dict[str, Any]] = []
    for item in items:
        if item["cohort"] != "transfer_regression":
            continue
        item_id = item["item_id"]
        for arm in ("baseline_outputs", "prior_outputs"):
            expected = reference_by_arm[arm].get(item_id)
            actual = actual_by_arm[arm].get(item_id)
            if expected is None or actual is None:
                if expected != actual:
                    differences.append(
                        {
                            "item_id": item_id,
                            "arm": arm,
                            "field": "$row",
                            "expected": expected,
                            "actual": actual,
                        }
                    )
                continue
            for field in sorted(set(expected) | set(actual)):
                expected_present = field in expected
                actual_present = field in actual
                expected_value = expected.get(field)
                actual_value = actual.get(field)
                if expected_present != actual_present or expected_value != actual_value:
                    differences.append(
                        {
                            "item_id": item_id,
                            "arm": arm,
                            "field": field,
                            "expected": expected_value,
                            "actual": actual_value,
                        }
                    )
    return differences


def _remote_compute(request: dict[str, Any]) -> dict[str, Any]:
    """Validate one approved continuation payload, then compare and train on one A10G."""
    import hashlib as remote_hashlib
    import importlib.metadata as metadata
    import json as remote_json
    import os
    import platform
    import random
    import re
    import time
    from pathlib import Path as RemotePath
    from stat import S_ISDIR, S_ISREG
    from typing import cast

    if type(request) is not dict or set(request) != {"payload", "payload_sha256", "runner_sha256"}:
        raise ValueError("remote verifier v2 request keys are invalid")
    payload = request["payload"]
    approved_payload, approved_runner = request["payload_sha256"], request["runner_sha256"]
    for digest in (approved_payload, approved_runner):
        if type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("remote verifier v2 approval hashes are invalid")
    if approved_payload != _EXPECTED_PAYLOAD_SHA256:
        raise ValueError("remote verifier v2 payload SHA-256 is not allowlisted")
    try:
        payload_bytes = remote_json.dumps(
            payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("remote verifier v2 payload is not canonical JSON") from exc
    if len(payload_bytes) > 2_000_000:
        raise ValueError("remote verifier v2 payload exceeds the byte limit")
    if remote_hashlib.sha256(payload_bytes).hexdigest() != approved_payload:
        raise ValueError("remote verifier v2 payload SHA-256 does not match approval")

    expected_keys = {
        "schema_version",
        "mode",
        "split",
        "source_sha256",
        "runner_sha256",
        "model",
        "settings",
        "prior_adapter",
        "counts",
        "items",
        "train_rows",
        "prior_reference_sha256",
        "prior_reference_outputs",
    }
    if type(payload) is not dict or set(payload) != expected_keys:
        raise ValueError("remote verifier v2 payload keys are invalid")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
        or payload["mode"] != "continue_train"
        or payload["split"] != "dev"
    ):
        raise ValueError("remote verifier v2 mode, split, or schema version is invalid")
    if payload["source_sha256"] != _EXPECTED_SOURCE_HASHES:
        raise ValueError("remote verifier v2 source hashes are not allowlisted")
    if payload["runner_sha256"] != approved_runner or approved_runner != _EXPECTED_RUNNER_SHA256:
        raise ValueError("remote verifier v2 runner SHA-256 does not match approval")
    if (
        payload["model"] != _EXPECTED_MODEL
        or payload["settings"] != _EXPECTED_SETTINGS
        or payload["prior_adapter"] != _EXPECTED_PRIOR_ADAPTER
        or payload["counts"] != _EXPECTED_COUNTS
    ):
        raise ValueError("remote verifier v2 model, settings, adapter, or counts changed")
    prior_reference_outputs = payload["prior_reference_outputs"]
    try:
        prior_reference_bytes = remote_json.dumps(
            prior_reference_outputs,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError(
            "remote verifier v2 prior reference outputs are not canonical JSON"
        ) from exc
    if (
        payload["prior_reference_sha256"] != _EXPECTED_PRIOR_REFERENCE_SHA256
        or remote_hashlib.sha256(prior_reference_bytes).hexdigest()
        != _EXPECTED_PRIOR_REFERENCE_OUTPUTS_SHA256
    ):
        raise ValueError("remote verifier v2 prior reference changed")

    items = payload["items"]
    if type(items) is not list or len(items) != 108:
        raise ValueError("remote verifier v2 requires exactly 108 evaluation items")
    seen: set[str] = set()
    cohort_counts = {name: 0 for name in _EXPECTED_COHORT_COUNTS}
    for item in items:
        if type(item) is not dict or set(item) != {"item_id", "prompt", "cohort"}:
            raise ValueError("remote verifier v2 item shape is invalid")
        item_id, prompt, cohort = item["item_id"], item["prompt"], item["cohort"]
        if (
            type(item_id) is not str
            or re.fullmatch(r"[A-Za-z0-9_-]+::[A-Za-z0-9_-]+", item_id) is None
            or item_id in seen
            or cohort not in cohort_counts
        ):
            raise ValueError("remote verifier v2 item IDs or cohorts are invalid")
        if type(prompt) is not str or not prompt.strip() or len(prompt.encode("utf-8")) > 100_000:
            raise ValueError("remote verifier v2 prompt is empty or exceeds its byte limit")
        seen.add(item_id)
        cohort_counts[cohort] += 1
    if cohort_counts != _EXPECTED_COHORT_COUNTS:
        raise ValueError("remote verifier v2 evaluation cohort counts changed")

    train_rows = payload["train_rows"]
    if type(train_rows) is not list or len(train_rows) != 192:
        raise ValueError("remote verifier v2 requires exactly 192 train rows")
    for row in train_rows:
        if type(row) is not dict or set(row) != {"prompt", "completion"}:
            raise ValueError("remote verifier v2 training row shape is invalid")
        prompt, completion = row["prompt"], row["completion"]
        if (
            type(prompt) is not str
            or not prompt.strip()
            or len(prompt.encode("utf-8")) > 100_000
            or type(completion) is not str
            or len(completion.encode("utf-8")) > 1000
        ):
            raise ValueError("remote verifier v2 training row exceeds its text bounds")
        try:
            target = remote_json.loads(completion)
        except (TypeError, ValueError) as exc:
            raise ValueError("remote verifier v2 completion is invalid JSON") from exc
        if (
            type(target) is not dict
            or set(target) != {"support", "eligibility"}
            or target["support"] not in {"supported", "unsupported", "uncertain"}
            or target["eligibility"] not in {"eligible", "ineligible", "uncertain"}
        ):
            raise ValueError("remote verifier v2 completion labels are invalid")

    # Check both mounted volumes and every prior artifact before importing Modal or ML packages.
    prior_mount = RemotePath(_EXPECTED_PRIOR_ADAPTER["volume_mount"])
    prior_root = RemotePath(_EXPECTED_PRIOR_ADAPTER_ROOT)
    try:
        canonical_prior = prior_mount.resolve(strict=True)
        if (
            not S_ISDIR(canonical_prior.stat().st_mode)
            or not S_ISDIR(prior_root.lstat().st_mode)
            or prior_root.resolve(strict=True)
            != canonical_prior / _EXPECTED_PRIOR_ADAPTER["relative_path"]
        ):
            raise ValueError("remote verifier v2 prior adapter directory is invalid")
        prior_expected = {
            "adapter_config.json": _EXPECTED_PRIOR_ADAPTER["config_sha256"],
            "adapter_model.safetensors": _EXPECTED_PRIOR_ADAPTER["safetensors_sha256"],
        }
        prior_hashes: dict[str, str] = {}
        for filename, expected_hash in prior_expected.items():
            path = prior_root / filename
            if not S_ISREG(path.lstat().st_mode):
                raise ValueError("remote verifier v2 prior adapter artifact is not a regular file")
            actual_hash = remote_hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                raise ValueError(f"remote verifier v2 prior {filename} SHA-256 mismatch")
            prior_hashes[filename] = actual_hash

        output_mount = RemotePath(_EXPECTED_OUTPUT_MOUNT)
        canonical_output = output_mount.resolve(strict=True)
        if not S_ISDIR(canonical_output.stat().st_mode):
            raise ValueError("remote verifier v2 output volume is invalid")
        runs_root = canonical_output / "runs"
        if runs_root.exists() and (
            not S_ISDIR(runs_root.lstat().st_mode)
            or runs_root.resolve(strict=True) != canonical_output / "runs"
        ):
            raise ValueError("remote verifier v2 output runs directory is invalid")
        run_path = runs_root / approved_payload
        if run_path.exists() or run_path.is_symlink():
            raise FileExistsError("remote verifier v2 run directory must be new")
    except (OSError, RuntimeError) as exc:
        raise ValueError(
            "remote verifier v2 mounted volumes or prior artifacts are unsafe"
        ) from exc

    import modal as modal_sdk
    import torch  # type: ignore[import-not-found]
    from peft import PeftModel, prepare_model_for_kbit_training  # type: ignore[import-not-found]
    from transformers import (  # type: ignore[import-not-found]
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    packages = (
        "torch",
        "transformers",
        "tokenizers",
        "accelerate",
        "peft",
        "bitsandbytes",
        "safetensors",
    )
    expected_versions = ("2.7.1", "4.51.3", "0.21.1", "1.6.0", "0.15.2", "0.45.5", "0.5.3")
    versions = {"modal": getattr(modal_sdk, "__version__", "injected runtime")}
    versions.update({name: metadata.version(name) for name in packages})
    if tuple(versions[name] for name in packages) != expected_versions:
        raise RuntimeError("remote verifier v2 dependency versions do not match the pinned image")
    if platform.python_version_tuple()[:2] != ("3", "12"):
        raise RuntimeError("remote verifier v2 requires Python 3.12")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("remote verifier v2 requires an A10G CUDA device with BF16 support")

    runs_root.mkdir(exist_ok=True)
    os.mkdir(run_path)
    adapter_path = run_path / "adapter"
    os.mkdir(adapter_path)
    try:
        started = time.perf_counter()
        torch.manual_seed(_EXPECTED_SETTINGS["seed"])
        torch.cuda.manual_seed_all(_EXPECTED_SETTINGS["seed"])
        device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device)
        torch.cuda.reset_peak_memory_stats(device)
        model_started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(
            _EXPECTED_MODEL["model_id"],
            revision=_EXPECTED_MODEL["revision"],
            trust_remote_code=False,
            local_files_only=False,
        )
        template = tokenizer.chat_template
        eos_id = tokenizer.eos_token_id
        if type(template) is not str or not template or type(eos_id) is not int or eos_id < 0:
            raise ValueError("pinned Qwen tokenizer template or EOS token is invalid")
        pad_id = tokenizer.pad_token_id if type(tokenizer.pad_token_id) is int else eos_id
        tokenizer_info = {
            "class": type(tokenizer).__name__,
            "is_fast": bool(tokenizer.is_fast),
            "chat_template_sha256": remote_hashlib.sha256(template.encode("utf-8")).hexdigest(),
            "eos_token_id": eos_id,
            "pad_token_id": pad_id,
        }
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            _EXPECTED_MODEL["model_id"],
            revision=_EXPECTED_MODEL["revision"],
            trust_remote_code=False,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
            attn_implementation=_EXPECTED_SETTINGS["attn_implementation"],
            quantization_config=quantization,
            device_map="cuda",
            local_files_only=False,
        )
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=True,
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
                enable_thinking=False,
                return_tensors="pt",
            )
            if not isinstance(encoded, torch.Tensor) or encoded.ndim != 2 or encoded.shape[0] != 1:
                raise ValueError("Qwen chat template returned an invalid token tensor")
            ids = encoded[0].detach().cpu().tolist()
            if not ids or len(ids) > _EXPECTED_SETTINGS["max_input_tokens"]:
                raise ValueError("verifier prompt exceeds 2048 tokens; truncation is disabled")
            if any(type(token_id) is not int or token_id < 0 for token_id in ids):
                raise ValueError("Qwen chat template returned invalid input token IDs")
            return cast(list[int], ids)

        prepared = [(item, encode_prompt(item["prompt"])) for item in items]

        def generate(active_model: Any) -> list[dict[str, Any]]:
            active_model.eval()
            outputs: list[dict[str, Any]] = []
            for item, input_ids in prepared:
                input_tensor = torch.tensor([input_ids], dtype=torch.long, device=input_device)
                with torch.inference_mode():
                    generated = active_model.generate(
                        input_ids=input_tensor,
                        attention_mask=torch.ones_like(input_tensor),
                        do_sample=_EXPECTED_SETTINGS["do_sample"],
                        num_beams=1,
                        max_new_tokens=_EXPECTED_SETTINGS["max_new_tokens"],
                        eos_token_id=eos_id,
                        pad_token_id=pad_id,
                    )
                if (
                    not isinstance(generated, torch.Tensor)
                    or generated.ndim != 2
                    or generated.shape[0] != 1
                ):
                    raise ValueError("Qwen generation returned an invalid token tensor")
                output_ids = generated[0, len(input_ids) :].detach().cpu().tolist()
                if any(type(token_id) is not int or token_id < 0 for token_id in output_ids):
                    raise ValueError("Qwen generation returned invalid output token IDs")
                ended = bool(output_ids and output_ids[-1] == eos_id)
                hit_max = len(output_ids) >= _EXPECTED_SETTINGS["max_new_tokens"]
                outputs.append(
                    {
                        "item_id": item["item_id"],
                        "raw_output": tokenizer.decode(output_ids, skip_special_tokens=True),
                        "finish_reason": "eos" if ended else "length" if hit_max else "unknown",
                        "input_tokens": len(input_ids),
                        "output_tokens": len(output_ids),
                        "input_token_ids": input_ids,
                        "output_token_ids": output_ids,
                        "ended_with_eos": ended,
                        "hit_max_new_tokens": hit_max,
                    }
                )
            torch.cuda.synchronize()
            return outputs

        baseline_started = time.perf_counter()
        baseline_outputs = generate(base_model)
        baseline_seconds = time.perf_counter() - baseline_started
        prior_started = time.perf_counter()
        prior_model = PeftModel.from_pretrained(
            base_model,
            str(prior_root),
            is_trainable=False,
            local_files_only=True,
        )
        if any(parameter.requires_grad for parameter in prior_model.parameters()):
            raise RuntimeError("frozen prior adapter unexpectedly has trainable parameters")
        prior_outputs = generate(prior_model)
        prior_seconds = time.perf_counter() - prior_started
        parity_differences = _collect_parity_differences(
            items,
            baseline_outputs,
            prior_outputs,
            prior_reference_outputs,
        )
        if parity_differences:
            torch.cuda.synchronize()
            return {
                "schema_version": 1,
                "status": "protocol_failed",
                "mode": "continue_train",
                "split": "dev",
                "payload_sha256": approved_payload,
                "runner_sha256": approved_runner,
                "source_sha256": _EXPECTED_SOURCE_HASHES,
                "model": _EXPECTED_MODEL,
                "settings": _EXPECTED_SETTINGS,
                "prior_adapter": _EXPECTED_PRIOR_ADAPTER,
                "counts": _EXPECTED_COUNTS,
                "prior_adapter_artifact_sha256": prior_hashes,
                "versions": {**versions, "python": platform.python_version()},
                "tokenizer": tokenizer_info,
                "modal_function_call_id": modal_sdk.current_function_call_id(),
                "baseline_outputs": baseline_outputs,
                "prior_outputs": prior_outputs,
                "parity_differences": parity_differences,
                "training_performed": False,
                "training": {"row_count": 0, "microsteps": 0, "optimizer_steps": 0},
                "timings": {
                    "model_load_seconds": round(model_load_seconds, 6),
                    "baseline_seconds": round(baseline_seconds, 6),
                    "prior_seconds": round(prior_seconds, 6),
                    "total_seconds": round(time.perf_counter() - started, 6),
                },
                "hardware": {
                    "gpu_name": gpu_name,
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                },
            }

        base_model = prior_model.unload()
        del prior_model
        import gc

        gc.collect()
        torch.cuda.empty_cache()
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
            raise RuntimeError("only prior LoRA adapter parameters may be trainable")
        input_device = tuned_model.get_input_embeddings().weight.device

        train_examples: list[tuple[list[int], list[int]]] = []
        for row in train_rows:
            prompt_ids = encode_prompt(row["prompt"])
            completion_ids = tokenizer.encode(row["completion"], add_special_tokens=False)
            if not completion_ids:
                raise ValueError("verifier v2 completion tokenization returned no tokens")
            sequence = prompt_ids + completion_ids + [eos_id]
            if len(sequence) > _EXPECTED_SETTINGS["max_input_tokens"]:
                raise ValueError("verifier v2 training sequence exceeds 2048 tokens")
            train_examples.append((sequence, [-100] * len(prompt_ids) + completion_ids + [eos_id]))
        if len(train_examples) != 192:
            raise ValueError("verifier v2 training example count changed after tokenization")

        def adapter_state_sha256() -> str:
            state = remote_hashlib.sha256()
            for name, parameter in tuned_model.named_parameters():
                if "lora_" in name:
                    state.update(name.encode("utf-8"))
                    state.update(
                        parameter.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
                    )
            return state.hexdigest()

        initial_hash = adapter_state_sha256()
        optimizer = torch.optim.AdamW(trainable, lr=_EXPECTED_SETTINGS["learning_rate"])
        optimizer.zero_grad(set_to_none=True)
        micro_losses: list[float] = []
        optimizer_losses: list[float] = []
        training_started = time.perf_counter()
        tuned_model.train()
        for epoch in range(_EXPECTED_SETTINGS["epochs"]):
            indices = list(range(len(train_examples)))
            random.Random(_EXPECTED_SETTINGS["seed"] + epoch).shuffle(indices)
            group_losses: list[float] = []
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
                    raise RuntimeError("verifier v2 training produced a non-finite loss")
                value = float(loss.detach().cpu().item())
                micro_losses.append(value)
                group_losses.append(value)
                (loss / _EXPECTED_SETTINGS["gradient_accumulation_steps"]).backward()
                if (position + 1) % _EXPECTED_SETTINGS["gradient_accumulation_steps"] == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_losses.append(sum(group_losses[-4:]) / 4)
        torch.cuda.synchronize()
        training_seconds = time.perf_counter() - training_started
        final_hash = adapter_state_sha256()
        if initial_hash == final_hash or len(micro_losses) != 384 or len(optimizer_losses) != 96:
            raise RuntimeError("verifier v2 adapter mutation or training step counts are invalid")
        trained_row_count = len(train_examples)
        prior_hashes_after: dict[str, str] = {}
        for filename in prior_expected:
            prior_hashes_after[filename] = remote_hashlib.sha256(
                (prior_root / filename).read_bytes()
            ).hexdigest()
        if prior_hashes_after != prior_hashes:
            raise RuntimeError("read-only prior adapter artifacts changed during continuation")

        tuned_model.save_pretrained(str(adapter_path), safe_serialization=True)
        artifact_hashes: dict[str, str] = {}
        for filename in sorted(os.listdir(adapter_path)):
            path = adapter_path / filename
            if filename.endswith((".bin", ".pt", ".pth")) or not S_ISREG(path.lstat().st_mode):
                raise RuntimeError("verifier v2 adapter emitted an unsafe artifact")
            artifact_hashes[filename] = remote_hashlib.sha256(path.read_bytes()).hexdigest()
        if not {"adapter_config.json", "adapter_model.safetensors"} <= set(artifact_hashes) or set(
            artifact_hashes
        ) - {"README.md", "adapter_config.json", "adapter_model.safetensors"}:
            raise RuntimeError("verifier v2 adapter artifact set changed")
        trainable_names = sorted(
            name for name, parameter in tuned_model.named_parameters() if parameter.requires_grad
        )
        trainable_parameter_count = sum(parameter.numel() for parameter in trainable)

        # Reload a clean base so the k-bit training preparation cast cannot affect inference.
        del optimizer, trainable, train_examples, tuned_model, base_model
        del loss, ids, label_tensor, adapter_state_sha256
        gc.collect()
        torch.cuda.empty_cache()
        reload_started = time.perf_counter()
        tuned_base = AutoModelForCausalLM.from_pretrained(
            _EXPECTED_MODEL["model_id"],
            revision=_EXPECTED_MODEL["revision"],
            trust_remote_code=False,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
            attn_implementation=_EXPECTED_SETTINGS["attn_implementation"],
            quantization_config=quantization,
            device_map="cuda",
            local_files_only=False,
        )
        tuned_base = prepare_model_for_kbit_training(
            tuned_base,
            use_gradient_checkpointing=True,
        )
        tuned_base.config.use_cache = True
        tuned_model = PeftModel.from_pretrained(
            tuned_base,
            str(adapter_path),
            is_trainable=False,
            local_files_only=True,
        )
        if any(parameter.requires_grad for parameter in tuned_model.parameters()):
            raise RuntimeError("reloaded verifier v2 adapter unexpectedly has trainable parameters")
        input_device = tuned_model.get_input_embeddings().weight.device
        reload_seconds = time.perf_counter() - reload_started
        after_reload_hashes = {
            filename: remote_hashlib.sha256((adapter_path / filename).read_bytes()).hexdigest()
            for filename in artifact_hashes
        }
        if after_reload_hashes != artifact_hashes:
            raise RuntimeError("verifier v2 adapter artifacts changed during inference reload")
        tuned_started = time.perf_counter()
        tuned_outputs = generate(tuned_model)
        tuned_seconds = time.perf_counter() - tuned_started
        return {
            "schema_version": 1,
            "status": "success",
            "mode": "continue_train",
            "split": "dev",
            "payload_sha256": approved_payload,
            "runner_sha256": approved_runner,
            "source_sha256": _EXPECTED_SOURCE_HASHES,
            "model": _EXPECTED_MODEL,
            "settings": _EXPECTED_SETTINGS,
            "prior_adapter": _EXPECTED_PRIOR_ADAPTER,
            "counts": _EXPECTED_COUNTS,
            "baseline_outputs": baseline_outputs,
            "prior_outputs": prior_outputs,
            "tuned_outputs": tuned_outputs,
            "prior_adapter_artifact_sha256": prior_hashes,
            "prior_adapter_artifact_sha256_after_training": prior_hashes_after,
            "versions": {**versions, "python": platform.python_version()},
            "tokenizer": tokenizer_info,
            "modal_function_call_id": modal_sdk.current_function_call_id(),
            "timings": {
                "model_load_seconds": round(model_load_seconds, 6),
                "baseline_seconds": round(baseline_seconds, 6),
                "prior_seconds": round(prior_seconds, 6),
                "training_seconds": round(training_seconds, 6),
                "tuned_seconds": round(tuned_seconds, 6),
                "tuned_reload_seconds": round(reload_seconds, 6),
                "total_seconds": round(time.perf_counter() - started, 6),
            },
            "training": {
                "row_count": trained_row_count,
                "epochs": _EXPECTED_SETTINGS["epochs"],
                "microsteps": len(micro_losses),
                "optimizer_steps": len(optimizer_losses),
                "learning_rate": _EXPECTED_SETTINGS["learning_rate"],
                "microstep_losses": micro_losses,
                "optimizer_step_losses": optimizer_losses,
                "adapter_path": f"{_EXPECTED_OUTPUT_MOUNT}/runs/{approved_payload}/adapter",
                "adapter_relative_path": f"runs/{approved_payload}/adapter",
                "trainable_parameter_count": trainable_parameter_count,
                "trainable_parameter_names": trainable_names,
                "adapter_artifact_sha256": artifact_hashes,
                "adapter_artifact_sha256_after_reload": after_reload_hashes,
                "adapter_mutation": {
                    "initial_state_sha256": initial_hash,
                    "final_state_sha256": final_hash,
                    "changed": True,
                },
            },
            "memory": {
                "gpu_name": gpu_name,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            },
        }
    finally:
        _REMOTE_OUTPUT_VOLUME.commit()
