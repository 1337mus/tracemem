"""Source-free remote inference implementation for the verifier v2 blind test."""

from __future__ import annotations

from typing import Any

_EXPECTED_PAYLOAD_COMPACT: dict[str, Any] = {}
_EXPECTED_PAYLOAD_SHA256 = ""
_EXPECTED_RUNNER_SHA256 = ""
_EXPECTED_HELPER_SHA256 = ""
_REMOTE_HELPER_PATH = ""


def _verify_adapter_mount(
    mount_path: str,
    relative_path: str,
    expected_hashes: dict[str, str],
) -> dict[str, str]:
    """Verify an adapter tree, allowing a symlink only at the mount root."""
    import hashlib
    import re
    from pathlib import Path
    from stat import S_ISDIR, S_ISREG

    if type(mount_path) is not str or type(relative_path) is not str:
        raise ValueError("adapter mount path is invalid or unsafe")
    parts = relative_path.split("/")
    if (
        not parts
        or any(not part or part in {".", ".."} for part in parts)
        or relative_path.startswith("/")
        or type(expected_hashes) is not dict
        or set(expected_hashes) != {"adapter_config.json", "adapter_model.safetensors"}
        or any(
            type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            for digest in expected_hashes.values()
        )
    ):
        raise ValueError("adapter path or artifact hashes are invalid")

    mount = Path(mount_path)
    try:
        canonical_mount = mount.resolve(strict=True)
        if not S_ISDIR(canonical_mount.stat().st_mode):
            raise ValueError("adapter mount is not a directory")
        current = canonical_mount
        for index, part in enumerate(parts):
            current = current / part
            if not S_ISDIR(current.lstat().st_mode) or current.resolve(
                strict=True
            ) != canonical_mount.joinpath(*parts[: index + 1]):
                raise ValueError("adapter path contains an unsafe symlink")

        actual_hashes: dict[str, str] = {}
        for filename, expected_hash in expected_hashes.items():
            artifact = current / filename
            if not S_ISREG(artifact.lstat().st_mode):
                raise ValueError("adapter artifact is unsafe or not a regular file")
            actual_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                raise ValueError(f"adapter artifact SHA-256 mismatch: {filename}")
            actual_hashes[filename] = actual_hash
        return actual_hashes
    except (OSError, RuntimeError) as exc:
        raise ValueError("adapter files are unavailable or unsafe") from exc


def _remote_compute(request: dict[str, Any]) -> dict[str, Any]:
    """Validate exact approvals, then compare base, prior, and continued adapters."""
    import hashlib as remote_hashlib
    import importlib.metadata as metadata
    import json as remote_json
    import platform
    import re
    import time
    from pathlib import Path as RemotePath
    from typing import cast

    if type(request) is not dict or set(request) != {
        "payload",
        "payload_sha256",
        "runner_sha256",
    }:
        raise ValueError("remote blind-test request keys are invalid")
    payload = request["payload"]
    approved_payload = request["payload_sha256"]
    approved_runner = request["runner_sha256"]
    for digest in (approved_payload, approved_runner):
        if type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("remote blind-test approval hashes are invalid")
    if approved_payload != _EXPECTED_PAYLOAD_SHA256:
        raise ValueError("remote blind-test payload SHA-256 is not allowlisted")
    try:
        payload_bytes = remote_json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("remote blind-test payload is not canonical JSON") from exc
    if len(payload_bytes) > 2_000_000:
        raise ValueError("remote blind-test payload exceeds the byte limit")
    if remote_hashlib.sha256(payload_bytes).hexdigest() != approved_payload:
        raise ValueError("remote blind-test payload SHA-256 does not match approval")
    if approved_runner != _EXPECTED_RUNNER_SHA256:
        raise ValueError("remote blind-test runner SHA-256 is not allowlisted")
    if type(_EXPECTED_PAYLOAD_COMPACT) is not dict:
        raise ValueError("remote blind-test compact expectations are invalid")

    expected_keys = {
        "schema_version",
        "mode",
        "split",
        "source_sha256",
        "runner_sha256",
        "model",
        "settings",
        "prior_adapter",
        "continued_adapter",
        "candidate_provenance",
        "counts",
        "items",
        "training_performed",
    }
    if type(payload) is not dict or set(payload) != expected_keys:
        raise ValueError("remote blind-test payload keys are invalid")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
        or payload["mode"] != "blind_test_inference"
        or payload["split"] != "test"
        or payload["training_performed"] is not False
    ):
        raise ValueError("remote blind-test mode, split, or schema version is invalid")
    if (
        type(payload["source_sha256"]) is not dict
        or payload["runner_sha256"] != approved_runner
        or any(
            payload[key] != _EXPECTED_PAYLOAD_COMPACT.get(key) for key in expected_keys - {"items"}
        )
        or payload["counts"] != {"test_cases": 24, "test_candidates": 48}
    ):
        raise ValueError("remote blind-test provenance, model, settings, or counts changed")

    source_hashes = payload["source_sha256"]
    helper_key = "experiments/verifier_v2_test_remote.py"
    if (
        set(source_hashes)
        != {
            "examples/synthetic/verifier-v2-test-v1.json",
            "training/verifier_data.py",
            helper_key,
        }
        or source_hashes.get(helper_key) != _EXPECTED_HELPER_SHA256
    ):
        raise ValueError("remote blind-test source hashes are not allowlisted")
    if payload["runner_sha256"] != _EXPECTED_RUNNER_SHA256:
        raise ValueError("remote blind-test runner SHA-256 does not match approval")

    items = payload["items"]
    if type(items) is not list or len(items) != 48:
        raise ValueError("remote blind test requires exactly 48 candidate prompts")
    seen: set[str] = set()
    for item in items:
        if type(item) is not dict or set(item) != {"item_id", "prompt"}:
            raise ValueError("remote blind-test item shape is invalid")
        item_id, prompt = item["item_id"], item["prompt"]
        if (
            type(item_id) is not str
            or re.fullmatch(r"[A-Za-z0-9_-]+::[A-Za-z0-9_-]+", item_id) is None
            or item_id in seen
        ):
            raise ValueError("remote blind-test item IDs are invalid or duplicated")
        if type(prompt) is not str or not prompt.strip() or len(prompt.encode("utf-8")) > 100_000:
            raise ValueError("remote blind-test prompt is empty or exceeds its byte limit")
        seen.add(item_id)

    helper_path = RemotePath(_REMOTE_HELPER_PATH)
    try:
        helper_hash = remote_hashlib.sha256(helper_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError("remote blind-test helper is unavailable") from exc
    if helper_hash != _EXPECTED_HELPER_SHA256:
        raise ValueError("remote blind-test helper SHA-256 does not match approval")

    prior = payload["prior_adapter"]
    continued = payload["continued_adapter"]
    prior_hashes = _verify_adapter_mount(
        prior["volume_mount"],
        prior["relative_path"],
        {
            "adapter_config.json": prior["config_sha256"],
            "adapter_model.safetensors": prior["safetensors_sha256"],
        },
    )
    continued_hashes = _verify_adapter_mount(
        continued["volume_mount"],
        continued["adapter_relative_path"],
        continued["adapter_artifact_sha256"],
    )

    # The exact payload, runner, helper, and both adapters are checked before ML imports.
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
    versions.update({package: metadata.version(package) for package in packages})
    if tuple(versions[package] for package in packages) != expected_versions:
        raise RuntimeError("remote blind-test dependency versions do not match the pinned image")
    if platform.python_version_tuple()[:2] != ("3", "12"):
        raise RuntimeError("remote blind test requires Python 3.12")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("remote blind test requires an A10G CUDA device with BF16 support")

    model_spec = payload["model"]
    settings = payload["settings"]
    started = time.perf_counter()
    torch.manual_seed(settings["seed"])
    torch.cuda.manual_seed_all(settings["seed"])
    device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device)
    gpu_capability = list(torch.cuda.get_device_capability(device))
    torch.cuda.reset_peak_memory_stats(device)

    model_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        model_spec["model_id"],
        revision=model_spec["revision"],
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
        use_gradient_checkpointing=True,
    )
    base_model.config.use_cache = True
    base_model.eval()
    model_load_seconds = time.perf_counter() - model_started
    input_device = base_model.get_input_embeddings().weight.device

    prepared: list[tuple[dict[str, Any], list[int]]] = []
    for item in items:
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": item["prompt"]}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=settings["enable_thinking"],
            return_tensors="pt",
        )
        if not isinstance(encoded, torch.Tensor) or encoded.ndim != 2 or encoded.shape[0] != 1:
            raise ValueError("Qwen chat template returned an invalid token tensor")
        input_ids = encoded[0].detach().cpu().tolist()
        if not input_ids or len(input_ids) > settings["max_input_tokens"]:
            raise ValueError("blind-test prompt exceeds the input limit; truncation is disabled")
        if any(type(token_id) is not int or token_id < 0 for token_id in input_ids):
            raise ValueError("Qwen chat template returned invalid input token IDs")
        prepared.append((item, cast(list[int], input_ids)))

    def generate(active_model: Any) -> list[dict[str, Any]]:
        active_model.eval()
        outputs: list[dict[str, Any]] = []
        for item, input_ids in prepared:
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
            ended_with_eos = bool(output_ids and output_ids[-1] == eos_id)
            hit_max = len(output_ids) >= settings["max_new_tokens"]
            outputs.append(
                {
                    "item_id": item["item_id"],
                    "raw_output": tokenizer.decode(output_ids, skip_special_tokens=True),
                    "finish_reason": "eos"
                    if ended_with_eos
                    else "length"
                    if hit_max
                    else "unknown",
                    "input_tokens": len(input_ids),
                    "output_tokens": len(output_ids),
                    "input_token_ids": input_ids,
                    "output_token_ids": output_ids,
                    "ended_with_eos": ended_with_eos,
                    "hit_max_new_tokens": hit_max,
                }
            )
        torch.cuda.synchronize()
        return outputs

    base_started = time.perf_counter()
    base_outputs = generate(base_model)
    base_seconds = time.perf_counter() - base_started

    prior_started = time.perf_counter()
    prior_model = PeftModel.from_pretrained(
        base_model,
        str(RemotePath(prior["volume_mount"]).resolve(strict=True) / prior["relative_path"]),
        is_trainable=False,
        local_files_only=True,
    )
    if any(parameter.requires_grad for parameter in prior_model.parameters()):
        raise RuntimeError("frozen prior adapter unexpectedly has trainable parameters")
    prior_outputs = generate(prior_model)
    prior_seconds = time.perf_counter() - prior_started

    base_model = prior_model.unload()
    del prior_model
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    continued_started = time.perf_counter()
    continued_root = (
        RemotePath(continued["volume_mount"]).resolve(strict=True)
        / continued["adapter_relative_path"]
    )
    continued_model = PeftModel.from_pretrained(
        base_model,
        str(continued_root),
        is_trainable=False,
        local_files_only=True,
    )
    if any(parameter.requires_grad for parameter in continued_model.parameters()):
        raise RuntimeError("frozen continued adapter unexpectedly has trainable parameters")
    continued_outputs = generate(continued_model)
    continued_seconds = time.perf_counter() - continued_started
    torch.cuda.synchronize()

    return {
        "schema_version": 1,
        "status": "success",
        "mode": "blind_test_inference",
        "split": "test",
        "payload_sha256": approved_payload,
        "runner_sha256": approved_runner,
        "source_sha256": source_hashes,
        "model": model_spec,
        "settings": settings,
        "prior_adapter": prior,
        "prior_adapter_artifact_sha256": prior_hashes,
        "continued_adapter": continued,
        "continued_adapter_artifact_sha256": continued_hashes,
        "candidate_provenance": payload["candidate_provenance"],
        "training_performed": False,
        "versions": {**versions, "python": platform.python_version()},
        "tokenizer": tokenizer_info,
        "modal_function_call_id": modal_sdk.current_function_call_id(),
        "base_outputs": base_outputs,
        "prior_outputs": prior_outputs,
        "continued_outputs": continued_outputs,
        "timings": {
            "model_load_seconds": round(model_load_seconds, 6),
            "base_seconds": round(base_seconds, 6),
            "prior_seconds": round(prior_seconds, 6),
            "continued_seconds": round(continued_seconds, 6),
            "total_seconds": round(time.perf_counter() - started, 6),
        },
        "hardware": {
            "gpu_name": gpu_name,
            "gpu_capability": gpu_capability,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
    }
