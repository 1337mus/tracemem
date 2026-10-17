"""Bounded source-free private inference runner."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from stat import S_IMODE, S_ISDIR
from types import FunctionType
from typing import Any

from experiments import modal_verifier_v2

# These names are injected into source-free FunctionType globals by the builder.
_APPROVED_PAYLOAD_SHA256: str
_APPROVED_RUNNER_SHA256: str
_EXPECTED_CONTINUED_ADAPTER: dict[str, Any]
_EXPECTED_MODEL: dict[str, Any]
_EXPECTED_PAYLOAD_SHA256: str
_EXPECTED_PRIOR_ADAPTER: dict[str, Any]
_EXPECTED_PROVENANCE: dict[str, Any]
_EXPECTED_RUNNER_SHA256: str
_EXPECTED_SETTINGS: dict[str, Any]
_REMOTE_COMPUTE: Callable[[dict[str, Any]], dict[str, Any]]
_verify_adapter_mount: Callable[[str, str, dict[str, str]], dict[str, str]]

_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_DIR = _ROOT / ".local"
_PRIVATE_OUTPUT_DIR = _LOCAL_DIR / "private-modal-verifier"
_MODEL = deepcopy(modal_verifier_v2._MODEL)
_SETTINGS = {
    "do_sample": False,
    "enable_thinking": False,
    "max_new_tokens": 128,
    "max_input_tokens": 2048,
    "attn_implementation": "eager",
    "seed": 17,
}
_PRIOR_ADAPTER = deepcopy(modal_verifier_v2._PRIOR_ADAPTER)
_CONTINUED_VOLUME_NAME = "tracemem-verifier-v2"
_CONTINUED_VOLUME_MOUNT = "/mnt/tracemem-verifier-v2"
_PRIVATE_SFT_VOLUME_NAME = "tracemem-private-sft"
_PRIVATE_SFT_VOLUME_MOUNT = "/mnt/tracemem-private-sft"
_EXPECTED_CANDIDATE_MANIFEST: dict[str, Any] = {
    "training_payload_sha256": "384f9a02c6db633c90421d436db877898c02f0d9389f902fd6f881efff754861",
    "training_receipt_sha256": "c5847c80375506c5cd143608141eec39d030052c3fa6ed39a6bf2c5244048832",
    "adapter_relative_path": (
        "runs/384f9a02c6db633c90421d436db877898c02f0d9389f902fd6f881efff754861/adapter"
    ),
    "adapter_artifact_sha256": {
        "adapter_config.json": "e681a3435ab4a39f801ac4336d3dbb5339d2cdc3ed3a2a3b84d2ad5fa8aa1c77",
        "adapter_model.safetensors": (
            "24be4cef389ffb1a249a010a90afd42876b8aabc6611ce47a7ffb5b8b6d3e0ce"
        ),
    },
}
_PROVENANCE = {
    "data_origin": "user_provided_private_conversations",
    "permitted_use": "explicit_user_authorized_private_evaluation_only",
    "license": "private_use_only_no_redistribution_license_asserted",
    "source_content_in_payload": "minimized_prompts_only",
}
_MAX_ITEMS = 32
_MAX_PROMPT_CHARS = 12_000
_MAX_PAYLOAD_BYTES = 180_000


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def payload_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def build_payload(
    items: list[dict[str, str]],
    candidate_manifest: dict[str, Any],
    *,
    private_sft: bool = False,
) -> dict[str, Any]:
    """Build the approved private inference payload."""
    if type(private_sft) is not bool:
        raise ValueError("private_sft must be an explicit boolean")
    if private_sft:
        manifest_keys = {
            "training_payload_sha256",
            "training_receipt_sha256",
            "adapter_relative_path",
            "adapter_artifact_sha256",
        }
        artifact_keys = {"adapter_config.json", "adapter_model.safetensors"}
        if type(candidate_manifest) is not dict or set(candidate_manifest) != manifest_keys:
            raise ValueError("candidate manifest fields are not allowlisted")
        payload_hash = candidate_manifest["training_payload_sha256"]
        receipt_hash = candidate_manifest["training_receipt_sha256"]
        artifacts = candidate_manifest["adapter_artifact_sha256"]
        if (
            type(payload_hash) is not str
            or re.fullmatch(r"[0-9a-f]{64}", payload_hash) is None
            or type(receipt_hash) is not str
            or re.fullmatch(r"[0-9a-f]{64}", receipt_hash) is None
            or candidate_manifest["adapter_relative_path"] != f"runs/{payload_hash}/adapter"
            or type(artifacts) is not dict
            or set(artifacts) != artifact_keys
            or any(
                type(artifacts[name]) is not str
                or re.fullmatch(r"[0-9a-f]{64}", artifacts[name]) is None
                for name in artifact_keys
            )
        ):
            raise ValueError("candidate manifest hashes or adapter path are invalid")
        manifest = deepcopy(candidate_manifest)
    else:
        if (
            type(candidate_manifest) is not dict
            or candidate_manifest != _EXPECTED_CANDIDATE_MANIFEST
        ):
            raise ValueError("candidate manifest is not the reviewed private adapter")
        manifest = deepcopy(_EXPECTED_CANDIDATE_MANIFEST)
    if type(items) is not list or not 1 <= len(items) <= _MAX_ITEMS:
        raise ValueError("private verifier requires 1 to 32 prompts")
    seen: set[str] = set()
    normalized_items: list[dict[str, str]] = []
    for item in items:
        if type(item) is not dict or set(item) != {"item_id", "prompt"}:
            raise ValueError("private verifier items must contain only item_id and prompt")
        item_id, prompt = item["item_id"], item["prompt"]
        if (
            type(item_id) is not str
            or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", item_id) is None
            or item_id in seen
        ):
            raise ValueError("private verifier item IDs must be unique opaque IDs")
        if (
            type(prompt) is not str
            or not prompt.strip()
            or len(prompt) > _MAX_PROMPT_CHARS
            or "\x00" in prompt
        ):
            raise ValueError("private verifier prompt is empty or exceeds its character limit")
        seen.add(item_id)
        normalized_items.append({"item_id": item_id, "prompt": prompt})

    if private_sft:
        frozen_manifest = _EXPECTED_CANDIDATE_MANIFEST
        prior_adapter = {
            "volume_name": _CONTINUED_VOLUME_NAME,
            "volume_mount": _CONTINUED_VOLUME_MOUNT,
            "relative_path": frozen_manifest["adapter_relative_path"],
            "safetensors_sha256": frozen_manifest["adapter_artifact_sha256"][
                "adapter_model.safetensors"
            ],
            "config_sha256": frozen_manifest["adapter_artifact_sha256"]["adapter_config.json"],
        }
        continued_volume_name = _PRIVATE_SFT_VOLUME_NAME
        continued_volume_mount = _PRIVATE_SFT_VOLUME_MOUNT
    else:
        prior_adapter = deepcopy(_PRIOR_ADAPTER)
        continued_volume_name = _CONTINUED_VOLUME_NAME
        continued_volume_mount = _CONTINUED_VOLUME_MOUNT
    provenance = deepcopy(_PROVENANCE)
    if private_sft:
        provenance["data_origin"] = "authorized_private_checkpoint_evaluation_prompts"
    continued_adapter = {
        "volume_name": continued_volume_name,
        "volume_mount": continued_volume_mount,
        "relative_path": manifest["adapter_relative_path"],
        "artifact_sha256": deepcopy(manifest["adapter_artifact_sha256"]),
    }
    payload = {
        "schema_version": 1,
        "mode": "private_inference_only",
        "provenance": provenance,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model": deepcopy(_MODEL),
        "settings": deepcopy(_SETTINGS),
        "candidate_manifest": manifest,
        "prior_adapter": prior_adapter,
        "continued_adapter": continued_adapter,
        "items": normalized_items,
        "training_performed": False,
    }
    if len(_canonical_bytes(payload)) > _MAX_PAYLOAD_BYTES:
        raise ValueError("private verifier payload exceeds its byte limit")
    return payload


def _validate_approved_payload(
    payload: dict[str, Any], approved_payload_sha256: str, approved_runner_sha256: str
) -> tuple[dict[str, Any], str, str]:
    if type(payload) is not dict:
        raise ValueError("private verifier payload is invalid")
    digest = payload_sha256(payload)
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if (
        type(approved_payload_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", approved_payload_sha256) is None
        or digest != approved_payload_sha256
    ):
        raise ValueError("private verifier payload SHA-256 does not match approval")
    if (
        type(approved_runner_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", approved_runner_sha256) is None
        or approved_runner_sha256 != runner_hash
        or payload.get("runner_sha256") != runner_hash
    ):
        raise ValueError("exact current private verifier runner SHA-256 approval is required")
    try:
        continued = payload.get("continued_adapter")
        private_sft = (
            type(continued) is dict and continued.get("volume_name") == _PRIVATE_SFT_VOLUME_NAME
        )
        expected = build_payload(
            payload["items"], payload["candidate_manifest"], private_sft=private_sft
        )
    except (KeyError, TypeError, ValueError):
        raise ValueError("private verifier payload differs from its fixed envelope") from None
    if _canonical_bytes(payload) != _canonical_bytes(expected):
        raise ValueError("private verifier payload differs from its fixed envelope")
    return expected, digest, runner_hash


def _prepare_private_output_path(output_path: Path) -> Path:
    if not isinstance(output_path, Path):
        raise TypeError("output_path must be a pathlib.Path")
    absolute_path = output_path if output_path.is_absolute() else _ROOT / output_path
    absolute_path = Path(os.path.abspath(absolute_path)).resolve(strict=False)
    private_dir = _PRIVATE_OUTPUT_DIR.resolve(strict=False)
    if absolute_path.parent != private_dir or not absolute_path.name:
        raise ValueError("private verifier output path must be under .local/private-modal-verifier")

    try:
        local_stat = _LOCAL_DIR.lstat()
    except FileNotFoundError:
        _LOCAL_DIR.mkdir(mode=0o700)
        local_stat = _LOCAL_DIR.lstat()
    if not S_ISDIR(local_stat.st_mode):
        raise ValueError("private verifier .local directory is unsafe")
    try:
        _PRIVATE_OUTPUT_DIR.mkdir(mode=0o700, exist_ok=True)
        output_stat = _PRIVATE_OUTPUT_DIR.lstat()
    except OSError:
        raise ValueError("private verifier output directory is unsafe") from None
    if (
        not S_ISDIR(output_stat.st_mode)
        or _PRIVATE_OUTPUT_DIR.resolve(strict=True) != private_dir
        or S_IMODE(output_stat.st_mode) != 0o700
    ):
        raise ValueError("private verifier output directory must be a non-symlink 0700 directory")
    return absolute_path


def modal_function_options() -> dict[str, Any]:
    """Return the pinned source-free single-container inference envelope."""
    from experiments.modal_verifier_transfer import modal_function_options as transfer_options

    options = transfer_options()
    options["timeout"] = 900
    return options


def _remote_compute(request: dict[str, Any]) -> dict[str, Any]:
    """Validate private approvals and compare the frozen base and adapter models."""
    import hashlib as remote_hashlib
    import importlib.metadata as metadata
    import json as remote_json
    import platform
    import re as remote_re
    import time as remote_time
    from pathlib import Path as RemotePath
    from typing import Any as RemoteAny
    from typing import cast

    if type(request) is not dict or set(request) != {
        "payload",
        "payload_sha256",
        "runner_sha256",
    }:
        raise ValueError("private verifier request keys are invalid")
    payload = request["payload"]
    approved_payload, approved_runner = request["payload_sha256"], request["runner_sha256"]
    for digest in (approved_payload, approved_runner):
        if type(digest) is not str or remote_re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("private verifier approval hashes are invalid")
    try:
        payload_bytes = remote_json.dumps(
            payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        raise ValueError("private verifier payload is not canonical JSON") from None
    if len(payload_bytes) > 180_000:
        raise ValueError("private verifier payload exceeds its byte limit")
    digest = remote_hashlib.sha256(payload_bytes).hexdigest()
    if digest != approved_payload or digest != _EXPECTED_PAYLOAD_SHA256:  # noqa: F821
        raise ValueError("private verifier payload SHA-256 does not match exact remote approval")
    if approved_runner != _EXPECTED_RUNNER_SHA256:  # noqa: F821
        raise ValueError("private verifier runner SHA-256 does not match exact remote approval")
    keys = {
        "schema_version",
        "mode",
        "provenance",
        "runner_sha256",
        "model",
        "settings",
        "candidate_manifest",
        "prior_adapter",
        "continued_adapter",
        "items",
        "training_performed",
    }
    if type(payload) is not dict or set(payload) != keys:
        raise ValueError("private verifier payload keys are invalid")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
        or payload["mode"] != "private_inference_only"
        or payload["training_performed"] is not False
        or payload["runner_sha256"] != approved_runner
        or payload["provenance"] != _EXPECTED_PROVENANCE  # noqa: F821
        or payload["model"] != _EXPECTED_MODEL  # noqa: F821
        or payload["settings"] != _EXPECTED_SETTINGS  # noqa: F821
        or payload["candidate_manifest"] != _EXPECTED_CANDIDATE_MANIFEST
        or payload["prior_adapter"] != _EXPECTED_PRIOR_ADAPTER  # noqa: F821
        or payload["continued_adapter"] != _EXPECTED_CONTINUED_ADAPTER  # noqa: F821
    ):
        raise ValueError("private verifier payload metadata is not allowlisted")
    items = payload["items"]
    if type(items) is not list or not 1 <= len(items) <= 32:
        raise ValueError("private verifier prompt count is invalid")
    seen: set[str] = set()
    for item in items:
        if type(item) is not dict or set(item) != {"item_id", "prompt"}:
            raise ValueError("private verifier items must contain only item_id and prompt")
        item_id, prompt = item["item_id"], item["prompt"]
        if (
            type(item_id) is not str
            or remote_re.fullmatch(r"[A-Za-z0-9_-]{1,64}", item_id) is None
            or item_id in seen
            or type(prompt) is not str
            or not prompt.strip()
            or len(prompt) > 12_000
            or "\x00" in prompt
        ):
            raise ValueError("private verifier prompt item is invalid")
        seen.add(item_id)

    prior, continued = payload["prior_adapter"], payload["continued_adapter"]
    prior_hashes = {
        "adapter_config.json": prior["config_sha256"],
        "adapter_model.safetensors": prior["safetensors_sha256"],
    }
    prior_actual = _verify_adapter_mount(  # noqa: F821
        prior["volume_mount"], prior["relative_path"], prior_hashes
    )
    continued_actual = _verify_adapter_mount(  # noqa: F821
        continued["volume_mount"], continued["relative_path"], continued["artifact_sha256"]
    )

    # Runtime and adapter checks precede all model imports.
    import modal as modal_sdk
    import torch  # type: ignore[import-not-found]
    from huggingface_hub.utils import disable_progress_bars  # type: ignore[import-not-found]
    from peft import PeftModel, prepare_model_for_kbit_training  # type: ignore[import-not-found]
    from transformers import (  # type: ignore[import-not-found]
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from transformers.utils import logging as transformers_logging  # type: ignore[import-not-found]

    transformers_logging.set_verbosity_error()
    disable_progress_bars()
    packages = (
        "torch",
        "transformers",
        "tokenizers",
        "accelerate",
        "peft",
        "bitsandbytes",
        "safetensors",
    )
    versions = {"modal": getattr(modal_sdk, "__version__", "injected runtime")}
    versions.update({name: metadata.version(name) for name in packages})
    if tuple(versions[name] for name in packages) != (
        "2.7.1",
        "4.51.3",
        "0.21.1",
        "1.6.0",
        "0.15.2",
        "0.45.5",
        "0.5.3",
    ):
        raise RuntimeError("private verifier dependencies do not match the pinned image")
    if platform.python_version_tuple()[:2] != ("3", "12"):
        raise RuntimeError("private verifier requires Python 3.12")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("private verifier requires an A10G CUDA device with BF16 support")

    started = remote_time.perf_counter()
    model_spec, settings = payload["model"], payload["settings"]
    torch.manual_seed(settings["seed"])
    torch.cuda.manual_seed_all(settings["seed"])
    device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device)
    torch.cuda.reset_peak_memory_stats(device)
    model_started = remote_time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        model_spec["model_id"], revision=model_spec["revision"], trust_remote_code=False
    )
    template, eos_id = tokenizer.chat_template, tokenizer.eos_token_id
    if type(template) is not str or not template or type(eos_id) is not int or eos_id < 0:
        raise ValueError("pinned Qwen tokenizer metadata is invalid")
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
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
    base_model.config.use_cache = True
    base_model.eval()
    model_load_seconds = remote_time.perf_counter() - model_started
    input_device = base_model.get_input_embeddings().weight.device
    prepared: list[tuple[dict[str, RemoteAny], list[int]]] = []
    for item in items:
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": item["prompt"]}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        )
        if not isinstance(encoded, torch.Tensor) or encoded.ndim != 2 or encoded.shape[0] != 1:
            raise ValueError("pinned Qwen tokenizer returned an invalid tensor")
        ids = encoded[0].detach().cpu().tolist()
        if not ids or len(ids) > settings["max_input_tokens"]:
            raise ValueError("private verifier prompt exceeds the token limit")
        prepared.append((item, cast(list[int], ids)))

    def generate(active_model: RemoteAny) -> list[dict[str, RemoteAny]]:
        active_model.eval()
        outputs: list[dict[str, RemoteAny]] = []
        for item, input_ids in prepared:
            tensor = torch.tensor([input_ids], dtype=torch.long, device=input_device)
            with torch.inference_mode():
                generated = active_model.generate(
                    input_ids=tensor,
                    attention_mask=torch.ones_like(tensor),
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=settings["max_new_tokens"],
                    eos_token_id=eos_id,
                    pad_token_id=pad_id,
                )
            output_ids = generated[0, len(input_ids) :].detach().cpu().tolist()
            ended = bool(output_ids and output_ids[-1] == eos_id)
            hit_max = len(output_ids) >= settings["max_new_tokens"]
            outputs.append(
                {
                    "item_id": item["item_id"],
                    "raw_output": tokenizer.decode(output_ids, skip_special_tokens=True),
                    "input_tokens": len(input_ids),
                    "output_tokens": len(output_ids),
                    "input_token_ids": input_ids,
                    "output_token_ids": output_ids,
                    "finish_reason": "eos" if ended else "length" if hit_max else "unknown",
                    "ended_with_eos": ended,
                    "hit_max_new_tokens": hit_max,
                }
            )
        torch.cuda.synchronize()
        return outputs

    base_started = remote_time.perf_counter()
    base_outputs = generate(base_model)
    base_seconds = remote_time.perf_counter() - base_started
    prior_started = remote_time.perf_counter()
    prior_model = PeftModel.from_pretrained(
        base_model,
        str(RemotePath(prior["volume_mount"]).resolve(strict=True) / prior["relative_path"]),
        is_trainable=False,
        local_files_only=True,
    )
    if any(parameter.requires_grad for parameter in prior_model.parameters()):
        raise RuntimeError("private prior adapter unexpectedly has trainable parameters")
    prior_outputs = generate(prior_model)
    prior_seconds = remote_time.perf_counter() - prior_started
    base_model = prior_model.unload()
    del prior_model
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    continued_started = remote_time.perf_counter()
    continued_root = (
        RemotePath(continued["volume_mount"]).resolve(strict=True) / continued["relative_path"]
    )
    continued_model = PeftModel.from_pretrained(
        base_model, str(continued_root), is_trainable=False, local_files_only=True
    )
    if any(parameter.requires_grad for parameter in continued_model.parameters()):
        raise RuntimeError("private continued adapter unexpectedly has trainable parameters")
    continued_outputs = generate(continued_model)
    continued_seconds = remote_time.perf_counter() - continued_started
    torch.cuda.synchronize()
    return {
        "schema_version": 1,
        "status": "success",
        "mode": "private_inference_only",
        "payload_sha256": approved_payload,
        "runner_sha256": approved_runner,
        "provenance": payload["provenance"],
        "candidate_manifest": payload["candidate_manifest"],
        "model": model_spec,
        "settings": settings,
        "prior_adapter": prior,
        "continued_adapter": continued,
        "adapter_artifact_sha256": {"prior": prior_actual, "continued": continued_actual},
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
            "total_seconds": round(remote_time.perf_counter() - started, 6),
        },
        "hardware": {
            "gpu_name": gpu_name,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
    }


def _remote_safe_entry(request: dict[str, Any]) -> dict[str, Any]:
    """Keep Modal failures from logging prompt or model-output text."""
    try:
        return _REMOTE_COMPUTE(request)  # noqa: F821
    except Exception as error:
        return {
            "schema_version": 1,
            "status": "remote_failed",
            "mode": "private_inference_only",
            "payload_sha256": _APPROVED_PAYLOAD_SHA256,  # noqa: F821
            "runner_sha256": _APPROVED_RUNNER_SHA256,  # noqa: F821
            "training_performed": False,
            "error": {"type": type(error).__name__, "message": "private verifier failed"},
        }


def _make_remote_function(payload: dict[str, Any]) -> FunctionType:
    from experiments.verifier_v2_test_remote import _verify_adapter_mount

    approved, digest, runner_hash = _validate_approved_payload(
        payload, payload_sha256(payload), payload["runner_sha256"]
    )
    verify_adapter = FunctionType(
        _verify_adapter_mount.__code__,
        {"__name__": "__main__"},
        name="verify_private_adapter_mount",
    )
    continued_adapter = deepcopy(approved["continued_adapter"])
    compute = FunctionType(
        _remote_compute.__code__,
        {
            "__name__": "__main__",
            "_EXPECTED_PAYLOAD_SHA256": digest,
            "_EXPECTED_RUNNER_SHA256": runner_hash,
            "_EXPECTED_PROVENANCE": deepcopy(approved["provenance"]),
            "_EXPECTED_MODEL": deepcopy(_MODEL),
            "_EXPECTED_SETTINGS": deepcopy(_SETTINGS),
            "_EXPECTED_CANDIDATE_MANIFEST": deepcopy(approved["candidate_manifest"]),
            "_EXPECTED_PRIOR_ADAPTER": deepcopy(approved["prior_adapter"]),
            "_EXPECTED_CONTINUED_ADAPTER": continued_adapter,
            "_verify_adapter_mount": verify_adapter,
        },
        name="private_verifier_compute",
    )
    function = FunctionType(
        _remote_safe_entry.__code__,
        {
            "__name__": "__main__",
            "_REMOTE_COMPUTE": compute,
            "_APPROVED_PAYLOAD_SHA256": digest,
            "_APPROVED_RUNNER_SHA256": runner_hash,
        },
        name="remote_private_verifier",
    )
    function.__doc__ = _remote_safe_entry.__doc__
    return function


def _build_modal_function(payload: dict[str, Any]) -> tuple[Any, Any]:
    """Build a direct-request, source-free app with read-only adapter mounts."""
    approved, _, _ = _validate_approved_payload(
        payload, payload_sha256(payload), payload["runner_sha256"]
    )
    import modal

    from experiments.modal_verifier_pairs import _IMAGE_PACKAGES

    image = modal.Image.debian_slim(python_version="3.12").pip_install(*_IMAGE_PACKAGES)
    app = modal.App("tracemem-private-verifier")
    options = modal_function_options()
    volumes: dict[str, Any] = {}
    for adapter in (approved["prior_adapter"], approved["continued_adapter"]):
        volume = modal.Volume.from_name(
            adapter["volume_name"], create_if_missing=False
        ).with_mount_options(read_only=True)
        volumes[adapter["volume_mount"]] = volume
    options["volumes"] = volumes
    function = app.function(image=image, **options)(_make_remote_function(approved))
    return app, function


def run_once(
    payload: dict[str, Any],
    *,
    approved_payload_sha256: str,
    approved_runner_sha256: str,
    output_path: Path,
) -> dict[str, Any]:
    """Run one explicitly approved private inference request."""
    expected, digest, runner_hash = _validate_approved_payload(
        payload, approved_payload_sha256, approved_runner_sha256
    )
    safe_output_path = _prepare_private_output_path(output_path)
    from experiments.sft_launch import (
        _acquire_launch_lock,
        _require_supported_python,
        _reserve_output,
        _write_receipt,
        verify_modal_inventory,
        verify_personal_workspace,
    )

    _require_supported_python()
    output_fd = _reserve_output(safe_output_path)
    lock_fd: int | None = None
    started = time.perf_counter()
    remote_started = False
    workspace_verified = False
    inventory_count: int | None = None
    try:
        lock_fd = _acquire_launch_lock()
        verify_personal_workspace()
        workspace_verified = True
        inventory = verify_modal_inventory()
        inventory_count = len(inventory)
        app, function = _build_modal_function(expected)
        with app.run(detach=False):
            remote_started = True
            result = function.remote(
                {"payload": expected, "payload_sha256": digest, "runner_sha256": runner_hash}
            )
        if type(result) is not dict or result.get("status") not in {"success", "remote_failed"}:
            raise TypeError("remote private verifier returned an invalid result")
        if (
            result.get("mode") != "private_inference_only"
            or result.get("payload_sha256") != digest
            or result.get("runner_sha256") != runner_hash
            or result.get("training_performed") is not False
        ):
            raise TypeError("remote private verifier result does not match approval")
        receipt = {
            **result,
            "modal_app_id": getattr(app, "app_id", None),
            "modal_app_name": "tracemem-private-verifier",
            "client_elapsed_seconds": round(time.perf_counter() - started, 6),
            "launch_checks": {
                "personal_workspace_verified": workspace_verified,
                "modal_inventory_checked": True,
                "modal_app_count": inventory_count,
            },
        }
        _write_receipt(output_fd, receipt)
        return receipt
    except Exception as error:
        failure = {
            "schema_version": 1,
            "status": "launch_failed",
            "mode": "private_inference_only",
            "payload_sha256": digest,
            "runner_sha256": runner_hash,
            "remote_started": remote_started,
            "client_elapsed_seconds": round(time.perf_counter() - started, 6),
            "error": {"type": type(error).__name__, "message": "private verifier failed"},
        }
        _write_receipt(output_fd, failure)
        return failure
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        os.close(output_fd)
