"""Bounded paired QLoRA verifier experiment runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import FunctionType
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_DATASET_PATH = _ROOT / "examples/synthetic/verifier-pairs-v1.json"
_RENDERER_PATH = _ROOT / "training/verifier_data.py"
_DATASET_SHA256 = "9442f2703ccf98c602579009f1759a41377fd860f5648626c2b26da2f8ea0358"
_RENDERER_SHA256 = "ada22823e053042d6d20e533f35410d852855607c8f6d51fd3cab93d8a4234c6"
_MODEL = {
    "model_id": "Qwen/Qwen3-4B",
    "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
    "quantization": {
        "load_in_4bit": True,
        "type": "nf4",
        "double_quant": True,
        "compute_dtype": "bfloat16",
    },
}
_SETTINGS = {
    "do_sample": False,
    "enable_thinking": False,
    "max_new_tokens": 128,
    "max_input_tokens": 2048,
    "attn_implementation": "eager",
    "seed": 17,
    "optimizer": "AdamW",
    "learning_rate": 0.0001,
    "gradient_accumulation_steps": 4,
    "epochs": 4,
    "microsteps": 192,
    "optimizer_steps": 48,
    "gradient_checkpointing": True,
    "completion_only_loss": True,
    "completion_eos_loss": True,
    "lora": {
        "rank": 16,
        "alpha": 32,
        "dropout": 0.05,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ],
    },
}
_VOLUME_NAME = "tracemem-verifier-pairs-v1"
_VOLUME_MOUNT = "/mnt/tracemem-verifier-pairs-v1"
_IMAGE_PACKAGES = (
    "torch==2.7.1",
    "transformers==4.51.3",
    "tokenizers==0.21.1",
    "jinja2==3.1.6",
    "huggingface-hub==0.30.2",
    "safetensors==0.5.3",
    "accelerate==1.6.0",
    "peft==0.15.2",
    "bitsandbytes==0.45.5",
)
_REMOTE_VOLUME: Any = None


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def payload_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def build_payload(mode: str, *, split: str = "dev") -> dict[str, Any]:
    """Build the exact offline payload approved for one verifier run."""
    if type(mode) is not str or mode not in {"baseline", "train"}:
        raise ValueError("mode must be baseline or train")
    if type(split) is not str or split != "dev":
        raise ValueError("only the dev split is available in this runner")
    if hashlib.sha256(_DATASET_PATH.read_bytes()).hexdigest() != _DATASET_SHA256:
        raise ValueError("verifier dataset SHA-256 is not allowlisted")
    renderer_hash = hashlib.sha256(_RENDERER_PATH.read_bytes()).hexdigest()
    if renderer_hash != _RENDERER_SHA256:
        raise ValueError("verifier prompt renderer SHA-256 is not allowlisted")

    from training.verifier_data import (
        render_candidate_prompt,
        render_training_rows,
        validate_verifier_dataset,
    )

    dataset = json.loads(_DATASET_PATH.read_text(encoding="utf-8"))
    cases = validate_verifier_dataset(dataset)
    items = [
        {
            "item_id": f"{case['case_id']}::{candidate['candidate_id']}",
            "prompt": render_candidate_prompt(case, candidate["candidate_id"]),
        }
        for case in cases
        if case["split"] == split
        for candidate in case["candidates"]
    ]
    train_rows = render_training_rows(dataset, "train") if mode == "train" else []
    if len(items) != 12 or len(train_rows) != (48 if mode == "train" else 0):
        raise ValueError("verifier dataset does not have the frozen candidate counts")

    return {
        "schema_version": 1,
        "mode": mode,
        "split": split,
        "source_sha256": {
            "examples/synthetic/verifier-pairs-v1.json": _DATASET_SHA256,
            "training/verifier_data.py": renderer_hash,
        },
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model": _MODEL,
        "settings": _SETTINGS,
        "items": items,
        "train_rows": train_rows,
    }


def modal_function_options() -> dict[str, Any]:
    """Return the fixed one-container A10G envelope for a paired verifier run."""
    return {
        "gpu": "A10G",
        "cpu": (4, 4),
        "memory": (32_768, 32_768),
        "min_containers": 0,
        "max_containers": 1,
        "buffer_containers": 0,
        "retries": 0,
        "startup_timeout": 600,
        "timeout": 3600,
        "single_use_containers": True,
        "serialized": True,
        "include_source": False,
    }


def _remote_compute(request: dict[str, Any]) -> dict[str, Any]:
    """Validate one approved payload, then run the fixed baseline or paired QLoRA train."""
    import hashlib as remote_hashlib
    import importlib.metadata as metadata
    import json as remote_json
    import platform
    import random
    import re as remote_re
    import time as remote_time
    from typing import cast

    if type(request) is not dict or set(request) != {
        "payload",
        "payload_sha256",
        "runner_sha256",
    }:
        raise ValueError("remote verifier request keys are invalid")
    payload = request["payload"]
    approved_payload = request["payload_sha256"]
    approved_runner = request["runner_sha256"]
    for digest in (approved_payload, approved_runner):
        if type(digest) is not str or remote_re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("remote verifier approval hashes are invalid")
    try:
        payload_bytes = remote_json.dumps(
            payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("remote verifier payload is not canonical JSON") from exc
    if len(payload_bytes) > 2_000_000:
        raise ValueError("remote verifier payload exceeds the byte limit")
    if remote_hashlib.sha256(payload_bytes).hexdigest() != approved_payload:
        raise ValueError("remote verifier payload SHA-256 does not match approval")

    expected_keys = {
        "schema_version",
        "mode",
        "split",
        "source_sha256",
        "runner_sha256",
        "model",
        "settings",
        "items",
        "train_rows",
    }
    if type(payload) is not dict or set(payload) != expected_keys:
        raise ValueError("remote verifier payload keys are invalid")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
        or payload["mode"] not in {"baseline", "train"}
        or payload["split"] != "dev"
    ):
        raise ValueError("remote verifier mode, split, or schema version is invalid")
    expected_sources = {
        "examples/synthetic/verifier-pairs-v1.json": (
            "9442f2703ccf98c602579009f1759a41377fd860f5648626c2b26da2f8ea0358"
        ),
        "training/verifier_data.py": (
            "ada22823e053042d6d20e533f35410d852855607c8f6d51fd3cab93d8a4234c6"
        ),
    }
    if payload["source_sha256"] != expected_sources:
        raise ValueError("remote verifier data or prompt renderer hashes are not allowlisted")
    if payload["runner_sha256"] != approved_runner:
        raise ValueError("remote verifier runner SHA-256 does not match approval")
    config = {"model": payload["model"], "settings": payload["settings"]}
    config_bytes = remote_json.dumps(
        config, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    config_hash = "dc87be522b34af996b458378449c68129977797cce1bb085d873a957fd82884b"
    if remote_hashlib.sha256(config_bytes).hexdigest() != config_hash:
        raise ValueError("remote verifier model or settings changed")
    expected_model, expected_settings = payload["model"], payload["settings"]

    items = payload["items"]
    if type(items) is not list or len(items) != 12:
        raise ValueError("remote verifier requires exactly 12 dev candidate items")
    seen: set[str] = set()
    for item in items:
        if type(item) is not dict or set(item) != {"item_id", "prompt"}:
            raise ValueError("remote verifier item shape is invalid")
        item_id, prompt = item["item_id"], item["prompt"]
        if (
            type(item_id) is not str
            or remote_re.fullmatch(r"[A-Za-z0-9_-]+::[A-Za-z0-9_-]+", item_id) is None
            or item_id in seen
        ):
            raise ValueError("remote verifier item IDs are invalid or duplicated")
        if type(prompt) is not str or not prompt.strip() or len(prompt.encode("utf-8")) > 100_000:
            raise ValueError("remote verifier prompt is empty or exceeds its byte limit")
        seen.add(item_id)
    train_rows = payload["train_rows"]
    expected_train_count = 48 if payload["mode"] == "train" else 0
    if type(train_rows) is not list or len(train_rows) != expected_train_count:
        raise ValueError("remote verifier training row count is invalid")
    for row in train_rows:
        if type(row) is not dict or set(row) != {"prompt", "completion"}:
            raise ValueError("remote verifier training row shape is invalid")
        prompt, completion = row["prompt"], row["completion"]
        if (
            type(prompt) is not str
            or not prompt.strip()
            or len(prompt.encode("utf-8")) > 100_000
            or type(completion) is not str
            or len(completion.encode("utf-8")) > 1000
        ):
            raise ValueError("remote verifier training row exceeds its text bounds")
        try:
            target = remote_json.loads(completion)
        except (TypeError, ValueError) as exc:
            raise ValueError("remote verifier training completion is invalid JSON") from exc
        if (
            type(target) is not dict
            or set(target) != {"support", "eligibility"}
            or target["support"] not in {"supported", "unsupported", "uncertain"}
            or target["eligibility"] not in {"eligible", "ineligible", "uncertain"}
        ):
            raise ValueError("remote verifier training completion labels are invalid")

    # Imports of Modal and ML runtimes stay after all approval and shape guards above.
    import modal as modal_sdk
    import torch  # type: ignore[import-not-found]
    from peft import (  # type: ignore[import-not-found]
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from transformers import (  # type: ignore[import-not-found]
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    pinned_packages = (
        "torch",
        "transformers",
        "tokenizers",
        "accelerate",
        "peft",
        "bitsandbytes",
        "safetensors",
    )
    versions = {
        "modal": getattr(modal_sdk, "__version__", "injected runtime"),
        **{package: metadata.version(package) for package in pinned_packages},
    }
    if tuple(versions[package] for package in pinned_packages) != (
        "2.7.1",
        "4.51.3",
        "0.21.1",
        "1.6.0",
        "0.15.2",
        "0.45.5",
        "0.5.3",
    ):
        raise RuntimeError("remote verifier dependency versions do not match the pinned image")
    if platform.python_version_tuple()[:2] != ("3", "12"):
        raise RuntimeError("remote verifier requires Python 3.12")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("remote verifier requires an A10G CUDA device with BF16 support")

    started = remote_time.perf_counter()
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device)
    torch.cuda.reset_peak_memory_stats(device)
    model_started = remote_time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        expected_model["model_id"], revision=expected_model["revision"], trust_remote_code=False
    )
    template = tokenizer.chat_template
    if type(template) is not str or not template:
        raise ValueError("pinned Qwen tokenizer chat template is unavailable")
    tokenizer_provenance = {
        "class": type(tokenizer).__name__,
        "is_fast": bool(tokenizer.is_fast),
        "chat_template_sha256": remote_hashlib.sha256(template.encode("utf-8")).hexdigest(),
        "eos_token_id": tokenizer.eos_token_id,
    }
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        expected_model["model_id"],
        revision=expected_model["revision"],
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        quantization_config=quantization,
        device_map="cuda",
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = True
    model.eval()
    model_load_seconds = remote_time.perf_counter() - model_started
    eos_id = tokenizer.eos_token_id
    if type(eos_id) is not int or eos_id < 0:
        raise ValueError("pinned Qwen tokenizer EOS token ID is invalid")
    pad_id = tokenizer.pad_token_id if type(tokenizer.pad_token_id) is int else eos_id
    input_device = model.get_input_embeddings().weight.device

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
        token_ids = encoded[0].detach().cpu().tolist()
        if not token_ids or len(token_ids) > 2048:
            raise ValueError("verifier prompt exceeds 2048 tokens; truncation is disabled")
        return cast(list[int], token_ids)

    prepared_items = [(item, encode_prompt(item["prompt"])) for item in items]
    if len({len(ids) for _item, ids in prepared_items}) == 0:
        raise ValueError("remote verifier has no tokenized dev items")

    def generate_outputs(
        active_model: Any, prepared: list[tuple[dict[str, Any], list[int]]]
    ) -> list[dict[str, Any]]:
        active_model.eval()
        outputs = []
        for item, input_ids in prepared:
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=input_device)
            attention = torch.ones_like(input_tensor)
            with torch.inference_mode():
                generated = active_model.generate(
                    input_ids=input_tensor,
                    attention_mask=attention,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=128,
                    eos_token_id=eos_id,
                    pad_token_id=pad_id,
                )
            output_ids = generated[0, len(input_ids) :].detach().cpu().tolist()
            if any(type(token_id) is not int for token_id in output_ids):
                raise ValueError("Qwen generation returned invalid output token IDs")
            ended_with_eos = bool(output_ids and output_ids[-1] == eos_id)
            hit_max = len(output_ids) >= 128
            outputs.append(
                {
                    "item_id": item["item_id"],
                    "raw_output": tokenizer.decode(output_ids, skip_special_tokens=True),
                    "finish_reason": (
                        "eos" if ended_with_eos else "length" if hit_max else "unknown"
                    ),
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

    baseline_started = remote_time.perf_counter()
    baseline_outputs = generate_outputs(model, prepared_items)
    baseline_seconds = remote_time.perf_counter() - baseline_started
    response: dict[str, Any] = {
        "schema_version": 1,
        "status": "success",
        "mode": payload["mode"],
        "split": "dev",
        "payload_sha256": approved_payload,
        "runner_sha256": approved_runner,
        "model": expected_model,
        "settings": expected_settings,
        "versions": {**versions, "python": platform.python_version()},
        "tokenizer": tokenizer_provenance,
        "modal_function_call_id": modal_sdk.current_function_call_id(),
        "baseline_outputs": baseline_outputs,
        "timings": {
            "model_load_seconds": round(model_load_seconds, 6),
            "baseline_seconds": round(baseline_seconds, 6),
            "total_seconds": round(remote_time.perf_counter() - started, 6),
        },
        "memory": {
            "gpu_name": gpu_name,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
    }

    if payload["mode"] == "train":
        volume = _REMOTE_VOLUME
        run_path = f"/mnt/tracemem-verifier-pairs-v1/runs/{approved_payload}"
        adapter_path = f"{run_path}/adapter"
        os = __import__("os")
        os.makedirs("/mnt/tracemem-verifier-pairs-v1/runs", exist_ok=True)
        os.mkdir(run_path)
        os.mkdir(adapter_path)
        train_examples = []
        for row in train_rows:
            prompt_ids = encode_prompt(row["prompt"])
            completion_ids = tokenizer.encode(row["completion"], add_special_tokens=False)
            if not completion_ids:
                raise ValueError("verifier completion tokenization returned no tokens")
            sequence = prompt_ids + completion_ids + [eos_id]
            if len(sequence) > 2048:
                raise ValueError("verifier training sequence exceeds 2048 tokens")
            train_examples.append((sequence, [-100] * len(prompt_ids) + completion_ids + [eos_id]))
        if len(train_examples) != 48:
            raise ValueError("verifier training example count changed after tokenization")

        torch.manual_seed(17)
        torch.cuda.manual_seed_all(17)
        model.config.use_cache = False
        if not model.is_gradient_checkpointing:
            model.gradient_checkpointing_enable()
        model = get_peft_model(
            model,
            LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=list(expected_settings["lora"]["target_modules"]),
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable or any(
            "lora_" not in name for name, p in model.named_parameters() if p.requires_grad
        ):
            raise RuntimeError("only LoRA adapter parameters may be trainable")

        def adapter_state_sha256() -> str:
            state_hash = remote_hashlib.sha256()
            for name, parameter in model.named_parameters():
                if "lora_" in name:
                    state_hash.update(name.encode("utf-8"))
                    state_hash.update(
                        parameter.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
                    )
            return state_hash.hexdigest()

        initial_adapter_hash = adapter_state_sha256()
        optimizer = torch.optim.AdamW(trainable, lr=0.0001)
        optimizer.zero_grad(set_to_none=True)
        microstep_losses: list[float] = []
        optimizer_losses: list[float] = []
        training_started = remote_time.perf_counter()
        model.train()
        for epoch in range(4):
            indices = list(range(48))
            random.Random(17 + epoch).shuffle(indices)
            step_losses: list[float] = []
            for position, index in enumerate(indices):
                sequence, labels = train_examples[index]
                ids_tensor = torch.tensor([sequence], dtype=torch.long, device=input_device)
                labels_tensor = torch.tensor([labels], dtype=torch.long, device=input_device)
                attention = torch.ones_like(ids_tensor)
                loss = model(
                    input_ids=ids_tensor,
                    attention_mask=attention,
                    labels=labels_tensor,
                ).loss
                if loss is None or not torch.isfinite(loss):
                    raise RuntimeError("verifier training produced a non-finite loss")
                value = float(loss.detach().cpu().item())
                microstep_losses.append(value)
                step_losses.append(value)
                (loss / 4).backward()
                if (position + 1) % 4 == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_losses.append(sum(step_losses[-4:]) / 4)
        torch.cuda.synchronize()
        training_seconds = remote_time.perf_counter() - training_started
        final_adapter_hash = adapter_state_sha256()
        if initial_adapter_hash == final_adapter_hash:
            raise RuntimeError("verifier adapter did not change during training")

        model.save_pretrained(adapter_path, safe_serialization=True)
        artifact_hashes: dict[str, str] = {}
        for filename in sorted(os.listdir(adapter_path)):
            artifact = f"{adapter_path}/{filename}"
            if filename.endswith(".bin") or filename.endswith(".pt") or filename.endswith(".pth"):
                raise RuntimeError("verifier adapter emitted a non-safetensors weight artifact")
            with open(artifact, "rb") as artifact_file:
                artifact_hashes[filename] = remote_hashlib.sha256(artifact_file.read()).hexdigest()
        if not {"adapter_config.json", "adapter_model.safetensors"} <= set(artifact_hashes) or set(
            artifact_hashes
        ) - {"README.md", "adapter_config.json", "adapter_model.safetensors"}:
            raise RuntimeError("verifier adapter artifact set changed")
        model.config.use_cache = True
        tuned_started = remote_time.perf_counter()
        tuned_outputs = generate_outputs(model, prepared_items)
        tuned_seconds = remote_time.perf_counter() - tuned_started
        volume.commit()
        response.update(
            {
                "tuned_outputs": tuned_outputs,
                "timings": {
                    **response["timings"],
                    "training_seconds": round(training_seconds, 6),
                    "tuned_seconds": round(tuned_seconds, 6),
                    "total_seconds": round(remote_time.perf_counter() - started, 6),
                },
                "training": {
                    "row_count": len(train_examples),
                    "epochs": 4,
                    "microsteps": len(microstep_losses),
                    "optimizer_steps": len(optimizer_losses),
                    "microstep_losses": microstep_losses,
                    "optimizer_step_losses": optimizer_losses,
                    "adapter_path": adapter_path,
                    "trainable_parameter_count": sum(p.numel() for p in trainable),
                    "adapter_artifact_sha256": artifact_hashes,
                    "adapter_mutation": {
                        "initial_state_sha256": initial_adapter_hash,
                        "final_state_sha256": final_adapter_hash,
                        "changed": True,
                    },
                },
                "memory": {
                    "gpu_name": gpu_name,
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                },
            }
        )
    return response


def _make_remote_function(volume: Any) -> FunctionType:
    function = FunctionType(
        _remote_compute.__code__,
        {"__name__": "__main__", "_REMOTE_VOLUME": volume},
        name="remote_verifier_pairs",
    )
    function.__doc__ = _remote_compute.__doc__
    return function


def _build_modal_function(payload: dict[str, Any]) -> tuple[Any, Any]:
    import modal

    image = modal.Image.debian_slim(python_version="3.12").pip_install(*_IMAGE_PACKAGES)
    app = modal.App("tracemem-verifier-pairs-v1")
    volume = (
        modal.Volume.from_name(_VOLUME_NAME, create_if_missing=True)
        if payload["mode"] == "train"
        else None
    )
    options = modal_function_options()
    options["volumes"] = {_VOLUME_MOUNT: volume} if volume is not None else {}
    function = app.function(image=image, **options)(_make_remote_function(volume))
    return app, function


def run_once(
    payload: dict[str, Any],
    *,
    approved_payload_sha256: str,
    approved_runner_sha256: str,
    output_path: Path,
) -> dict[str, Any]:
    """Run one approved payload with fresh workspace, inventory, lock, and receipt guards."""
    if type(payload) is not dict:
        raise ValueError("payload differs from the current fixed payload")
    try:
        expected_payload = build_payload(payload["mode"], split=payload["split"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("payload differs from the current fixed payload") from error
    if _canonical_bytes(payload) != _canonical_bytes(expected_payload):
        raise ValueError("payload differs from the current fixed payload")
    digest = payload_sha256(payload)
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if (
        approved_payload_sha256 != digest
        or approved_runner_sha256 != runner_hash
        or payload.get("runner_sha256") != runner_hash
    ):
        raise ValueError("exact current payload and runner SHA-256 approvals are required")
    if not isinstance(output_path, Path):
        raise TypeError("output_path must be a pathlib.Path")
    from experiments.sft_launch import (
        _acquire_launch_lock,
        _reserve_output,
        _write_receipt,
        verify_modal_inventory,
        verify_personal_workspace,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_fd = _reserve_output(output_path)
    lock_fd: int | None = None
    started = time.perf_counter()
    remote_started = False
    try:
        lock_fd = _acquire_launch_lock()
        verify_personal_workspace()
        inventory = verify_modal_inventory()
        app, function = _build_modal_function(payload)
        with app.run(detach=False):
            remote_started = True
            result = function.remote(
                {
                    "payload": payload,
                    "payload_sha256": digest,
                    "runner_sha256": runner_hash,
                }
            )
        if type(result) is not dict or result.get("status") != "success":
            raise TypeError("Modal returned an invalid verifier result")
        receipt = {
            **result,
            "modal_app_id": getattr(app, "app_id", None),
            "modal_app_name": "tracemem-verifier-pairs-v1",
            "client_elapsed_seconds": round(time.perf_counter() - started, 6),
            "launch_checks": {
                "personal_workspace_verified": True,
                "modal_inventory_checked": True,
                "modal_app_count": len(inventory),
            },
        }
        _write_receipt(output_fd, receipt)
        return receipt
    except Exception as error:
        from experiments.scenario_payload import safe_exception_message

        failure = {
            "schema_version": 1,
            "status": "launch_failed",
            "mode": payload["mode"],
            "payload_sha256": digest,
            "runner_sha256": runner_hash,
            "remote_started": remote_started,
            "client_elapsed_seconds": round(time.perf_counter() - started, 6),
            "error": {"type": type(error).__name__, "message": safe_exception_message(error)},
        }
        _write_receipt(output_fd, failure)
        raise
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        os.close(output_fd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="modal-verifier-pairs")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="build an offline, reviewable payload")
    prepare.add_argument("--mode", choices=("baseline", "train"), required=True)
    prepare.add_argument("--split", choices=("dev",), default="dev")
    prepare.add_argument("--show-payload", action="store_true")
    launch = commands.add_parser("launch", help="run one exactly approved Modal payload")
    launch.add_argument("--mode", choices=("baseline", "train"), required=True)
    launch.add_argument("--split", choices=("dev",), default="dev")
    launch.add_argument("--approve-payload-sha256", required=True)
    launch.add_argument("--approve-runner-sha256", required=True)
    launch.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = build_payload(args.mode, split=args.split)
        digest = payload_sha256(payload)
        runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if payload["runner_sha256"] != runner_hash:
            raise ValueError("current runner SHA-256 changed while preparing the payload")
        if args.command == "prepare":
            result: dict[str, Any] = {
                "mode": args.mode,
                "split": args.split,
                "item_count": len(payload["items"]),
                "train_row_count": len(payload["train_rows"]),
                "source_sha256": payload["source_sha256"],
                "payload_sha256": digest,
                "runner_sha256": runner_hash,
                "remote_started": False,
            }
            if args.show_payload:
                result["payload"] = payload
            print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
            return 0
        if args.approve_payload_sha256 != digest or args.approve_runner_sha256 != runner_hash:
            raise ValueError("exact current payload and runner SHA-256 approvals are required")
        receipt = run_once(
            payload,
            approved_payload_sha256=args.approve_payload_sha256,
            approved_runner_sha256=args.approve_runner_sha256,
            output_path=args.output,
        )
        print(
            json.dumps(
                {
                    "receipt_path": str(args.output),
                    "payload_sha256": digest,
                    "runner_sha256": runner_hash,
                    "remote_started": True,
                    "modal_app_id": receipt.get("modal_app_id"),
                    "modal_function_call_id": receipt.get("modal_function_call_id"),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except Exception as error:
        from experiments.scenario_payload import safe_exception_message

        print(f"{type(error).__name__}: {safe_exception_message(error)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
