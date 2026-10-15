"""Lazy remote QLoRA training implementation; imported only by the remote entry point."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any


def run_training(request: Any, support: Any) -> dict[str, Any]:
    """Preflight, train, evaluate, and save only the adapter and tokenizer assets."""
    started = time.perf_counter()
    versions = support._check_dependency_versions(support._TRAIN_DEPENDENCIES)
    import torch  # type: ignore[import-not-found]
    from peft import (  # type: ignore[import-not-found]
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from transformers import (  # type: ignore[import-not-found]
        AutoModelForCausalLM,
        BitsAndBytesConfig,
    )

    rows = request["train_rows"]
    dev_prompts = request["dev_prompts"]
    tokenizer = support._load_pinned_tokenizer()
    tokenizer_provenance = support._tokenizer_provenance(tokenizer, versions)
    preflight = support.preflight_train_rows(
        rows, tokenizer, dev_prompts=dev_prompts, selected_in_order=True
    )
    response: dict[str, Any] = {
        "mode": "train",
        "model_id": support.MODEL_ID,
        "revision": support.MODEL_REVISION,
        "dependency_versions": versions,
        "tokenizer_provenance": tokenizer_provenance,
        "preflight": preflight,
    }
    if preflight["status"] != "ready":
        return {**response, "status": preflight["status"]}

    train_examples = [support.encode_sft_example(row, tokenizer) for row in rows]
    dev_records = [
        {
            "case_id": case_id,
            "prompt": prompt,
            "input_ids": support._encode_user_prompt(prompt, tokenizer),
        }
        for case_id, prompt in support._validated_dev_prompts(dev_prompts)
    ]
    run_dir = _reserve_run_directory(Path(request["remote_run_dir"]))
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("remote QLoRA requires a CUDA device with BF16 support")
    torch.manual_seed(support.SEED)
    torch.cuda.manual_seed_all(support.SEED)
    torch.cuda.reset_peak_memory_stats()

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        support.MODEL_ID,
        revision=support.MODEL_REVISION,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization,
        device_map="auto",
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    input_device = model.get_input_embeddings().weight.device
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    if not getattr(model, "is_gradient_checkpointing", False):
        raise RuntimeError("gradient checkpointing was not enabled")
    model.config.use_cache = False
    model.eval()
    base_started = time.perf_counter()
    base_outputs = _generate_outputs(
        model, tokenizer, dev_records, torch, input_device, support._DEV_MAX_NEW_TOKENS
    )
    torch.cuda.synchronize()
    base_seconds = time.perf_counter() - base_started
    base_parameter_count = int(model.num_parameters())
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=list(support._LORA_TARGETS),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    trainable = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise RuntimeError("only LoRA adapter parameters may be trainable")
    trainable_parameter_count = sum(parameter.numel() for _, parameter in trainable)
    initial_adapter_hash = _adapter_state_hash(model, torch)
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable], lr=support.LEARNING_RATE
    )
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    accumulated_loss = 0.0
    micro_examples = 0
    consumed_examples = train_examples[
        : support.MAX_TRAIN_STEPS * support.GRADIENT_ACCUMULATION_STEPS
    ]
    training_started = time.perf_counter()
    model.train()
    for example in consumed_examples:
        input_ids = torch.tensor([example["input_ids"]], dtype=torch.long, device=input_device)
        attention_mask = torch.tensor(
            [example["attention_mask"]], dtype=torch.long, device=input_device
        )
        labels = torch.tensor([example["labels"]], dtype=torch.long, device=input_device)
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        loss_value = float(loss.detach().float().cpu())
        if not math.isfinite(loss_value):
            raise RuntimeError(f"non-finite training loss at microbatch {micro_examples + 1}")
        (loss / support.GRADIENT_ACCUMULATION_STEPS).backward()
        accumulated_loss += loss_value
        micro_examples += 1
        if micro_examples % support.GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(accumulated_loss / support.GRADIENT_ACCUMULATION_STEPS)
            accumulated_loss = 0.0
            if len(losses) >= support.MAX_TRAIN_STEPS:
                break
    torch.cuda.synchronize()
    training_seconds = time.perf_counter() - training_started
    if len(losses) != support.MAX_TRAIN_STEPS:
        raise RuntimeError(f"expected {support.MAX_TRAIN_STEPS} optimizer steps; got {len(losses)}")
    final_adapter_hash = _adapter_state_hash(model, torch)
    _require_adapter_mutation(initial_adapter_hash, final_adapter_hash)

    adapter_dir = run_dir / "adapter"
    tokenizer_dir = run_dir / "tokenizer"
    adapter_dir.mkdir()
    tokenizer_dir.mkdir()
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(tokenizer_dir)
    _write_tokenizer_provenance(tokenizer, tokenizer_dir, tokenizer_provenance)
    artifact_hashes = _hash_artifacts(run_dir)

    model.config.use_cache = True
    tuned_started = time.perf_counter()
    tuned_outputs = _generate_outputs(
        model, tokenizer, dev_records, torch, input_device, support._DEV_MAX_NEW_TOKENS
    )
    torch.cuda.synchronize()
    tuned_seconds = time.perf_counter() - tuned_started
    total_seconds = time.perf_counter() - started
    memory = {
        "device_name": torch.cuda.get_device_name(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    return {
        **response,
        "status": "completed",
        "generation": {
            "max_new_tokens": support._DEV_MAX_NEW_TOKENS,
            "do_sample": False,
            "base_seconds": base_seconds,
            "tuned_seconds": tuned_seconds,
            "base_outputs": base_outputs,
            "tuned_outputs": tuned_outputs,
        },
        "training": {
            "optimizer": "AdamW",
            "learning_rate": support.LEARNING_RATE,
            "max_steps": support.MAX_TRAIN_STEPS,
            "optimizer_steps": len(losses),
            "microbatch_size": support.MICROBATCH_SIZE,
            "gradient_accumulation_steps": support.GRADIENT_ACCUMULATION_STEPS,
            "examples_consumed": micro_examples,
            "input_tokens_consumed": sum(
                len(example["input_ids"]) for example in consumed_examples
            ),
            "supervised_tokens_consumed": sum(
                sum(label != -100 for label in example["labels"]) for example in consumed_examples
            ),
            "losses": losses,
            "mean_loss": sum(losses) / len(losses),
            "base_parameters": base_parameter_count,
            "trainable_parameters": trainable_parameter_count,
            "adapter_changed": True,
            "adapter_hash_before_training": initial_adapter_hash,
            "adapter_hash_after_training": final_adapter_hash,
            "seconds": training_seconds,
        },
        "timing": {"total_seconds": total_seconds},
        "memory": memory,
        "artifacts": {"sha256": artifact_hashes},
    }


def _reserve_run_directory(path: Path) -> Path:
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("remote_run_dir must be a normalized absolute path")
    path.mkdir(parents=True, exist_ok=False)
    return path


def _require_adapter_mutation(before: str, after: str) -> None:
    if before == after:
        raise RuntimeError("LoRA adapter parameters did not change during training")


def _adapter_state_hash(model: Any, torch: Any) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if "lora_" not in name:
            continue
        tensor = parameter.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _generate_outputs(
    model: Any,
    tokenizer: Any,
    prompts: list[dict[str, Any]],
    torch: Any,
    device: Any,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    model.eval()
    for prompt in prompts:
        prompt_ids = prompt["input_ids"]
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        token_ids = generated[0, len(prompt_ids) :].detach().cpu().tolist()
        outputs.append(
            {
                "case_id": prompt["case_id"],
                "prompt_token_ids": prompt_ids,
                "completion_token_ids": token_ids,
                "raw_text": tokenizer.decode(token_ids, skip_special_tokens=False),
                "response_text": tokenizer.decode(token_ids, skip_special_tokens=True),
                "generated_tokens": len(token_ids),
                "max_new_tokens": max_new_tokens,
                "reached_generation_limit": len(token_ids) == max_new_tokens
                and (not token_ids or token_ids[-1] != tokenizer.eos_token_id),
            }
        )
    return outputs


def _write_tokenizer_provenance(
    tokenizer: Any, directory: Path, tokenizer_provenance: dict[str, Any]
) -> None:
    provenance = {
        **tokenizer_provenance,
        "tokenizer_class": type(tokenizer).__name__,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "padding_side": tokenizer.padding_side,
    }
    (directory / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _hash_artifacts(run_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(run_dir)
        if path.name in {"pytorch_model.bin", "model.safetensors"} or (
            path.name.startswith("model-") and path.suffix in {".bin", ".safetensors"}
        ):
            raise RuntimeError("refusing to save or hash base model weights")
        if not relative_path.parts or relative_path.parts[0] not in {"adapter", "tokenizer"}:
            raise RuntimeError("only adapter and tokenizer artifacts may be saved")
        hashes[relative_path.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not hashes:
        raise RuntimeError("adapter and tokenizer artifacts were not written")
    return hashes
