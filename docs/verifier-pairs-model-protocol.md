# Paired-verifier model experiment

Frozen before model outputs on 2026-10-16. This tests a verifier, using fixed
claims from the [fictional paired dataset](verifier-pairs-data-card.md).
It does not measure extraction recall or downstream question answering.

The model is Qwen3-4B at revision
`1cfa9a7208912126459214e8b04321603b3df60c`. Both arms use NF4 double
quantization, BF16 compute, identical QLoRA preparation, greedy generation,
no thinking, a 2,048-token input limit and a 128-token output limit. Inputs
are never truncated. The seed is 17. Labels stay local during inference.

First run the untuned model on 12 development candidates. If it makes errors,
train once on the 48 training candidates: four shuffled epochs, microbatch
size one, accumulation four, 48 AdamW steps at learning rate 0.0001. LoRA
uses rank 16, alpha 32, dropout 0.05 and the seven attention/MLP projections.
Only completion tokens and EOS contribute to training loss. Each call uses
one personal Modal A10G, no retries and a 3,600-second timeout. Model weights
stay off the Mac. The training call measures its own untuned control before
adding the adapter, then repeats the same development prompts after training.

The primary development criterion requires all six unsupported claims caught,
at least five of six supported claims retained, all 12 responses valid and
untruncated, more unsupported claims caught than the matched untuned arm,
and no fewer supported claims retained. Only an explicit `unsupported`
verdict earns catch credit. Invalid output earns no credit. Uncertainty is
reported separately.

Scope is scored independently on eight decisive references; the four
uncertain references remain separate. A secondary scope diagnostic requires
more decisive scope answers correct without reducing either support count,
with every response valid. It does not replace the primary criterion.

Release the test split only after the primary development criterion passes
and the adapter is frozen. Otherwise leave test inference unrun. The test is
curator-visible, not blind. Zero human reviewers have validated these labels;
neither criterion selects a product model or establishes generalization.

The corpus SHA-256 is
`9442f2703ccf98c602579009f1759a41377fd860f5648626c2b26da2f8ea0358`.
The reviewed runner SHA-256 is
`1a43e8185fcf0f1ed1002c1a9dfde812a48b6919d9a98d0e89cab134cbc9d9ea`.
The successful-baseline payload SHA-256 is
`91bbc871de912b0295c145d68ec1f741e44f3dadb4781b6daac2f17551818695`;
the training payload is
`534139fb7e076e7511e85928c51dfe80b2a625138f3bebe50e36be3b77a24106`.
The first baseline attempt failed before inference on Modal runtime metadata;
the repaired launch changed no prompts, labels, model settings or criteria.

Offline preparation (no model download):

```sh
PYTHONPATH=src:. .venv/bin/python -m experiments.modal_verifier_pairs \
  prepare --mode train --show-payload
```

Launching requires both exact approval hashes and a new receipt destination.
The launcher rebuilds the reviewed inputs, checks the personal workspace and
stopped-job inventory, and acquires the shared operational lock. It uploads
no source tree. Private transcripts and the sealed longer-conversation
candidate are unused. Raw receipts and the original pre-output freezes are
preserved under the ignored local experiment directory.
