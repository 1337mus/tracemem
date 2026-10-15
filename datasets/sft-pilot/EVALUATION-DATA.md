# SFT pilot evaluation data

This fixture is a small, fictional evaluation set for the TraceMem claim contract. It is separate from training examples and contains no imported conversations or real people.

## Files and splits

- `eval-seeds.json` contains the eight authored development scenario families.
- `eval.json` is generated from those seeds and contains 32 development cases.
- The eight held-out test families, their seeds, and the 32 generated test cases live only under `.local/sft-pilot-20261015/evaluation-private/` on the authoring machine. They are not included in tracked files.
- `training/build_pilot_eval.py` builds the development set by default. It reads test seeds and writes test output only when given an explicit `--test-seeds` path.
- `tests/test_pilot_eval_data.py` reads tracked development seeds only and does not require private files; its test-split check adapts those seeds in memory.
- The private audit at `.local/sft-pilot-20261015/evaluation-private/test_pilot_eval_private_data.py` is local-only and reads held-out seeds.

Each partition has eight scenario families with four controlled variants per family. Variants within a family share a scenario group and entity IDs, so they are correlated examples. Keep every family in one partition and calculate evaluation summaries by family where possible. Do not treat the four variants as four independent ground-truth cases.

The development and test families use separate names, settings, text templates, family IDs, and entity IDs. IDs begin with `EVALDEV-` or `EVALTEST-`.

## Annotation scope

Every case has six conversation messages and an explicit list of expected claims. Positive cases include one or more exact evidence targets. Eight cases per partition have no eligible target claim. Those empty cases test abstention when the conversation only contains unresolved references, quoted instructions, assistant decoys, questions, or non-personal context.

Cases tagged `selective_claims_non_exhaustive` document that annotations select targets and do not enumerate every incidental claim. The current evaluator does not interpret this tag: it exact-matches predictions against declared targets and counts undeclared outputs as unmatched extras. A positive case still requires its annotated target claim. Empty cases have no eligible target claim within the authored scope.

Treat scores as diagnostics of agreement with this curated target set, not as estimates of semantic precision, semantic recall, or complete memory quality. Entity IDs in `case_metadata` only group related variants; targets use `user` or names visible in source messages because the prompt omits metadata.

The authored families cover decision certainty, reported attribution, historical and current changes, tentative and completed states, negation, uncertain coreference, quoted instructions and assistant decoys, explicit causation, and location versus ownership. Evidence spans are generated from exact quotes in the authored message templates. The builder checks every case with the existing challenge validator.

## Provenance and limits

The files declare `license: MIT`, `annotation_status: agent_authored`, and `provenance: newly_authored_fiction`. No human has labeled or reviewed these examples. The challenge validator checks structure, labels, and evidence-span integrity. It does not establish that the claim interpretation is correct. These files support a curation pilot only; they do not support held-out quality claims.

The private test packet should remain local during training. The root task owner performs an unblinded semantic audit, so this separation is procedural and does not establish independent ground truth.

## Rebuild and check

From the repository root, build development data with:

```sh
PYTHONPATH=src python3 training/build_pilot_eval.py
```

To build the local test file during curation, pass the private seed file explicitly:

```sh
PYTHONPATH=src python3 training/build_pilot_eval.py \
  --test-seeds .local/sft-pilot-20261015/evaluation-private/test-seeds.json \
  --test-output .local/sft-pilot-20261015/evaluation-private/test.json
```

Run only the targeted tracked-data check with:

```sh
python3 tests/test_pilot_eval_data.py -v
python3 .local/sft-pilot-20261015/evaluation-private/test_pilot_eval_private_data.py -v
```
