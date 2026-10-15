# TraceMem fictional SFT pilot data

This directory contains a small, agent-authored pilot dataset for claim extraction. It contains only newly written fictional dialogue. The expected labels are provisional examples and have no human annotation. A 128-row sample from this dataset was used in the completed technical training pilot; see the [pilot results](../../docs/sft-pilot-results.md).

## Size and independence

[`train-seeds.json`](train-seeds.json) contains **32 distinct scenario families**. The deterministic builder creates 16 value substitutions for each family, producing **512 training rows** in [`train.json`](train.json). These are 512 rows but only 32 authored situations; the variants within a family are correlated and must not be described or sampled as 512 independent scenarios. Each family keeps one `scenario_group` and `template_group` across its variants.

Eight families are authored as abstention cases with empty expected claim lists. Their 16 variants each produce **128 abstention rows**, exactly 25% of the generated set. These cases cover unsupported causal adjacency, assistant-only assertions, quoted prompt-injection text, unresolved references, questions, counterfactual hypotheticals, unrelated weather, and unknown ownership.

## Scenario coverage

The seed families cover direct decisions and preferences; negative facts; tentative consideration; intentions versus completed actions; exact and imprecise dates; third-party reports and tool observations; explicit causality versus adjacency; location versus ownership; multi-session referents; corrections that retain historical evidence; assistant-only and quoted-instruction decoys; unresolved pronouns; questions, hypotheticals, and empty claims. The full wording and targets are in the seed file.

## Labels and evidence

Generated cases use the existing explicit-v2 claim schema and predicate set. Claims preserve speaker role, attribution, modality, polarity, and exact event-date bounds. Recorded message time is not used as an event date; year-only or otherwise imprecise dates have null bounds. Questions, unresolved referents, counterfactual hypotheticals, and unsupported relationships abstain. Every non-empty target cites exact message text with zero-based Unicode code-point offsets and exclusive ends. The builder computes spans from the final rendered text, requires each quote to occur exactly once in its cited message, and validates every case with `tracemem.challenge._validate_case` before writing.

The output uses the required root metadata: `version: 1`, `license: MIT`, `annotation_status: agent_authored`, and `provenance: newly_authored_fiction`. Every row has `split: train`. IDs for cases, scenarios, messages, sessions, groups, and entities use the `TRAIN` namespace.

## Regeneration and checks

From the repository root, regenerate the fixture with:

```sh
python3 training/build_pilot_train.py
```

The builder uses only the Python standard library and the fixed seed in `train-seeds.json`; it makes no model or network calls. The focused contract and deterministic-regeneration checks are:

```sh
PYTHONPATH=src python3 tests/test_pilot_train_data.py -v
```

This dataset has not been independently human-reviewed. The completed pilot is a technical execution and does not validate the semantic labels. Review them before relying on the data for further training or model-quality claims.
