# SFT pilot results

Updated: 2026-10-15

The bounded QLoRA run completed, but the tuned adapter did not improve exact
claim matching on the eight fictional development prompts. Base and tuned
outputs each had zero true-positive claims. This is evidence that the training
and evaluation path ran; it is not evidence of model quality or production
readiness.

## Run and evaluation

The run fine-tuned `Qwen/Qwen3-4B` at revision
`1cfa9a7208912126459214e8b04321603b3df60c` on 128 selected examples from the
512-row, agent-authored fictional corpus (seed 17). QLoRA used 4-bit NF4 with
BF16 compute, rank 16, alpha 32, dropout 0.05, learning rate `2e-4`, and 32
optimizer steps. The worker consumed 140,995 input tokens and supervised
14,167 target tokens; 33,030,144 parameters were trainable.

The comparison used the same eight fixed dev prompts for base and tuned
generation, with deterministic decoding and a 1,024-token limit. The private
32-case test split was unused. Labels are provisional, agent-authored, and have
zero human reviewers. Gate A remains unmet.

| Raw output metric | Base | Tuned |
| --- | ---: | ---: |
| Exact-match true positives / false positives / false negatives | 0 / 9 / 10 | 0 / 10 / 10 |
| Claims with intact citations / predicted claims | 0 / 9 | 0 / 10 |
| Parse failures | 2 | 0 |
| Schema failures | 1 | 1 |
| Outputs reaching the generation limit | 2 / 8 | 0 / 8 |
| Unknown-case abstentions | 1 / 2 | 2 / 2 |

Exact matching is intentionally strict: targets are selective and
non-exhaustive, and extra predictions count as false positives. It does not
judge semantic support. The aggregate exact-match score was uniformly poor,
and raw evidence spans were invalid for every predicted claim.

Several cases illustrate the failure modes. Both base and tuned outputs for
`EVALDEV-F02-V01` treated a report about Rin's software use as a direct user
claim. The tuned output for `EVALDEV-F01-V01` inferred event dates from message
timestamps. Tuned outputs omitted the negative-use case `EVALDEV-F05-V01` and
the intention case `EVALDEV-F04-V01`. In `EVALDEV-F08-V01`, the tuned output
inferred user ownership from a cabinet location even though the source assigns
the ledger to the North Pier team and denies user ownership.

## Quote-anchoring ablation

A separate post-hoc ablation applied deterministic unique-quote anchoring to
the existing model responses. It repaired offsets only; it did not change or
validate claim meaning.

| After anchoring | Base | Tuned |
| --- | ---: | ---: |
| Claims with intact citations / predicted claims | 9 / 9 | 8 / 10 |
| Exact-match true positives | 0 | 0 |

Anchoring improved citation-span integrity for those outputs but did not
produce an exact claim match. It is not a semantic validation result.

## Execution, artifact, and source checks

Optimization took 426.58 seconds, the remote worker ran for 768.13 seconds,
and the client call took 855.38 seconds. The adapter model file is 132,187,888
bytes in the owner's personal Modal Volume and was not downloaded. The Modal
app was stopped with zero active tasks at the post-run inventory check.

The quoted resource rate was `$1.545984` per execution hour. Applying it to
the worker runtime gives an estimate of about `$0.33` before startup, build,
and storage costs; actual provider billing is unavailable, so this is not an
invoice amount. The one-hour timeout was not a billing cap.

The final source check exited successfully: 702 tests passed and one optional
Context Kit test was skipped. Ruff reported 155 files formatted and mypy
reported no issues in 41 source files. These results validate the source
snapshot used for the run, not the provisional annotations.

## Offline replay and audit files

The fictional prepared job and completed remote receipt are included as
auditable examples: [job](../examples/synthetic/sft-pilot-job-v1.json) and
[receipt](../examples/synthetic/sft-pilot-remote-v1.json). The copies contain
fictional pilot data and generated outputs; no checkpoint or private test data
is included. Recreate the raw development report locally with Python on CPU;
this command reads the recorded outputs and makes no model or GPU call. The
output path must be new because the evaluator creates it exclusively:

```sh
PYTHONPATH=src:. .venv/bin/python -m training.evaluate_sft \
  --job examples/synthetic/sft-pilot-job-v1.json \
  --receipt examples/synthetic/sft-pilot-remote-v1.json \
  --output .local/sft-dev-report-replay.json
```

The aggregate metrics on this page are also available in
[machine-readable JSON](sft-pilot-results.json). Original local receipts and
the raw dev report remain in the author's `.local/sft-pilot-20261015/` folder.

## What follows

The next useful work is failure analysis: improve the training examples or
representation, compare the base checkpoint behavior, and obtain human review
of labels and evidence. This report does not schedule or imply another
training launch.
