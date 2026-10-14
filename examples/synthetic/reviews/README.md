# Fictional scenario review artifacts

These files contain newly authored fictional conversations and agent judgments
only. No private conversation excerpts, independent human labels, or accepted
memories are present. `manifest.json` records immutable byte hashes. Originals
remain in the ignored local working directories.

The scenario bundle was prepared from the two frozen inference receipts with
seed1729. It includes84 schema-valid candidates. Parse/schema failures remain
explicit exclusions in provenance. An agent reviewed the complete blind bundle;
root then read all84 complete conversations, candidates and rationales and made
nine separate adjudication changes. Root was aware of model identities and the
first review, so its responses are not a second independent blind review.

Keep `*-answer-key.json` and `scenario-review-provenance.json` away from future
blind reviewers. The separate controls contain8 authored positive examples and
8 simple dev-derived foils. Their16/16 agreement does not establish calibration.

Reproduce the generated-candidate accounting from the workspace root:

```sh
PYTHONPATH=src .venv/bin/python -m experiments.summarize_scenario_reviews \
  --receipts examples/synthetic/qwen3-4b-scenarios-remote.json \
    examples/synthetic/phi-3.5-mini-scenarios-remote.json \
  --responses examples/synthetic/reviews/scenario-review-agent-responses.json \
    examples/synthetic/reviews/scenario-review-root-responses.json \
  --seed 1729 --output /tmp/tracemem-reviewed-support-new.json
```

The output path must not already exist. Expected accounting is preserved in
`docs/scenarios-reviewed-support.json`; it reports each reviewer/model/split
separately and keeps uncertain judgments and failure exclusions visible.
