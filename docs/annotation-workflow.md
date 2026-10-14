# Local annotation review

Generated-receipt tooling also supports a separate accounting step:

```sh
PYTHONPATH=src .venv/bin/python -m experiments.summarize_scenario_reviews \
  --receipts examples/synthetic/qwen3-4b-scenarios-remote.json \
  --responses "$REVIEW_DIR/agent-responses.json" \
  --seed 1729 --output "$REVIEW_DIR/model-accounting.json"
```

Use the same receipt sequence and shuffle seed used to prepare the review bundle.
Omit `--responses` to report zero reviewers. The command regenerates task/model
provenance from validated receipts, counts repeated claim origins, and keeps
models, splits and reviewers separate. Parse/schema exclusions differ from valid
empty outputs. Support fractions use explicit reviewed/decisive denominators,
with null on zero denominators; they never establish acceptance or human truth.
Its output is created exclusively and cannot alias an input. Completed scenario
receipts must exist before this command can run; it launches no inference.

The workflow exports a blind bundle for human or agent review, then compares
complete reviewer responses with the separate authored-label answer key. It reads
the packaged scenario set and makes no network calls. Keep the answer key private
from reviewers.

Create a fresh output directory, then export the tasks:

```sh
REVIEW_DIR="$(mktemp -d)"
PYTHONPATH=src .venv/bin/python -m experiments.review_annotations export \
  --bundle "$REVIEW_DIR/blind.json" \
  --answer-key "$REVIEW_DIR/answer-key.json" \
  --seed 123
```

To include local foil or generated candidates, pass JSON objects that map case IDs
to candidate-claim lists:

```sh
PYTHONPATH=src .venv/bin/python -m experiments.review_annotations export \
  --bundle "$REVIEW_DIR/blind.json" \
  --answer-key "$REVIEW_DIR/answer-key.json" \
  --seed 123 \
  --foils "$REVIEW_DIR/foils.json" \
  --unlabeled "$REVIEW_DIR/generated-candidates.json"
```

The blind bundle contains messages and candidates with opaque task IDs. It does not
contain case IDs, scenario IDs, split names, group metadata, candidate membership,
or authored labels. The answer key maps tasks to their case, split, membership, and
reference label. It also lists cases with no tasks. A case with no gold, foil, or
unlabeled candidate creates no review task.

This is procedural blinding, not guaranteed anonymity. Message text and timestamps
are preserved for evidence review. In this fixture, timestamps track case and split
order, so someone familiar with the source set may infer a task's case or split.

## Response format

Each reviewer supplies one complete JSON document with exactly one response per
task. Every response in that document uses the same reviewer ID and kind. The
timestamp must be UTC in `YYYY-MM-DDTHH:MM:SSZ` form. The importer rejects missing,
duplicate, unknown, or extra task responses and rejects extra or missing fields.
The example shows one response row; repeat that object once for each task.

```json
{
  "version": 1,
  "bundle_sha256": "<copy bundle_sha256 from blind.json>",
  "responses": [
    {
      "task_id": "<copy task_id from blind.json>",
      "label": "supported",
      "rationale": "The conversation supports the claim.",
      "reviewer_id": "reviewer-01",
      "reviewer_kind": "human",
      "timestamp": "2026-10-14T19:00:00Z"
    }
  ]
}
```

`label` is `supported`, `unsupported`, or `uncertain`. Judge whether the conversation
supports the claim. Candidate schema validation only checks that fields have valid
shapes and enum values. It does not prove semantic support or citation integrity.
Treat those as separate questions: a claim can be supported by the conversation
while its cited message ID, offset, or quote is wrong. Note citation problems in
the rationale; do not use the support label as an evidence-integrity code. Use
`uncertain` when the conversation does not allow a confident support judgment.

`reviewer_kind` is self-attested. The workflow does not verify human identity or
distinguish a human from an agent. Report counts are split by this declared kind.
Supported, unsupported, and uncertain response counts include unlabeled generated
candidates. Agreement is calculated only for decisive responses on gold and foil
tasks. Generated candidates have `membership: "unlabeled"` and
`reference_label: null`; they are never treated as unsupported foils and do not
enter the agreement denominator. Reports compare labels descriptively and never
change or promote the authored labels.

## Local report

After saving each complete response document, pass all reviewer files to `report`:

```sh
PYTHONPATH=src .venv/bin/python -m experiments.review_annotations report \
  --bundle "$REVIEW_DIR/blind.json" \
  --answer-key "$REVIEW_DIR/answer-key.json" \
  --responses "$REVIEW_DIR/human.json" "$REVIEW_DIR/agent.json" \
  --output "$REVIEW_DIR/report.json"
```

Export and report refuse to overwrite existing output files. They reject output
paths that alias any input, including symlink-resolved paths, and use exclusive
file creation. SHA-256 values bind the dataset and bundle contents for consistency
checks. They do not authenticate the dataset author, reviewer, bundle producer, or
provenance.
