# Dataset isolation audit

Run the local lexical and declared-identifier audit across the scenario corpus,
the current longitudinal development slice, and the bundled claim challenge:

```sh
PYTHONPATH=src .venv/bin/python -m experiments.audit_dataset \
  src/tracemem/data/scenarios_v1.json \
  examples/synthetic/longitudinal-dev-v2.json \
  src/tracemem/data/claim_challenge.json \
  > docs/dataset-isolation-report.json
```

The report records input SHA-256 digests in command-line order and omits source
paths. With no paths, the command preserves its original bundled-challenge report
and `fixture_sha256` field.

Each `case_metadata` row declares a `scenario_group`, `template_group`, and list
of `entity_ids`. Reuse of one of these literal identifiers across splits flags
cases to inspect for scenario, template, or entity leakage. The audit keeps the
assigned split labels. Inputs without metadata still contribute to lexical
duplicate checks; if any input declares metadata, uncovered cases are listed and
the report marks metadata coverage incomplete. The claim challenge has no
metadata, so its 24 diagnostic cases are uncovered in this combined report.

The measured 84-case pool has 36 dev, 24 diagnostic, and 24 test cases, with no
train cases. Metadata covers 60 cases and is incomplete. The report found zero
declared-identifier overlaps, exact duplicate conversations, or cross-split
near-duplicate pairs at the default 0.8 threshold. These results describe this
fixture set and these literal IDs only. Assigned test labels do not establish
held-out evaluation.

`scenarios_v1.json` is a separate corpus from `longitudinal-dev-v2.json`. Do not
include `longitudinal-dev-v1.json` alongside v2; it is an earlier version of the
same development slice. Renaming identifiers can hide conceptual reuse, so zero
identifier overlaps do not prove semantic isolation. The report excludes raw
message and claim text, while declared identifiers may still be private. It does
not establish Gate A readiness, rights, human label quality, or sample adequacy.

See the generated [dataset isolation report](dataset-isolation-report.json) and
the [audit implementation](../src/tracemem/dataset_audit.py).
