# Longitudinal development slice data card

Updated 2026-10-15 UTC.

## Contents and provenance

- **Fixture:** [`examples/synthetic/longitudinal-dev-v1.json`](../examples/synthetic/longitudinal-dev-v1.json), version 1, MIT, `annotation_status: agent_authored`.
- **Separate foils:** [`examples/synthetic/longitudinal-dev-v1-foils.json`](../examples/synthetic/longitudinal-dev-v1-foils.json).
- **SHA-256:** fixture `d75b2237e739bfd8ff72be25d234f62004b6641a48a8bd9893442d64e8bc8fda`; foils `26f65c6c80f8e1abcb4c9d7eaa1e7215dc41d114113f33f781550612d77f91be`.
- **Counts:** 12 cases, 146 messages, 13 provisional target claims, and 13 foils. Each case has 12 or 14 messages, spans 4 or 5 sessions, and covers 36 to 55 calendar days. Five targets cite evidence across multiple sessions. Every case is `dev`; this fixture is separate from the fixed 48-case `load_scenarios()` set.
- **Origin:** all conversations and labels were newly authored as fiction for this project. No private source or external corpus was used. Names, settings, organizations, and conversation dates are fictional, not observed event truth.

## Intended use and limits

This is a small development slice for reviewing evidence-grounded claim candidates over longer timelines. It includes corrections, conditional plans, role attribution, quoted instructions, negation, explicit time boundaries, same-name ambiguity, uncertain third-party reports, and unsupported causal leaps. Five target claims cite supporting details across sessions by construction. Message count, elapsed time, and metadata do not establish a natural long-horizon benchmark or semantic independence.

Target labels are selective and non-exhaustive. The two empty-`expected` cases mean that no target claim was selected for this slice. They do not mean that the conversations contain no extractable facts or that a system should abstain on the whole conversation. All labels are agent-authored and provisional. No human has reviewed or adjudicated them. Integrity checks verify structure and exact cited spans; they do not verify semantic support, annotation quality, or generalization. The declared scenario groups, template groups, and entity IDs are bookkeeping only. This slice has no inference results and no Gate A approval.

## Local checks

The [readiness report](longitudinal-readiness.json) records 49 sessions and no
exact duplicate conversations or cross-pool lexical matches at Jaccard 0.8
against the earlier 72 extraction cases. That lexical check cannot establish
semantic independence. All 13 targets and all 13 foils pass citation checks and
remain `REVIEW`; none is automatically accepted. The 26 review tasks include
both empty-target cases through their foils.

An always-empty output misses all 13 selected targets. Replaying the reference
matches 13/13; replaying the authored foils matches 0/13 while retaining 13/13
valid citations. These are checks of the scoring setup, not model results.

## Prepare a blind review bundle

Run this from the repository root. It writes only a blind bundle into a new temporary directory. Give reviewers only `blind.json`; keep the source fixture and any separately generated answer key with the curator.

```sh
PYTHONPATH=src .venv/bin/python - <<'PY'
import hashlib
import json
import tempfile
from pathlib import Path

from tracemem.annotations import export_review_bundle

fixture_path = Path("examples/synthetic/longitudinal-dev-v1.json")
foils_path = Path("examples/synthetic/longitudinal-dev-v1-foils.json")
raw = fixture_path.read_bytes()
payload = json.loads(raw)
foils = json.loads(foils_path.read_text(encoding="utf-8"))
blind = export_review_bundle(
    payload["cases"],
    hashlib.sha256(raw).hexdigest(),
    foils_by_case=foils,
    seed=1729,
    case_metadata=payload["case_metadata"],
)[0]
out_dir = Path(tempfile.mkdtemp(prefix="longitudinal-blind-review-"))
output_path = out_dir / "blind.json"
with output_path.open("x", encoding="utf-8") as output:
    json.dump(blind, output, ensure_ascii=False, indent=2)
    output.write("\n")
print(output_path)
PY
```

The exporter binds task IDs and bundle hashes to the fixture bytes and keeps case identity and candidate membership out of the blind bundle. This procedural blinding does not validate labels or guarantee anonymity.
