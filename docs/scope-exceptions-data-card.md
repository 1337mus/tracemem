# Scoped defaults and exceptions data card

Updated 2026-10-15 UTC.

## Purpose and files

This slice covers default/exception branches, source attribution, dated exceptions, canceled proposals, and a prerequisite that does not decide the later choice. Earlier fixtures include `dev-ferry-bike`,
`test-station-elevator`, `dev-coffee-after-lunch`, and `long-dev-case-02`.

- Fixture: [`scope-exceptions-dev-v1.json`](../examples/synthetic/scope-exceptions-dev-v1.json), SHA-256 `58af765d06497afaad3c5cfbc276704b33d3a3828f63d5a6d888297c527b50dc`.
- Foils: [`scope-exceptions-dev-v1-foils.json`](../examples/synthetic/scope-exceptions-dev-v1-foils.json), SHA-256 `0bdfff504e40c360c6f2af4cfe6607f3e52acd8ba9c00b3bd25eaba9bebbf37a`.
- Scope: 6 dev cases, 30 messages, 12 sessions, 10 provisional targets, 12 foils, 22 blind tasks.
- Cases: performed map-tube exception to an archive default; quoted handbook rule; reported email/SMS duty exception; dated entrance exception and return to default; canceled single-sided print proposal; written-approval prerequisite without a press choice.

## Measured evidence

- [Source replay report](scope-exceptions-report.json), SHA-256 `9370c4b3099f1c2d429810bf8ac6d7f2a11e59f40b1443833507d0c8d3e867f0`: 26 cited spans; `REVIEW` integrity for all 10 targets and 12 foils; zero model calls.
- [Combined audit](validation-scope-exceptions.json): 90 cases (42 dev, 24 diagnostic, 24 assigned test, 0 train); metadata covers 66, with 24 challenge cases missing metadata. No lexical duplicate pairs or declared cross-split identifier overlaps were found; this does not establish semantic isolation or independence. Gate A remains unmet.
- [Unblinded root review](scope-exceptions-semantic-audit.json) provisionally supports 10 targets and rejects 12 foils. Membership was visible; this is neither independent nor human validation.
- Root's local `make check` passed 617 tests with one optional skip; source and installed focused runs each passed 5 tests. These are local results; verify signed delivery and exact-head hosted CI in `.local/active-cycle.json`.
- TDD record: static fixture/CI/docs work is exempt; new tests exercise existing runtime behavior. No runtime behavior or RED cycle is claimed.

## Export a blind packet and HTML

Run from the repository root; share only `reviewer/` and retain `curator/` separately.

```sh
PYTHONPATH=src .venv/bin/python - <<'PY'
import hashlib, json, tempfile
from pathlib import Path
from tracemem.annotations import export_review_bundle
from tracemem.claims import parse_candidates, verify_candidate
from tracemem.review_html import render_review_html
from tracemem.schema import EvidenceRecord
root=Path.cwd(); path=root/'examples/synthetic/scope-exceptions-dev-v1.json'; raw=path.read_bytes()
data=json.loads(raw); foils=json.loads((path.parent/'scope-exceptions-dev-v1-foils.json').read_text())
fields=('session_id','message_id','role','text','recorded_at','content_kind')
for case in data['cases']:
    records=[EvidenceRecord(source='scope-review',part_index=0,**{k:m[k] for k in fields}) for m in case['messages']]
    for claim in [*case['expected'],*foils[case['case_id']]]:
        parsed=parse_candidates({'claims':[claim]})[0]
        if verify_candidate(parsed,records).status!='REVIEW': raise ValueError(case['case_id'])
blind,key=export_review_bundle(data['cases'],hashlib.sha256(raw).hexdigest(),foils,seed=1729,case_metadata=data['case_metadata'])
out=Path(tempfile.mkdtemp(prefix='scope-exceptions-')); reviewer=out/'reviewer'; curator=out/'curator'; reviewer.mkdir(); curator.mkdir()
opts=dict(indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)
(reviewer/'blind.json').write_text(json.dumps(blind,**opts)+'\n',encoding='utf-8')
(reviewer/'review.html').write_text(render_review_html(blind),encoding='utf-8')
(curator/'answer-key.json').write_text(json.dumps(key,**opts)+'\n',encoding='utf-8')
print('Reviewer files:',reviewer,'; answer key:',curator/'answer-key.json')
PY
```

Combined lexical audit using the existing CLI:

```sh
PYTHONPATH=src .venv/bin/python experiments/audit_dataset.py \
  src/tracemem/data/scenarios_v1.json examples/synthetic/longitudinal-dev-v2.json \
  src/tracemem/data/claim_challenge.json examples/synthetic/scope-exceptions-dev-v1.json
```

## Limits

Fictional, MIT, dev-only, and `agent_authored`; labels are provisional. Citation integrity is not
semantic truth; there are zero human labels, held-out cases, Gate A approval, or extraction-quality
measurement. This slice establishes no benchmark quality, generalization, independence, or real-world truth.
