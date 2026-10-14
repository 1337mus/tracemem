# Frozen synthetic QA analysis implementation

Root brief, 2026-10-14. Implement the existing frozen rules in
`docs/synthetic-qa-answer-protocol.md`, without observing any model answers first.
Files: `experiments/analyze_synthetic_qa.py`, `tests/test_synthetic_qa_analysis.py`,
`docs/synthetic-qa-analysis.md`, `.local/synthetic-qa-analysis-worker-*` raw logs.
No edits elsewhere. No network/model/Modal calls, training, full suite, commits or
push. Preserve all historical files. Target <=300 production LOC; propose a split
before exceeding500. Use one genuine behavior RED/GREEN at a time, capturing raw
output with PYTHONPATH=src; setup/import failure is not behavioral RED. Do not
reconstruct logs. Tests may call real fixed preparation and construct fictional
receipts; no runtime model libraries.

Implement `analyze_receipt(receipt, *, approved_runner_sha256,
approved_template_sha256)` returning an ordinary JSON-compatible dict, plus a CLI
with --receipt, --approve-runner-sha256, --approve-template-sha256, --output NEWPATH.
Regenerate only the fixed prepare_synthetic_qa() requests/references, pin canonical
request 85d1334d0401ef2bdc920bf316f066bc3e74e9423ee5bec0aa63f4a559be2084,
references70769ec36ac4c1e6a226d7432ea88241225e35d91d803cdc0a678fe2589226d2,
fixturea4d94c8e727c71e8c8b723e97c9674b1d0008ac9f35ba675485a7ea796349d78.
Regenerate build_payload() from experiments.modal_synthetic_qa and require exact
80e058c629de8f8a06ca4caa8f1093a8308a299c976b018ab7e8c5c62493b2ac. No arbitrary
request/reference inputs. Check valid lowercase SHA256 approval values, exact runner
and template matches. These are reviewed provenance checks, not remote attestation.

Success envelope contract: schema_version1, payload_sha256, runner_sha256,
client_elapsed_seconds,result. Failure/error or malformed receipts refuse analysis.
Require exact payload digest also inside result; model_id Qwen/Qwen3-4B and revision
1cfa9a7208912126459214e8b04321603b3df60c; config_sha256 is canonical payload.model
hash; exact template approval; package_versions torch2.7.1 (permit +CUDA build suffix),
transformers4.51.3,accelerate1.6.0; nonblank modal_call_id. Request_count=72 and
exactly72 unique allowlisted outputs, each required to have request_id,raw_output,
input_tokens,generated_tokens,stop_reason,hit_max_new_tokens,truncated. Strict ints
(no bool), prompt1..2048, generated0..128; raw_output str (empty valid), booleans
strict, stop_reason eos/length/other. Enforce length implies128 and truncated=true;
eos impliesgenerated>=1 and truncated=false; hit_max_new_tokens iff generated128;
other remains in denominators and flagged. Verify finite nonnegative client timing.
Other runtime metadata remains retained in raw receipt, not construed as attestation.

Only normalization is ' '.join(text.casefold().split()). No punctuation removal,
substring matching, aliases, semantic judging or tuning. Empty and length outputs
count in all relevant denominators. Reference UNKNOWN exactly for unknown questions.
Return18 cells separately under dev/test (each nine cells,3modes x3 byte ceilings),
each n4,answerable n3,unknown n1. Each has literal agreement numerator/denominator
for all,answerable,unknown; zero-context literal UNKNOWN numerator/denominator at
ceiling0 includes all4 (else denominator0); finish/hit-limit/truncated/empty counts;
actual input/generated token min/max/sum. Include eight distinct questions and72
matched repeated requests, provisional agent-only label and no-semantic/no-matched-
token/no-independent-scenario limitations. No pooled dev/test score or ranking.
Preserve per-request outcomes in the report to independently reproduce cell counts;
raw text stays in the immutable input receipt. Report source receipt canonical hash,
request/reference/payload/runner/template/config and analyzer source hashes.

CLI read bounded2MiB JSON, reject duplicate keys/nonfinite constants. Output must be
new mode0600, no overwrite/symlink target; sanitized errors and nonzero status.
Verification: exact18-cell oracle on constructed receipt, normalization/empty/length/
zero-context distinctions, shuffled outputs yield same report, duplicate/missing/extra
IDs fail, provenance/invalid types fail, input/output failure paths. Ruff and strict
mypy focused files. Release hashes, exact commands and honest raw log chronology.
