# Provisional synthetic QA answer diagnostic

Frozen before any QA inference, 2026-10-14. This is a proposed offline analysis
contract for the exact72 reviewed requests. No answers have been generated or
scored. It does not change the answering prompt, fixture, retrieval or payload.

The first analysis will report literal reference agreement, not semantic QA
accuracy or upstream LongMemEval scores. Normalize each answer by Python
`' '.join(text.casefold().split())` only. Do not remove punctuation, extract a
substring, rewrite answers or add aliases after observing outputs. Preserve every
raw answer. References are provisional agent-authored, not independent human truth.
No judge-model call is part of this diagnostic.

Require one raw output for every reviewed request ID, with no duplicate, missing
or extra IDs. Verify exact payload/runner/model provenance and reference/request
bundle hashes before scoring. Invalid or partial receipts fail analysis rather
than silently changing denominators. Empty outputs disagree and are counted.
Length-stopped outputs remain in literal-agreement denominators and have separate
finish/truncation counts. No inference retry or test-output salvage is authorized
by the evaluator.

Report dev and test separately, then each retrieval mode and byte ceiling:18cells
with four questions each, three answerable and one unknown. Include counts and
literal-agreement denominators for all questions, answerable questions and unknown
questions. The unknown-question reference is exactly UNKNOWN. Also report the
literal UNKNOWN rate for every zero-byte control, including answerable questions;
appropriate abstention with empty context is distinct from question-reference
agreement. Record actual input/generated token counts and lengths from the remote
receipt; byte ceilings are not tokenizer-matched budgets. Matched cells repeat
eight questions and must not be pooled as72 independent scenarios.

This tiny fixture tests protocol behavior. It cannot establish model superiority,
calibration, real benchmark compatibility, semantic correctness, cost per correct
answer or training readiness. Any later prompt, alias or metric revision needs a
new explicitly development-labeled diagnostic; do not retune this frozen test.
