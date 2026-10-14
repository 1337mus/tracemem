# Phase 2 implementation slice

2026-10-13. Continue the authorized local prototype with inexpensive synthetic-only
Modal inference. Private inputs stay untouched and off remote services. No training,
publishing, background deployment or whole-repository upload in this slice.

Acceptance: tested candidate schema and span/role checks; at least 24 fictional
challenge cases covering distinct failure types; exact-match extraction diagnostics;
a reviewed, bounded remote inference runner; measured embedding/extraction baselines
when the account permits. Preserve prior retrieval diagnostics for comparison.

## Candidate interface

`claims.py`: frozen `EvidenceSpan(session_id, message_id, char_start, char_end, quote)`;
`Claim(subject, predicate, object, speaker_role, attribution, modality, polarity,
valid_from, valid_to, evidence)`; `parse_candidates(payload) -> tuple[Claim,...]`.
Payload is exactly `{"claims": [...]}`; an empty list is abstention. Reject unknown
keys, bool-as-int offsets, blank strings, invalid labels, invalid date bounds,
empty evidence and excessive sizes. Predicates: PREFERS, USES, DECIDED, PLANS,
COMPLETED, REPORTS, LOCATED_IN, CAUSES, HAS. Roles: user, assistant, tool, unknown.
Attributions: direct_user_statement, assistant_statement, tool_observation,
third_party_report, quoted_text, unclear. Modalities: asserted, tentative, preference,
intention, hypothetical, question, reported, completed, historical.
Polarity: positive/negative. Valid times are ISO dates or null.

`verify_candidate(claim, records) -> VerificationResult(status, errors, citations)`.
Reject unknown/ambiguous references, bad Unicode-codepoint bounds, unequal quotes,
source-role inconsistencies and attribution-role contradictions. Valid integrity
returns REVIEW, never ACCEPT: an exact quote does not establish semantic entailment.
Citations bind record ID and revision. This does not resolve quoted-user authorship.

## Challenge and evaluation

`challenge.py`: `load_challenge()` returns authored case dictionaries and a file
hash. A case contains case_id, scenario_id, split, tags, messages (session_id,
message_id, role, text, recorded_at, content_kind), and expected claim dictionaries.
Whole scenarios stay within one split; variants of a template are not independent
scenarios. The initial set is a diagnostic challenge, not training-ready data.

`evaluate_outputs(cases, outputs)` consumes case_id -> raw generated string or
parsed object. Report parse/schema failures, exact normalized structured claim
precision/recall/F1, evidence integrity, label errors, unknown-case abstention,
case breakdowns and deterministic scenario-bootstrap intervals. Exact structured
match is not semantic support precision. Reject duplicate case IDs or missing
outputs. Parse whole JSON; never silently salvage fragments or chain of thought.

## Remote work and ownership

- Evidence worker: claims.py and test_claims.py, about 300 production LOC.
- Challenge worker: challenge.py, data/claim_challenge.json and test_challenge.py,
  about 300 production LOC excluding authored fixture data.
- Compute worker: experiments/modal_baseline.py, remote_vectors.py and focused
  tests, at most 500 production LOC. The current runner exceeds that target.
  Keep its remote function in the same script so Modal's cloudpickle serializes
  the `__main__` function by value while `include_source=False` disables source
  mounts. Moving it to an importable helper would default to by-reference
  serialization unless explicitly registered by value and proven source-free.
  Treat this as a one-experiment exception; revisit before reusing the runner.
- Root: architecture, fixture audit, prompts/runner review, actual launch, results,
  adjudication and final validation. Workers do not launch remote jobs.

Run RED/GREEN on meaningful behavior, then focused tests. Do not rerun the full
suite during construction. Keep implementation paths disjoint. No changes to
existing ingestion/storage unless a reproduced defect requires them.

The runner must disable implicit source mounts, use only explicit synthetic
fixture payloads and a pinned public checkpoint, and verify the selected personal
Modal workspace without exposing credentials. One T4 container, min containers 0,
max containers 1, no retries/detach/schedule, startup timeout <=300 seconds, runtime
<=900 seconds per call. At most two initial calls. Operational target <=US$2
for this inference slice; this is a conservative resource envelope, not a billing
cap. Record actual GPU seconds and current provider pricing when available.
Do not launch expensive training under this envelope.

Use Qwen3-4B revision from configs/model-candidates.json for extraction, greedy
decoding with thinking disabled and <=512 new tokens/case. First run at most 24
cases. Embeddings use revision-pinned all-MiniLM-L6-v2 after license metadata is
verified. Download weights remotely. Return versions, input hashes, generated
outputs or synthetic vectors, timing and peak GPU memory. Run the existing
retrieval benchmark locally with precomputed remote vectors; keep Torch off Mac.
