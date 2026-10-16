# Private verifier annotation and export contract

This local pilot uses a separate version 1 corpus for private conversation
records. It preserves private_source provenance and the
private-not-for-redistribution license. Do not pass this corpus through the
synthetic verifier validator or change that validator's MIT/fiction gates.

## API

validate_private_verifier_dataset(payload, source_records) validates the
private corpus against a local source-record mapping and returns validated
cases. source_records is keyed by record_id and must contain the adapter
snapshot fields record_id, revision, session_id, message_id, text,
text_sha256, source_locator, line_number, role, recorded_at, and
content_kind. Additional adapter fields are allowed. The text hash is
recomputed; every case message must match its local record's content,
identity, session, and hash.

render_private_training_rows(payload, source_records) returns only train
rows, each with a label-free prompt and a JSON completion containing the
separate support and eligibility labels.

render_private_eval_prompts(payload, source_records, split) accepts dev or
test and returns prompt strings only. Both renderers use the shared
verifier prompt renderer after private validation. Prompts include only the
message fields and candidate citation needed by that renderer; source
locators, hashes, privacy clearance, provenance, reviewer identities, status,
and rationales stay out of the prompt.

These functions return in-memory values and perform no file or network writes.
The caller is responsible for keeping source records and exports in the
authorized local environment.

## Corpus shape

The root has exactly these fields and literals:

- version: integer 1
- license: private-not-for-redistribution
- annotation_status: agent_provisional
- provenance: private_source
- scope_policy: personal-memory-research-v1
- cases: a non-empty list

Each case contains case_id, source session_id, split (train, dev, or
test), unique window_id, privacy_clearance, messages, and candidates.
Every window from one source session must use the same split. Exact or
NFKC/casefold/whitespace-normalized message duplicates of at least 160
characters may not cross splits.

privacy_clearance is an explicit object with status (cleared, pending,
or rejected), reviewer_id, reviewer_type (agent or human), and
reviewed_at. Pending clearance may have a null timestamp. Only cleared cases
are rendered.

Each message stores message_id, source_record_id, revision, session_id,
text, text_sha256, source_locator, line_number, role, recorded_at, and
content_kind. Unknown recorded_at remains null. Roles and content kinds
follow EvidenceRecord; a user message cannot be marked as tool_output.

Each candidate stores candidate_id, claim, message_id, source_record_id,
quote, char_start, char_end, valid_from, valid_to, support, eligibility,
separate support_reason and eligibility_reason, label_status, annotator,
and reviews. Offsets are Python string indices into the exact local source
text, with an exclusive end; the cited slice must equal quote and stay within
the source record. Support values are supported, unsupported, or uncertain.
Eligibility values are eligible, ineligible, or uncertain. Neither axis is
inferred from the other.

annotator and each review record retain identity plus reviewer_type, so agent
and human judgments remain distinguishable. Reviews record their own support
and eligibility labels and rationales. Pending reviews carry null
labels/rationales; completed reviews carry both labels and rationales.
label_status is agent_provisional, accepted, pending, or rejected.
Accepted labels require at least one independent reviewer whose completed
labels agree on both axes. Provisional disagreements stay recorded but are
withheld from exports until reconciled.

No row with pending or rejected privacy clearance, label status, or review
status is exported. An uncertain support or eligibility value remains a
valid label and may be exported once the privacy and review gates are clear.
