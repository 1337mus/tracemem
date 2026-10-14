# Source-retention impact diagnostic

Bounded continuation after the turn-retrieval slice. This implements a pure,
non-mutating dependency diagnostic before any erasure mechanism. Original stores,
Context Kit, private conversations and exports remain untouched. No inference,
network, GPU, migrations or deletion. No claim that erasure is implemented.

Owned files: `src/tracemem/retention.py`, `tests/test_retention.py`,
`docs/source-retention.md`. Do not edit existing runtime modules or fixtures.

API: `plan_source_erasure(records: Sequence[EvidenceRecord],
entries: Sequence[CandidateEntry], *, source: str, session_id: str) -> ErasureImpact`.
Frozen dataclasses are acceptable. Inputs are an explicit in-memory snapshot; the
function opens no files. Accept sequence values, not strings/generators; validate
all records/entries before computing any result, using exact expected types.
Validate selector as nonempty UTF-8 strings; errors never echo selectors or text.
Reject duplicate `(record_id, revision)` keys and duplicate candidate entry IDs.
Validate candidate IDs and supersedes as canonical SHA256 IDs; reject dangling
lineage and cycles. This is diagnostic integrity, not re-verification of a ledger.

Return deterministic sorted tuples for selected `(record_id, revision)` pairs,
direct candidate IDs, conservative span-only ambiguous candidate IDs, transitive
dependent candidate IDs, and union affected candidate IDs. Include all historical
revisions supplied, not only current heads. Match selected source AND session
exactly; part indices and different source providers must remain distinct.

Direct means any canonical verification citation identifies a selected
record/revision, using existing citation parser (inspect `citations.py`). Validate
all provided verification citations: malformed ones fail closed with sanitized
ValueError. Do not rely on verification status: rejected entries may retain quotes.
Span-only ambiguity means an evidence span's `(session_id, message_id)` matches
any selected record, but the entry has no exact selected citation. Span schema
lacks provider/revision/part information, so report this conservatively rather
than claiming proof of source provenance. Include it even when a citation points
at another provider or revision. Transitive dependent means an entry supersedes
any affected entry; compute complete descendant closure, excluding directly or
ambiguously affected entries from this separate category. Ancestors/siblings do
not become affected merely because a descendant is affected.

Do not return raw source/session selectors, claim fields, evidence quotes, source
paths, verification errors, or transcript text. IDs and hashes are still private.
The report must explicitly say it is a snapshot dependency diagnostic, deletion
is not performed, vector-cache keys correspond to selected revisions but counts
and physical deletion are not audited, and external exports/backups/unlinked
copies/free SQLite pages/in-memory caches cannot be discovered. Inputs must
include all historical evidence and ledger entries for a complete local snapshot.
No guarantee of semantic dependence or complete information-flow detection.

Use meaningful TDD: first assert absent behavior on a stub after import succeeds,
then implement that behavior. Add focused behavioral tests incrementally. Required
coverage: all revisions, exact source/session isolation, Unicode, rejected-entry
quotes, ambiguous spans (including other-provider citations), valid canonical
citations with precise revision matching, multi-hop and branched closure,
unrelated/ancestor retention, empty target, order invariance, duplicate inputs,
invalid selectors/types/citations/lineage/cycles, no caller input mutation.
Use real EvidenceStore/CandidateLedger in at least one fixture and verify their
bytes and existing source/citation access survive unchanged. All test data newly
authored and fictional. Run focused unittest, Ruff/check+format, strict mypy.
Return exact commands/results and release ownership. Do not stage/commit/push.
