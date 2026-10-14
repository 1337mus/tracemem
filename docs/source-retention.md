# Source-retention impact diagnostic

`tracemem.retention.plan_source_erasure` reports which candidate entries in an
explicit in-memory evidence and ledger snapshot refer to a selected source and
session. It is a dependency diagnostic only. It does not delete records, alter
the evidence store or candidate ledger, perform inference, or contact an
external service.

```python
from tracemem.retention import plan_source_erasure

impact = plan_source_erasure(
    evidence_store.records(current_only=False),
    candidate_ledger.entries(),
    source="conversation-export",
    session_id="session-42",
)
```

Pass sequences containing `EvidenceRecord` and `CandidateEntry` values. The
caller chooses the snapshot; include every historical evidence revision and
candidate ledger entry for a complete local analysis. The diagnostic rejects
malformed records, claims, citations, and lineage before computing the result.
It does not re-verify a ledger or infer semantic truth.

The result contains sorted tuples of selected `(record_id, revision)` pairs,
direct candidate IDs, ambiguous candidate IDs, transitive dependent candidate
IDs, and the union of affected candidate IDs. A direct dependency has a
canonical verification citation for an exact selected record and revision.
An ambiguous dependency has a matching `(session_id, message_id)` evidence
span but no exact selected citation; spans lack provider, revision, and part
identity, so this is a conservative signal. A dependent entry supersedes an
affected entry. Ancestors and siblings are not included merely because a
descendant is affected. Rejected entries are considered because they may
preserve citations or quoted evidence.

The result includes explicit limits: vector-cache keys correspond to selected
record/revision pairs, but cache counts and physical deletion are unaudited;
external exports, backups, unlinked copies, free SQLite pages, and in-memory
caches cannot be discovered. Semantic dependence and complete information-flow
detection are not guaranteed. The result does not contain source or session
selectors, claim fields, quotes, source paths, verification errors, or
transcript text. Candidate and record IDs are still private identifiers and
should be handled accordingly.
