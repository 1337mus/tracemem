# Candidate ledger implementation brief

2026-10-14. Build a separate local SQLite candidate ledger without changing the
evidence database schema or any original source. This is an engineering slice
of Phase 4; it does not satisfy the dataset or training gates.

The ledger preserves proposed claims, deterministic integrity outcomes, pinned
source citations and explicit correction lineage. It never establishes semantic
truth, makes ACCEPT decisions or feeds candidates into default retrieval.

API: context-managed CandidateLedger(path); append(claim, records,
extractor_id, recorded_at, supersedes=()) -> immutable entry; get(entry_id);
entries(). Use frozen dataclasses, tuple fields, canonical JSON and content
SHA-256 IDs. Store the exact claim snapshot, verifier result, citations,
extractor provenance and caller-supplied canonical UTC timestamp. Run
verify_candidate internally; do not accept caller-supplied verification. Store
both REVIEW and REJECT candidates. Reject invalid metadata. Supersedes is
explicit proposal lineage, never an automatic statement that the old claim is
false or current. Targets must already exist, cannot repeat or self-reference;
multiple branches remain possible. Do not mutate prior rows. Re-appending an
identical entry is idempotent. Any metadata change creates a different entry.

Use a dedicated application ID/schema version and fail closed when passed any
non-ledger file. Create new parent folders 0700 and files 0600. Reject symlinks
and nonregular database paths. Preflight existing SQLite files read-only before
opening writable. Prefer reusing tested private-file primitives if that does
not couple ledger schema validation to EvidenceStore. EvidenceStore itself must
remain unchanged. SQL must be parameterized; transactionally insert entry and
lineage. WITHOUT ROWID and insert/update/delete protection must prevent ordinary
SQL replacement or mutation. SQLite owner access can drop triggers; do not claim
cryptographic tamper resistance. This ledger stores private derived content when
used on private input, and belongs in excluded local storage.

Limit production implementation to roughly 350 physical lines, tests to behavior
needed for the slice. Explicitly defer erasure, ledger ingestion CLI, acceptance
policy, temporal truth resolution and retrieval integration. If this cannot fit,
report the partition before implementation.

Tests: persistence roundtrip/reopen; deterministic verification computed locally;
REJECT preserved and never upgraded; exact pinned citation survives changed
source revisions; append idempotence and changed metadata; correction lineage
preserves both entries; unknown/duplicate targets fail with no partial write;
SQL UPDATE/DELETE/REPLACE fail; alien database unchanged; path permissions and
symlink rejection. Use TDD and finish focused tests, Ruff and source strict mypy.
