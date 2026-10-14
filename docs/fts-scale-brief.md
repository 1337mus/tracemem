# Indexed FTS head maintenance

2026-10-14. The 50k synthetic setup took 129.55s (10k: 4.06s). Both head
triggers delete FTS rows by unindexed record_id. Implement a new-store schema
version 3 projection using stable head rowids for indexed FTS maintenance.
No original data, existing database or historical receipt may be changed.

Own only src/tracemem/storage.py, the EVIDENCE_SCHEMA_VERSION constant in
src/tracemem/schema.py, and tests/test_storage.py. Preserve evidence rows as
WITHOUT ROWID and all existing immutability/collision checks and transactions.
Do not alter EvidenceRecord identities, ledger schema, APIs, adapters or retrieval.
No migrations/rebuilds, network, models, full-suite or performance jobs.

Use evidence_heads' existing SQLite rowid as the FTS rowid. INSERT/UPDATE
triggers must INSERT FTS with rowid=new.rowid and delete the existing row by
rowid only. Add an AFTER DELETE head trigger to remove its FTS row by old.rowid
so stale rows cannot survive head removal/reinsert. Add this trigger to required
schema validation. Advance the evidence version to 3 so version 2 stores are
rejected through the read-only preflight without mutation or sidecars. Do not
silently reinterpret old FTS layouts. The code must never migrate an old store.

TDD: first a genuine temporary SQLite test showing FTS rowid tracks head rowid
after updating an earlier head when later heads exist; old implementation must
fail this behavior. Pin one FTS entry per head, no obsolete searchable text,
exact revision fetch surviving updates, Unicode/current-rank retrieval and
idempotent duplicate ingests. Add real head deletion/reinsert checks for no
stale FTS rows and new-rowid correctness, including another head surviving.
Reject schema-2 stamped DB before any writes with byte-identical original and
no sidecars. Preserve atomic rollback on mixed valid/invalid ingest and all
SQL-replacement evidence immutability tests. Use EXPLAIN QUERY PLAN on the
actual rowid deletion shape (not timing thresholds) to verify indexed virtual
table lookup rather than record_id full scan. Root will inspect complete SQL
and run new before/after measurements after approval of the diff.

Only focused storage/retrieval/citation/ledger tests, Ruff and source mypy now.
Report exact RED/GREEN evidence and version compatibility risk. Root owns the
final full-suite and new installed-wheel validation after both workers settle.
