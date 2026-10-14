# Exact source retrieval

2026-10-14. Next local engineering slice while the synthetic comparisons run.
Preserve the evidence and ledger schemas, previous receipts, and original inputs.

Implement `src/tracemem/citations.py` and `tests/test_citations.py` using TDD.
No CLI, network, model inference, writes to source stores, acceptance decisions,
or current-truth resolution. Approximately 180 production lines maximum.

Expose a frozen `ResolvedCitation` containing citation, full EvidenceRecord,
char_start, char_end and quote. `fetch_citation(store, citation)` must parse only
the existing canonical format `record_id@revision:start:end`, using full lowercase
SHA-256 identities and canonical nonnegative decimal offsets (no leading zeros).
Require a nonempty span, fetch that exact revision using EvidenceStore.get, verify
the returned record identities by recomputation, and check Unicode code-point
bounds. Never search another revision or silently use current heads. Missing
revisions raise KeyError; invalid/corrupt citations raise ValueError. The returned
quote is the exact source slice, not a supplied paraphrase. Keep strings immutable.

`fetch_entry_evidence(store, entry)` returns the citations in claim-evidence order.
Only integrity-clean REVIEW entries with one citation per evidence span qualify;
REJECT entries fail explicitly without returning a partial success. Resolve each
citation and verify session/message IDs, offsets and quote match its claim span.
Re-run verify_candidate using the resolved records and require an identical REVIEW
result, including exact citations; this catches role and attribution mismatches.
Do not describe REVIEW or successful source fetching as semantic support. This
helper accepts an already loaded CandidateEntry and an already opened store;
it does not bind ledger and source file lifecycles or make either self-contained.

Tests must use actual temporary SQLite stores and ledgers: Unicode exact slice;
old revision survives ingesting a newer revision and close/reopen; unavailable
revision never falls back; malformed IDs/offsets/empty/out-of-range spans rejected;
multi-message entry resolves in order; mismatched quote/identity/role rejected;
REJECT candidate refused; no database writes during fetch (total_changes or trace
and unchanged records). Include a corrupt stored-row identity test if possible.
Run focused suite, Ruff, strict source mypy. Root reviews the complete diff.

## Installed-wheel smoke

From the repository root, build and install a wheel into a fresh environment, then run the
smoke from an external working directory with a fresh SQLite work directory:

```sh
set -e
repo_root="$PWD"
smoke_tmp="$(mktemp -d)"
mkdir "$smoke_tmp/wheels" "$smoke_tmp/run" "$smoke_tmp/work"
python3 -m pip wheel --no-deps --wheel-dir "$smoke_tmp/wheels" "$repo_root"
python3 -m venv "$smoke_tmp/venv"
"$smoke_tmp/venv/bin/python" -m pip install "$smoke_tmp/wheels/tracemem-0.1.0-py3-none-any.whl"
(cd "$smoke_tmp/run" && env -u PYTHONPATH -u PYTHONHOME \
  "$smoke_tmp/venv/bin/python" "$repo_root/experiments/smoke_installed.py" \
  --source-root "$repo_root" --work-dir "$smoke_tmp/work")
```
