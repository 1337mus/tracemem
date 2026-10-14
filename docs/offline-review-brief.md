# Offline review continuation

Updated 2026-10-14. Prototype scope; no training, private-data uploads, or GPU calls.

Build a dependency-free offline HTML review page over the existing validated blind
bundle. This removes JSON-editing friction for independent reviewers; it does not
create human labels or satisfy Gate A. Preserve the frozen dataset and all prior
reviews. Keep models, keys, origins and split membership out of the review page.

Architecture: `tracemem.review_html.render_review_html(bundle)` validates through
the existing annotation validator, serializes only blind content into a safe inert
JSON block, and returns a self-contained document. An experiment CLI accepts only
bundle/output paths and uses strict JSON reading plus exclusive file creation.
No original-source writes, external assets, network, localStorage or service worker.
Use textContent for all untrusted strings and a restrictive CSP with static-script
and style hashes. Draft import/export uses explicit files. Complete responses use
the existing version-1 contract and remain subject to Python validation.

Review page: one task at a time; complete source messages and full candidate;
supported/unsupported/uncertain labels with no default; required rationale;
explicit reviewer ID and kind with no kind preselected; previous/next and progress;
draft save/restore; final export only when every task has a valid response. Import
must validate the full draft before replacing current state, reject wrong-bundle,
unknown/duplicate task IDs and malformed fields. Preserve Unicode and offsets.
Clearly distinguish semantic support from citation integrity and self-attested
human identity. Warn that unsaved work is lost when closing/reloading.

Allowed worker files: `src/tracemem/review_html.py`,
`experiments/export_review_html.py`, `tests/test_review_html.py`,
`docs/offline-review.md`. Root handles delivery records, README links and review.
Apply red/green tests before behavior; record exact RED and GREEN commands.
Acceptance: safe script embedding including `</script>` and Unicode; rejection of
invalid bundles; output/source preservation; no answer-key dependency; draft and
complete response behavior exercised in JS runtime or browser; existing validator
accepts the complete exported document. Root reads all new source/tests, runs a
real browser smoke, and validates final checks before signed private push.
