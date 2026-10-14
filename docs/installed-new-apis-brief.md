# Installed budget and offline review API smoke

Updated 2026-10-14. Root-reviewed private prototype. No new model, GPU, network,
package installs or private-data operations. Root owns final validation and push.

Existing experiments/smoke_installed.py runs in CI against a fresh wheel outside
checkout, but does not exercise the shipped budget_benchmark/review_html APIs.
Root already tested both manually from installed site-packages; make that check
permanent in the existing installed smoke. Preserve all prior smoke behavior.

Worker owns ONLY experiments/smoke_installed.py and new tests/test_installed_new_apis.py.
No commits, pushes, other edits except .local/ logs. Root is committing the prior
uncertainty experiment concurrently; do not touch its files. Follow TDD with real
red/green evidence before logic changes, no mocks that substitute target behavior.

Add a narrow helper usable by main after preflight. It reads only tracked fictional
inputs under source-root and writes SQLite artifacts only under supplied work-dir.
Use installed tracemem.benchmark.load_synthetic_data, EvidenceStore and
budget_benchmark.run_context_budget_benchmark. Compare every field to the recorded
docs/baseline-context-budgets.json except environment.python and environment.sqlite;
record both actual and recorded environment values explicitly, compare the exact
set of environment fields. No broad metadata/latency exclusion. Preserve the recorded
report and fixture SHA. A changed packed citation, query/ID, configuration, metric
or environment schema must fail. Use a fresh work-dir database filename.

Read examples/synthetic/reviews/scenario-review-blind.json (84 tasks) from source,
bind its raw-file SHA, render via installed tracemem.review_html.render_review_html,
and compare to root-approved final HTML SHA07521f971f5554e5476d15a1299eb75cf0b5b20fbc6854c79b6053a1aecc474e.
Pin raw bundle SHA65ed4a2e0bbd227c3a521899676c4db98085f855083ec2d2a16aecbdda55bda5
and budget report raw SHA (compute/verify from docs before pinning). Do not generate
or load answers, labels or model origins for the review UI check. No browser or
network calls. The HTML need not be persisted: hash the rendered text directly.

Move/recheck installed_module_paths after the new module calls so all loaded
tracemem modules must resolve to installed site-packages outside source-root.
Output concise new receipt fields for module paths, source/raw fixture hashes,
HTML hash/task count, stable budget fields match and differing environment values.
Keep existing receipt keys and all prior behavior. Fail with clear, specific
SystemExit/error on mismatch. Owner-only files if creating additional exports.

Meaningful tests cover full real fictional helper success, preservation of input
bytes, changed budget metric/citation/config rejected, py/sql env differences
allowed but environment-schema differences rejected, altered review bundle/HTML
hash rejected. Avoid building wheels in each unit test: pass a normal temporary
source-root with copied fictional files; end-to-end fresh installed-wheel validation
is root/CI's separate check. For hash mismatch behavior, seed fixture hash via
copying committed fixture bytes, mutate temp-only copies; no production-input edits.
Return exact commands/test results and any limitations; root reviews complete diff.
