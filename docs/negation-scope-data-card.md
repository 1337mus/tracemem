# Negation and scope fictional data

This MIT-licensed set is independently authored fiction; it does not rework
private conversations. The [train fixture](../examples/synthetic/negation-scope-train.json)
and [development fixture](../examples/synthetic/negation-scope-dev.json) are
byte-for-byte copies of the reviewed fictional source files.

The training split has 24 candidates across 12 scenarios. The author, a blind
agent review, and root review agreed on 24/24 support labels and 24/24
eligibility labels. All 24 candidates remain train-only and are excluded from
evaluation.

The development split has eight candidates across four scenarios. Author labels
matched root's blind labels on 8/8 support and 8/8 eligibility judgments; root
then adjudicated. These eight candidates remain dev-only and are excluded from
training. There were zero human reviewers.

The fiction uses clear cues and leading assistant questions, which limits its
realism. The new candidate used 14 rows from cases 01, 02, 05, 09, 10, 11, and
12; the other ten training rows remain unused. It joined the earlier 242-row
mix for a 256-row run starting from synthetic-v2. The run also changed row order
and optimizer steps, so it cannot isolate the 14 fictional rows' effect. See
the [contrast results](private-sft-contrast-results.md); the frozen gate failed.
