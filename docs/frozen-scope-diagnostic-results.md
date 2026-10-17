# Frozen-scope inference diagnostic

Updated 2026-10-17. This was an inference-only, three-arm comparison. It added
eight independently authored fictional candidates across four scenarios and
replayed the existing 16 private development prompts as controls. No training
was performed.

| Arm | Fictional support | Fictional eligibility |
| --- | ---: | ---: |
| Base | 8/8 | 2/8 |
| Synthetic-v2 adapter | 8/8 | 8/8 |
| Latest private adapter | 8/8 | 8/8 |

The narrow diagnostic gate passed: the latest private adapter matched the
synthetic-v2 adapter on support and eligibility and made no unsupported
autoaccept on these fictional cases. The result covers only eight cases with
provisional agent labels; it does not establish a broader performance gain or
justify selecting a model. The fiction uses clear cues, including leading
assistant questions, and received no human review.

All 72 responses were valid JSON and ended at EOS. On the original 16 private
controls, all three arms exactly matched the earlier receipt's input and output
token sequences. The latest private adapter's earlier errors remain: 0/2
uncertain-scope cases correct and one unsupported claim autoaccepted.

Ten bounded source scans yielded no new real-source admissions. This zero yield
does not show that a larger archive is unusable. The 14-row fictional challenge
later joined the earlier 242-row mix in a separate comparison from synthetic-v2;
see the [contrast results](private-sft-contrast-results.md). The reserved test
source remains untouched. Extraction and downstream QA were not measured, and
no model was selected. See the [aggregate report](frozen-scope-diagnostic-results.json)
and [fictional data card](negation-scope-data-card.md).
