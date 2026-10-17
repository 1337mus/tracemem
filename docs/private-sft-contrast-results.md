# Private SFT contrast results

Updated 2026-10-17. The new candidate used 256 training rows: the earlier
242-row mix plus 14 newly authored fictional rows. Training started from the
synthetic-v2 adapter, not the previous private adapter. Row order and optimizer
steps also changed, so this comparison cannot isolate the effect of the 14 new
rows.

| Arm | Useful accepted | Supported retained | Unsupported rejected | Uncertain scope exact | JSON + EOS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base, same call | 4/7 | 7/8 | 7/8 | 2/2 | 16/16 |
| Synthetic-v2, same call | 4/7 | 7/8 | 8/8 | 2/2 | 16/16 |
| Previous private, historical | 6/7 | 8/8 | 7/8 | 0/2 | 16/16 |
| New candidate, same call | 7/7 | 8/8 | 7/8 | 0/2 | 16/16 |

Base, synthetic-v2 and the new candidate were the same-call arms; the previous
private adapter is a historical reference.

The new training pass used 256 microsteps and 64 optimizer steps, taking
335.533268 seconds. The earlier 242-row comparison used 242 microsteps, 61
optimizer steps and 316.333 seconds. For each of the base and synthetic-v2
arms, input and output token sequences matched that arm’s earlier receipt on all
16 prompts.
All 48 same-call responses were valid JSON and ended at EOS.

The prospective gate failed. The new candidate met the useful-acceptance
threshold at 7/7 and retained all 8/8 supported claims, but rejected only 7/8
unsupported claims and got 0/2 uncertain-scope labels right. Those two scope
labels remain policy-sensitive. No human reviewed these provisional labels.

This adaptive development slice contains 16 candidates from eight windows in
two sessions; all seven useful references come from one session. The reserved
test remains unused. This result does not establish extraction or downstream QA
efficacy, and no model was selected. See the [aggregate report](private-sft-contrast-results.json).
