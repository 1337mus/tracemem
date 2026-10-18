# Balanced-support recovery results

Updated 2026-10-17 UTC.

The first approved launch failed during runtime setup before training and
consumed no scheduled rows. The original error cause remains unestablished. A
separate CPU-only public-checkpoint cache and offline loader succeeded, staging
ten public files (about 8.06 GB) in remote-only storage. A reviewed recovery
then completed a 256-row training run and matched comparison.

The deterministic schedule exposed 85 supported, 85 unsupported, and 86
uncertain rows, including repeats. Training consumed 193 unique source rows,
including 37 private rows out of 50 available. Repeated exposures are not new
independent examples. Training used 64 optimizer steps and took 334.441137
seconds.

The comparison used 16 private development controls and 16 fictional
development candidates. Counts compare the historical 256-row adapter with the
balanced-support recovery candidate.

| Measure | Private historical | Private recovery | Fictional historical | Fictional recovery |
| --- | ---: | ---: | ---: | ---: |
| Supported claims retained | 8/8 | 8/8 | 7/8 | 7/8 |
| Unsupported claims rejected | 7/8 | 8/8 | 6/6 | 6/6 |
| Useful references accepted | 7/7 | 6/7 | 6/7 | 6/7 |
| Eligibility labels exactly correct | 14/16 | 13/16 | 15/16 | 16/16 |
| Uncertain eligibility labeled uncertain (private) | 0/2 | 0/2 | — | — |
| Uncertain support labeled uncertain (fictional) | — | — | 0/2 | 2/2 |
| Unsupported eligible claims automatically accepted | 1/7 | 0/7 | 0/5 | 0/5 |

Both frozen gates failed. The recovery improved unsupported rejection in the
private controls and fictional eligibility and uncertain-support counts, while
private useful acceptance and exact eligibility declined from the historical
adapter. No model was selected or promoted.

All 96 outputs across the three comparison arms and two cohorts were valid JSON
and ended with EOS. Base and prior input and output token arrays matched frozen
inference on 32/32 prompts each; the recovery candidate's input arrays matched
on 32/32 prompts.

The aggregates were independently recomputed across all arms and cohorts, with
168 checks passing. Raw receipts were unchanged. Labels are provisional agent
judgments with zero human reviewers. These are small adaptive development sets,
not held-out evaluation; the reserved verifier test was unused. The results do
not support causal, general efficacy, product-quality, or downstream QA claims.

An eligibility-focused audit found 67 uncertain-eligibility exposures. A
54-candidate, 27-scenario fictional training set with joint 3×3 balance is being
authored; blind labels are required before training. Claude metadata
verification is complete: five older leads included three already represented
in development and two unverified; no new private rows were admitted. The
[machine-readable aggregate](balanced-support-results.json) records the
shareable counts.
