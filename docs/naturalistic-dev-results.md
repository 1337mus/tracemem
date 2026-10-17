# Naturalistic fictional development results

This inference-only comparison scored 16 fictional candidates across eight
scripted scenarios. The run also included 16 existing private controls; the
results below describe only the fictional development cohort. No training was
performed, and the dataset is permanently excluded from training.

| Measure | Base | Synthetic-v2 adapter | New 256-row adapter |
| --- | ---: | ---: | ---: |
| Useful accepted | 5/7 | 6/7 | 6/7 |
| Supported retained | 8/8 | 7/8 | 7/8 |
| Unsupported rejected | 5/6 | 6/6 | 6/6 |
| Uncertain support correct | 1/2 | 0/2 | 0/2 |
| Eligibility exact | 8/16 | 14/16 | 15/16 |
| Unsupported eligible autoaccepts | 0/5 | 0/5 | 0/5 |

All 48 fictional-cohort outputs were valid and ended at EOS. The labels include
eight supported, six unsupported and two uncertain support references. There
were no uncertain-eligibility references.

The prospective dev gate passed: the new adapter matched the prior adapter on
supported retention and unsupported rejection, improved eligibility exact
match from 14/16 to 15/16, and made zero unsupported eligible autoaccepts. This
narrow pass is separate from the failed private-training gate; it does not
reverse that result or select a model. Useful acceptance is descriptive, not a
criterion in this gate.

Labels are agent-provisional with zero human reviewers. These scripted,
nonrepresentative scenarios are development data, mostly based on final-turn
evidence, and are not a final test. The result makes no product claim and does
not measure extraction efficacy or downstream answer quality. In particular,
both adapters got 0/2 uncertain-support labels correct, below the base at 1/2.

See the [aggregate JSON](naturalistic-dev-results.json) and the separate
[private SFT contrast](private-sft-contrast-results.md), whose training gate
remains failed.
