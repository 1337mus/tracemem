# Eligibility transfer diagnostic

One inference-only comparison covered 16 fictional train claims in eight
scenarios and 16 unchanged controls, across all nine support-by-eligibility
cells.

| Cohort / arm | Supported retained | Unsupported rejected | Useful accepted | Unsupported-eligible false accepts | Eligibility exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fictional / base | 7/7 | 3/4 | 2/2 | 0/1 | 8/16; uncertain 5/8 |
| Fictional / prior | 6/7 | 4/4 | 1/2 | 0/1 | 12/16; uncertain 6/8 |
| Fictional / latest | 6/7 | 4/4 | 1/2 | 0/1 | 12/16; uncertain 4/8 |
| Adaptive controls / base | 7/8 | 7/8 | 4/7 | 0/7 | 6/16; uncertain 2/2 |
| Adaptive controls / prior | 7/8 | 8/8 | 4/7 | 0/7 | 9/16; uncertain 2/2 |
| Adaptive controls / latest | 8/8 | 8/8 | 6/7 | 0/7 | 13/16; uncertain 0/2 |

On fiction, uncertain-support exactness was 5/5 base, 0/5 prior, and 4/5 latest;
eligibility was 5/8, 6/8, and 4/8, so the prospective target failed. Latest
tied prior on overall eligibility (12/16), retention (6/7), rejection (4/4),
and useful acceptance (1/2). This train-split comparison does not establish
held-out effectiveness. Post-hoc action
counts—not a frozen gate—showed uncertain-fiction autoaccepts of 3/8, 2/8,
2/8 (base/prior/latest); uncertain controls 0/2, 0/2, 1/2; ineligible fiction
2/4, 0/4, 0/4. Wrong labels differ from unsafe autoaccepts. All 96 outputs were
valid JSON+EOS; each arm's input/output tokens matched on 16/16 controls.

One inference call; no training, promotion, or human review. Training-split
claims and adaptive controls are not held-out. Remote/client times were
180.274/198.722 seconds; billing is unknown. The reserved test remains unused.

At the time of this inference-only diagnostic, the reviewed
52-claim/26-scenario set had not yet been trained. A later bounded comparison
has completed; see the [eligibility-transfer results](eligibility-transfer-results.md)
for aggregate scores, run counts and limitations. Independent real-session
evaluation, downstream QA and human label calibration remain necessary; the
reserved test remains unused.
