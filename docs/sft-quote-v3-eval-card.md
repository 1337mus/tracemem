# TraceMem quote-v3 synthetic evaluation card

Updated: 2026-10-15 (evaluation freeze v2)

This evaluation consists of separately authored synthetic-fiction dev (24 cases) and test (48 cases) fixtures. The cases were authored for this experiment and are labeled `agent_authored` / `newly_authored_fiction`. Zero human review has been performed. Semantic isolation from any training material or prior model output is not proven. The fixture author did not inspect the restricted training fixtures, prior model outputs, or prior review results while authoring these files.

The training data was frozen before a root review made narrow semantic corrections to the dev/test targets and rubric; this curation happened before any model outputs were generated or reviewed. The 48-case test split contains cases across chronology, subject attachment, and retention/general control (16 each), including 12 deliberately empty cases (4 in each category). The planned test size is small (`N=48`); results are descriptive on this authored fixture and must not be presented as held-out population accuracy or independent human-reviewed performance. The dev fixture contains 24 cases (8 per category), including 6 deliberately empty cases, and remains weighted toward short cases. Nonempty test cases include longer multi-message examples.

The cases target speech/report corrections versus event chronology, ownership versus borrowing or custody, subject attribution and third-party reports, and retention/access versus general control. Expected claims are concise, provisional, non-exhaustive targets. `eval-rubric.json` states per-case distinctions, expected-fact coverage, atomic outcome-critical facts, and paraphrase constraints, but is not an automatic semantic scoring oracle.

The quote-v2 validator now accepts the native `test` split. Both dev and test fixtures are validated directly with their own split names, and all 56 expected quote spans cite the complete referenced user message so the cited text contains the gold claim’s relevant facts, time, and subject. This verifies structural and exact-source constraints but does not establish semantic correctness.
