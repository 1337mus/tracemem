# Claim challenge data card

**Fixture:** [`src/tracemem/data/claim_challenge.json`](../src/tracemem/data/claim_challenge.json)
**SHA-256:** `d1a7e6f91b8ea81c60f318f0472fb85f4e230d3ac5742e1d6d76c0148f6ffe67`
**Snapshot:** 24 cases, 24 distinct scenario IDs, all marked `diagnostic`; 4 have empty expected claims (no retained memory under the challenge policy).

The Qwen3-4B V1 and explicit-v2 prompt diagnostics both used this same snapshot. They
are prompt-development runs, not independent or held-out evaluations. The bounded
[dataset readiness audit](dataset-readiness.json) found zero exact duplicate pairs, but
there are no assigned train/dev/test/held-out cases, so cross-split near-duplicate checks
are vacuous. Distinct scenario IDs do not prove entity, template, or lexical independence.

## Provenance and terms

These cases were newly authored as fictional examples. They contain no copied
private-conversation excerpts and do not represent any person's conversation
history. The fixture and evaluator are distributed under the project’s MIT
license; the license covers these authored synthetic examples as project data.
It does not grant rights to private conversations. See [data governance](data-governance.md)
for the separate rules governing private inputs.

## Annotation conventions

Each case groups exact message records with gold claims. A claim records a
subject, predicate, object, speaker role, attribution, modality, polarity,
optional valid-time dates, and one or more evidence spans. Span offsets count
Unicode codepoints in the exact message text, and the quote must match that
slice. Multiple spans are used when a later message depends on an earlier
referent. Attribution distinguishes direct statements from assistant text,
tool observations, third-party reports, and quoted material.

Valid-time fields accept an exact calendar date or null. A year-only statement
such as “in 2023” does not establish January 1 through December 31: the schema
has no year-precision representation, so both bounds are null. Empty expected
claims denote a no-memory annotation, not proof that the transcript contains no
facts. This policy excludes ambiguous references, questions, and hypotheticals
from retained memory even though the schema includes question and hypothetical
labels. Assistant decoys are omitted as user memories; the assistant’s
assertion in the mural case is not treated as the user’s decision.

## Known limitations

The cases are diagnostic examples, not audited held-out data. They are not
training data, and Gate A has not been met. The evaluator’s exact structured
score compares each output with one gold rendering. A different but faithful
paraphrase, or an extra faithful fact omitted from that rendering, can be
penalized as a mismatch or false positive; exact-match precision is not a
semantic-support judgment.

The initial extraction prompt gives the output schema and citation requirements
but does not define detailed canonicalization rules for subjects and objects.
The challenge also contains no direct case with a malicious instruction
embedded in evidence text. Its 24 distinct scenarios and scenario-bootstrap
intervals support diagnostic comparison only; they do not establish
generalization or significance beyond this authored set. No training is
authorized by this data card. The readiness audit does not establish legal rights,
annotation quality, sample adequacy, or Gate A approval.
