# TraceMem handoff

Updated 2026-10-15 UTC. Verify by: 2026-10-15. This checkpoint covers a static
longitudinal v2 data revision. Delivery and exact-head CI state live in
`.local/active-cycle.json`; verify its SHA, signature, and CI result before
calling the slice delivered. Read that file and verify fresh git, process,
worker, and remote-job state before resuming any operational work.

## Resume safely

Keep Context Kit, original conversations, historical receipts, and
`~/.llm-context` read-only. Keep private conversation data, vectors, claims,
and checkpoints local. Reviewed code, docs, and fictional fixtures
may be signed and pushed to `1337mus/tracemem` after a fresh check confirms the
repository is PRIVATE. Do not launch inference, training, Modal, or GPU work
for this slice. Recheck active jobs before any future runtime work; never
duplicate a live call. Training still requires separate approval.

## Current phase and gate

Phase 0 and Phase 1 infrastructure are complete; Phase 2 remains synthetic
diagnostics. No model is selected or trained. No inference was launched for
the v2 slice, and it adds no human labels. Gate A remains unmet with zero
independently verified human labels. The project README architecture and
plain-language explanation are complete.

## Longitudinal v2

V2 is a revision of the frozen v1 slice, not additional independent cases. It
has 12 fictional dev cases, 146 messages, 49 case-scoped sessions, 13
provisional targets, 12 foils, one unlabeled candidate, and 26 review tasks.
Case 10 drops evidence message `m06` from its second target; case 7 moves one
unresolved candidate from foil to unlabeled. Four targets cite multiple
sessions, which does not prove four distinct reasoning requirements. Preserve
the v1 fixtures and prior responses unchanged; all 24 historical artifact pins
still match.

V1 labels are agent-authored and provisional. Its earlier blind review came
from the same Luna model family as the label-authoring model, so it is not
independent validation. V2 has no fresh independent review and no human
adjudication. See the [revision card](longitudinal-revision-v2.md) and
[validation record](validation-longitudinal-v2.json).

## Verification and delivery

The final local source check passed 599 of 600 tests, with one skip; Ruff,
formatting across 133 files, and mypy across 32 files passed. The installed
focused run passed 3 v2 and 5 v1 tests. All 37 runtime `.py`/`.json` files
matched source, the recorded wheel, and installed site-packages byte-for-byte.
Source and installed v2 exports matched exactly: blind SHA-256
`f20591b39319ae2d986243aa3f844b4646f9f47098f6f415c6439623a84a3cd2`, answer
key SHA-256 `88dc55032ce43879af0f1c196a9b3b5746fd83f80c9c6e3804d3a5b98cdf5ebe`.
Use the raw-captured local receipt named in the validation record; the earlier
installed receipt's reconstructed, doubled-brace stdout was later edited and
is superseded, not proof.

Delivery and exact-head CI state live in `.local/active-cycle.json`. Before
treating this work as delivered, verify that its SHA matches the intended tree,
its signature is valid, and its CI result succeeded. A prior delivery's CI is
historical and does not establish the current head's state.

## Next useful work

After exact-head CI, add distinct license-clean scenarios and obtain
independent annotation. Do not build another wrapper around these fixtures or
repeat inference. Check the documentation index for detailed architecture,
experiment receipts, and validation records before repeating earlier work.

## History

The byte-identical pre-compaction handoff is preserved in
[handoff-history-2026-10-15-v2.md](handoff-history-2026-10-15-v2.md). The
[documentation index](README.md) links the detailed records.
