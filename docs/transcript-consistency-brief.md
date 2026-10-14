# Next bounded investigation: quiescent transcript imports

Root architecture checkpoint after boundedJSONL reader. Existing documented reader
is streaming and does not guarantee a whole-file snapshot. Next useful Phase1 step:
use fictional append/truncate/inplace-edit/rename-replacement probes during iteration
to characterize which partial inputs could be committed. Do not read real transcripts
or modify existing sources/ContextKit/llm-context. No inference,network orGPU needed.

Start with source code/currentAPI and focused fictional probes, not a redesign. Root
owns riskdecision, acceptance and review; Luna/medium may gather evidence and implement
from an amended file-level brief. No currently active worker. Do not duplicate anyjob.
Preserve the default API contract; do not silently make all live-session reads fail.
A likely useful optional contract is require_quiescent=True plus an explicitCLIflag,
requiring a regular nonsymlink input and descriptor/path metadata agreement at end of
iteration. This can detect ordinary changes but cannot prove absence of same-size,
metadata-restored edits or provide an atomic sourcefilesystem snapshot. Do not claim
stronger guarantees. Reuse existing source-descriptor guards if appropriate after
reading their semantics, rather than copying unrelated database assumptions.

Before implementing, root must pin exact optionalflag, regularfile/symlink policy,
nonblockingopen behavior forspecialfiles, identity/stat checks, error messages and
rollback tests. Keep <=150productionLOC target,partition before500. Genuine initial
RED/GREEN perbehavior; no retroactive baseline replay presented as originalTDD.
Late consistency refusal must trigger existing EvidenceStore transactionrollback.
No schema migration,sourcecopy orpersistent privatecorpus. Root independently checks
unchangeddefault outputs, explicitoptin mutation/refusal andlogicalrollback,then
fullsuiteonce,new approvedsource/wheel,isolatedtests,PRIVATEassert andsigneddelivery.
