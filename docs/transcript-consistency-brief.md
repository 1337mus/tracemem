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

## Root-pinned implementation contract

Fictional observation completed: append/truncate/inplaceedit/pathreplacement can be
committed with unchangedpathlocator; no defaultconsistency promise currently exists.
Evidence `.local/transcript-consistency-observation.py/json/log`; no realinputs.
Implement only src/tracemem/adapters/_transcript_source.py (newhelper),
src/tracemem/adapters/__init__.py,src/tracemem/cli.py,
tests/test_transcript_consistency.py,docs/transcript-consistency.md and
.local/transcript-consistency-worker-* logs. Rootowns all otherroot/oracle/state files.
No staging/commits/push/fullsuite. Productiontarget<=150LOC,partitionbefore500.

API read_transcript(...,require_quiescent:bool=False). Strict bool validation before
opening/reading source; invalid=>TypeError('require_quiescent must be a boolean').
Existing defaultbranch uses Path.open('rb') exactly and existing bound/decode/parse/
record logic,stats,identities unchanged. Keep newdescriptor mechanics in helpermodule
returning contextmanagedBinaryIO. Optin: rawleafpath lstatmust be regularnonsymlink;
os.open O_RDONLY|O_NONBLOCK plus O_NOFOLLOW/O_CLOEXEC ifavailable; initialfstat regular
and agreeswith initiallstat. Compare tuple(dev,ino,size,mtime_ns,ctime_ns,mode) initially
and at natural fulliterationexhaustion for BOTH open descriptor and currentpathlstat.
Pathmissing/replaced/symlink/specialfile/ordinarymetadatachange or OSError =>
ValueError('transcript source is not a quiescent regular file') without paths/content.
No readsourcechanges. Close alldescriptors on success/failure/earlyclose. Validate
at natural successfulEOF only, not iteratorclose after partialconsumption; preserve
primarydecode/linebound errors. Fullconsumption required for successfulfinalcheck.
No promiseagainst metadatarestoration/atomic snapshots or allancestor symlinkraces.

CLI ingest adds --require-quiescent actionstore_true. Keep existing resolvedsource
for DBoutside-source guard; optin passes original args.input to reader so leafsymlink
is not hiddenby _resolve_source_path.resolve. Defaultpasses resolvedsource asbefore.
CLI alreadychecks issupportedfile; directories/FIFOrefused; missing/alias errors
sanitizedexistingcontract. NewemptyDB schema mayremain onrefusal,asdocumented.
No newreceipt/statsfield,storagechange or schema migration.

Tests selfcontainedfictional realfiles/stores: stableClaude+Codex default==optin
records/stats exact; strictinvalidbool preopen; all four mutators appliedwhileactual
store consumesgenerator andrefusal rollsback existingheads/newrevisions/rows; default
mutationsstillaccepted perexistingcontract; API+CLIleafsymlink refusedoptin while
existingdefault stillworks; FIFO/directory/missing refusedwithoutblocking;
O_NOFOLLOW/nonblocking flag and openrace regular->special refusal; source unchanged
fornonmutatingreads; iteratorclose releasesfd,originalinvalidUTF8/oversize errors
preserved; CLIexplicitflag path reachesreader and sanitizedfailures. Avoid bulk
boilerplate or mockonlyproofs; use narrowstream/descriptor instrumentation onlywhere
needed. Genuineincremental RED/GREEN perbehavior,not retrospectiveTDD claim. Root
probe .local/transcript-consistency-root-oracle.py already authored; don'tedit/run.
Focused new+existingadapter/bounds tests,Ruffcheck/format,strictsourcemypy. Release
files/hashes/exactchronology. Rootowns independentreplay/fullsuite/wheel/delivery.
