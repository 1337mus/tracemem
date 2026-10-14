# LongMemEval dataset rights metadata

Historical metadata-only audit, checked 2026-10-14 against the public Hugging Face dataset API, pinned tree metadata, and dataset README for [`xiaowu0162/longmemeval-cleaned`](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned). The resolved dataset commit is [`98d7416c24c778c2fee6e6f3006e7a073259d48f`](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/tree/98d7416c24c778c2fee6e6f3006e7a073259d48f).

**What the public metadata says.** The dataset API returns the tag `license:mit` and `cardData.license: mit`; the README YAML front matter also says `license: mit`. This is a dataset-card declaration. The pinned tree lists `.gitattributes`, `README.md`, and three benchmark JSON files, with no standalone `LICENSE`, `LICENSE.md`, `LICENSE.txt`, or `COPYING` file. Therefore no license text or license terms were available to inspect in the dataset repository at this commit.

The API description and README say this dataset replaces the original LongMemEval dataset and removes noisy history sessions that interfere with answer correctness. The README points to a Google Sheets change log for more detail. That external document was not fetched. The inspected metadata does not identify the provenance, authorship, or rights and permissions for every constituent conversation or record.

**Assessment.** This adds a Hugging Face MIT card declaration and a narrow description of the cleaning relationship to the original benchmark. Constituent provenance and third-party permissions remain unresolved. The MIT license found in the LongMemEval code repository supplies no additional evidence about those permissions.

The tree lists oracle at 15,388,478 bytes, S at 277,383,467 bytes, and M at 2,737,100,077 bytes. At the time of this metadata audit, S and M exceeded the prototype bytes adapter's 64 MiB input cap. A streaming file scanner has since been added and tested on synthetic inputs within its declared limits, but neither real S nor M has been tested. These tree sizes are metadata, not measured full-release parsing or retrieval results.

**Historical fetch scope.** The metadata audit fetched only public API metadata, the tree listing, and the 626-byte dataset README. At that time, no benchmark JSON or conversation content, model weights, linked change log, or upstream source code was fetched or executed. Tree metadata reported JSON object sizes without requesting file bodies. The machine-readable [metadata-only receipt](longmemeval-rights.json) remains unchanged and records that historical scope.

A later local-only format-compatibility probe fetched the pinned oracle object. Its first restricted request ended with `URLError`, before an HTTP status or body; one root network-enabled GET returned HTTP 200 and verified the expected Git LFS size and SHA-256. That probe is described in [LongMemEval format compatibility](longmemeval-compatibility.md). It does not change the historical metadata receipt or add evidence about constituent-record permissions. No benchmark content or question identifiers are reproduced here, and no new rights conclusion is made.

Exact request URLs, response sizes, HTTP statuses, SHA-256 digests, and cached metadata are recorded in [the machine-readable receipt](longmemeval-rights.json) and `.local/longmemeval-rights/receipt.json`. The raw cache is local and ignored by git.
