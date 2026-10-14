# Private GitHub prototype validation

2026-10-14. Raj explicitly requested a project under his personal GitHub profile,
private until he chooses to publish later. Root created and verified private
1337mus/tracemem; no files uploaded yet. Local prototype, no Linear/PR required
for initial repository bootstrap. No private sources or .local files committed.

Own only .github/workflows/ci.yml and docs/ci.md. Add one push/pull_request/manual
GitHub Actions validation job on Ubuntu with Python3.12 (dev locks target3.12),
contents:read, no secrets, no Modal credentials/models/inference/training. Use
pinned full official checkout/setup-python action commit SHAs, verifying tags
through read-only public GitHub metadata. No third-party actions, uploads or
private Context Kit checkout. Do not expose caches/artifacts of private content.

Set up .venv, install requirements-dev.lock + requirements-modal.lock, then
make check PYTHON=.venv/bin/python. The one optional real Context Kit integration
may skip because ctxkit is intentionally not installed; the real Modal SDK
serialization test should run offline without credentials. Inspect test/runner
code to ensure no network/GPU call before finalizing workflow.

Build a wheel with installed setuptools via local build backend or pip wheel
--no-build-isolation --no-deps; install --no-deps into a separate fresh venv.
Run console + module demos and experiments/smoke_installed.py from outside checkout
with PYTHONPATH/PYTHONHOME unset. Use absolute source root, system-temp fresh work
directory to satisfy smoke guard; do not weaken guard. Include installed new
ChatGPT adapter/snapshot test modules copied outside checkout when stable.
Use set -euo pipefail so test/build errors fail job. No silent failure/continue-on-error.
No extra broad reruns: root runs final full suite/package locally once production
sources stable, then watches first CI after push.

This is workflow configuration, TDD exemption. Validate YAML parse and shell
syntax using tools already available. Do not install random tools just to lint.
Document exact CI scope, optional Context Kit skip, no cloud inference, inability
to claim hosted CI passed before push, and failure handling. Keep compact. No
README edits, no source/test changes, no commit/push/PR from worker. Return stable
files and inspected command/network evidence for root review.
