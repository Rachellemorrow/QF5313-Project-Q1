# M0 environment decision and verification record

## Scope and authorization

The user approved repository-local environment provisioning and narrowly scoped
shared configuration changes. Only Q1 development is authorized. M0 contains no
financial models, market-data downloads, Q2/Q3 implementations, or shared model
utilities. `report.md` remains reserved for the student-authored report.

The examination is preserved unchanged in `docs/project_spec.md` (SHA-256:
`A3EB42B218A35351804F7B71AAEB54521105C710E06F8C2DA2A3DAB3E3F35340`).
The missing former root examination copy cannot be compared retroactively.

## Authentic Git history

- `ff743281c0225e5083eb49ca013aead7fb2974e7`: initial commit; `AGENTS.md` was empty.
- `41ba1b92755ca4eaee9405d74202cbe67b9be454`: populated group contract, committed
  by the user before this provisioning step; the working tree was clean on entry.
- The first assessment session encountered an empty contract. Full-contract-before-
  first-session compliance therefore cannot be claimed. No history or timestamps
  have been rewritten. This document is not a substitute for an authentic session
  export; group-level `SESSION.md` remains a later deliverable.

## Shared configuration decisions

- Pin all nine examination libraries, pytest, uv, and the managed Python patch
  version. Commit `uv.lock` with transitive pins and artifact hashes.
- Keep the supported project range at Python 3.13/3.14 (not 3.15). Actual M0 runtime
  verification targets standard, GIL-enabled CPython 3.13 on Windows x86-64 only.
- Use the public Aliyun PyPI mirror and prebuilt wheels (`no-build = true`), avoiding local
  compilers and source build side effects. No custom GPU index or CUDA toolkit is
  required. GPU acceleration is not a prerequisite for Q1 or other group work.
- `scripts/project.py` uses the installed host Python 3.11 only as a standard-
  library bootstrap/launcher. The `-B` flag prevents writes of host bytecode. It
  verifies the mirrored uv wheel against the official PyPI pinned SHA-256 before
  extracting its executable. The mirror was selected after the direct upstream
  download was too slow; a complete mirrored wheel passed the same pinned hash.
- Project work must run through the launcher. Environment settings apply only to
  child processes; the parent's PATH, profile, registry, and execution policy are
  unchanged. Windows Python registry registration/discovery and executable-bin
  installation are explicitly disabled. No environment activation is needed.
- `.tools/` contains uv and managed Python; `.venv/` contains installed packages;
  `.uv-cache/`, `.cache/`, `.tmp/`, and `.pytest_cache/` hold generated artifacts.
  All are gitignored. Tests check configured paths and actual temporary locations.
  This is application-level isolation, not an OS filesystem sandbox or exhaustive
  audit of every Windows background-service write.
- Host-specific `UV_*`, `PYTHONPATH`, `PYTHONHOME`, and active-venv overrides are
  ignored by the child launcher configuration. Network proxy/TLS settings are not
  changed or logged. pytest third-party plugin autoload is disabled for predictable
  environment checks; future approved plugins must be loaded explicitly.
- Other members may use the common dependencies; no Q2/Q3 source/test/data/report
  files are created or changed. Future dependency additions require coordination.

## Reproduction commands

Run from the root in PowerShell; no policy change is required:

```powershell
py -3.11 -B scripts/project.py setup
py -3.11 -B scripts/project.py verify
```

`setup` verifies/downloads uv, installs the Python patch pinned by `.python-version`
with `--no-bin --no-registry`, and runs `uv sync --locked`. First setup requires
internet and sufficient disk space; managed Python/environments are not portable
between clones and must be recreated, not copied or committed.

`verify` runs uv/Python version checks, `uv lock --check --offline`,
`uv sync --locked --check --offline`, `uv pip check`, and the Q1 environment-only
pytest file through the managed Python. Offline verification works only after
successful provisioning; a fully offline clean-clone installer is not provided.

## Execution record and status

Environment provisioning and verification succeeded. No M1 work was performed.

### Installed versions

| Component | Verified version |
|---|---|
| CPython (standard GIL-enabled build) | 3.13.18 |
| uv | 0.13.7 (316c827e0, x86_64-pc-windows-msvc) |
| NumPy | 2.5.4 |
| SciPy | 1.18.1 |
| pandas | 3.1.0 |
| scikit-learn | 1.9.1 |
| Plotly | 7.1.0 |
| Rich | 15.0.0 |
| NetworkX | 3.7 |
| AKShare | 1.20.27 |
| PyTorch | 2.15.0 (CPU build; CUDA unavailable) |
| pytest | 9.2.0 |

Hardware reported by the runtime: AMD64 Family 25 Model 68 Stepping 1,
AuthenticAMD; Windows 11, build 26200. All ten imports resolved inside `.venv`.
uv resolved 70 lockfile entries across supported environment markers and installed
50 distributions on this Windows/Python combination. No dependency conflicts were
reported by `uv pip check`.

### Executed provisioning and verification commands

These are the actual operational commands, not planned Q1 model commands. Initial
Git/file inspections and the failed earlier attempts are also in the authentic
session transcript; this decision log is not a replacement session export.

```powershell
# Completed mirrored download; checksum matched official PyPI metadata.
curl.exe --fail --location --silent --show-error --connect-timeout 15 --max-time 120 --output ".tmp/uv-0.13.7-win_amd64-aliyun.whl" --write-out "Downloaded %{size_download} bytes at %{speed_download} bytes/second\n" "https://mirrors.aliyun.com/pypi/packages/8e/4f/041b59d2a38807a3e31ef41064cf3e45d2752fe0409d89040d52778049ae/uv-0.13.7-py3-none-win_amd64.whl"
(Get-FileHash -Algorithm SHA256 -LiteralPath .tmp/uv-0.13.7-win_amd64-aliyun.whl).Hash
py -3.11 -B scripts/project.py bootstrap
py -3.11 -B scripts/project.py uv python list --only-downloads --all-versions 3.13
py -3.11 -B scripts/project.py uv python install 3.13.18 --no-bin --no-registry
py -3.11 -B scripts/project.py uv lock
py -3.11 -B scripts/project.py setup
py -3.11 -B scripts/project.py verify
py -3.11 -B scripts/project.py uv --version
py -3.11 -B scripts/project.py uv run --locked --offline python --version
py -3.11 -B scripts/project.py uv lock --check --offline
py -3.11 -B scripts/project.py uv sync --locked --check --offline
py -3.11 -B scripts/project.py uv pip check --python .venv/Scripts/python.exe
py -3.11 -B scripts/project.py uv run --locked --offline python -m pytest tests/q1/test_environment.py -v -s
py -3.11 -B scripts/project.py uv cache dir
py -3.11 -B scripts/project.py uv python dir
```

The uv wheel SHA-256 is
`2835e2ab397d0db86323b9975dce75ecd76aa9cfa0d6a2aea84734aae7f8c9b3`.
The launcher additionally compares an existing uv executable to the verified wheel
before reusing it. `uv lock` is a maintainer operation used to create the initial
lock, not part of ordinary clean-clone setup; use `setup` with `--locked` instead.

### Actual smoke-test results

| Check | Result |
|---|---|
| Exact Python patch, uv version, managed interpreter and `.venv` paths | PASS |
| All nine required libraries and pytest: imports and exact pins | PASS |
| NumPy solve; SciPy Brent root and numerical integral | PASS |
| PyTorch float64 tensor arithmetic and autograd | PASS |
| Plotly inline JavaScript HTML, no external script sources | PASS |
| Rich panel/table rendering, displayed in the `-s` run | PASS |
| pytest execution | PASS: 18 tests, 0 failures |
| `uv lock --check --offline` | PASS |
| `uv sync --locked --check --offline` | PASS: would make no changes |
| `uv pip check` | PASS: all 50 installed packages compatible |
| Configured caches, actual temp files, HTML, evidence and bytecode paths | PASS |
| Parent environment and PATH preserved by launcher | PASS (host-side assertion) |

The initial suite completed in 112.63 seconds; the explicit `-v -s` run repeated
all 18 tests successfully in 72.61 seconds. The latter emitted version/hardware
evidence to `.tmp/pytest-of-15479/pytest-1/test_environment_evidence0/environment.json`
and a self-contained HTML artifact to
`.tmp/pytest-of-15479/pytest-1/test_plotly_self_contained_htm0/smoke.html`.
These are intentionally ignored and regenerated by subsequent test runs.
The README's combined setup-and-verification command was then executed successfully:
all 18 tests passed again in 17.37 seconds. The lockfile SHA-256 was unchanged
before and after this repeat:
`8CC3F7B2C53449904895B4D288A2B00F7837953721FC2386FAB3D24A8931AE31`.
A structured lockfile review checked 70 package entries, 505 SHA-256 artifact
references, and only HTTPS mirror URLs without embedded usernames or passwords.

The launcher, uv diagnostics, and tests verified all configured and observed
project output locations under the repository. No operation intentionally wrote
project files outside it, and no global software installation or configuration
change was performed. There was no whole-machine filesystem tracing baseline, so
this is not a claim of an exhaustive forensic audit of external filesystem writes.

### Failures encountered

1. Direct execution of newly created PowerShell `.ps1` helpers was blocked by the
   existing execution policy. The policy was not changed. Those unused helpers
   were replaced with the standard-library Python launcher.
2. The first uv-wheel download exceeded the shell's 120-second timeout and left
   a partial file inside `.tmp/`. A repository-local resumable curl download was
   started, but was superseded by the faster mirrored download, whose pinned hash
   passed before uv execution. Only this session's redundant upstream curl process
   was subsequently stopped; its partial artifact remains untouched in `.tmp/`.
3. An auxiliary metadata-review one-liner failed because of PowerShell/Python
   quote escaping. Re-running the check using standard input succeeded; no project
   configuration or dependency change was needed.

These issues are resolved workarounds, not current environment blockers. No files
were deleted or Git history rewritten during the continuation.

### Remaining validation boundaries

- Import/numerical smoke tests do not establish historical data availability,
  AKShare endpoint behavior, or scientific/model correctness. Those belong to M1+.
- Future GPU behavior, other platforms, Python 3.14, and Q2/Q3 code are not tested
  by M0. Residual training nondeterminism must be measured in later milestones.
- Do not start M1 until M0 succeeds and the student authorizes that milestone.
