# Q1 M8 — Integration, scientific verification and compliance audit

## Post-M8 final completion addendum

M8 baseline `5ea9e47` is preserved. Final bounded recovery confirms **0 → 0**
eligible real quotes; no real date/expiry was recovered. Exact primary waterfall:
606 input → exclude 94 index → exclude 23 unknown-underlying SSE → reject 489 for
unverified date-effective metadata → 0. Overlapping unit/convention failures are
not added to these counts. Even candidate ETF slices have at most three strikes
(one paired strike), below existing SVI's five-strike minimum. q=0, SHIBOR
interpolation and volume-based filters are permissible documented choices, but
cannot repair contract evidence or strike support. No empirical fitting, new
downloads or synthetic retraining was justified. See `q1_final_recovery.md` and
`q1_final_recovery_evidence.json` for exact counts, external-field gaps, CFFEX
assessment, minimum viable sample, commands and hard-stop decision.

The session inventory now has two local Q1 sessions; use the updated two-session
export instructions in `q1_final_recovery.md`, not the older one-session snapshot
in §5 below. Historical M8 tests/results remain unchanged. Final targeted results
are separately recorded in `q1_final_verification.json`. Verdicts remain unchanged.

## Verdict and scope

**SOFTWARE: PARTIAL — EMPIRICAL: BLOCKED — GROUP_SUBMISSION: NOT_READY.**

Equivalent original labels: **SOFTWARE_PARTIAL / EMPIRICAL_BLOCKED /
SUBMISSION_NOT_READY**. Passing software tests does not establish historical-data,
market-performance, global arbitrage-free or group-deliverable compliance.

Q1 only. M4/M6 software checks passed; M5/M7 limitations are preserved. This audit
does not retune or retrain completed experiments, rewrite commits, create a
submission tag, modify Q2/Q3, author `report.md`, or fabricate `SESSION.md`.
The only shared implementation addition is the explicitly requested read-only
`code/compare.py`; group members must review its input contract before integration.

## 1. Actual evidence and comparison contract

The audit reads actual M1/M1.5 manifests, M2 data/coverage/registry, M4/M5/M6/M7
metrics and checkpoints. Original artifact SHA-256/size manifests are validated.
`tests/q1/fixtures/verified_runs/` contains **lossless gzip snapshots** of eight
canonical metrics/gate runs, their original manifests, M4 diagnostics and seven
neural checkpoints. These are explicitly synthetic SOFTWARE evidence and blocked
gate records, not historical market-data replacements. The capture verified 102
original indexed artifacts and stored 24 snapshot files. Large original HTML is
indexed, not copied or regenerated. The original ignored experimental outputs are
preserved. A clean checkout can replay the tracked numeric evidence and perform
checkpoint inference without any original ignored files or model fitting.

`code/compare.py` validates the course `{params,metrics,fit_seconds,history}` schema,
requires question-prefixed dataset IDs, rejects contradictory synthetic/empirical
provenance and isolates question/mode partitions. Rows retain run ID, dataset,
date/expiry/maturity, model, seed, cohort, observed/eligible/train/validation counts,
expected/predicted counts, coverage, relative median/p95/max, price/IV MAE/RMSE,
separate butterfly definitions, calendar/forward counts and diagnostic domain,
time, termination, convergence, failed fits and reasons. Missing quantities remain
null. Neural MAX_EPOCHS is not labelled optimizer convergence. Shared neural fit
time is attributed across expiries and repeated fold rows are marked, not summed.

No ranking mixes questions, datasets, modes, runs or cohorts. Scoped M7 rankings
require identical held-out-ID hashes and **all four models at 100% coverage** on
the same run/date/expiry/seed; the failed L-BFGS cohort is not ranked. Repeated M5
source runs remain distinct, never averaged. Other questions' schema-compatible
metrics are preserved generically without assigning Q1 semantics or changing
their files. Missing/unknown provenance is not called empirical.

The main checkout initially contains nine saved metrics files (eight canonical
runs plus the preserved initial M5 run). Canonical report tables use M5's final
reporting run, not an average of its two timing measurements. The comparison has
131 normalized rows and zero input-schema/provenance issues; canonical snapshot
replay has eight runs. Original figure references are included in the JSON index.

## 2. Examination compliance matrix (§§0–3)

Legend: PASS = verified within stated scope; PARTIAL = implemented/evidenced with
gaps; FAIL = current submission/data do not meet the requirement; UNVERIFIED =
not established. Software and empirical columns are deliberately independent.

| Exact examination requirement | Software / implementation | Empirical / delivery | Source | Relevant test | Actual artifact / remaining gap |
|---|---|---|---|---|---|
| §0.1 / §1: complete agent contract before first session | Current contract PASS | Historical compliance FAIL | `AGENTS.md`; `docs/decisions/m0-environment.md` | Git history inspection | Initial `ff74328` contract was empty; populated in `41ba1b9` after initial assessment. Preserve deviation |
| §0.2 / §2.2: structured repository | Q1 PASS | Group UNVERIFIED | `code/q1/`, both Q1 approach scripts | Q1 CLI tests | Q2/Q3 and all six approaches require their owners' integration |
| §0.3: student-authored report | Support material PASS | FAIL in current checkout / not authored here | `docs/q1_report_support.md` | Human review | Root `report.md` is absent; agent materials do not substitute for it |
| §0.4 / §1: authentic complete sessions | PARTIAL | Group export PARTIAL / root file absent | `scripts/q1_session_export.py` | `test_final_integration.py` Markdown round-trip | One local session identified, JSON/Markdown verified; final M8 cutoff and other members still need collection |
| §2.1: Python 3.13/3.14, uv, pinned dependencies | PASS | PASS in tested Windows environment | `scripts/project.py`, `pyproject.toml`, `uv.lock` | `test_environment.py` | Python 3.13.18, uv 0.13.7; 70 locked entries, 50 compatible installed distributions |
| §2.3: snapshots, provenance, 50 MB policy | PASS checksums | PARTIAL rights/coverage | `q1/provenance.py`, `q1/market_data.py` | `test_market_data.py`, source-audit tests | Committed M1/M1.5/M2 archives; license/redistribution review remains student responsibility |
| §2.5: seed, split, schemas, Rich, self-contained Plotly | PASS | Empirical runs BLOCKED | Q1 CLIs; `code/compare.py` | CLI, artifact, integration tests | Default seed 42; M7 fixed 42/43/44; standard schema; original HTML integrity checked |
| §2.5: comparison and ranked table | PASS with scope limits | No empirical ranking | `code/compare.py`, `q1/comparison.py` | `test_final_integration.py` | `results/comparison.html/.json/.csv`; only fully matched complete M7 cohorts ranked |
| §2.2: <=10-line README and reproduction instructions | PARTIAL | Group review pending | `README.md`, this document | Direct file inspection | Existing 9-line README still describes M0; complete current Q1 commands are here, shared README not changed without coordination |
| §2.6: initial and milestone commits | PASS milestones | Initial full-contract timing FAIL | Git history | Git inspection | Authentic incremental M0–M7 commits preserved; no timestamp repair/tag |
| §3.1: real Chinese ETF, not substituted index data | PASS separation | PARTIAL | `q1/market_data.py` | `test_market_data.py` | 489 known 50ETF rows, 94 separate index rows and 23 unknown-metadata old rows; not a compliant panel |
| §3.1: one month of daily chains | PASS coverage gate | FAIL current archive | `q1/market_data.py` | coverage tests | `data/clean/q1/m2_pipeline/coverage_summary.json`: `complete_month_verified=false` |
| §3.1: at least four maturities on required samples | PASS synthetic fixture | PARTIAL candidates, FAIL strictly eligible scope | M1 inventory, M2 coverage | audit/coverage tests | Four candidate expiries are not four verified complete historical chains; eligible expiries 0 |
| §3.1: at least five dates spanning two years | PASS audit machinery | FAIL current archive | M1/M1.5 audit, M2 coverage | coverage tests | Target ETF evidence 2026-04-23–2026-11-04, 130 sparse dates; no five verified multi-year chains |
| §3.1: identifiers, strikes, expiry, units, historical prices | PASS temporal gates | FAIL required metadata validity | `q1/market_data.py` | dated-metadata tests | Verified effective-dated metadata rows 0; current master cannot be projected backward |
| §3.1: underlying spot and SHIBOR alignment | PASS numerical/data joins | PARTIAL | `q1/rates.py`, `q1/market_data.py` | `test_rates.py`, alignment tests | Exact-date evidence exists; approved zero-curve/day-count/fixing/dividend conventions unresolved |
| §3.2: both calls and puts | PASS | PARTIAL raw evidence, not eligible chains | `q1/pricing.py`, `q1/approach_a.py` | `test_pricing.py`, preparation tests | Known ETF 261 calls / 228 puts; synthetic 164 each; all real calibration rows blocked |
| §3.2: examination put–call parity | PASS | UNVERIFIED calibrated empirical panel | `q1/pricing.py`, `q1/approach_a.py` | parity tests | Original residual and dividend-adjusted residual separately recorded; no price repair |
| §3.2: liquidity/spread/moneyness filtering | PASS implemented | PARTIAL unit/liquidity evidence | `q1/market_data.py`, `q1/approach_a.py` | filtering tests | Volume>=1, available relative spread<=.25; missing bid/ask/OI not fabricated; 604 rows lack these fields |
| §3.2: forward BS call/put pricing | PASS | Convention approval UNVERIFIED | `q1/pricing.py`, `q1/neural_surface.py` | pricing/neural-math tests | NumPy/log-CDF and Torch agree; F/D/r/q are explicit, not silently inferred |
| §3.2: Brent IV on [1e-4,5], divergence reporting | PASS | No eligible empirical inversion attempted | `q1/pricing.py` | IV bound/failure tests | Per-quote IV diagnostics and causes; failures never assigned arbitrary IV |
| §3.2: fixed log-moneyness delivery grid | PASS | UNVERIFIED empirical deliverable | `q1/svi.py`, evaluation modules | grid tests | 201 nodes on [-.5,.5]; observed fixture support [-.4,.4], wings labelled |
| §3.3.1: constrained SciPy SVI, initialization/settings/time | PASS | BLOCKED | `q1/classical_svi.py` | `test_classical_svi.py` | M4 metrics/diagnostics and seven figures; four synthetic fits, no market fit |
| §3.3.1: positive total variance | PASS checks | UNVERIFIED real surfaces | `q1/svi.py`, SVI transforms | positivity/gradient tests | Analytical global minimum and grid checks; a itself need not be positive |
| §3.3.2: examination butterfly, locations/magnitudes | PASS implementation | UNVERIFIED real surfaces | `q1/svi.py`, evaluation adapters | butterfly/neural tests | Complete formula with w''/2 separately labelled, never substituted |
| §3.3.2 / §3.4.4: calendar arbitrage | PASS detection, model outcome PARTIAL | BLOCKED | shared diagnostics; `q1/final_audit.py` | dense finite-difference integration test | M7 seed44 has genuine between-expiry calendar violations; preserve all locations |
| §3.3.3: linear total-variance interpolation | PASS | UNVERIFIED empirical term structure | `q1/svi.py`, `q1/classical_svi.py` | interpolation tests | Linear w, no implicit extrapolation/anchor; neural interpolation supplement not substituted for MLP |
| §3.3.3: negative forward variance | PASS detection, model outcome PARTIAL | BLOCKED | shared forward diagnostics | grid/dense audit tests | M7 dense neural seed44 has 272 negative-forward locations |
| §3.3.4: repricing max and cross-date median/p95 | PASS aggregation | FAIL required multi-date empirical evidence | `q1/classical_svi.py`, comparison adapters | multi-date arithmetic and comparison tests | Original slice/date summaries preserved; one artificial date is not empirical cross-date validation |
| §3.4.1: differentiable SVI, gradients | PASS implementation | BLOCKED | `q1/torch_svi.py` | `test_torch_svi.py` | CPU float64 NumPy/autograd/finite-difference checks |
| §3.4.1: same constraints/objective, convergence comparison | PARTIAL | BLOCKED | `q1/torch_comparison.py` | matched-objective tests | Identical observations/RSS/weights/starts; closed SciPy vs interior Torch domains and stopping rules differ |
| §3.4.1: optimization/recovery outcome | PARTIAL | BLOCKED | M5/M7 ledgers | nonconvergence tests | M5 3/4 L-BFGS fits; Adam 0/4 at original strict M4 recovery tolerance; M7 one L-BFGS failure |
| §3.4.2: <=4-layer positive-volatility MLP | PASS | BLOCKED | `q1/neural_surface.py` | positive-output/layer tests | Default 3 affine layers including output, 337 parameters; smooth activations |
| §3.4.2: squared option-price objective | PASS | BLOCKED | `q1/neural_training.py` | loss/gradient tests | No IV MSE or hidden SVI-target pretraining |
| §3.4.2: exact exam-g and fixed-k calendar penalties | PASS calculus, outcome PARTIAL | BLOCKED | neural calculus/training | analytic/finite-difference tests | Input normalization stays in graph; soft penalties do not ensure global no-arbitrage |
| §3.4.2: mandatory coefficient ablation | PASS execution, limited evidence | BLOCKED | `q1/neural_experiment.py` | four-case tests | A/B identical, C/D identical; butterfly never activates on benign M6 fixture; all end at max epochs |
| §3.4.3: 30% missing-quote study, leakage prevention | PASS protocol, fit outcome PARTIAL | BLOCKED | `q1/robustness*.py` | M7 protocol/runner tests | Quote-level nearest-integer 25/82, seeds42/43/44; same held-out IDs; retained-only preprocessing/validation |
| §3.4.3: interpolation/extrapolation comparison | PARTIAL statistical scope | BLOCKED | M7 support metrics | classification tests | Only two extrapolation legs at one strike/seed; no broad extrapolation conclusion |
| §3.5: indicative <2% and zero interior butterfly | Software diagnostics only | UNVERIFIED / not achieved as market evidence | M4–M7 outputs | numerical tests are not benchmarks | No liquid-market benchmark claim; neural large relative maxima and calendar violations remain |
| §§2.3,2.5: offline clean-checkout reproduction | PARTIAL, scope-qualified | Historical validity BLOCKED | bundle/replay scripts | isolated replay and integrity tests | Tracked snapshots + source can replay results/rebuild M2/infer/check gates; independent retraining deliberately not rerun; first dependency setup needs provisioning |

## 3. Mathematical findings and known limitations

Detailed equations/protocols/result tables and statement-of-work material are in
`docs/q1_report_support.md`. The M5 constraints/stopping caveat, M6 zero butterfly
activation/MAX_EPOCHS and M7 incomplete L-BFGS coverage are not defects hidden by
the comparison. Complete-cohort predictive errors remain unavailable where
predictions fail; available errors always carry coverage.

The seed44 neural dense-grid anomaly is audited using the **existing checkpoint**:
recompute the M6 15-maturity x201-k grid, compare saved counts, differentiate
physical T at fixed k, central-difference with steps 1e-3/1e-4/1e-5/1e-6, and
independently evaluate an offending calendar pair without autograd. The result is
**genuine model behavior, not a verification sign error**. The matched four-expiry
grid has zero available calendar/forward violations but the dense grid has 402
negative-w_T locations, 448 calendar-pair and 272 negative-forward locations.
Soft penalties and sparse expiry-node evaluation do not prove monotonicity.
No weights, coefficients, seed choices or budgets are changed after this finding.

The real archive has 606 observations including non-ETF/unknown records; 489 known
50ETF observations span only 130 recent dates and three strikes. No row has fully
verified historical date-effective metadata. Price units and rate/dividend
conventions remain unapproved. Four candidate expiries do not establish a month
of complete daily chains or five verified dates spanning two years. M1.5 provider
documentation does not constitute a retrieved authorized dataset. Students must
obtain licensed/authorized dated metadata/price snapshots and review conventions.

## 4. Reproduction and integrity

Run from repository root using the managed launcher (Windows AMD64):

```powershell
# First provisioning needs network or a separately provisioned authorized cache.
py -3.11 -B scripts/project.py setup
py -3.11 -B scripts/project.py uv lock --check --offline
py -3.11 -B scripts/project.py uv sync --locked --check --offline
py -3.11 -B scripts/project.py uv pip check --python .venv/Scripts/python.exe

# Read original saved metrics; no training. Fresh output path if one already exists.
py -3.11 -B scripts/project.py uv run --locked --offline python code/compare.py

# Source-only artifact replay: source snapshots are tracked, original results are unnecessary.
py -3.11 -B scripts/project.py uv run --locked --offline python code/q1/validation_bundle.py restore --destination .tmp/q1_replay
py -3.11 -B scripts/project.py uv run --locked --offline python code/compare.py --results-dir .tmp/q1_replay/results --output .tmp/q1_replay/comparison.html

# Equivalent isolated clean checkout from staged/tracked source, no ignored outputs.
py -3.11 -B scripts/project.py uv run --locked --offline python scripts/q1_isolated_replay.py --destination .tmp/q1_m8_clean_checkout

# Full suite ONCE after integration; reuse manifest-verified M4/M5 full CLI runs.
# Small numerical/unit-fit tests still execute; no completed experiment is retrained.
py -3.11 -B scripts/project.py uv run --locked --offline python -m pytest tests/q1 -q --reuse-q1-artifacts
```

Default tests without `--reuse-q1-artifacts` retain their original from-scratch CLI
fixture behavior. The audit-specific flag does not skip mathematical assertions:
it substitutes verified completed CLI artifacts for two full M4/M5 experiment
fixtures, retaining hash/schema/numerical/figure/no-overwrite checks. It requires
the original full HTML/manifests in the main checkout. On a fresh checkout use
the isolated replay/inference audit; do not claim a new full training experiment.

The isolated test shares the already-provisioned pinned dependency environment
read-only, but exports source/archives from the Git index to an empty directory,
blocks network, rebuilds the real M2 CSV from tracked M1/M1.5 snapshots, restores
all canonical numeric results/checkpoints with exact hashes, regenerates the
comparison, rechecks dense violations and executes all four strict empirical gates.
This is **artifact/inference reproducibility**, not a second independent model
training replication. Original milestone HTML is not redundantly regenerated;
its original hashes/index and source renderers remain available. No ignored
parent results, private data, old .tmp files or M7 fit caches are inputs.

## 5. Session export and group handover

Installed OpenCode was inspected: **v2.2.5**, `session list --format json` and
`session export <ID>` (JSON). The documented TUI `session.export` command is
**Export session to editor** (default Ctrl+X then X); no Markdown CLI format flag
is advertised. TUI editor export was not invoked automatically (external-editor
location/tool verbosity cannot be assumed complete).

The project-wide API inventory, including a second empty pagination page, found
one session: **ses_e58f9132affeaic2LlD5RPAtxS**, titled “M0 repository and environment
setup”. Its actual user prompts cover M0, M1, M1.5, M2A, M2 and M4–M8. Mathematical
M3 duties were implemented under M2A; no fictitious separate M3 session is invented.
Other members' machines/projects/sessions are not certified by this inventory.

A raw byte-preserving native JSON export was tested and a mechanical Markdown
render was verified by decoding **every fenced message back to the original**.
The inspected snapshot had 337 messages, 13 user prompts and 436 tool parts,
including full exported tool states. No AI summary replaces the transcript. A
PowerShell line/text capture corrupted one long JSON line; direct subprocess byte
capture fixed it. Preserve this caveat rather than claiming the failed file valid.

Exact final export command (after the final commit/response, into a NEW directory):

```powershell
opencode.cmd session list --format json --max-count 1000
py -3.11 -B scripts/project.py uv run --locked --offline python scripts/q1_session_export.py --session ses_e58f9132affeaic2LlD5RPAtxS --opencode-executable "C:/Users/15479/AppData/Roaming/npm/node_modules/@opencode/cli/bin/opencode.exe" --output-dir .tmp/q1_sessions_final
```

This writes native `session.json`, complete lossless `session.md` and
`export_verification.json`; it never writes root `SESSION.md`. Existing exports
are not overwritten. Inspect/redact actual credentials explicitly, preserve
redaction records and any referenced tool/file attachments, collect other members'
authentic exports, then coordinate the final group `SESSION.md`. The current M8
session is live; its final message cannot already be present in a prior snapshot.
No claim of completed group export is made.

## 6. Measured M8 audit/reproduction evidence

- Targeted integration: **5 passed in 9.96 seconds**; extended integration plus
  reused M4/M5 artifact checks: **16 passed in 79.60 seconds**. No full M4–M7 model
  experiment was retrained. Existing unit/numerical fit assertions still execute.
- Environment: Python **3.13.18**, uv **0.13.7**, offline lock/sync consistency
  passed with no environment changes; **50 installed distributions compatible**.
- Source/data audit verified **110 original source files** under pinned manifests.
  Dataset remains 606 observations / **0 eligible**.
- Lossless verification bundle: **24 compressed snapshots, 6,060,847 bytes** total;
  largest restored file 30,046,173 bytes (<50 MB). Q1-local `.gitattributes` protects
  new snapshot/manifest bytes; it does not alter other questions or old datasets.
- Isolated staged-source checkout: **PASS in 51.4185 seconds**. Snapshot restoration
  0.5669s; true M2 regeneration 9.8793s with byte-identical `quotes.csv`; eight-run,
  119-row comparison regeneration 3.9324s; checkpoint/finite-difference audit
  0.8614s. Four independent empirical CLI gates returned expected BLOCKED exit 2
  in 7.2435 / 12.6032 / 8.2281 / 8.1036s. No ignored original result or model cache
  was an input. The pre-provisioned pinned dependency environment was shared.
- Dense anomaly: worst saved point k=.415,T=1 gives autograd w_T
  **-0.023403562931940847**. Central differences for h=1e-3/1e-4/1e-5/1e-6 are
  -0.0234033988349 / -0.0234035612912 / -0.0234035629140 / -0.0234035629071.
  A separate interior-k calendar pair at k=.355,T=.84703196347→1 drops w by
  **0.001629876721397805**. This is far larger than floating-point sign noise.
- Literal high-confidence secret scan: **233 then-staged/tracked files**, including
  decompressed verification snapshots, and **11 existing Git commits** had zero
  findings for private-key blocks/AWS access-key IDs/long sk-prefixed secrets.
  This is a bounded pattern scan, not proof of absence of all credentials; final
  human session/source review remains required.
- A development comparison draft was preserved in `.tmp/m8_comparison_initial/`;
  the final dashboard adds only safe complete-cohort rankings. No original M4–M7
  HTML was regenerated or overwritten. Full-suite results and actual artifact
  hashes are recorded in `docs/q1_m8_verification.json` after execution.
- Full Q1 regression ran **once**: **264 passed, 0 failed in 205.00 seconds** via
  `--reuse-q1-artifacts`. Mathematical/unit-fit tests still execute; verified full
  M4/M5 CLI artifacts replace unnecessary completed-experiment reruns. These
  passing tests do not erase the scientific/data/submission limitations above.

## 7. Remaining responsibilities and stop

Students must approve real pricing/curve/dividend conventions, obtain/verify the
missing historical ETF data and reuse rights, interpret the numerical findings,
write the scientific report and statement of work, review the shared comparator,
collect final authentic sessions and independently integrate Q2/Q3. Group checks
must cover all six approaches, final report and submission policy before any final
tag. **No submission tag is created here. Stop after M8.**
