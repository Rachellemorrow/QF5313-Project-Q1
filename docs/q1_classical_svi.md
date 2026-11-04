# Q1 M4 — Classical SVI calibration and arbitrage diagnostics

## Outcome and evidence boundaries

- **SOFTWARE_VERIFICATION: PASS** — constrained calibration, numerical recovery,
  diagnostics, reporting, figures, CLI and the strict empirical gate were exercised.
- **EMPIRICAL_REQUIREMENTS: BLOCKED** — the registered M2 archive still has zero
  strictly eligible calibration records. No real-market SVI fit was attempted.

All synthetic inputs/results are software fixtures, not historical market data or
empirical benchmarks. One artificial valuation date cannot establish the course's
multi-year or daily-chain requirements. M1/M1.5/M2 archives, M2A libraries, shared
configuration, dependencies and Q2/Q3 were preserved without modification.

## 1. Prerequisite audit against examination §3.2

The existing numerical suite was run **before implementation: 88 passed**.
Passing that suite does not establish empirical compliance with every requirement.

| §3.2 requirement | Existing implementation | M4 action / remaining evidence gap |
|---|---|---|
| Examination put–call parity residual per strike | `q1.pricing.examination_parity_residual` | Added dated expiry/strike/unit call–put pairing and a parity ledger; missing/ambiguous pairs and failures remain explicit |
| Dividend convention kept distinct | Separate adjusted residual already exists | Report both residuals; never replace the examination expression; synthetic q is explicit, empirical conventions still blocked |
| Quote filtering rule justified by available liquidity | M2 implements quality/volume/spread rules, optional moneyness and rejection ledgers | Strict M2 eligibility retained; M4 adds declared log-moneyness support and paired-side policy. Missing spreads are warned about, not invented. Real unit/convention validation remains unresolved |
| Forward-price Black–Scholes call and put equations | `q1.pricing.black_scholes`, call/put helpers and F/D helpers | Reused unchanged for inversion/repricing; no duplicate pricing model |
| Brent inversion on `[1e-4,5]` for both sides, with divergence cases | `q1.pricing.implied_volatility` and its diagnostic result | Added per-input IV ledger, including failures and reasons; unsuccessful inversions are not silently dropped |
| Fixed log-moneyness grid per expiry and deliverable surface | `q1.svi.fixed_log_moneyness_grid` and interpolation primitives | Added calibrated slice evaluation and total-variance-interpolated surface on a documented common grid; only synthetic verification was executed |

Small software prerequisite gaps were therefore **integration/reporting gaps**:
pairing, per-slice quote preparation, surface orchestration and end-to-end reporting.
The formulas themselves were reused, not refactored. Empirical verification duties
remain unfulfilled because data readiness is not implied by mathematical tests.

## 2. Files and separation of responsibilities

- `code/approach1_q1_svi.py`: course-compliant CLI and output writer.
- `code/q1/classical_svi.py`: constrained least squares, analytic residual Jacobian,
  repricing, arbitrage/term-structure verification and cross-date summaries.
- `code/q1/approach_a.py`: mode boundaries, strict M2 gate, pairing/IV ledgers,
  grouping, weights and metrics orchestration.
- `code/q1/approach_a_figures.py`: seven self-contained Plotly HTML views, each
  clearly labelled with execution origin.
- `code/q1/verification_fixtures.py`: explicitly synthetic development generator.
- `tests/q1/fixtures/synthetic/m4_svi_verification.json`: labelled fixture definition;
  no synthetic observations are written to the real-data archive.
- `tests/q1/test_classical_svi.py`, `test_approach_a.py`: new software verification.
- `docs/q1_m4_verification.json`: compact evidence extracted from the actual saved
  runs, with source artifact hashes and separate software/empirical statuses.

Pipeline: **load -> verify -> split -> fit -> evaluate -> save**. Splitting means
exact valuation-date/expiry slices, not a random train/test holdout. Repricing is
explicitly **in-sample**; it is not an out-of-sample experiment or ablation.

## 3. Calibration objective, constraints and solver

Fit total variance, not volatility or option prices, for Approach A:

```text
k_i = ln(K_i/F(T))
w_i = sigma_IV_i^2 * T
w_model = a + b*[rho*(k-m) + sqrt((k-m)^2 + eta^2)]
residual_i = sqrt(weight_i) * (w_model_i - w_i)
reported objective = sum_i residual_i^2
SciPy cost = reported objective / 2
```

Both call and put inversions are retained. Each paired strike receives total
weight one, split equally between its observations, preventing a duplicated side
from silently increasing that strike's influence. Weights and their policy are
recorded. No hidden butterfly/calendar penalty or arbitrage repair is included.

`scipy.optimize.least_squares` uses `method='trf'`, analytic Jacobian,
`x_scale='jac'`, linear loss, `ftol=xtol=gtol=1e-12`, default `max_nfev=2000`.
At least **five distinct strikes** are required; many repeated C/P observations
at fewer strikes do not pass that gate. Five is only a numerical minimum, not a
claim of reliable parameter identification from a sparse market smile.

The optimized coordinates are `(v_min,b,rho,m,eta)` with:

```text
a = v_min - b*eta*sqrt(1-rho^2)
v_min >= 1e-10, b >= 0
-1+1e-8 <= rho <= 1-1e-8
eta >= 1e-8; m unbounded
```

Thus the analytic global minimum of w is positive, rather than incorrectly
requiring `a>=0` or relying only on a few grid nodes. Raw parameter reconstruction,
global positivity and positivity at quotes/evaluation points are checked after
optimization. Float64 cancellation or an unsuccessful solver is a failure, not a
silently accepted fit. These numerical safety margins are documented software
settings, not empirically optimized hyperparameters.

Default initialization uses `v_min=max(0.5*median(w),10*positive_floor)`, b=0.1,
rho=0, m=median(k), eta=0.2. It does not use fixture truth. Explicit initial raw
parameters are supported and rejected if infeasible, not silently projected.

Every attempted slice has a record: initialization, raw/transformed parameter
order, constraints, objective/weights, solver configuration, nfev/njev, actual
objective-evaluation history, final RSS/SciPy cost, fitted candidate parameters,
global minimum, wall time, convergence and failure reason. Evaluation history is
labelled as function evaluations, **not invented accepted-iteration counts**.
Nonconverged candidates remain visible but are not used as successful surfaces.

## 4. Quote preparation, parity and fixed grids

For this software protocol the common evaluation grid is **201 points on
[-0.5,0.5]** in ln(K/F); the same interval is the declared additional fitting
moneyness band. Development quotes cover 41 paired strikes on [-0.4,0.4]. Curve
evaluation outside that observed strike support is model-wing evaluation, not
additional observations. A later empirical protocol must review grid support and
filter choices rather than treating synthetic verification as methodological approval.

Empirical mode loads only `q1_510050` through M2's registered, checksum-verified
loader. It rechecks strict eligibility, real origin, units, complete effective-dated
metadata and bounds, prepared numerical quantities, source digests, target ETF and
daily-close provenance. It never uses `partial_diagnostic_quotes.csv` as a fallback.
Timestamped latest-quote snapshots are not silently mixed with daily closes.

Within a slice, F/T/D/spot/r/q must agree. The numerical relationship between
spot, r/q, F and D is checked. Additional filtering requires reported volume >=1;
available bid/ask must be ordered and have relative spread <=0.25. Unavailable
bid/ask stays unavailable and is warned about, not imputed. This does not relax
any stronger upstream M2 filter or approve its unresolved source units.

Pairing uses valuation date, exact expiry, strike and contract unit. Missing or
ambiguous C/P multiplicities are recorded. If one side fails inversion/filtering,
the counterpart is flagged rather than silently creating a one-sided fitted
strike. Original examination parity and separately labelled dividend-adjusted
parity are reported; neither residual is used to repair prices or automatically
force parity. Every failed inversion includes the reused Brent diagnostic.

## 5. Arbitrage and term structure

Existing M2A SVI values/derivatives, positivity and both butterfly functions are
reused unchanged. For every successful fitted slice, diagnostics retain the full
grid/values, violation counts, locations and magnitudes for:

- **Examination butterfly formula**, without the curvature term.
- **Separately labelled complete butterfly formula**, including w''/2.
- Positive total variance on the grid and the analytic global minimum.

All ordered maturity pairs at each valuation date are compared at common k.
Adjacent total-variance slopes are checked for negative forward variance and every
location/magnitude is retained. Nothing is clipped, arbitrage-repaired or suppressed.

Term interpolation reuses the M2A piecewise-linear **total variance** routine.
Queries are successful node maturities and their midpoints; endpoint consistency
is measured explicitly. No implicit T=0 anchor or time extrapolation is allowed.
Failures remain in the slice ledger; their parameters are never filled. A surface
uses only successful maturity nodes and exposes missing/failed slices. Fewer than
two valid distinct maturities yields an explicit insufficient-term-structure status.

These are finite-grid checks, not a global arbitrage-free proof. Butterfly checks
are reported on fitted expiry slices; they do not establish all possible k/T values
or guarantee absence of new butterfly violations at every interpolated maturity.

## 6. Repricing and cross-date statistics

At each valid input strike, fitted w gives `sigma_fit=sqrt(w/T)` and the existing
forward Black–Scholes routine reprices the original call or put. Absolute premium
errors and relative errors are retained quote-by-quote.

Policy for small premiums: positive prices below **1e-8 per underlying unit** retain
absolute errors but have **null relative error**, with an explicit excluded-near-zero
denominator status. The denominator is not floored and errors are not capped.
Nonpositive/invalid prices and repricing failures are reported separately.

Per-slice maximum relative errors are reported first. Cross-date output includes:

1. Per-expiry median/p95 of complete slice maxima across valuation dates.
2. For each date, the maximum across available expiry slices and completeness flags.
3. Primary median/p95 across **complete date maxima**. Dates with failed slices,
   rejected input quotes, failed repricing or undefined denominators are disclosed
   and excluded from that primary statistic, not silently counted as zero.
4. Separately labelled pooled-quote median/p95, not substituted for cross-date
   statistics. Partial slice/date values remain visible in the ledger.

## 7. Execution modes, outputs and comparison isolation

Required CLI options are supported: `--dataset`, `--seed` (default 42),
`--output-dir`, `--help`; additional controls are `--mode` and `--max-nfev`.
Dataset/mode combinations are enforced; an empirical dataset cannot be selected
in development mode. Seed is recorded; no random holdout/noise is injected and
initialization is deterministic. Fit wall time is measured, not reproducibly fixed.

Development output:

`results/verification/approach1/q1_synthetic_svi_verification/`

This is deliberately **outside** the standard empirical `results/approachN/*`
comparison path. `params.empirical_comparison_eligible` is false and every figure
has a conspicuous synthetic-not-empirical title/annotation. Future comparison code
must continue to check mode/status/eligibility, not indiscriminately recurse over
all `metrics.json` files.

Empirical output:

`results/approach1/q1_510050/`

Both modes use exactly the examination metrics top-level schema:
`{params: {}, metrics: {}, fit_seconds: 0.0, history: {}}`. Detailed diagnostics,
all slice failures and quote preparation reasons are retained. Output runs are
immutable: existing directories are refused rather than overwritten. Files have
an artifact manifest with SHA-256 hashes. Runtime `results/` is already gitignored;
the code/fixtures and compact labelled verification record are committed, not large
inline-JavaScript copies. Recreate results using the documented commands.

### Seven development figures produced

- `volatility_smile.html`
- `total_variance.html`
- `butterfly.html`
- `calendar.html`
- `volatility_surface.html`
- `market_model_prices.html`
- `repricing_error_distribution.html`

All use `include_plotlyjs=True`, do not open a browser, and were checked for inline
JavaScript/no external script sources. Failed/insufficient views are annotated,
not replaced by invented data. The surface does not extrapolate time. Repricing
plots preserve call/put labels. Full diagnostics and `metrics.json` are also saved.

With zero eligible empirical rows, the empirical run writes only a BLOCKED metrics
record, eligibility report and artifact manifest. Fit count/time are zero and
repricing statistics are null. **No empirical figures or calibrated parameters are
invented.** A blocked run's own software status is PARTIAL because it only exercised
the gate; the overall SOFTWARE_VERIFICATION PASS below comes from the separate
development run and full tests, not a claim of an empirical fit.

## 8. Actual verification evidence

Environment: locked Python **3.13.18**, SciPy **1.18.1**, NumPy **2.5.4**,
Plotly **7.1.0**, pytest **9.2.0**, uv **0.13.7** on Windows AMD64.

- Prerequisite numerical suite before implementation: **88 passed**.
- New Approach A suite: **23 passed**, 30.06 seconds.
- Full relevant Q1 suite: **206 passed, 0 failed**, 94.30 seconds.
- Saved metrics schema, all artifact hashes, seven self-contained figures, origin
  isolation and actual strict empirical blockage were inspected.

Synthetic smoke: **328 call/put inputs**, 164 paired strikes, **one artificial
valuation date**, four artificial expiries. All four fits converged, with no IV
or preparation failures. No evaluation-grid butterfly (either definition), calendar
or negative-forward-variance violations occurred for this benign fixture; separate
tests deliberately construct violations and assert their locations/magnitudes.

| Synthetic expiry label | nfev | Weighted RSS | Global minimum fitted w |
|---|---:|---:|---:|
| 2000-02-02 | 11 | 1.4414974867186315e-26 | 0.019447270417005263 |
| 2000-04-02 | 11 | 1.0113609965054691e-26 | 0.036463632031884685 |
| 2000-07-01 | 10 | 2.2774231033348154e-26 | 0.060772720053135755 |
| 2001-01-02 | 18 | 4.7189609179845875e-26 | 0.09723635208501996 |

Measured maximum raw-parameter recovery error: **2.2937207688755734e-12**.
Maximum total-variance error versus known fixture: **1.4066525722000733e-13**.
Maximum relative repricing/date-max error: **2.4260657831879317e-12**.
Median/p95 date maxima coincide because there is only **one synthetic valuation
date**; a separate multi-date unit example tests the aggregation arithmetic.
Measured total fit time in this actual run: **0.05559239999274723 seconds**.
These are software verification numbers, not market calibration benchmarks.

Actual empirical run: **606 input rows, zero strictly eligible rows, zero attempted
slices**, status **BLOCKED**, zero empirical figures. The partial diagnostic preview
was not used and metadata/rate/dividend/unit gates were not relaxed.
No real quotes were inverted or repriced. Zero IV/fit failure counters in this
blocked run mean **no attempts**, not that the 606 input quotes passed numerical
or empirical validation; the eligibility report records their rejection reasons.

## 9. Exact executed commands and exit behavior

```powershell
py -3.11 -B scripts/project.py uv run --locked --offline python -m pytest tests/q1/test_pricing.py tests/q1/test_svi.py tests/q1/test_math_accuracy.py -q
py -3.11 -B scripts/project.py uv run --locked --offline python -m pytest tests/q1/test_classical_svi.py tests/q1/test_approach_a.py -q
py -3.11 -B scripts/project.py uv run --locked --offline python code/approach1_q1_svi.py --help
py -3.11 -B scripts/project.py uv run --locked --offline python code/approach1_q1_svi.py --mode development --dataset q1_synthetic_svi_verification --seed 42 --output-dir results
py -3.11 -B scripts/project.py uv run --locked --offline python code/approach1_q1_svi.py --mode empirical --dataset q1_510050 --seed 42 --output-dir results
py -3.11 -B scripts/project.py uv run --locked --offline python -m pytest tests/q1 -q
```

Reruns must use a **new repository-local** output base: the original saved run is
preserved. This milestone performed no downloads or dependency changes.

The direct CLI returns **2** for expected empirical BLOCKED, 0 for a successful
development run, and 1 for calibration/verification failure. The pre-existing shared
`scripts/project.py` launcher collapses any child nonzero status to **1** while its
message retains child status 2. An initial PowerShell smoke check incorrectly
expected the wrapper to preserve 2; that command check failed. We then verified the
saved BLOCKED status/zero eligibility and the direct CLI exit through tests. The
shared launcher was **not modified**. Treat the wrapped empirical command's nonzero
status as expected only after checking its saved BLOCKED record, not as blanket
permission to ignore other errors.

## 10. Remaining limitations and stop

SVI is nonlinear and can be ill-conditioned/non-identifiable on sparse or nearly
flat smiles. A single deterministic initialization and the numerical lower-bound
margins are recorded, not claimed universally optimal. Solver success is distinct
from arbitrage absence, repricing quality or out-of-sample validity. No violations
are suppressed and no empirical <2%/zero-butterfly target is claimed as achieved.

The real-data blockers remain date-effective historical metadata/adjustments,
complete daily and two-year ETF chains, verified units, approved pricing/curve and
dividend conventions, and authorized offline data rights. The implemented classical
software does not resolve those prerequisites or complete group Q2/Q3.

**Stop after M4. No PyTorch SVI, neural surface or next milestone has started.**
