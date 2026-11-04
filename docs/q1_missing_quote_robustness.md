# Q1 M7 — Quote-level missing-quote robustness

**SOFTWARE_VERIFICATION: PARTIAL** — pipeline/test/artifact verification completed,
but one L-BFGS fit did not converge and supplementary neural checks find calendar
arbitrage between expiry nodes. **EMPIRICAL_REQUIREMENTS: BLOCKED**.

## Scope and prespecified complete protocol

Q1 only, examination §3.4.3. M4/M5 solvers and M6 mathematics/Adam training are
reused, not reimplemented. No external source search, new model architecture,
paired-deletion sensitivity, hyperparameter search or M8 integration is included.
The working tree was clean at baseline `0a46178bc87b3dde0846558c4dc5cf81b564945a`.
The strict registered archive was checked: **606 inputs, zero eligible quotes**.

The following complete configuration was specified before the expensive experiment:

- Existing labelled M4 fixture: 328 quotes, one artificial date, four expiries,
  41 strikes and both C/P legs per expiry. Never described as historical data.
- **Seeds 42, 43, 44**. Each date/expiry independently samples individual quote IDs
  uniformly without replacement, using one seeded generator over sorted slices.
- Drop `floor(0.30*N+0.5)` quotes per slice (nearest integer, half up). With N=82,
  this means **25 dropped / 57 retained**, not an invented exact 30%. Thus each
  seed has **100 dropped / 228 retained**, realized deletion **30.4878%**.
- Same persisted mask/target IDs for SciPy SVI, Torch Adam SVI, Torch L-BFGS SVI
  and one fresh dated neural surface. Quote-level deletion legitimately separates
  call/put counterparts; no additional post-deletion paired-side rejection occurs.
- CPU float64; one Torch CPU thread. M4 SciPy TRF, analytic Jacobian, initialization,
  safety bounds, max_nfev=2000 and ftol/xtol/gtol=1e-12 remain unchanged.
- M5 defaults unchanged: Adam lr=.03, at most 10,000 updates, decay .5 every 2,000;
  L-BFGS lr=1, at most 1,000 native iterations/2,000 native evaluations, history 50,
  strong-Wolfe, native gradient/change tolerances 1e-12/1e-18. Declared Torch success
  remains RSS<=1e-14 or physical half-RSS gradient infinity norm<=1e-10.
- M6 architecture **2→16→16→1**, tanh, positive softplus, **337 parameters**. Fresh
  initialization per mask/seed. Adam lr=.01, batch size 256, at most 600 epochs,
  validation patience 100 and min_delta=1e-8; train-only normalization/collocation.
- Neural validation: reserve 20% (floor, at least one) of retained strike groups
  per expiry, keeping any available same-strike C/P legs together. Remaining
  retained groups train; each train expiry requires at least five strike groups.
- **User-approved fixed D coefficients (1,1)**, chosen before M7 performance was
  inspected. No M7 coefficient selection. M6's selected C used this same full
  synthetic universe and could have seen future M7 held-out quotes in validation;
  it is therefore NOT reused as a supposedly leak-free selected configuration.
  D is the original both-penalty examination/M6 preset, approved explicitly in
  this session. No M6 weights/checkpoint are loaded for training or initialization.

New implementation files: `code/q1/robustness_protocol.py` (masks/metrics),
`code/q1/robustness.py` (shared model runner/cache), `code/q1/robustness_cli.py`
(execution/artifacts), and `code/q1/robustness_figures.py` (eight figure groups).
New tests are `tests/q1/test_robustness_protocol.py` and
`tests/q1/test_robustness_runner.py`. Only the Approach B entrypoint and the narrow
M6 retained-only split interface are changed; documentation/evidence are this
file and `docs/q1_m7_verification.json`.

Different objectives/feasible boundary faces/stopping rules remain explicit:
SVI fits total-variance RSS; neural fits option-price MSE plus the examination
butterfly and calendar derivative penalties. This is not a perfectly controlled
optimizer benchmark. SVI fits all 228 retained observations; neural must reserve
internal validation, so its parameter-training count is smaller and reported.

## Isolation, preparation and support

The cleaned universe is frozen using the unchanged M4 preparation/eligibility
rules BEFORE deletion. Individual IV inversion uses verified exogenous forward,
discount, rate/dividend and maturity conventions. Full-data parity is descriptive
only: it never determines fitted inputs or refit initialization. Both instruments
remain eligible after masking even if their counterpart was dropped. Re-running
M4's paired filter on the retained subset would incorrectly delete extra quotes.

Dropped rows are not passed to SVI fitting or neural training/validation functions,
normalization, checkpoint selection or fit-cache keys. No full-data fitted SVI
parameters initialize a refit. Torch gets SciPy's **initial**, not fitted, raw
parameters on the identical retained observations/weights. Each retained strike
has total SVI weight one, divided over its available legs: singleton weight 1,
two-leg weights .5/.5. No missing side is imputed or inferred by parity.

M6's original trainer required three nonempty folds. A narrow backward-compatible
extension `retained_only=True` supports a two-fold train/validation partition of
retained rows only; its default M6 validation and optimization are unchanged.
Incomplete C/P groups are valid here. Same-strike available legs still cannot
cross the internal training/validation boundary. Target tests confirm held-out
price/IV/forward changes cannot alter fitted parameters, histories or normalization.

Every dropped quote has a common support classification based on the **retained**
strike range in its expiry: interpolation, extrapolation, or insufficient support
(<5 distinct retained strikes). Neural's actual parameter-training support is
reported additionally, since some retained quotes are reserved for validation.
Too-sparse neural training groups are explicitly unavailable, not silently filled.

Random deletion mostly tests interpolation and can retain the opposite leg at an
otherwise dropped quote's strike, carrying nearly identical IV information under
parity. It is a weak/optimistic extrapolation stress test, especially for noiseless
synthetic quotes. Empty extrapolation groups have null errors, not invented zeros.

## Evaluation, failure accounting and aggregation

Every method attempts exactly the same dropped IDs, including failed slices.
Nonconverged SVI candidates remain in the fit ledger but are not scored as valid
predictions. Neural MAX_EPOCHS/validation stopping is not stationary/global
convergence: its convergence field is null and usable-checkpoint status separate.
One dated neural fit's time is equally attributed to its expiry ledger rows to
avoid multiplying its wall time; the original dated training record is retained.

Per-quote output includes market/predicted IV and premium, signed/absolute IV
error, absolute pricing error, relative pricing error, support classes, failure
reason and status. Premiums below **1e-8** retain absolute errors but have null
relative error. The denominator is never floored; observations are not hidden.

Per seed/date/expiry and support category, metrics include IV MAE/RMSE, price
MAE/RMSE, relative mean/median/p95/max, expected/predicted counts and coverage.
Complete-cohort error fields are null if any target prediction is missing.
Available-prediction errors are explicitly conditional, always paired with
coverage/failures. A **secondary intersection** of successfully predicted IDs
across all four methods permits matched-subset comparison; its IDs/counts are
explicit, and it never replaces the original cohort's failures or denominator.

Per-seed and per-date/expiry aggregation includes fit success/failures, time and
arbitrage availability. Calendar pairs/forward intervals are attributed to their
later expiry. Across seeds, report arithmetic means and **sample std (ddof=1)**,
with available/expected seed counts. These are variability summaries, not claims
of independent market samples or statistical significance.

The locked CPU environment and explicit seeds permit deterministic verification;
wall times still vary, and bitwise numeric portability across CPU/library/thread
configurations is not guaranteed.

Cross-method arbitrage uses the **same 201-point k grid [-.5,.5] and original
expiry nodes**. Shared M4 checks report examination butterfly, separately labelled
complete butterfly, all maturity pairs and adjacent negative forward variance,
with locations/magnitudes and missing-expiry coverage. Neural's existing M6 dense
grid check is a separately labelled supplement, not substituted for the common
grid. Missing diagnostics cannot certify zero arbitrage; no repairs are applied.

The full-data SVI reference is for figure illustration ONLY; it sees dropped
quotes and is never a competitive held-out method or initializer. An available
hash-verified M4 reference with the identical prepared universe/configuration is
reused. Otherwise a clean clone computes the cheap deterministic reference once,
not once per seed. Saved provenance distinguishes these cases.

## Artifacts, restart and reproduction

The original M4/M5/M6 results are immutable. Component-qualified M7 paths:

- Synthetic: `results/verification/approach2/q1_synthetic_missing_quotes/`.
- Empirical gate: `results/approach2/q1_510050_missing_quotes/`.
- Debugging uses a separate `_smoke` suffix and tiny budgets, never final findings.

Files include course-schema `metrics.json` (`params`, `metrics`, `fit_seconds`,
`history`), `robustness_summary.json`, `failure_ledger.json`,
`arbitrage_summary.json`, held-out/per-seed CSVs, `configuration.json`, three
`split_manifests/seed_*.json`, full per-fit records/neural checkpoints in `fits/`,
and a recursive SHA-256/byte-size artifact manifest. Quote IDs, masks, counts,
expiry coverage, seeds, split/configuration/implementation hashes are persisted.

Completed fit jobs are cached independently (including genuine nonconvergence).
On interrupted identical-config execution, `--resume` reuses successful AND failed
completed jobs and computes only unsaved jobs; no seed, budget or tolerance changes
are hidden. An already completed run is merely integrity-checked/read, without
training or plotting again. Figures are atomically written and completed figures
are retained on resume. Cache keys include the applicable model implementation,
retained inputs and settings; corrupted/mismatched evidence is refused.

```powershell
# Complete fixed configuration, exactly the mandatory three seeds.
py -3.11 -B scripts/project.py uv run --locked --offline python code/approach2_q1_torch.py --component robustness --mode development --dataset q1_synthetic_missing_quotes --seeds 42 43 44

# Resume an interrupted identical-config study, or verify an already completed one.
py -3.11 -B scripts/project.py uv run --locked --offline python code/approach2_q1_torch.py --component robustness --mode development --dataset q1_synthetic_missing_quotes --seeds 42 43 44 --resume

# Unchanged strict registered 510050 data; no external search/download.
py -3.11 -B scripts/project.py uv run --locked --offline python code/approach2_q1_torch.py --component robustness --mode empirical --dataset q1_510050_missing_quotes --seeds 42 43 44

# Optional DEBUG command only, separated from the complete configuration.
py -3.11 -B scripts/project.py uv run --locked --offline python code/approach2_q1_torch.py --component robustness --mode development --dataset q1_synthetic_missing_quotes_smoke --seeds 42 --smoke

py -3.11 -B scripts/project.py uv run --locked --offline python -m pytest tests/q1/test_robustness_protocol.py tests/q1/test_robustness_runner.py -q
# Once, after implementation stabilizes:
py -3.11 -B scripts/project.py uv run --locked --offline python -m pytest tests/q1 -q
```

Robustness uses `--seeds`; the existing singular `--seed` remains for M5/M6.
Complete M7 model settings are the fixed validated defaults above; unrelated
component-specific optimizer/epoch flags do not retune this protocol. Fresh runs
refuse existing directories without `--resume`. Direct exit 1 means PARTIAL, 2
means BLOCKED/invalid arguments; the existing launcher maps nonzero child exits to 1.

Eight self-contained inline-JS groups are generated once: full-data implied IV,
retained/dropped quotes, full-data SVI reference, retained SVI refits, neural
surface predictions, held-out errors by k, held-out errors by T, and robustness
comparison table. Each synthetic figure prominently says NOT EMPIRICAL MARKET
RESULTS. Conditional errors/failure coverage are visible, including failed fits.

## Verification evidence and limitations

Before complete execution: five protocol/masking/leakage tests passed in **12.75s**.
A small seed-42 smoke with Adam/L-BFGS two-step budgets attempted 100 dropped IDs
per method: SciPy/neural predicted 100, both deliberately under-budgeted Torch SVI
methods predicted zero with visible failures. That smoke used the initial debug
calendar-only preset before the user-approved fixed D complete protocol; it is
not promoted to a final comparison. The expanded targeted run (M7 plus existing
M6 training regression) passed **22 tests in 62.55s**; the stabilized 12 M7 tests
passed in **39.36s**, including resume-with-no-fitting and eight-figure contracts.
Tests capture the Plotly HTML contract rather than repeatedly generating large
figure files. Actual complete-run measurements are recorded in
`docs/q1_m7_verification.json`, including every convergence failure.

### Actual complete experiment (one execution, no tuning/restarts)

All four methods attempted 100 identical targets per seed, 300 overall.
SciPy/Adam/neural predicted **300/300**; L-BFGS predicted **275/300**. Each SVI
method had 12 slice fits. Neural had **three dated fits**, not 12 independent fits;
its per-expiry readiness rows refer to those shared models. Each neural run used
600 epochs and MAX_EPOCHS termination, with best epochs 600 / 598 / 596 for seeds
42 / 43 / 44. Train/validation counts were 182/46, 185/43 and 186/42 respectively,
always from the 228 retained quotes. No held-out early stopping or coefficient
selection occurred.

The following are **means ± sample std across seeds of available-prediction
errors**, with original coverage explicitly retained. L-BFGS's seed-42 errors are
conditional on only 75/100 predictions, not comparable full-cohort claims.

| Method | Coverage | IV MAE | IV RMSE | Price MAE | Relative-price MAE (ratio) | Fit seconds / seed |
|---|---:|---:|---:|---:|---:|---:|
| SciPy | 300/300 | 1.19e-12 ± 1.94e-12 | 2.54e-12 ± 4.20e-12 | 1.06e-11 ± 1.61e-11 | 3.86e-12 ± 6.34e-12 | 0.0559 ± 0.0133 |
| Adam | 300/300 | 7.09e-8 ± 4.66e-9 | 9.86e-8 ± 7.82e-9 | 8.87e-7 ± 5.04e-8 | 2.53e-7 ± 2.00e-8 | 35.3348 ± 4.6855 |
| L-BFGS | 275/300 | 1.67e-9 ± 2.12e-9 | 3.51e-9 ± 4.71e-9 | 2.13e-8 ± 2.97e-8 | 6.92e-9 ± 9.31e-9 | 2.5433 ± 2.4139 |
| Neural | 300/300 | 0.00307 ± 0.00144 | 0.00640 ± 0.00239 | 0.02507 ± 0.01315 | 0.01453 ± 0.00985 | 7.6757 ± 0.0701 |

The secondary common-success cohort contains **75/100/100 targets** across seeds
for **every method**. On that matched subset, mean IV RMSE / price MAE were:
SciPy `2.5473e-12 / 1.0768e-11`, Adam `9.9382e-8 / 8.7365e-7`, L-BFGS
`3.5148e-9 / 2.1253e-8`, neural `0.00660705 / 0.0255057`. This deliberately narrower
cohort is secondary and never replaces the 25 missing predictions/failure.
The synthetic truth is itself SVI and noise-free, structurally favoring correctly
specified SVI; these numbers do not establish general market robustness rankings.

Only **two extrapolation targets** occurred: seed 44, the call/put legs at one
365-day upper boundary strike (`SYNTHETIC_365_40_C/P`). Seeds 42 and 43 had none.
There were 298 interpolation targets. Neural pooled interpolation IV RMSE / price
MAE were `0.00670267 / 0.0244835`; extrapolation values were `0.00431629 / 0.112082`
on just **two legs of one strike**, not robust extrapolation evidence. L-BFGS
covered 273/298 interpolation and 2/2 extrapolation targets. All other methods
covered every target; no relative-price denominator exclusions occurred in this
fixture. Per-seed/date/expiry and category results, null empty groups and seed
availability counts remain in the machine-readable summary.

The sole convergence failure was **L-BFGS seed 42, expiry 2000-04-02**: 57 retained
quotes, 1,000 native iterations, 1,492 closure calls and 1,493 objective/gradient
evaluations including the post-check. Final RSS `1.9668009e-12` and physical
half-RSS gradient infinity norm `2.9708356e-8` failed the unchanged declared
thresholds. Its 25 targets are explicitly uncovered and its diagnostics missing;
the fitted candidate remains visible but is not promoted to success.

**Arbitrage requires a two-level reading.** On the shared original-expiry × 201-k
grid, all available examination/complete butterfly, calendar and negative-forward
counts were zero. The failed L-BFGS expiry remains unverified. However, the
supplementary neural **15-maturity × 201-k** check found the following:

| Seed | Examination / complete butterfly | Negative partial_T(w) locations | Calendar-pair locations | Negative forward locations |
|---|---:|---:|---:|---:|
| 42 | 0 / 0 | 0 | 0 | 0 |
| 43 | 0 / 0 | 0 | 0 | 0 |
| 44 | 0 / 0 | **402** | **448** | **272** |

Thus the original expiry grid missed **between-expiry calendar violations**.
All locations and magnitudes are preserved under the neural seed-44
`dense_neural_supplement` in `metrics.json`. The soft penalties did not guarantee
calendar monotonicity. These results were **not** used to change coefficients,
increase budgets, choose a more favorable seed, or regenerate tuned figures.
Supplementary dense-grid mean ± sample std across seeds were 134 ± 232.095
negative-w_T locations, 149.333 ± 258.653 calendar-pair locations and 90.667 ±
157.039 negative-forward locations. These supplement, rather than overwrite,
the matched original-expiry grid results.

Total measured refitting time was **136.8288 seconds**, excluding the descriptive
full-data reference, which was reused from hash-verified M4 output. There were
39 fresh model fits (36 SVI plus three neural) and one cached reference record.
Eight actual HTML groups were generated once. All **61 artifact hashes/sizes**,
eight inline-JavaScript/no-external-source figures, three masks and all matched-ID
checks passed inspection. Three restored neural checkpoints exactly reproduced
saved dropped-quote volatilities; normalization buffers exactly matched their
actual training-only input statistics. Previous M5/M6 saved metrics hashes were
unchanged. Empirical output remains BLOCKED: 606 inputs, zero eligible quotes,
zero fits/predictions/figures and zero fit time.

### Final regression and integrity

The complete Q1 regression suite was run **once** after stabilization:
**259 passed, 0 failed in 252.61 seconds** (247 pre-existing plus 12 new M7 tests).
The complete three-seed experiment and final eight-figure set each ran once;
cached reference reuse and targeted small configurations avoided repeated full
training. The staged Q1-only change set passed `git diff --check`; M4/M5 algorithms,
prior results, shared configuration and Q2/Q3 were preserved. Passing tests confirms
correct numerical/protocol/failure accounting, not that every fit converged or
all arbitrage conditions were satisfied. M7 therefore remains **PARTIAL**, with
empirical requirements **BLOCKED**.

Historical metadata/units, complete daily/multi-year ETF chains and approved real
pricing/curve/dividend conventions remain unresolved. Strict empirical eligibility
is not relaxed. Synthetic verification is not empirical performance, a global
arbitrage-free proof, or completion of the group project.

**Stop after M7. Do not begin M8 automatically.**
