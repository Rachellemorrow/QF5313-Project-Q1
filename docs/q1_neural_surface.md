# Q1 M6 — Neural volatility surface and mandatory penalty ablation

**SOFTWARE_VERIFICATION: PASS** for the neural component's synthetic numerical and
pipeline acceptance. **EMPIRICAL_REQUIREMENTS: BLOCKED**. This does not upgrade
M5's PARTIAL status or claim completion of the whole Q1/group project.

## Scope and prespecified verification protocol

Q1 only; implements examination §3.4.2. M4/M5 numerical algorithms and historical
results are preserved. M5 remains PARTIAL (one default L-BFGS failure, Adam recovery
precision and domain/stopping-rule differences); M6 does not try to fix it.
No M7 30%-missing-quote robustness study or external data search is included.

New Q1 implementation files: `code/q1/neural_surface.py` (MLP/pricing/calculus),
`neural_training.py` (split/Adam/checkpoints), `neural_evaluation.py` (shared
diagnostic adapters/repricing), `neural_experiment.py` (four-case orchestration),
`neural_figures.py` (six plots) and `neural_cli.py` (component execution/writer).
Only `code/approach2_q1_torch.py` is modified: it adds a neural dispatch and CLI
flags while keeping the existing SVI branch unchanged. New tests are
`tests/q1/test_neural_math.py` and `tests/q1/test_neural_training.py`; this document
and `docs/q1_m6_verification.json` contain the protocol and measured evidence.

The complete synthetic configuration below was specified **before** its execution.
Small earlier runs are debugging/unit-test evidence, not empirical findings.

| Item | Complete synthetic verification configuration |
|---|---|
| Source | unchanged, labelled M4 mathematical fixture; 328 C/P quotes, one artificial date, four expiries |
| Seed / precision / execution | 42; CPU float64; one Torch CPU thread in neural CLI |
| Architecture | 2 → 16 → 16 → 1, tanh hidden activations; **337 trainable parameters** |
| Maximum layer convention | affine/Linear transformations including output; default three, configurable maximum four; no batchnorm/dropout |
| Output | `sigma=1e-6+softplus(logit)`; positive, no upper clipping |
| Inputs | physical `k=ln(K/F)` and positive T in years |
| Preprocessing | population mean/std from training quotes only; zero feature std replaced by 1 |
| Split | seeded per-expiry paired strikes: train 25, validation 8, held-out test 8 of 41; both C/P in the same fold |
| Fold sizes | 200 train, 64 validation, 64 test quotes |
| Adam | learning rate 0.01, batch size 256, maximum 600 epochs; default betas/epsilon, no weight decay or gradient clipping |
| Early stopping | validation **price MSE**, patience 100, absolute min_delta=1e-8; restore lowest-validation-price checkpoint, including epoch 0 |
| Training penalties | 17 × 7 mid-cell collocation points, bounds from TRAIN k/T support only |
| Independent evaluation | fixed M4 201 k nodes on [-0.5,0.5]; 13 evenly spaced positive maturities over observed support plus exact expiries |
| Four coefficient pairs | A=(0,0), B=(1,0), C=(0,1), D=(1,1) |

The only coefficient-selection grid is the **mandatory** `{0,1} × {0,1}` ablation,
not an additional hyperparameter search. The nonzero value 1 is prespecified in
raw premium-per-underlying-unit loss units, not asserted universal or optimal.
Select the pair with minimum best-validation price MSE, ties A→B→C→D. All four
trainings/validation selections finish **before** held-out prices or independent
arbitrage diagnostics are evaluated. No coefficients are tuned on evaluation quotes.
Dataset, initialization, architecture, seed, splits, preprocessing, collocation and
evaluation grids are identical across all four cases. Validation price error,
not test error or arbitrage counts, selects the highlighted model.

Software acceptance: targeted numerical/gradient/split/checkpoint/failure/schema
tests pass; all four complete runs remain finite, have evaluable checkpoints and
improve validation price MSE, with no quote-preparation rejections. Reaching a
budget is **not mathematical convergence**. Accuracy and arbitrage benchmarks are
reported separately, never forced to pass or substituted for empirical results.

## Model, pricing and exact objective

Training-only normalization `(x-mean_train)/std_train` occurs **inside** the model
graph. The stable softplus output is positive even for a strongly negative logit.
The model has no cross-sample operations, so differentiating the sum of outputs
produces pointwise input derivatives. Raw physical k and T are independent autograd
leaves: `partial_T(w)` holds k fixed and includes both explicit T in w and the
normalization/model chain rule. It is not a derivative holding strike K fixed.

```text
sigma = MLP(k,T) > 0
w = sigma^2*T
g_exam = (1-k*w_k/(2*w))^2 - (w_k^2/4)*(1/w+1/4)
g_complete = g_exam + w_kk/2              # separately labelled diagnostic only
L_price = mean((BS_forward_price(sigma)-observed_price)^2)
L_butterfly = mean(relu(-g_exam)^2)       # training collocation grid
L_calendar = mean(relu(-partial_T(w))^2)  # training collocation grid
L_total = L_price + lambda1*L_butterfly + lambda2*L_calendar
```

There is no IV MSE, SVI-target pretraining, hidden quote reweighting, price
normalization, arbitrage repair or replacement of the examination g formula.
Neural fitting uses price MSE, unlike M4/M5's total-variance RSS. The penalties are
soft constraints: neither training-grid nor finite evaluation-grid checks prove
global no-arbitrage. Zero coefficients skip unneeded training derivative graphs;
their uncomputed penalty history entries are **null**, not claimed zero violations.
Every model still receives the full independent diagnostic evaluation.

`black_scholes_torch` mirrors the verified `pricing.black_scholes`: caller-supplied
positive forward F and discount D, log-CDF evaluation of the OTM side, parity for
the ITM side, exact-ATM erf identity, and intrinsic T=0/sigma=0 limits. Both C/P
and mixed batches are supported; only analytic roundoff guards are used. No rate,
dividend, day count, contract multiplier or empirical convention is inferred.

## Split discipline, fitting and checkpoints

Paired-strike splitting is within each dated expiry, not a time-series forecast.
It keeps C/P counterparts together and requires nonoverlapping complete folds.
For multiple dates the pipeline trains a separate two-input surface per date,
rather than conflating different dates without a date feature. Rejected quotes,
IV failures and parity diagnostics remain in the reused M4 preparation ledger.

Only training k/T determine learned preprocessing and collocation bounds. Validation
inputs/prices are used for early stopping/selection only. Held-out values cannot
affect fitting, initialization, normalization or selection. Split validation may
inspect group identifiers to check no overlaps; it does not learn from held-out
features or labels. Evaluation labels include interpolation versus extrapolation
relative to **training** strike support. This is not the M7 missing-quote experiment.

Training histories include epoch 0, actual epochs, optimizer updates, backward
calls, training-objective and validation-price evaluations, best epoch, stopping
reason, gradient norm and failure reason. Training curves are weighted averages
of **pre-update minibatch** objectives, not mislabelled full-dataset post-update
losses. Restored-checkpoint training and validation losses are evaluated separately.
Nonfinite losses/gradients/parameters fail explicitly; failed training is not saved
as a usable checkpoint. MAX_EPOCHS and early stopping are termination reasons, not
proofs of convergence to a stationary/global optimum.

Each successful ablation saves its best-validation inference checkpoint: state
dict, configuration, normalization buffers, seed, coefficients, best epoch/value,
and deterministic tensor fingerprint. Loading uses CPU `weights_only=True`, checks
format/fingerprint and restores all buffers. The writer reloads every checkpoint
and verifies its tensor fingerprint. Checkpoints are for inference, not resumable
Adam training; optimizer/RNG continuation state is not stored. No file is overwritten.

## Reused evaluation and reporting

M4 quote preparation, strict empirical gate, Brent inversions and parity are reused
unchanged. NumPy Black–Scholes independently reprices each predicted volatility.
The existing `repricing_summary` computes per-slice maxima and complete-date
median/p95 separately for train/validation/test, retaining undefined near-zero
relative denominators. Its legacy `fit.converged` adapter flag means **usable
checkpoint for repricing only**, not claimed neural optimizer convergence.

NumPy `svi.GridDiagnostics`, `calendar_diagnostics`, `forward_variance_diagnostics`
and `interpolate_total_variance` are reused on neural values. Direct autograd gives
neural derivatives (SVI-specific analytic derivative routines are not applicable).
All evaluation g values, calendar differences/locations/magnitudes, derivative
violations and negative forward-variance locations remain visible. Both butterfly
definitions are separately labelled. Every ordered evaluation-maturity pair is
checked at common k. Shared linear-w interpolation is a diagnostic, not a silent
replacement of the neural function between maturities.

Training collocation and evaluation are separately constructed grids with different
resolutions/purposes, not necessarily mathematically disjoint at isolated nodes.
Evaluation includes model wings outside quote/training support; these are not new
observations. Dense intermediate maturity checks supplement per-expiry reports.

Each run uses the required `{params, metrics, fit_seconds, history}` schema and
retains the four ablation results, histories, initialization/split fingerprints,
quote ledgers, predictions/fold metrics and complete diagnostics. Selection and
data-origin flags are explicit. Runtime Python/NumPy/Torch/OS/CPU/thread facts are
recorded; wall times and cross-platform floating-point arithmetic are not promised
bitwise deterministic. This is not an isolated repeated timing benchmark.

## Commands, artifacts and separation

```powershell
# Complete documented synthetic verification: execute once; six figures generated once.
py -3.11 -B scripts/project.py uv run --locked --offline python code/approach2_q1_torch.py --component neural --mode development --dataset q1_synthetic_neural_verification --seed 42

# Strict registered q1_510050 data, component-qualified output ID preserves M5 artifacts.
py -3.11 -B scripts/project.py uv run --locked --offline python code/approach2_q1_torch.py --component neural --mode empirical --dataset q1_510050_neural --seed 42

# Targeted development tests; small configurations, no repeated complete training or HTML.
py -3.11 -B scripts/project.py uv run --locked --offline python -m pytest tests/q1/test_neural_math.py tests/q1/test_neural_training.py -q

# Run once after implementation stabilizes.
py -3.11 -B scripts/project.py uv run --locked --offline python -m pytest tests/q1 -q
```

Synthetic artifacts: `results/verification/approach2/q1_synthetic_neural_verification/`.
Empirical gate: `results/approach2/q1_510050_neural/`. Both use new directories and
refuse overwrites; use a new repository-local `--output-dir` for an authorized
rerun. The latter loads the unchanged registered **q1_510050** ETF dataset; the
suffix distinguishes the neural component, not a substitute financial instrument.
M5 SVI remains the default CLI component and its previous paths are untouched.

Outputs: `metrics.json`, `ablation_comparison.csv`, four successful dated `.pt`
checkpoints, SHA-256 artifact manifest, and six self-contained inline-JS figures:
`training_validation.html`, `neural_smiles.html`, `neural_surface.html`,
`observed_predicted_prices.html`, `arbitrage_heatmaps.html`, `penalty_ablation.html`.
Smiles/surface highlight the validation-selected ablation; all four are retained
in metrics and comparison/diagnostic plots. All synthetic figures explicitly say
NOT EMPIRICAL MARKET RESULTS. Generated large outputs remain gitignored; source,
tests and compact actual verification evidence are committed for offline reproduction.

The direct CLI returns 0 for completed software acceptance, 1 for PARTIAL/failure,
2 for empirical BLOCKED or invalid arguments. The existing launcher maps child
nonzero status to 1. Inspect saved status rather than ignoring unrelated errors.
Blocked empirical runs generate no training/checkpoints/figures.

## Evidence and limitations

Prerequisites: float64 operations and the seven initial targeted mathematical tests
passed (7.68 seconds). A separate 12-epoch D debugging smoke fit reduced validation
price MSE from 2.83394763145547 to 0.35303651101202826, with 337 parameters and 12
updates; it is not a final fit. Expanded targeted tests passed **19 tests in 34.34
seconds**, including active-penalty backward gradients, nonfinite-gradient detection,
finite differences, leakage prevention, checkpoint restoration and tiny four-case
ablation execution. Tests capture the Plotly write contract instead of repeatedly
writing the same large HTML files. Complete-run results were recorded from the
actual saved artifacts in `docs/q1_m6_verification.json`.

### Actual complete configuration (one execution)

All four runs remained finite, reached **600 epochs / 600 updates**, selected epoch
600 as their best validation checkpoint, and terminated with **MAX_EPOCHS**. The
budget was not extended and no stationary/global convergence is claimed. Each run
recorded 602 training-objective evaluations, 602 validation-price evaluations and
600 backward calls, including explicit initial/restored checks as appropriate.

| Case | (lambda1,lambda2) | Restored train price MSE | Best validation MSE | Held-out price MSE | Fit seconds |
|---|---|---:|---:|---:|---:|
| A | (0,0) | 0.000319263 | 0.000507458 | 0.000898303 | 9.7013 |
| B | (1,0) | 0.000319263 | 0.000507458 | 0.000898303 | 7.8970 |
| C | (0,1) | 0.000313142 | 0.000499245 | 0.000886606 | 7.8046 |
| D | (1,1) | 0.000313142 | 0.000499245 | 0.000886606 | 7.6993 |

Common initial validation MSE was **2.83394763145547**. Validation selected **C**,
tied with D. A/B checkpoints and C/D checkpoints respectively have identical tensor
fingerprints: the butterfly term never activated here (B/D history maximum 0).
The calendar term did activate (C/D history maximum `8.0918264e-6`). This benign
fixture cannot demonstrate general butterfly-penalty effectiveness or a broad
arbitrage/accuracy trade-off. No speculative coefficient search was performed.

Every case had **zero examination butterfly, complete butterfly, calendar
derivative, all-pair calendar and negative-forward-variance violations** on the
15-maturity × 201-k independent grid. Per-expiry reports and all locations/magnitudes
are saved, including deliberately violating unit-test examples. Selected C's
minimum evaluated volatility was **0.31182275922725305**; positivity is not a
global arbitrage-free guarantee.

Held-out complete-date maximum relative repricing errors were **10.1112%** (A/B)
and **9.9662%** (C/D). Median and p95 date maxima coincide because there is just one
artificial date. These outputs do **not** establish the indicative 2% empirical
pricing target; they are not pooled-median quote errors or market findings.

Actual saved-checkpoint Torch/NumPy repricing maximum absolute difference over 328
quotes: **2.842170943040401e-14** premium units. On trained selected C at physical
`k=(-0.13,0.17)`, `T=(0.2,0.7)`, central step `1e-5`, maximum finite-difference
errors were `8.68e-11` for w_k, `4.45e-11` for w_T, and `8.93e-7` for w_kk
(inspection tolerances `1e-7`, `1e-7`, `1e-5`). These are separate from training
stopping/acceptance thresholds. All four checkpoint restorations reproduced all
328 saved volatilities **exactly**. All 12 artifact hashes/sizes and six actual
HTML files passed inspection for inline Plotly/no external script sources.

Empirical smoke: **606 rows, zero eligible rows**, no fitting, checkpoint or figure.
The component-qualified output preserves all M5 results; their saved metrics
hashes were independently checked unchanged. The expected direct CLI BLOCKED exit
was 2 (launcher maps it to 1), not an unreported download/training failure.

### Final regression and integrity

The full Q1 suite was run **once after implementation stabilized**: **247 passed,
0 failed in 227.07 seconds** (228 existing tests plus 19 new M6 tests). The complete
600-epoch four-case configuration and the actual six-figure set were each generated
once. Small debugging/unit configurations were not promoted to final experiment
results. Total measured complete-run fitting time was about **33.10 seconds**;
these single-run timings include setup/logging and are not a repeated benchmark.

The final inspected Git change set contains six new Q1 modules, two new Q1 test
files, two new Q1 documents, and only the neural CLI extension to the existing
Approach B entrypoint. `git diff --check` passed. M4/M5 mathematical/calibration
modules, the original fixture, data, shared configuration and Q2/Q3 were unchanged.

The historical archive remains blocked until effective-dated metadata, verified
units, complete historical ETF chains and approved pricing/curve/dividend conventions
are available. Synthetic price/volatility and arbitrage results cannot establish
market performance, multi-year coverage or the examination's empirical benchmark.
Student review and scientific interpretation remain required.

**Stop after M6. Do not begin M7 automatically.**
