# Q1 report-support materials — NOT the student-authored final report

## Final handover and student report outline

The final empirical outcome is **0 eligible quotes before / 0 after recovery**;
no empirical IV, SVI, neural, repricing or arbitrage result is claimed. The 606
archived observations and all synthetic results below are preserved. Exact
rejection/strike-support tables, scope-qualified replay commands and authentic
two-session export instructions are in `q1_final_recovery.md`; row-level primary
attribution is in `q1_final_recovery_evidence.json`. There are no new market figures.

Suggested structure for the **students to author**, not a completed report:

| Section | Verified support to use | Required student contribution |
|---|---|---|
| Data and eligibility | Final recovery waterfall; M2 source hashes and candidate expiry table | Explain why evidence cannot support empirical claims; review source rights |
| Mathematical models | Forward BS, Brent, SVI and two butterfly formulas below | Verify equations, state assumptions and objective rationale |
| Protocol | M4–M7 configuration table below | Discuss synthetic-only scope, splits and comparability caveats |
| Results | Actual M5/M6/M7 tables below; comparator with coverage | Interpret measured findings without treating fixtures as market evidence |
| Verification | M8 264-test record, isolated artifact replay, final targeted checks | Distinguish software tests, checkpoint inference and independent replication |
| Limitations | Missing historical coverage, failed L-BFGS fit, dense neural calendar violations | Discuss implications; no universal no-arbitrage or market benchmark claim |
| Reproducibility and work division | Final recovery commands and two authentic session IDs | Write statement of work; combine Q2/Q3 and authentic exports |

Q1 compliance checklist: implemented numerical methods/provenance/diagnostics and
synthetic ablations **evidenced**; fitted-model outcomes **partial**; real ETF
calibration, complete historical month/four verified expiries/five multi-year
sample chains and empirical <2% benchmark **not demonstrated**. Group report,
complete `SESSION.md`, six-approach integration and submission policy remain
student/group actions. Do not fill missing empirical cells with synthetic values.

## Mathematical model descriptions

**Forward Black–Scholes.** With positive exogenous F(T), discount D(T), strike K,
T in years and volatility sigma, set v=sigma*sqrt(T), d1=ln(F/K)/v+v/2 and d2=d1-v.
Calls are D[F N(d1)-K N(d2)] and puts D[K N(-d2)-F N(-d1)]. The stable NumPy/Torch
kernels use log-CDF OTM evaluation and parity for ITM prices. IV uses bracketed
Brent on [1e-4,5], with explicit failed inversion diagnostics. Price units are
per underlying unit, not automatically cash per contract. The unadjusted
examination residual C-P-[S-K exp(-rT)] and the dividend-adjusted residual are
distinct. Real SHIBOR interpretation/dividends remain unapproved; synthetic r=.02,
q=.01 and spot=100 are fixture conventions, not Chinese market estimates.

**SVI.** w(k)=a+b[rho(k-m)+sqrt((k-m)^2+eta^2)], k=ln(K/F), w=sigma_IV^2*T.
b>=0, |rho|<1, eta>0. Global minimum w is a+b*eta*sqrt(1-rho^2), not a alone.
SciPy uses constrained least squares in minimum-variance coordinates; Torch maps
latent coordinates through softplus/tanh. Their safe boundary faces differ and
their stopping criteria are not equivalent. Both minimize the same weighted
total-variance RSS, using one total weight per strike. SciPy cost is half RSS.
SVI analytic first/second k derivatives are checked against autograd and finite
differences. Neither positivity nor solver convergence implies no-arbitrage.

**Diagnostics.** Examination g=(1-k*w_k/(2w))^2-(w_k^2/4)(1/w+1/4). Complete g
adds w_kk/2 and is separately labelled. Calendar checks compare w at common k;
term interpolation is linear in w, not sigma, and negative forward-variance
slopes are retained. Finite grids do not prove global inequalities.

**Neural surface.** Default 2→16→16→1 tanh network has 337 trainable parameters
and three affine layers including output (maximum four under the documented
counting convention). sigma=1e-6+softplus(logit). Input normalization uses training
data only and stays in the differentiation graph. Loss is mean squared option
pricing error plus lambda1*mean(relu(-g_exam)^2) and
lambda2*mean(relu(-partial_T(w))^2), with T differentiated at fixed physical k.
The complete butterfly expression is not silently substituted in the loss.

## Experimental protocols

| Experiment | Observations / split | Objective / settings | Evaluation limitations |
|---|---|---|---|
| M4 | 328 synthetic C/P quotes, 164 paired strikes, four expiries, one artificial date | SciPy TRF, analytic Jacobian, defaults in classical_svi.py | In-sample only; no empirical benchmark |
| M5 | Exact M4 targets/weights/raw starts, CPU float64 | SciPy vs Adam/L-BFGS; unchanged objectives and recorded tolerances | Different coordinates/domains/stopping rules; L-BFGS 3/4; no universal speed/accuracy conclusion |
| M6 | 200 train / 64 validation / 64 test; C/P pairs grouped | 600 epochs, Adam .01, batch256, patience100; four coefficient pairs (0,0),(1,0),(0,1),(1,1) | Validation chooses C, tied D; all max epochs, butterfly never active; one benign synthetic date |
| M7 | Seeds42/43/44; 25/82 individual quotes deleted per expiry | Unchanged SVI budgets, fresh MLP per seed, user-approved fixed D=(1,1); retained-only internal validation | M6 winner not reused as leak-free selection; opposite retained leg can carry near-identical IV information |

## Actual numerical result tables (distinct cohorts, do not merge)

M5 four-slice synthetic recovery: SciPy 4/4, Adam 4/4 declared convergence,
L-BFGS 3/4. Maximum candidate RSS: 4.71896e-26 / 9.98838e-15 / 7.10249e-11.
Under the unchanged original strict M4 recovery thresholds (variance1e-9,
parameters1e-6), passing slices are **4 / 0 / 3**, not 4/4 for all methods.
M5's looser CLI recovery tolerance is not equivalent to the original standard.

| M6 ablation | Validation price MSE | Held-out price MSE | Budget termination |
|---|---:|---:|---|
| A (0,0) | 0.0005074582153352174 | 0.0008983031269098603 | 600 epochs |
| B (1,0) | 0.0005074582153352174 | 0.0008983031269098603 | 600 epochs |
| C (0,1) | 0.000499245492129125 | 0.0008866055198775561 | 600 epochs |
| D (1,1) | 0.000499245492129125 | 0.0008866055198775561 | 600 epochs |

M6 held-out date-max relative errors: 10.1112% (A/B), 9.9662% (C/D). A/B and C/D
respective checkpoints match: butterfly penalty was inactive, not shown effective.

| M7 method | Target coverage | IV RMSE mean ± sample std | Price MAE mean ± sample std | Fit seconds / seed mean |
|---|---:|---:|---:|---:|
| SciPy | 300/300 | 2.541e-12 ± 4.202e-12 | 1.058e-11 ± 1.612e-11 | .05586 |
| Adam | 300/300 | 9.863e-8 ± 7.823e-9 | 8.874e-7 ± 5.039e-8 | 35.33479 |
| L-BFGS | 275/300 | 3.515e-9 ± 4.710e-9 | 2.125e-8 ± 2.972e-8 | 2.54327 |
| Neural | 300/300 | .006399 ± .002389 | .025068 ± .013153 | 7.67568 |

These errors are conditional on available predictions. L-BFGS's 25 missing
seed42/April-expiry targets remain failures, not zero errors; complete-cohort
scores are null. Secondary matched-success intersection counts are 75/100/100
per seed for every model. All per-expiry median/p95/max pricing errors and
coverage are in `results/comparison.json/.csv` and the original detailed metrics.
Only two M7 extrapolation targets exist, the C/P legs at one upper boundary
strike in seed44. Neural pooled interpolation/extrapolation price MAE is
.0244835/.112082; this does not support broad extrapolation claims.

The common original-expiry grid misses between-expiry neural violations: seed44's
dense grid has **402 negative-w_T, 448 calendar-pair and 272 negative-forward
locations**. Direct checkpoint inference/central differences confirm genuine
behavior. Butterfly counts remain zero. Do not describe this as an arbitrage-free
surface or adjust coefficients after observing the held-out findings.

## Figure index and suggested captions

The machine-readable complete original index is
`tests/q1/fixtures/verified_runs/manifest.json` → `original_artifact_inventory`
(filter `.html`), and the comparator's `figure_index`. Paths refer to preserved
ignored originals; the numeric evidence is losslessly tracked for clean replay.

| Source directory | Figures / caption guidance |
|---|---|
| `results/verification/approach1/q1_synthetic_svi_verification/` | Seven M4 figures: smile, total variance, butterfly, calendar, surface, market/model prices, repricing distribution. Caption: synthetic in-sample recovery, not market evidence |
| `results/m5_final/verification/approach2/q1_synthetic_svi_optimizer_comparison/` | Six optimizer views: objective vs iteration/evaluation, objective vs time, parameters, convergence, repricing, arbitrage. Caption: native iterations unavailable for SciPy; show L-BFGS failure and setup/timing caveats |
| `results/verification/approach2/q1_synthetic_neural_verification/` | Six M6 views: training/validation, smiles, surface, observed/predicted prices, arbitrage heatmaps, penalty ablation. Caption: validation-selected C, finite-grid checks, max epochs, inactive butterfly penalty |
| `results/verification/approach2/q1_synthetic_missing_quotes/` | Eight M7 groups: full IV, mask, full SVI reference, retained SVI refits, neural surfaces, held-out error vs k/T, robustness table. Caption: reference sees full data and is illustration only; report missing targets and dense violations |
| `results/comparison.html` | Question/mode-separated integrated tables; no global ranking or synthetic/empirical pooling |

## Sources/provenance and student decisions

The M1 archive contains actual bounded AKShare/Sina/SSE/Eastmoney/Jin10 observations,
not a complete licensed historical panel. M1.5 records provider documentation,
access and redistribution stops; no qualifying new sample was recovered. M2
retains exact source hashes, dates and rejection flags. Current public masters do
not verify historical adjusted strikes/units. ETF, index and unknown instrument
rows remain separate; SHIBOR quote interpolation is not automatically an approved
zero curve. Refer to `q1_data_feasibility.md`, `q1_source_recovery.md` and
`q1_market_data_pipeline.md` for source-level facts and unresolved rights.

Suggested statement-of-work facts (students must review/edit, not present as an
already completed student declaration): the agent implemented Q1 numerical/data
software, fixtures, tests, diagnostics, plots, audit/comparison/replay tools and
recorded actual failed experiments. The student authorized milestones, scope,
data-independent work and the M7 fixed-D decision. Humans remain responsible for
model/method review, source rights and conventions, scientific interpretation,
the final report, group integration, authentic session collection and submission.
No empirical finding, student conclusion or completed group report is fabricated.
