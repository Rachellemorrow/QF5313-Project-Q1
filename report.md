# Question 1 — Implied Volatility Surfaces for Chinese ETF Options

## 1. Objective and scope

We examine two approaches to volatility-surface construction: classical constrained SVI calibration and differentiable calibration using PyTorch, including a neural volatility surface with arbitrage penalties. Both approaches use the same pricing kernel, data checks and evaluation targets. Our intended application is Chinese ETF options.

The examination requires a month of daily chains with at least four expiries and at least five sample dates spanning two years. The available archive does not satisfy this requirement. We therefore report the data audit separately from the synthetic experiments. No empirical ETF calibration, market repricing benchmark or empirical arbitrage-free surface is claimed. The synthetic experiments also reveal limitations: some optimizations fail, and the neural surface violates calendar constraints between sampled expiries.

## 2. Data, provenance and eligibility

The saved archive contains 606 real observations obtained through the documented AKShare/provider interfaces. Source identity, retrieval information, schemas and checksums are retained in data/metadata.json and the Q1 pipeline evidence. Of these observations, 94 are CFFEX index-option records, 23 have an unidentified historical SSE underlying, and 489 are identified 50ETF candidates. Index observations are retained separately rather than silently substituted for ETF options.

| Primary eligibility stage | Rejected | Remaining |
|---|---:|---:|
| Archived input | — | 606 |
| Non-ETF instrument family | 94 | 512 |
| Unknown/non-target underlying | 23 | 489 |
| Invalid/nonpositive target-candidate price | 0 | 489 |
| Unverified date-effective contract metadata | 489 | 0 |
| Eligible empirical quotes | — | **0** |

This is a reporting precedence: downstream zero counts do not mean those requirements passed. Independently overlapping flags include unverified premium units on 606 rows and unapproved dividend/rate conventions on 606 rows. These counts cannot be added to the primary rejection counts. Missing bid/ask or open interest is disclosed as a warning, not the explanation for zero eligibility. The single zero-close/inconsistent-OHLC observation belongs to the separate index archive and is not repaired.

The 489 ETF candidates span 130 dates from 2026-04-23 to 2026-11-04. They contain only three distinct strikes overall and at most three daily strikes in any date/expiry slice. Only one strike has both call and put observations. Four candidate expiry labels therefore do not establish four verified, usable chains. The existing SVI implementation requires at least five distinct strikes, so approving rate/dividend approximations alone would not make these slices calibratable.

Historical contract mappings, effective strikes and units, adjustment histories and source reuse rights remain unresolved. Same-date ETF spot and eight SHIBOR tenor levels are available for the identified candidates; the main limitations are historical contract verification and strike coverage. The examination permits zero dividends with justification, but the empirical dividend and rate conventions remain unresolved. The implemented rate adapter supports alternative explicit rate interpretations; SHIBOR quotes are not automatically a validated zero curve.

The filter design checks instrument identity, date-effective metadata, positive prices, unit consistency and approved conventions before model preparation. Liquidity and support checks are documented in docs/q1_market_data_pipeline.md. The existing paired-quote requirement is conservative; individual BS inversion need not require both legs. Removing that restriction would still leave inadequate strike support. Empirical inversions and fits are skipped, not reported as zero-error successes or optimizer failures.

The explicit additional fitting rule is volume >=1, ordered nonnegative available bid/ask with (ask-bid)/mid <=0.25, and log-moneyness in [-0.5,0.5]. Missing spreads are warned about rather than imputed. The volume rule excludes inactive quotes; the spread cap removes visibly wide quotes when observed; the moneyness band limits unsupported wings. These thresholds limit inactive quotes, wide spreads and poorly supported wings. Historical metadata remains a separate limitation.

## 3. Mathematical specification

### 3.1 Forward pricing, parity and implied volatility

Let T be maturity in years, K the strike, F(T)>0 the exogenous forward and D(T) the discount factor. Under constant continuous rates and dividend yield, F(T)=S_0 exp((r-q)T) and D(T)=exp(-rT). Define

$$d_1=\frac{\log(F/K)}{\sigma\sqrt T}+\frac{\sigma\sqrt T}{2},\qquad d_2=d_1-\sigma\sqrt T.$$

Then

$$C=D[FN(d_1)-KN(d_2)],\qquad P=D[KN(-d_2)-FN(-d_1)].$$

Implied volatility solves V_BS(sigma)-V_observed=0 using bracketed Brent on [10^-4,5]. Invalid prices and unsuccessful brackets/inversions are retained as diagnostics. The kernels use stable OTM evaluation and parity reconstruction. Prices are per underlying unit and must not be interpreted automatically as cash per contract.

The examination parity residual is

$$R_0=C-P-[S_0-Ke^{-rT}].$$

The dividend-adjusted residual is separately labelled:

$$R_q=C-P-D(F-K).$$

These coincide for q=0 under the constant-rate convention. The synthetic fixture uses S_0=100, r=0.02 and q=0.01; these are test assumptions, not Chinese market estimates.

### 3.2 Classical and differentiable SVI

With k=log(K/F(T)) and w=sigma_IV^2 T, the per-expiry SVI function is

$$w(k)=a+b[\rho(k-m)+\sqrt{(k-m)^2+\eta^2}],\qquad b\ge0,\quad |\rho|<1,\quad \eta>0.$$

For u=k-m and s=sqrt(u^2+eta^2),

$$w_k=b(\rho+u/s),\qquad w_{kk}=b\eta^2/s^3.$$

The global minimum is a+b eta sqrt(1-rho^2), so positivity cannot be checked from a alone. Calibration minimizes weighted total-variance residual sum of squares,

$$J(\theta)=\sum_i\omega_i[w_\theta(k_i)-\sigma_{IV,i}^2T]^2.$$

Call/put observations share one total weight per strike. SciPy uses constrained trust-region reflective least squares with an analytic Jacobian in minimum-variance coordinates; its reported cost is J/2. PyTorch uses differentiable softplus/tanh transformations with Adam or L-BFGS. Matched targets, weights and raw starts support comparison, but feasible boundary faces and stopping rules differ. Exact per-slice starts, fitted parameters, tolerances, evaluations and termination records are preserved in the tracked M4/M5 metrics snapshots; Section 4.3 reports the calibration details. Solver convergence is distinct from strict parameter recovery.

The default initialization is v_min=max(0.5 median(w),10 positive_floor), b=0.1, rho=0, m=median(k), eta=0.2, with a reconstructed from v_min. The SciPy numerical bounds are v_min>=1e-10, b>=0, |rho|<=1-1e-8 and eta>=1e-8. Solver settings are TRF, analytic Jacobian, x_scale=jac, linear loss, max_nfev=2000 and ftol=xtol=gtol=1e-12. M5 Adam uses learning rate 0.03, at most 10000 steps and halving every 2000 steps; L-BFGS uses learning rate 1, max_iter=1000, max_eval=2000, history_size=50, tolerance_grad=1e-12 and tolerance_change=1e-18. The recorded physical-gradient and objective acceptance thresholds are 1e-10 and 1e-14. Full settings remain in M5 params.torch_optimizer_config.

### 3.3 Arbitrage and repricing diagnostics

The examination expression is evaluated without substitution:

$$g_{exam}(k)=\left(1-\frac{k w_k}{2w}\right)^2-\frac{w_k^2}{4}\left(\frac1w+\frac14\right).$$

The complete expression is separately reported as g_complete=g_exam+w_kk/2. Calendar checks require w(k,T_2)>=w(k,T_1) at common k. Between maturities, interpolation is linear in w:

$$w(k,T)=(1-\alpha)w(k,T_1)+\alpha w(k,T_2),\quad \alpha=(T-T_1)/(T_2-T_1).$$

The common fixed grid is k_j=-0.5+0.005j, j=0,...,200. Synthetic observations lie on 41 strikes in [-0.4,0.4]; grid wings are model evaluations, not observations. Recover volatility as sigma(k,T)=sqrt(w(k,T)/T), with K=F(T) exp(k). No unsupported maturity extrapolation is implied.

Negative slopes [w(k,T_2)-w(k,T_1)]/(T_2-T_1) are retained as negative forward variance. Violation locations and magnitudes are recorded, and finite-grid success is not a global no-arbitrage proof.

For eligible observations, relative repricing error is |V_fit-V_observed|/V_observed. Slice maxima and cross-date median/p95 summaries must retain failed-slice coverage. There are no eligible empirical dates on which to establish the examination's median <2% benchmark.

### 3.4 Neural volatility surface

The network has architecture 2→16→16→1, tanh hidden activations and 337 parameters. Counting affine layers including the output gives three layers, within the limit of four. Its positive output is sigma_hat=10^-6+softplus(f_theta(k,T)). Normalization statistics use training data only, with the transformation retained in the differentiation graph.

For w_theta=T sigma_hat^2, the objective is

$$L=\frac1n\sum_i[V_{BS}(\hat\sigma_i)-V_i]^2+\lambda_1\operatorname{mean}[\max(0,-g_{exam})^2]+\lambda_2\operatorname{mean}[\max(0,-\partial_Tw_\theta)^2].$$

Autograd evaluates derivatives at fixed physical k. The complete butterfly expression is not substituted into the training loss. Penalty satisfaction at training points does not guarantee satisfaction between expiries.

## 4. Synthetic protocols and measured numerical results

All experiments in this section are synthetic. Default seed is 42; the missing-quote study uses seeds 42, 43 and 44. Mathematical tests and checkpoint evidence are distinct from independent empirical validation.
### 4.1 Synthetic experimental protocols

| Experiment | Observations / split | Objective / settings | Evaluation limitations |
|---|---|---|---|
| M4 | 328 synthetic C/P quotes, 164 paired strikes, four expiries, one artificial date | SciPy TRF, analytic Jacobian, defaults in classical_svi.py | In-sample only; no empirical benchmark |
| M5 | Exact M4 targets/weights/raw starts, CPU float64 | SciPy vs Adam/L-BFGS; unchanged objectives and recorded tolerances | Different coordinates/domains/stopping rules; L-BFGS 3/4; no universal speed/accuracy conclusion |
| M6 | 200 train / 64 validation / 64 test; C/P pairs grouped | 600 epochs, Adam .01, batch256, patience100; four coefficient pairs (0,0),(1,0),(0,1),(1,1) | Validation chooses C, tied D; all max epochs, butterfly never active; one benign synthetic date |
| M7 | Seeds42/43/44; 25/82 individual quotes deleted per expiry | Unchanged SVI budgets, fresh MLP per seed, fixed penalty configuration D=(1,1); retained-only internal validation | M6 winner not reused as leak-free selection; opposite retained leg can carry near-identical IV information |

### 4.2 Results by experiment

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
behavior. Butterfly counts remain zero, but these calendar violations show that the fitted neural surface is not arbitrage-free. The reported results retain the original penalty coefficients.

### 4.3 Saved synthetic calibration and diagnostic detail

All rows below are synthetic, observation date 2000-01-03. Numbers are rounded for display; full precision is in the named metrics. Initial/final vectors use (a,b,rho,m,eta).

| M4 expiry | Initial vector | Fitted vector | RSS | nfev | Fit seconds | Max relative repricing error |
|---|---|---|---:|---:|---:|---:|
| 2000-02-02 | (-0.00792285773, 0.1, 0, 0, 0.2) | (0.008, 0.06, -0.3, 0.04, 0.2) | 1.44149749e-26 | 11 | 0.0133801 | 2.42606578e-12 |
| 2000-04-02 | (0.00264464176, 0.1, 0, 0, 0.2) | (0.015, 0.1125, -0.3, 0.04, 0.2) | 1.011361e-26 | 11 | 0.0125872 | 1.39260465e-12 |
| 2000-07-01 | (0.0177410696, 0.1, 0, 0, 0.2) | (0.025, 0.1875, -0.3, 0.04, 0.2) | 2.2774231e-26 | 10 | 0.0100833 | 8.93452564e-14 |
| 2001-01-02 | (0.0403857114, 0.1, 0, 0, 0.2) | (0.04, 0.3, -0.3, 0.04, 0.2) | 4.71896092e-26 | 18 | 0.0195418 | 5.1036148e-13 |

M4 records four successful slices, no IV failures or quote-preparation rejections, and zero examination/complete butterfly, calendar and negative-forward-variance locations on its finite grid. The median and p95 of date maxima both equal 2.42606578e-12 (fractional relative error). Only one synthetic date exists: these equal summaries are not multi-date market validation. The near-zero relative-price denominator floor is 1e-8; excluded/failed observations remain in coverage ledgers.

| M5 optimizer | Expiry | Converged | Final candidate RSS | Native iterations | Objective evaluations | Fit seconds |
|---|---|---|---:|---:|---:|---:|
| scipy | 2000-02-02 | True | 1.44149749e-26 | N/A | 11 | 0.0153175 |
| scipy | 2000-04-02 | True | 1.011361e-26 | N/A | 11 | 0.012201 |
| scipy | 2000-07-01 | True | 2.2774231e-26 | N/A | 10 | 0.0154099 |
| scipy | 2001-01-02 | True | 4.71896092e-26 | N/A | 18 | 0.0277265 |
| adam | 2000-02-02 | True | 9.93602028e-15 | 3502 | 3503 | 13.472588 |
| adam | 2000-04-02 | True | 9.98838498e-15 | 3318 | 3319 | 7.4736166 |
| adam | 2000-07-01 | True | 9.98819222e-15 | 4193 | 4194 | 9.6872716 |
| adam | 2001-01-02 | True | 9.96684415e-15 | 4256 | 4257 | 10.1959052 |
| lbfgs | 2000-02-02 | True | 5.81297715e-20 | 38 | 41 | 0.1495379 |
| lbfgs | 2000-04-02 | False | 7.10249286e-11 | 1000 | 1509 | 4.605613 |
| lbfgs | 2000-07-01 | True | 4.23920292e-20 | 33 | 40 | 0.1426735 |
| lbfgs | 2001-01-02 | True | 1.11992073e-19 | 41 | 48 | 0.1878871 |

SciPy native accepted iterations are unavailable, so function-evaluation history is not relabelled as iterations. Timings are measured machine-dependent values, not reproducible constants; setup and stopping conventions limit speed comparisons. The M5 failed L-BFGS slice is 2000-04-02. Its candidate RSS remains visible above, while failed candidates do not enter successful surface statistics. M5 complete-date median/p95 of date-max relative error are 2.42606578e-12 for SciPy and 5.77087543e-06 for Adam; both statistics are null for L-BFGS because the date is incomplete. Zero reported finite-grid violations for diagnosed fits do not certify the missing slice.

M7 removes 25 of 82 individual quotes per expiry (30.49% after integer rounding), leaving 228 retained quotes and 100 held-out targets per seed. Errors in the preceding M7 table are means and sample standard deviations of seed-level conditional errors, not pooled RMSEs. M7 neural training reaches MAX_EPOCHS; completion is not an optimization-convergence claim.

For the synthetic seed44 dense audit, docs/q1_m8_verification.json records w_T=-0.023403562931940847 at k=0.415,T=1. Central differences at h=1e-4 give -0.023403561291157615. An interior-k pair at k=0.355, T=0.8470319634703196 to 1 decreases total variance by 0.001629876721397805. Thus the saved 402/448/272 dense violations represent model behavior rather than sign noise. Full location/magnitude arrays remain in the M7 dense diagnostics, indexed by the saved artifact manifests; counts alone are not a claim that all locations are reproduced in this text.

### 4.4 Figures

The machine-readable complete original index is
`tests/q1/fixtures/verified_runs/manifest.json` → `original_artifact_inventory`
(filter `.html`), and the comparator's `figure_index`. Paths refer to preserved
ignored originals; the numeric evidence is losslessly tracked for clean replay.

| Source directory | Figures and interpretation |
|---|---|
| `results/verification/approach1/q1_synthetic_svi_verification/` | Seven M4 figures: smile, total variance, butterfly, calendar, surface, market/model prices, repricing distribution. Interpretation: synthetic in-sample recovery, not market evidence |
| `results/m5_final/verification/approach2/q1_synthetic_svi_optimizer_comparison/` | Six optimizer views: objective vs iteration/evaluation, objective vs time, parameters, convergence, repricing, arbitrage. Interpretation: native iterations unavailable for SciPy; L-BFGS failure and setup/timing differences affect comparison |
| `results/verification/approach2/q1_synthetic_neural_verification/` | Six M6 views: training/validation, smiles, surface, observed/predicted prices, arbitrage heatmaps, penalty ablation. Interpretation: validation-selected C, finite-grid checks, max epochs, inactive butterfly penalty |
| `results/verification/approach2/q1_synthetic_missing_quotes/` | Eight M7 groups: full IV, mask, full SVI reference, retained SVI refits, neural surfaces, held-out error vs k/T, robustness table. Interpretation: reference sees full data and is illustration only; report missing targets and dense violations |
| `results/comparison.html` | Question/mode-separated integrated tables; no global ranking or synthetic/empirical pooling |


The following figures show the synthetic experiments. The original HTML files are available in the local results directories; their inclusion in a separate checkout depends on transferring those files.

- Figure Q1-1: [butterfly](results/verification/approach1/q1_synthetic_svi_verification/butterfly.html).
- Figure Q1-2: [calendar](results/verification/approach1/q1_synthetic_svi_verification/calendar.html).
- Figure Q1-3: [market model prices](results/verification/approach1/q1_synthetic_svi_verification/market_model_prices.html).
- Figure Q1-4: [repricing error distribution](results/verification/approach1/q1_synthetic_svi_verification/repricing_error_distribution.html).
- Figure Q1-5: [total variance](results/verification/approach1/q1_synthetic_svi_verification/total_variance.html).
- Figure Q1-6: [volatility smile](results/verification/approach1/q1_synthetic_svi_verification/volatility_smile.html).
- Figure Q1-7: [volatility surface](results/verification/approach1/q1_synthetic_svi_verification/volatility_surface.html).
- Figure Q1-8: [arbitrage summary](results/m5_final/verification/approach2/q1_synthetic_svi_optimizer_comparison/arbitrage_summary.html).
- Figure Q1-9: [objective vs iteration](results/m5_final/verification/approach2/q1_synthetic_svi_optimizer_comparison/objective_vs_iteration.html).
- Figure Q1-10: [objective vs wall time](results/m5_final/verification/approach2/q1_synthetic_svi_optimizer_comparison/objective_vs_wall_time.html).
- Figure Q1-11: [optimizer convergence](results/m5_final/verification/approach2/q1_synthetic_svi_optimizer_comparison/optimizer_convergence.html).
- Figure Q1-12: [parameter comparison](results/m5_final/verification/approach2/q1_synthetic_svi_optimizer_comparison/parameter_comparison.html).
- Figure Q1-13: [repricing comparison](results/m5_final/verification/approach2/q1_synthetic_svi_optimizer_comparison/repricing_comparison.html).
- Figure Q1-14: [arbitrage heatmaps](results/verification/approach2/q1_synthetic_neural_verification/arbitrage_heatmaps.html).
- Figure Q1-15: [neural smiles](results/verification/approach2/q1_synthetic_neural_verification/neural_smiles.html).
- Figure Q1-16: [neural surface](results/verification/approach2/q1_synthetic_neural_verification/neural_surface.html).
- Figure Q1-17: [observed predicted prices](results/verification/approach2/q1_synthetic_neural_verification/observed_predicted_prices.html).
- Figure Q1-18: [penalty ablation](results/verification/approach2/q1_synthetic_neural_verification/penalty_ablation.html).
- Figure Q1-19: [training validation](results/verification/approach2/q1_synthetic_neural_verification/training_validation.html).
- Figure Q1-20: [full data implied volatility](results/verification/approach2/q1_synthetic_missing_quotes/full_data_implied_volatility.html).
- Figure Q1-21: [full data svi reference](results/verification/approach2/q1_synthetic_missing_quotes/full_data_svi_reference.html).
- Figure Q1-22: [held out errors maturity](results/verification/approach2/q1_synthetic_missing_quotes/held_out_errors_maturity.html).
- Figure Q1-23: [held out errors moneyness](results/verification/approach2/q1_synthetic_missing_quotes/held_out_errors_moneyness.html).
- Figure Q1-24: [neural predictions](results/verification/approach2/q1_synthetic_missing_quotes/neural_predictions.html).
- Figure Q1-25: [retained dropped quotes](results/verification/approach2/q1_synthetic_missing_quotes/retained_dropped_quotes.html).
- Figure Q1-26: [retained svi refits](results/verification/approach2/q1_synthetic_missing_quotes/retained_svi_refits.html).
- Figure Q1-27: [robustness comparison](results/verification/approach2/q1_synthetic_missing_quotes/robustness_comparison.html).

## 5. Discussion

Classical SVI recovers the synthetic generating family extremely accurately, which is useful numerical verification but also an unusually favorable setting for SVI. It cannot establish superiority on noisy market surfaces. Adam's convergence declarations do not imply that it passes the original strict parameter-recovery tolerance. L-BFGS coverage failures prevent interpreting conditional error as a complete-cohort score.

The neural calendar penalty slightly reduces the measured M6 held-out price MSE in this fixture. The identical A/B and C/D checkpoints show that the butterfly penalty is inactive, so this experiment provides no evidence that the butterfly penalty improves learning. All configurations reach their epoch budget. These findings do not justify a universal claim about penalty effectiveness or optimization convergence.

The missing-quote study tests individual-quote deletion, not removal of all information at a strike: a retained opposite option leg can carry closely related volatility information. Only two extrapolation targets are present. The interpolation/extrapolation difference is descriptive, with insufficient support for a general extrapolation conclusion. Dense-grid calendar and forward-variance violations demonstrate why an original-expiry-only check is insufficient.

The main empirical limitation is independent of these numerical results: the archived source evidence does not establish a model-ready historical ETF panel. No inference about Chinese ETF smiles, market pricing efficiency, empirical calibration speed or market benchmark attainment follows from the synthetic tables.

## 6. Verification and reproducibility

The project pins Python and dependencies through .python-version, pyproject.toml and uv.lock, using scripts/project.py to isolate environments and caches inside the repository. The prior M8 verification records 264 passing tests; this is historical evidence, not a newly executed full suite. The final handover ran two targeted recovery tests (2 passed in 2.11 seconds), offline source verification, 24-snapshot restoration, and offline lock/no-change synchronization checks. No training or data download occurred during that handover.

The tracked compressed bundle supports exact numerical artifact replay and checkpoint inference, not an independent training replication. Original self-contained HTML figures are indexed but ignored by Git; users of a clean clone must not assume that those original HTML files are included. The comparison page can be regenerated from restored metrics. Use fresh output paths:

```powershell
# Initial provisioning may need network; subsequent commands are offline.
py -3.11 -B scripts/project.py setup
py -3.11 -B scripts/project.py uv run --locked --offline python data/fetch.py --question q1 --offline --verify-only
py -3.11 -B scripts/project.py uv run --locked --offline python code/q1/final_recovery.py --output .tmp/q1_report_recovery.json
py -3.11 -B scripts/project.py uv run --locked --offline python code/q1/validation_bundle.py restore --destination .tmp/q1_report_replay
py -3.11 -B scripts/project.py uv run --locked --offline python code/compare.py --results-dir .tmp/q1_report_replay/results --output .tmp/q1_report_replay/comparison.html
py -3.11 -B scripts/project.py uv run --locked --offline python -m pytest tests/q1/test_final_recovery.py -q
```

## 7. Conclusion

The implemented Q1 software covers classical and differentiable SVI, neural price-space fitting, penalty ablation, quote-removal experiments and explicit arbitrage diagnostics. The evidence supports numerical implementation and qualified synthetic findings. It does not establish completion of the historical-data requirement or attainment of the empirical repricing benchmark. The main contribution is therefore a numerical comparison with explicit failure diagnostics. Applying these methods to Chinese ETF options still requires a verified historical panel with adequate strike and maturity coverage.

## 8. Statement of work

OpenCode assisted with Q1 implementation, data checks, synthetic experiments, tests, figures and reproducibility tools. Recorded student directions determined the milestone scope, continuation with synthetic verification and the fixed penalty configuration for the missing-quote experiment. Codex assisted with handover materials and the wording and organization of this report. These activities are distinguished from mathematical review and final authorship; no unrecorded student contribution is attributed here.

## 9. Supporting evidence

Data provenance and the eligibility audit are recorded in `data/metadata.json`, `docs/q1_final_recovery_evidence.json` and `docs/q1_final_verification.json`. Mathematical and experimental checks are documented in `docs/q1_m4_verification.json` through `docs/q1_m8_verification.json`. The numerical snapshots and original artifact inventory are indexed in `tests/q1/fixtures/verified_runs/manifest.json`.

The historical handover audit reports successful hash and decompression checks for 24 compressed snapshots and matching sizes and SHA-256 hashes for 27 original HTML figures. These are previously recorded checks, rather than new tests performed during report editing.

## 10. Coverage and remaining limitations

| Examination requirement | Report coverage | Evidence/remaining limitation |
|---|---|---|
| 3.1 Sources, curves, dividend assumption and historical scope | Sections 1–2 | Real archive and rejection waterfall reported; zero eligible observations; required historical chains and approved empirical conventions remain unavailable |
| 3.2 Per-strike parity, filtering, Brent for calls/puts, fixed-grid surface | Sections 2 and 3.1–3.3 | Equations, thresholds and grid explicit; synthetic ledgers preserve failures; no empirical parity or IV table exists because gate is blocked |
| 3.3.1 SVI constraints, initialization and solver | Sections 3.2 and 4.3 | Synthetic initial/fitted parameters, settings and calibration table supplied |
| 3.3.2 Butterfly/calendar checks by date/expiry, locations | Sections 3.3 and 4.2–4.3 | Synthetic checks, missing-fit caveats and dense violation example supplied; full arrays in saved diagnostics; no empirical diagnostic claims |
| 3.3.3 Total-variance interpolation and negative forward variance | Sections 3.3 and 4.3 | Formula and measured dense failures retained |
| 3.3.4 Repricing maxima, cross-date median/p95 | Sections 3.3 and 4.3 | Synthetic slice maxima and date aggregation supplied; one artificial date does not establish historical-date coverage |
| 3.4.1 Torch SVI and convergence comparison | Sections 3.2 and 4.1–4.3 | Matched protocol, objective/evaluation/time tables and profiles; L-BFGS failure and unavailable SciPy iterations preserved |
| 3.4.2 Neural architecture, price-space loss, ablations | Sections 3.4 and 4.1–4.2 | Synthetic architecture, coefficients, held-out tables and inactive butterfly penalty disclosed |
| 3.4.3 Random 30% removal and extrapolation comparison | Sections 4.1–4.3 and 5 | Three synthetic seeds, rounded fraction, coverage, retained-leg information and only two extrapolation targets disclosed |
| 3.4.4 Same arbitrage verification for B | Sections 3.3 and 4.2–4.3 | Original-grid and dense-grid results distinguished; no arbitrage-free neural claim |
| 3.5 Indicative benchmarks and all-date calibration table | Sections 1, 4.3 and 7 | Synthetic evidence only; empirical <2% target, empirical zero interior violations and required historical calibration table are NOT demonstrated |
