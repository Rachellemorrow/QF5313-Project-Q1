# Q1 M5 — Differentiable SVI calibration (Approach B, Part 1)

## Outcome and scope

- **SOFTWARE_VERIFICATION: PARTIAL**: differentiable SVI, both optimizers, matched
  comparison, shared diagnostics, reporting and the empirical gate are implemented.
  The default L-BFGS run does **not** recover all four M4 fixture slices to the
  declared tolerance: one reaches its 1,000-iteration budget. This is not relabelled
  as convergence, even though its residual is small.
- **EMPIRICAL_REQUIREMENTS: BLOCKED**: the registered historical archive has 606
  observations and zero strictly eligible calibration rows. No empirical fit,
  repricing accuracy or arbitrage-free market surface is claimed.

Q1 only. No M6 neural surface, penalties, ablations or missing-quote experiment was
started. Approach A, M2A mathematics, original M4 fixtures, upstream data, shared
configuration, dependencies and Q2/Q3 remain unchanged. This document supports the
student's report; it is not the group report or an authentic session export.

## Files and reuse

New implementation:

- `code/q1/torch_svi.py`: float64 CPU model, differentiable transforms, Adam/L-BFGS,
  evaluation accounting, stopping checks and explicit failure records.
- `code/q1/torch_comparison.py`: matched-input orchestration and temporary serial
  SciPy timing wrappers; runtime information and course-schema metrics.
- `code/q1/torch_comparison_figures.py`: six self-contained Plotly views.
- `code/approach2_q1_torch.py`: `argparse` CLI, immutable local output and manifests;
  `--component svi` leaves room for later integration without implementing M6.
- `tests/q1/test_torch_svi.py`, `tests/q1/test_torch_comparison.py`: numerical,
  optimizer, failure, baseline-preservation, diagnostic, artifact and CLI tests.
- `docs/q1_m5_verification.json`: compact evidence from the actual saved runs.

Reuse is deliberate: `approach_a.calibrate_quotes` performs the existing quote
preparation, pairing, Brent IV inversion, slice grouping, weights and SciPy fit.
`classical_svi.verify_surface`, `reprice_quote` and `repricing_summary` verify the
Torch fits without a second implementation of arbitrage, interpolation or pricing.
`svi` supplies validated raw parameters and NumPy verification; `pricing` supplies
the original forward Black–Scholes calls/puts. `development_quotes` uses the
unchanged explicitly synthetic M4 fixture. The strict `empirical_gate` is unchanged.

Pipeline: **load → verify → split → fit → evaluate → save**. Split means exact
valuation-date/expiry slices, not a random holdout. All quoted repricing accuracy
is **in-sample**, not an out-of-sample comparison.

## Model and parameter domains

The same raw SVI formula is evaluated entirely with Torch operations:

```text
w(k) = a + b*[rho*(k-m) + sqrt((k-m)^2 + eta^2)]
v_min = 1e-10 + softplus(z0)
b     = softplus(z1)
rho   = (1-1e-8)*tanh(z2)
m     = z3
eta   = 1e-8 + softplus(z4)
a     = v_min - b*eta*sqrt((1-rho)*(1+rho))
```

For real finite latent coordinates this makes the analytic global minimum
`a+b*eta*sqrt(1-rho^2)` strictly positive. Requiring only `a>=0` would be unnecessarily
restrictive; merely checking a few quotes would not guarantee positive wings.
For b>0, solving `w'(k)=0` gives
`k_min=m-rho*eta/sqrt(1-rho^2)`; substitution yields that global minimum, since
`w''(k)=b*eta^2/((k-m)^2+eta^2)^(3/2)>=0`. The b=0 boundary is a flat smile in
SciPy's domain but is excluded from the exact smooth inverse used here.
Raw reconstruction, global minimum and positive variance on quotes/the M4 grid
are checked. Invalid/nonfinite intermediate variance or gradients fail explicitly;
no clipping, projection or repair is hidden in the objective.

**The domains are not mathematically equivalent.** SciPy permits the closed faces
`b=0`, `v_min=1e-10`, `eta=1e-8`, `rho=±(1-1e-8)`; finite real softplus/tanh cannot
represent these endpoints. Torch covers their strict interior, with m unbounded.
Finite-precision saturation may numerically reach a safety margin, and cancellation
in reconstructed a is still possible. Such numerical effects are not a proof of
domain equivalence. Stable inverse-softplus/atanh encode the exact shared raw
initialization to roundoff; a boundary initialization is rejected, never nudged.

## Matched objective, initialization and execution

All methods use the same float64 CPU observations, per-strike weights, raw initial
parameters, log-moneyness and total-variance targets:

```text
RSS = sum_i weight_i * (w_model(k_i) - sigma_IV_i^2*T)^2
```

Both call and put inversions remain observations. A paired strike has total weight
one (0.5 per side in this fixture). There are no price-space, butterfly or calendar
penalties in M5 SVI; those must not be confused with the future neural objective.
Input fingerprints, weights and raw starts are retained for every dated slice.
The M4 initialization does not use fixture truth:

```text
v_min=max(0.5*median(target_w),10*positive_floor)
b=0.1, rho=0, m=median(k), eta=0.2
```

The existing SciPy solver is unchanged: analytic-Jacobian TRF, `x_scale='jac'`,
linear loss, `max_nfev=2000`, `ftol=xtol=gtol=1e-12`. Its native cost is half RSS.

Torch defaults:

| Setting | Adam | L-BFGS |
|---|---|---|
| Learning rate | 0.03 | 1.0 |
| Budget | 10,000 updates | 1,000 native iterations / 2,000 native function evaluations |
| Other settings | default betas (0.9,0.999), eps=1e-12, no weight decay | history=50, strong-Wolfe line search |
| LR schedule | multiply by 0.5 every 2,000 updates | none |
| Native tolerances | no native convergence test | latent RSS gradient=1e-12; change=1e-18 |
| Declared success | RSS<=1e-14 **or** physical half-RSS gradient infinity norm<=1e-10 | same post-run check |

These tolerances are **not equivalent stopping rules**. SciPy uses bound-scaled
optimality and relative function/step tests in physical coordinates. Torch native
L-BFGS uses latent coordinates; the physical post-check is not its internal stopping
rule. Small latent gradients can result from transform saturation. Both latent
RSS and physical half-RSS gradient norms are reported, not silently interchanged.
The synthetic CLI additionally requires **all four** fits to converge, grid
variance error <=1e-6 and raw parameter error <=1e-4. These tolerances were not
loosened after the observed failure.

These M5 CLI recovery thresholds are **looser than the original M4 fixture's**
`1e-9` total-variance / `1e-6` parameter recovery thresholds. The source fixture
remains unchanged and its original thresholds are not silently redefined. Applying
the original M4 thresholds to the saved M5 candidates gives **SciPy 4/4, Adam 0/4,
L-BFGS 3/4** converged slices meeting both. Adam's declared RSS stopping threshold
is insufficient for M4-level recovery precision on these slices. Thus even Adam's
4/4 optimizer success is **not** a claim of M4-precision recovery. Both standards
and actual errors must be retained in any student-authored comparison.

## Accounting and timing limitations

- Adam records each actual objective/backward evaluation, including initial and
  terminal checks. Its update count differs from evaluation count by one on normal
  completion. Every history row includes the actual update count, RSS, gradients,
  learning rate and elapsed optimization time.
- L-BFGS counts every closure trial (including line-search probes), native `n_iter`
  and `func_evals`, one outer step call, and a separate final objective/gradient
  evaluation. Trial histories are not called accepted optimizer iterates. The exact
  native stop-reason enum is not exposed by PyTorch and is marked unavailable.
  Native evaluation limits are passed through as configuration; a strong-Wolfe
  search may perform additional probes before checking a budget. Actual counts,
  rather than an assumed hard cap, are the evidence.
- SciPy retains its actual nfev/njev and objective history. Its accepted optimizer
  iteration count is unavailable (`null`), not fabricated from nfev. A transparent
  temporary wrapper times residual evaluations without changing numerical values,
  Jacobian or solver settings. The existing extra post-fit residual call is counted
  separately; traces are segmented by fit invocation, even for identical inputs.
  Wrappers restore on exceptions and are **serial-only**, not thread-safe tooling.
- `fit_seconds` includes setup, logging, optimization and final validation. Torch
  setup and optimization time are separately visible. Objective wall-time plots
  start at the first evaluation, excluding setup; the convergence table retains
  total fit/setup times. The first optimizer use has substantial setup overhead.

Runs use one deterministic initialization, fixed solver order, CPU float64 and
seed 42 (CLI override available). There is no stochastic mini-batching. Runtime
Python/library/OS/CPU/thread information is recorded; timing is not deterministic
and bitwise portability across libraries/CPUs/thread settings is not guaranteed.
This is a **software comparison**, not a repeated, isolated performance benchmark.
Different coordinates, feasible boundaries, stopping rules and instrumentation
prevent a claim of a perfectly controlled comparison or universal superiority.

## Actual synthetic result and failure

The unchanged M4 fixture contains 328 call/put quotes, 164 paired strikes, one
artificial observation date and four artificial expiries. These are **not market
observations** and cannot satisfy the required multi-year/daily-chain coverage.

The first saved run at the default verification path produced:

| Method | Converged | Largest final RSS, including failed candidate | Largest known grid w error | Largest raw parameter error | Total fit seconds |
|---|---:|---:|---:|---:|---:|
| SciPy | 4/4 | 4.7189609e-26 | 1.4066526e-13 | 2.2937208e-12 | 0.054517 |
| Adam | 4/4 | 9.9883850e-15 | 1.0437453e-7 | 8.2268113e-6 | 37.877484 |
| L-BFGS | 3/4 | 7.1024929e-11 | 8.0709562e-6 | 3.4191390e-4 | 4.799106 |

The final reporting run and its hashes/times are separately recorded in
`docs/q1_m5_verification.json`; timing varies. Initial output is preserved rather
than overwritten. Reporting was strengthened after inspection so aggregate maxima
include **all available candidates**, not only successful fits; successful-only RSS
is separately labelled. The full failed candidate was always present in its slice
record and comparison rows.

| Synthetic expiry | SciPy nfev (iterations unavailable) | Adam updates / objective evals | L-BFGS native iterations / closure calls / total objective evals |
|---|---:|---:|---:|
| 2000-02-02 | 11 | 3502 / 3503 | 38 / 40 / 41 |
| 2000-04-02 | 11 | 3318 / 3319 | **1000 / 1508 / 1509 (failed)** |
| 2000-07-01 | 10 | 4193 / 4194 | 33 / 39 / 40 |
| 2001-01-02 | 18 | 4256 / 4257 | 41 / 47 / 48 |

The failed L-BFGS candidate has final physical half-RSS gradient infinity norm
`3.3153416e-7`, well above the declared `1e-10`. Its rho/eta errors are about
`3.42e-4` / `3.10e-4`. It was not accepted based solely on its small objective.
No restart, higher budget or rescaled-loss tuning was silently substituted for the
primary comparison. Determining a more reliable configuration is a remaining
software limitation, not evidence against all uses of L-BFGS.

SciPy/Adam complete-date maximum relative repricing errors are `2.4260658e-12` /
`5.7708754e-6` (ratios, not percentages). L-BFGS's maximum over its three successful
slices is `1.4688401e-8`, but its **complete-date median/p95 are null** because one
slice failed. One synthetic date cannot support empirical cross-date inference.

## Shared arbitrage verification

The exact M4 grid is 201 common log-moneyness points on [-0.5,0.5]. Synthetic quote
support is [-0.4,0.4]; outer nodes are model-wing evaluation, not observations.
Every successful fit is checked for raw validity/global and grid positivity,
examination-defined butterfly g, separately labelled complete g including w''/2,
all maturity-pair calendar inequalities, adjacent negative forward variance and
piecewise-linear total-variance interpolation. Violations retain locations and
magnitudes. No repair or suppression is performed.

The benign fixture has zero butterfly (both definitions), calendar and negative
forward-variance violations **among successful fitted slices**. L-BFGS's failed
expiry is explicitly missing from these diagnostics, not certified arbitrage-free
or filled using another method. Tests also inject deliberately arbitrage-violating
synthetic candidates and confirm reported locations and magnitudes survive the
comparison orchestration. Finite-grid checks are not a global no-arbitrage proof.

## Outputs, commands and expected exits

Each run uses `{params, metrics, fit_seconds, history}`. Full optimization histories,
initial/final parameters, gradients, failures, quote/IV/parity ledgers, diagnostics,
input fingerprints and comparison rows are retained in `metrics.json`. Additional
files: `optimizer_comparison.csv`, `parameter_comparison.json`, SHA-256
`artifact_manifest.json`, and six inline-JavaScript Plotly HTML views:

- `objective_vs_iteration.html` (SciPy evaluations are explicitly not iterations)
- `objective_vs_wall_time.html`
- `parameter_comparison.html`
- `optimizer_convergence.html`
- `repricing_comparison.html`
- `arbitrage_summary.html`

Synthetic titles explicitly say **NOT EMPIRICAL MARKET RESULTS**. Development
artifacts are outside `results/approachN/*`; their empirical eligibility is false.
Large generated HTML/history files are gitignored; code, fixtures and compact
labelled verification evidence are committed. Recreate them offline:

```powershell
py -3.11 -B scripts/project.py uv run --locked --offline python code/approach2_q1_torch.py --help
py -3.11 -B scripts/project.py uv run --locked --offline python code/approach2_q1_torch.py --mode development --dataset q1_synthetic_svi_optimizer_comparison --seed 42
py -3.11 -B scripts/project.py uv run --locked --offline python code/approach2_q1_torch.py --mode empirical --dataset q1_510050 --seed 42
# Final reporting run: preserves the initial evidence and includes runtime/all-candidate aggregate reporting.
py -3.11 -B scripts/project.py uv run --locked --offline python code/approach2_q1_torch.py --mode development --dataset q1_synthetic_svi_optimizer_comparison --seed 42 --output-dir results/m5_final
py -3.11 -B scripts/project.py uv run --locked --offline python -m pytest tests/q1 -q
```

An existing output directory is refused; reruns require another repository-local
base. The default synthetic comparison currently returns **1** (PARTIAL), not 0.
The direct empirical CLI returns **2** (BLOCKED); the existing launcher maps child
nonzero exits to 1. These are expected only after inspecting the actual saved
status; unrelated failures must not be ignored. The blocked run saves eligibility,
metrics and manifest only, with zero fit time and no fabricated figures/parameters.

## Verification process and intermediate failure

The pre-implementation M2A/M4 numerical suite passed **111 tests**. The initial
new model/gradient/short-optimizer suite passed **12 tests in 14.05 seconds**.
The first comparison-test run had **4 passed and 5 setup errors in 105.36 seconds**:
the module fixture incorrectly assumed the full synthetic CLI would exit 0.
The actual L-BFGS failure correctly made it exit 1. The tests were corrected to
assert the observed nonconvergence, retained candidate, strict stopping thresholds,
missing diagnostics and incomplete-date repricing; the optimizer budgets and
acceptance tolerances were **not changed** to make the run pass.

An intermediate full regression run had **227 passed, 1 failed in 212.82 seconds**.
The new cross-library objective-parity test used only an absolute `1e-24` tolerance;
Torch/NumPy RSS differed by `1.54e-23` (about `2.2e-13` relative) on the failed
L-BFGS candidate. A follow-up saved-artifact check showed that an initial correction
using relative `1e-10` plus absolute `1e-24` still rejected Adam's `2.31e-24` RSS
roundoff on the longest expiry. Relative error in squared near-zero residuals is
ill-conditioned. The final assertion compares `sqrt(RSS)` to the recomputed
weighted residual norm, with absolute tolerance
`64*float64_epsilon*max(1,norm(sqrt(weights)*target))` in residual units.
This explicit scale/operation-count roundoff budget does not change the fitted
objective, optimizer, stopping thresholds or surface/parameter acceptance criteria,
and does not promote the failed fit.
The already-running intermediate relative-RSS correction finished with **227
passed, 1 failed in 226.17 seconds**; the final norm-based correction was verified
on all 12 saved candidates and sent through a separate complete regression run.

Tests separately check actual successful-slice recovery, function/derivative
parity, physical-gradient agreement with the M4 analytic Jacobian, latent-gradient
finite differences, constraints, deterministic initialization, insufficient and
invalid inputs, baseline preservation, temporary-wrapper restoration, matched
weights/starts/fingerprints, all shared diagnostics/repricing, inline HTML and file
hashes, origin separation, immutable runs, CLI arguments and strict empirical
blockage. The complete final regression outcome is recorded in the companion
verification JSON. Passing tests that verify correct failure reporting is not a
claim that all calibration acceptance criteria were met.

**Final complete Q1 regression: 228 passed, 0 failed in 228.02 seconds**, including
all 206 existing tests and 22 new M5 tests. The five final saved-artifact validation
functions also passed on the actual reporting output, including all six inline
figures and their manifest hashes. `git diff --check` passed; the inspected change
set contains only the eight new Q1 implementation/test/documentation files.

## Remaining blockers and stop

Historical complete daily/two-year ETF chains, effective-dated metadata and units,
approved curve/forward/dividend conventions and offline data rights remain
unresolved upstream. Strict eligibility is not relaxed. Empirical pricing targets
and arbitrage benchmarks remain untested. Student approval of methodological
choices is still required for subsequent scientific experiments.

**Stop after M5. Do not start M6 without authorization.**
