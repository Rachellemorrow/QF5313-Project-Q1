# Q1 M2A — Data-independent mathematical foundations

## Scope and status

Implemented and verified numerical foundations only. The student explicitly
authorized this data-independent milestone while M1/M1.5 historical recovery
remains **BLOCKED**. This does not complete the original M2 data pipeline or any
real-market experiment. No historical prices, market snapshots, audit records,
dependency pins, Q2/Q3 files, calibration scripts or neural models were changed.

All synthetic inputs are software-verification cases. They are not historical
observations and cannot replace the examination's real-market experiments or
justify its empirical repricing/arbitrage benchmarks.

## Files and import convention

- `code/q1/pricing.py`: forward/spot Black–Scholes, bounds, Brent IV, parity and a
  minimal numerical quote adapter.
- `code/q1/svi.py`: SVI values/derivatives, positivity, distinct butterfly formulas,
  calendar differences, total-variance interpolation and forward-variance checks.
- `code/q1/_validation.py`: common Q1-only float64/broadcast validation.
- `code/q1/__init__.py`: Q1 package marker; no data loading on import.
- `tests/q1/conftest.py`: Q1 test import path and explicitly synthetic fixture loader.
- `tests/q1/test_pricing.py`, `test_svi.py`, `test_math_accuracy.py`: mathematical
  unit/accuracy verification, separate from the existing environment/data audits.
- `tests/q1/fixtures/synthetic/m2a_cases.json`: labelled deterministic software
  fixtures; no observation dates or empirical market-result records.

Future top-level approach scripts under `code/` can import `q1.pricing` and
`q1.svi`. The Q1 test configuration adds the repository's `code/` directory to the
test import path. No shared package/build configuration was changed. Data loading,
calibration, pricing, verification and experimental reporting remain separate.

## Pricing conventions and edge cases

For caller-supplied positive forward F, strike K, discount factor D and time T in
years, define v = sigma*sqrt(T), d1 = ln(F/K)/v + v/2 and d2 = d1-v:

```text
C = D * [F*N(d1) - K*N(d2)]
P = D * [K*N(-d2) - F*N(-d1)]
```

These are the examination's forward-price equations. `call_price`, `put_price`
and `black_scholes` broadcast NumPy float64 inputs; scalar inputs return Python
floats. Explicit scalar wrappers reject array inputs and share the same numerical
kernel, preventing scalar/vector convention drift. The kernel evaluates the OTM
side using log normal CDFs and `expm1`, then uses parity for the ITM side. At exact
ATM an `erf` identity retains tiny positive premiums. Computed model prices are
bounded against floating-point roundoff; **input market prices and arbitrage
diagnostics are never clipped, repaired or discarded**.

- Prices are **per underlying unit**, not cash per option contract. A contract unit
  or multiplier is metadata, not an automatic premium scaling instruction.
- Rates/dividend yields are annual continuously compounded **decimals**, not
  percentage quotes. `discount_factor(r,T) = exp(-r*T)`.
- `forward_from_spot(S,r,q,T) = S*exp((r-q)*T)` requires an explicit q.
  `black_scholes_from_spot` derives F and D from the same explicit r/q/T.
- Supplying F/D directly permits a future independently constructed term curve.
  D greater than one is allowed for negative rates. No SHIBOR-to-zero-curve method,
  dividend estimate, discrete-dividend treatment or empirical day count is chosen.
- `log_moneyness(K,F)` returns ln(K/F) without forming a potentially overflowing
  ratio. Evaluation-grid limits/count must be explicitly supplied.
- T=0 or sigma=0 uses the intrinsic-value limit; D must be exactly one at T=0.
  At expiry, both price bounds equal the unique payoff.
- Nonfinite/complex inputs, nonpositive F/K/D, negative time/volatility,
  incompatible shapes, unrepresentable results and standard-deviation
  overflow/underflow are rejected explicitly. Extremely small OTM premiums may
  underflow to zero; they must not be interpreted as identifiable IV observations.

## Brent implied volatility contract

`implied_volatility` calls `scipy.optimize.brentq` using **[1e-4, 5]** for calls
and puts. Default xtol/rtol are 1e-12; the separate absolute premium-residual check
is 1e-10 in the caller's per-unit premium scale. `maxiter` defaults to 100.

Every call returns an `IVResult` with volatility or `None`, convergence flag,
reason/message, iterations, evaluation count, bracket/prices, premium residual and
SciPy flag. Evaluation counts include bracket checks and final repricing. Exact
bracket endpoint matches have zero iterations rather than relying on solver endpoint
iteration bookkeeping. Identical inputs/settings return identical diagnostics in
the tested environment.

Distinct failures include invalid inputs/prices, economic price-bound violations,
expired/unidentifiable boundary quotes, flat float64 brackets, valid prices outside
the mandated volatility bracket, iteration exhaustion and failed residual checks.
Failures do not carry a successful volatility. In particular, an intrinsic or
saturated premium is not assigned an arbitrary bracket endpoint. `implied_volatility_batch`
broadcasts numeric inputs and returns one diagnostic object per quote, preserving
failures and ordering. Invalid batch shapes/nonnumeric arrays fail at the batch
boundary rather than silently dropping records.

## Parity: preserve the examination diagnostic

```text
examination_parity_residual:
    C - P - [S - K*exp(-r*T)]

dividend_adjusted_parity_residual (separate function and label):
    C - P - [S*exp(-q*T) - K*exp(-r*T)]
```

The first formula is unchanged. If synthetic prices use nonzero q, its residual
need not be zero; that does not authorize substituting the adjusted diagnostic.
Both diagnostics support broadcast arrays and validate their inputs.

## SVI calculus and diagnostics

For x=k-m and R=hypot(x,eta):

```text
w(k)   = a + b*[rho*x + R]
w'(k)  = b*[rho + x/R]
w''(k) = b*eta^2/R^3
```

`SVIParameters` enforces finite scalar parameters, b>=0, |rho|<1 and eta>0.
Negative a is permitted if the required variance domain remains positive.
Default variance evaluation rejects nonpositive values on the supplied grid.
`variance_positivity` instead exposes raw values and every nonpositive location;
it also reports the analytical global minimum:

```text
min_k w(k) = a + b*eta*sqrt(1-rho^2)
```

Parameter constraints alone do **not** prove positivity or absence of arbitrage.
Butterfly diagnostics require positive w and retain separate labels:

```text
g_examination = (1 - k*w'/(2*w))^2 - (w'^2/4)*(1/w + 1/4)
g_complete    = g_examination + w''/2
```

Returned diagnostics include full values, violation masks, counts, locations and
magnitudes. A test deliberately produces a negative examination diagnostic while
the complete expression is positive; another produces violations of both. The
complete expression never overwrites the examination result. Tolerance affects
the violation flag only, not the retained raw value; default tolerance is zero.

Calendar differences are `w_later(k)-w_earlier(k)` on a **common supplied
log-moneyness grid** with ordered maturities. They are not comparisons at the same
strike when forwards differ, and the routine does not remap strike grids.

`interpolate_total_variance` is piecewise linear in w, not sigma, with maturity as
the first array axis. It supports multiple grid dimensions and query maturities.
It rejects extrapolation, duplicate/unsorted maturities, invalid values and an
explicit T=0 anchor with nonzero variance. It never adds a zero anchor implicitly.
Nonmonotone positive variances are retained, not repaired. Forward variance is the
interval slope `(w[j+1]-w[j])/(T[j+1]-T[j])`; negative slopes are reported with
interval endpoints, grid indices and magnitudes. A T=0 interpolation anchor is
distinct from a positive-maturity SVI slice.

## Interface for a future real-data pipeline

`ForwardQuote.from_record` accepts the existing audit-style names `price`,
`strike`, `option_type` (C/P) and optional `contract_id`, **plus explicitly prepared**
`forward`, `time_to_expiry` and `discount_factor`. It does not load CSVs, derive
calendar times, infer missing attributes, scale by `contract_unit`, select dividends
or certify provenance/eligibility. Missing numerical fields are errors.

This is a minimal numerical adapter, not a claim that the final canonical data
pipeline schema is complete. The future loader must retain observation/expiry
dates, source, units and adjustment validity; normalize premium/contract units;
validate date-effective metadata and quote quality; and supply approved F/T/D.
Unknown historical attributes and the M1 `model_ready=False` flags are **not**
overridden by this library. Out-of-bound premiums can be retained by a caller and
reported as failed inversions, rather than silently filtered at loading time.

## Actual verification results

Locked environment: Python **3.13.18**, NumPy **2.5.4**, SciPy **1.18.1**,
pytest **9.2.0**, uv **0.13.7**, Windows AMD64.

- Final mathematical suite: **88 passed, 0 failed**, 3.48 seconds.
- Full Q1 suite including all M0/M1/M1.5 regressions: **130 passed, 0 failed**,
  25.41 seconds.
- Earlier development checks also passed (82 tests before additional edge checks,
  then 85); they are not additional independent market experiments.

Measured errors on **512 clearly synthetic call/put cases**, NumPy seed 42:

| Software-verification quantity | Measured maximum absolute error | Acceptance bound |
|---|---:|---:|
| Price versus independent scalar `math.erfc` evaluation | 7.105427357601002e-14 | 2e-12 |
| Recovered IV versus known synthetic sigma | 4.4220183070819985e-13 | 5e-10 |
| IV repricing premium residual | 1.5372592088169768e-11 | 1e-10 |
| Forward call–put parity | 7.105427357601002e-15 | 5e-13 |
| SVI analytical first derivative versus centered difference | 5.375523637329849e-11 | 1e-9 |
| SVI analytical second derivative versus centered difference | 4.549757715022196e-08 | 1e-6 |

Finite-difference steps were 1e-5 and 1e-4, respectively. The independent pricing
reference evaluates the specified equations directly; the production kernel uses
log-CDF/OTM/parity evaluation. Separate tests check textbook reference prices,
scalar/vector agreement, nonzero dividends, negative rates, zero time/volatility,
very small ATM prices, extreme log-moneyness, bounds/bracket failures, actual Brent
iteration exhaustion, residual failure, parameter constraints, positivity and
deliberately constructed butterfly/calendar/forward-variance violations.

The temporary accuracy JSON is labelled `SYNTHETIC_SOFTWARE_VERIFICATION_ONLY` and
written beneath repository-local pytest `.tmp/` output. No synthetic observations,
metrics or figures are placed in `data/`, empirical `results/`, or audit archives.
Small last-bit differences on other hardware/library versions are possible; tests
use documented tolerances rather than claiming cross-platform bitwise identity.

## Exact executed commands

Run from the repository root in PowerShell:

```powershell
py -3.11 -B scripts/project.py uv lock --check --offline
py -3.11 -B scripts/project.py uv sync --locked --check --offline
py -3.11 -B scripts/project.py uv run --locked --offline python -m pytest tests/q1/test_pricing.py tests/q1/test_svi.py -q
py -3.11 -B scripts/project.py uv run --locked --offline python -m pytest tests/q1/test_pricing.py tests/q1/test_svi.py tests/q1/test_math_accuracy.py -q -s
py -3.11 -B scripts/project.py uv run --locked --offline python -m pytest tests/q1 -q
```

These commands require no data download, new package or global environment change.

## Remaining limitations and next-milestone boundary

- IV sensitivity may be too small for a market quote's precision; a converged
  floating-point root is not a claim of statistical identification or liquidity.
  Extreme tails can underflow and saturated premiums are not inverted successfully.
- Float64 cannot represent arbitrary scales. Invalid/unrepresentable cases raise
  errors or IV failure records; users must supply appropriate units and tolerances.
- No discrete-dividend model, day-count/calendar policy, SHIBOR curve bootstrap,
  historical-contract adjustment policy, calibration objective/solver, optimizer
  initialization, SVI constraint transformation or arbitrage repair is implemented.
- Grid diagnostics are finite-grid checks, not a global proof that a fitted
  surface is arbitrage-free. No calibration or empirical benchmarks are claimed.
- The numerical foundations are ready for **Approach A calibration implementation
  after student authorization**. Actual market calibration/experiments still need
  licensed complete historical chains, effective-dated metadata/multipliers and
  the source/coverage/permission decisions documented in the unchanged M1/M1.5 audits.

Stop after M2A. Neither the next implementation milestone nor historical experiments
are automatically authorized by these mathematical tests.
