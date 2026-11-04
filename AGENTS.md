# AGENTS.md — QF5313 Advanced Programming 2026

## 1. Project Overview

This repository contains the group project for **Advanced Programming 2026**, Master of Quantitative Finance.

The group consists of four students and has selected three examination questions:

- **Question 1:** Implied Volatility Surface of China ETF Options
- **Question 2:** Monte Carlo Pricing and Hedging on Chinese Futures
- **Question 3:** China Macro Panel: Regimes, Nowcasting, and Indicator Clustering

For each selected question, both Option A and Option B must be implemented.

The repository must therefore contain **six complete approaches**, along with the required data pipelines, mathematical verification, experiments, comparisons, and final deliverables.

### Responsibility Boundary

**The current user's assigned responsibility is Question 1.**

The coding agent working under this user's session must prioritize and implement Q1 only.

Q2 and Q3 belong to other group members unless the user explicitly authorizes work on them.

The agent must understand the complete group project architecture but must NOT independently implement, modify, refactor, overwrite, or delete Q2/Q3 code, datasets, experimental results, or reports.

Group-wide configuration changes must be proposed for review before execution if they may affect other members.

The human group members remain responsible for mathematical methodology, verification, experimental interpretation, and the final written report.

---

## 2. Group Project Structure

Use the following repository organization.

```text
repo/
├── AGENTS.md
├── .gitignore
├── README.md
├── pyproject.toml
├── uv.lock
│
├── data/
│   ├── fetch.py
│   ├── metadata.json
│   ├── raw/
│   │   ├── q1/
│   │   ├── q2/
│   │   └── q3/
│   └── clean/
│       ├── q1/
│       ├── q2/
│       └── q3/
│
├── code/
│   ├── approach1_q1_svi.py
│   ├── approach2_q1_torch.py
│   ├── approach1_q2_mc.py
│   ├── approach2_q2_hedging.py
│   ├── approach1_q3_classical.py
│   ├── approach2_q3_sequence.py
│   ├── compare.py
│   │
│   ├── q1/
│   ├── q2/
│   ├── q3/
│   └── shared/
│
├── tests/
│   ├── q1/
│   ├── q2/
│   └── q3/
│
├── results/
│   ├── approach1/
│   ├── approach2/
│   └── comparison.html
│
├── docs/
│   ├── project_spec.md
│   ├── decisions/
│   └── integration/
│
├── report.md
└── SESSION.md
```

### Structure Rules

1. The six top-level approach scripts must preserve the naming convention `approach1_*.py` and `approach2_*.py`.
2. Question-specific implementation modules must remain inside `code/q1/`, `code/q2/`, or `code/q3/`.
3. Raw and cleaned datasets must be separated by question.
4. Tests must be grouped by question.
5. All experiments must use the examination's standard `metrics.json` schema.
6. Each approach must save results under `results/approachN/<dataset>/`, using unique question-prefixed dataset identifiers such as `q1_510050`.
7. `code/compare.py` must aggregate all available approaches without confusing datasets from different questions.
8. Shared modules should be introduced only when genuinely reusable and approved by affected group members.
9. The final `report.md` and `SESSION.md` are group-level deliverables.
10. The original examination document must be preserved under `docs/project_spec.md` before the root `report.md` is used for the group's final report.

The repository tree is the intended target structure. Files and directories should be created progressively as milestones are implemented.

---

## 3. Environment and Dependencies

### Mandatory Environment

- Python 3.13 or Python 3.14.
- Prefer Python 3.13.
- Package manager: `uv`.
- All dependencies must be pinned and reproducible.
- Use `pyproject.toml` and a committed `uv.lock`.

Common libraries required by the examination include:

- numpy
- scipy
- pandas
- scikit-learn
- plotly
- rich
- networkx
- akshare
- torch

Additional packages may be used when justified.

### Environment Isolation

All project-created virtual environments, caches, temporary files, test outputs, and generated artifacts must remain inside the repository root.

Examples include:

- `.venv/`
- `.cache/`
- `.pytest_cache/`
- `.uv-cache/`
- `.tmp/`

Configure tools to avoid writing outside the repository.

Do not install packages globally or modify user-level configuration without explicit approval.

### Reproducibility

Use:

- Default random seed: `42`.
- CLI override: `--seed`.
- Deterministic preprocessing and experiment splits.
- Seeded NumPy, Python, and PyTorch operations.
- Documented numerical tolerances.
- Recorded hardware and software versions.
- Explicit disclosure of residual nondeterminism.

Each executable approach must use `argparse` and follow:

load → verify → split → fit/train → evaluate → save

Each approach must write:

`results/approachN/<dataset>/metrics.json`

with the schema:

```json
{
  "params": {},
  "metrics": {},
  "fit_seconds": 0.0,
  "history": {}
}
```

Use Rich for terminal output.

Use Plotly for interactive figures, with `include_plotlyjs=True`.

The comparison script must aggregate available metrics and generate `results/comparison.html`.

### Executable Environment Commands (M0)

Verified in M0 on Windows x86-64: CPython **3.13.18**, uv **0.13.7**,
all pinned imports/numerical checks, and **18 passing environment smoke tests**.
Lockfile consistency and no-change synchronization checks also passed offline.

Run these PowerShell commands from the repository root on Windows x86-64.
The bootstrap prerequisite is the existing Python 3.11 installation and `py`
launcher; it runs only the standard-library launcher with bytecode writes disabled.
Project code runs with repository-managed Python 3.13, not that host interpreter.

```powershell
# Clean-clone setup: verifies a pinned uv wheel, installs the pinned Python,
# and creates/synchronizes .venv from the committed uv.lock.
py -3.11 -B scripts/project.py setup

# Offline verification after setup: versions, lock check, sync check,
# dependency consistency, and environment-only pytest tests.
py -3.11 -B scripts/project.py verify

# Individual diagnostics; always use the launcher for local cache isolation.
py -3.11 -B scripts/project.py uv --version
py -3.11 -B scripts/project.py uv run --locked --offline python --version
py -3.11 -B scripts/project.py uv lock --check --offline
py -3.11 -B scripts/project.py uv sync --locked --check --offline
py -3.11 -B scripts/project.py uv pip check --python .venv/Scripts/python.exe
py -3.11 -B scripts/project.py uv run --locked --offline python -m pytest tests/q1/test_environment.py -v -s
```

Use the launcher before every project command: it configures child-process-only
environment variables for `.venv/`, `.tools/`, `.uv-cache/`, `.cache/`, and `.tmp/`.
It disables Windows Python registry registration/discovery and user site packages.
It does not change the host PATH, execution policy, profile, or system Python.
Do not use bare system `python`, `pip`, `uv`, or activation scripts for project work.
The Python patch version is pinned in `.python-version`, uv in the launcher and
`pyproject.toml`, direct dependencies in `pyproject.toml`, and transitive dependencies
with artifact hashes in `uv.lock`. The initial setup requires network access.
Package downloads use the public Aliyun PyPI mirror; the uv wheel is checked
against its pinned official-PyPI SHA-256. Generated HTML, environment evidence,
and pytest temporary files were verified under `.tmp/` inside the repository.
Execution evidence, verification status, and shared configuration decisions are
recorded in `docs/decisions/m0-environment.md`.

**Planned Q1 commands — not implemented or verified in M0:**

```powershell
py -3.11 -B scripts/project.py uv run --locked python code/approach1_q1_svi.py --seed 42
py -3.11 -B scripts/project.py uv run --locked python code/approach2_q1_torch.py --seed 42
```

These are future CLI contracts, not claims that the scripts currently exist.
Dataset arguments and reproduction commands will be finalized in later milestones.
No Q2/Q3 commands or implementations are introduced by this session.

---

## 4. Question 1 — Implied Volatility Surface

### Ownership

**Question 1 is the current user's primary and authorized development scope.**

Implement both Option A and Option B completely.

The agent may autonomously inspect and edit Q1-specific files within the permitted repository boundaries.

Do not proceed to Q2 or Q3 implementation unless explicitly instructed.

### 4.1 Data Requirements

Use real Chinese ETF option market data.

The examination identifies:

- SSE ETF option chains.
- CFFEX CSI 300 index-option data as an available source.
- Underlying ETF/index history.
- SHIBOR domestic interest-rate data.
- Related LPR and FR007 information.

Required historical scope:

- One month of daily option chains.
- At least four expiries.
- At least five sample dates spanning at least two years.

The precise interpretation of the sampling scope must be documented.

Before model development:

1. Verify historical contract identifiers.
2. Verify strikes, expiries, option types, and contract units.
3. Verify historical market prices.
4. Verify underlying spot prices.
5. Verify interest-rate availability and date alignment.
6. Establish historical sample coverage.
7. Document dividends and forward-price conventions.
8. Prepare verifiable offline snapshots.

Do not fabricate historical observations.

Do not silently substitute index options for ETF options.

If historical data coverage is insufficient, report the gap and propose alternatives for student approval.

### 4.2 Common Mathematical Requirements

Implement Black–Scholes pricing using the forward-price formulation.

Define:

- Forward price: F(T)
- Log-moneyness: k = ln(K/F(T))
- Total variance: w(k,T) = sigma_IV(k,T)^2 * T

Implied volatility must be obtained using bracketed Brent root finding:

sigma in [1e-4, 5]

Both call and put quotations must be considered.

Record unsuccessful inversions and their causes.

Compute put–call parity residuals:

R = C - P - (S0 - K*exp(-r*T))

If dividend-adjusted parity is relevant, report it separately.

Implement and justify quote filtering based on available liquidity information, including volume, spreads, and moneyness.

Construct an explicitly defined fixed log-moneyness evaluation grid.

### 4.3 Option A — Classical SVI Calibration

Implement the SVI total variance model:

w(k) = a + b * [rho*(k-m) + sqrt((k-m)^2 + eta^2)]

Required constraints:

- b >= 0
- |rho| < 1
- eta > 0

Use constrained nonlinear least squares with SciPy.

Report:

- Initial parameters.
- Objective function.
- Constraints.
- Solver settings.
- Fitted parameters.
- Optimization status.
- Calibration time.

Ensure fitted total variance is positive in the relevant domain.

#### Butterfly Arbitrage

Implement the formula specified in the examination:

g(k) =
(1 - k*w'(k)/(2*w(k)))^2
- (w'(k)^2/4)*(1/w(k) + 1/4)

Also compute the complete standard expression, which includes `w''(k)/2`, as a separately labelled diagnostic.

Do not silently replace the examination's formula.

Report every violation, including its location and magnitude.

#### Calendar Arbitrage

Check:

w(k,T2) >= w(k,T1), for T2 > T1

Evaluate on a common log-moneyness grid.

Report violations per expiry and observation date.

#### Term Structure

Interpolate linearly in total variance.

Report negative forward variance.

#### Repricing Verification

Reprice eligible observed market quotes using fitted implied volatility.

Compute the examination-required relative pricing errors, including per-slice maxima and cross-date median and 95th percentile summaries.

Do not silently omit failed calibrations or invalid observations.

### 4.4 Option B — Differentiable Calibration

#### Differentiable SVI

Implement SVI calibration in PyTorch using differentiable parameter transformations.

Support Adam and/or L-BFGS optimization.

Compare against SciPy using:

- Matched observations.
- Comparable objectives.
- Documented initializations.
- Consistent numerical precision.
- Convergence iterations.
- Objective evaluations.
- Final objective values.
- Wall-clock time.

#### Neural Volatility Surface

Implement:

sigma_hat = MLP(k,T)

The MLP must respect the examination's maximum of four layers.

Use a positive volatility output.

The objective must be based on **squared option pricing error**, not implied-volatility MSE.

Required objective:

L_total = L_price + lambda1*L_butterfly + lambda2*L_calendar

where:

- L_price: squared Black–Scholes price error.
- L_butterfly: squared penalty for negative examination-defined g(k).
- L_calendar: squared penalty for negative partial_T(w).

Use PyTorch automatic differentiation as needed.

Perform mandatory ablation experiments on penalty coefficients.

#### Missing-Quote Robustness

Randomly remove 30% of eligible quotes.

Compare SVI refitting against neural-surface predictions on a matched evaluation set.

Use reproducible random masks.

Prevent training/evaluation leakage.

Distinguish interpolation within observed strike support from extrapolation outside it.

Apply the same arbitrage verification to Option B as to Option A.

### 4.5 Question 1 Expected Outputs

Generate:

- Historical market-data coverage report.
- Clean option quote datasets.
- Put–call parity diagnostics.
- IV inversion diagnostics.
- SVI parameter and calibration tables.
- Fitted volatility smiles and surfaces.
- Butterfly arbitrage diagnostics.
- Calendar arbitrage diagnostics.
- Total variance term-structure figures.
- Repricing error summaries.
- SciPy versus PyTorch convergence comparisons.
- Neural training curves.
- Penalty ablation results.
- Missing-quote robustness results.
- Approach A versus B comparison.

The examination's indicative benchmark includes median liquid-strike repricing relative error below 2% and zero interior butterfly violations.

These are targets, not permission to manipulate, hide, or fabricate results.

---

## 5. Question 2 — Monte Carlo Pricing and Hedging

### Ownership

Question 2 is assigned to other group members.

The current user's OpenCode agent must NOT implement or modify Q2 without explicit permission.

This section exists to establish repository-level requirements and integration compatibility.

### Option A — Monte Carlo Engine

The designated group member must implement:

- Historical futures data calibration.
- Risk-neutral GBM simulation.
- Arithmetic Asian call pricing.
- Discretely monitored up-and-out call pricing.
- Digital call pricing.
- Naive Monte Carlo.
- Antithetic variates.
- Geometric Asian control variates.
- Pathwise and finite-difference Greeks.
- RMSE convergence analysis.
- Vectorization, profiling, and performance comparison.

Simulation must use at least 252 steps per year.

European Monte Carlo validation must be reported against the Black–Scholes closed-form result, including the standard-error z-score.

### Option B — Deep Hedging

The designated group member must implement:

- PyTorch hedging strategy.
- Proportional transaction costs.
- Terminal hedging P&L.
- CVaR-based training objective.
- Independent training and validation randomness.
- Black–Scholes delta-hedging benchmarks.
- Rebalancing-frequency comparison.
- Transaction-cost sensitivity analysis.

Use the experiment sizes and benchmark configurations specified in the examination.

Q2 should use the same group-level CLI, metrics, seed, logging, and reproducibility conventions.

---

## 6. Question 3 — China Macro Panel

### Ownership

Question 3 is assigned to other group members.

The current user's OpenCode agent must NOT implement or modify Q3 without explicit permission.

### Common Requirements

Build a monthly Chinese macroeconomic panel with:

- At least 20 indicators.
- At least 10 years of observations.
- Documented publication lags.
- Explicit missing-value handling.
- Documented transformations and outlier rules.
- Training-only normalization.
- Expanding-window walk-forward validation.

Prevent publication-lag leakage.

Report Brier score, AUC, accuracy, and prior-frequency baseline.

### Option A — Classical Models

The designated group member must implement:

- PCA factor model.
- Rank selection using cross-validated nowcast error.
- GDP or industrial-output nowcasting bridge.
- Probit regime classification.
- Coefficient estimates and standard errors.
- Rolling indicator correlations.
- Ward hierarchical clustering.
- Dendrograms.
- Adjusted Rand index cluster-stability analysis.
- Lead/lag correlation interpretation.

### Option B — Sequence Learning

The designated group member must implement:

- LSTM or GRU with at most two layers.
- Explicit class-imbalance treatment.
- Training and validation discipline.
- Latent macro-state embeddings.
- K-means latent-state clustering.
- ARI/NMI comparisons where mathematically applicable.
- Probit versus sequence-model comparison.
- Mandatory intentional leakage ablation.
- Three-seed evaluation with mean and standard deviation.

Q3 must follow the same group-level results and reproducibility conventions.

---

## 7. Cross-Question Isolation and Collaboration

### 7.1 File Ownership

Default ownership:

| Area | Owner |
|---|---|
| `code/q1/` and Q1 approach scripts | Q1 student |
| `code/q2/` and Q2 approach scripts | Q2 assigned members |
| `code/q3/` and Q3 approach scripts | Q3 assigned members |
| `tests/q1/` | Q1 student |
| `tests/q2/` | Q2 assigned members |
| `tests/q3/` | Q3 assigned members |
| `data/raw/qN/`, `data/clean/qN/` | Corresponding question members |
| Group-level configuration and shared files | Shared, coordinated ownership |

No agent may modify another question's owned files without explicit authorization.

### 7.2 Shared Files

Treat the following as integration-sensitive:

- AGENTS.md
- pyproject.toml
- uv.lock
- .gitignore
- README.md
- data/fetch.py
- data/metadata.json
- code/compare.py
- code/shared/
- report.md
- SESSION.md

Changes to shared files must be narrowly scoped, documented, and reviewed for potential impact on all questions.

If a shared change could disrupt another student's work, the agent must stop and request permission.

### 7.3 Concurrent Development

Use question-specific Git branches where practical:

- `q1-development`
- `q2-development`
- `q3-development`
- `integration`

Prefer isolated worktrees or separate working copies when group members work concurrently.

Do not automatically merge branches or rewrite shared history.

Destructive operations such as force-push, hard reset, deleting branches, or overwriting another member's work require explicit approval.

### 7.4 Integration

Before merging a question's work:

1. Run its automated tests.
2. Verify its required metrics schema.
3. Verify offline reproducibility.
4. Check compatibility with pinned dependencies.
5. Check for conflicts in shared files.
6. Record remaining limitations.
7. Obtain appropriate group approval.

All six approaches must be covered by the final comparison and submission verification.

---

## 8. Data Integrity and Offline Reproduction

Every external data source must have:

- Source/interface identification.
- Retrieval date.
- Dataset schema.
- Relevant units.
- SHA-256 checksum.
- Explicit cleaning transformations.
- Offline fallback snapshot.

Follow the examination's 50 MB snapshot policy.

Do not commit secrets or credentials.

Do not fabricate missing observations, successful downloads, or benchmark results.

The repository must be reproducible from a clean clone with documented commands.

The final submission must not depend on private, unrecorded local files or an active internet connection.

---

## 9. Git and Session Integrity

The examination requires:

1. Git initialization and an initial commit before code development.
2. `AGENTS.md` committed before the first terminal coding-agent session.
3. Incremental Git commits after milestones.
4. Conventional commit messages.
5. Final tag `submission`.
6. Authentic agent session exports.

### Historical Compliance

If any agent session occurred before `AGENTS.md` existed, preserve the actual session history.

Do not create fake commits, change timestamps, or falsely claim compliance.

Document the deviation accurately.

### Session Export

The final `SESSION.md` must contain the complete authentic Markdown export of the relevant terminal coding-agent sessions, including:

- User prompts.
- Agent responses.
- Tool calls.
- Execution records.

Redact credentials and secrets.

Do not replace the export with an AI-generated summary.

Because multiple group members may use separate agents, coordinate collection of their authentic session exports before producing the group-level `SESSION.md`.

---

## 10. Mandatory Group Deliverables

The project must include all four deliverables:

### Deliverable 1: AGENTS.md

A complete working contract for terminal coding agents.

### Deliverable 2: Code Repository

Includes:

- Data pipelines.
- All six approaches.
- Mathematical verification.
- Experimental results.
- Comparison layer.
- Pinned dependencies.
- Reproduction commands.

### Deliverable 3: report.md

Student-authored report containing:

- Mathematical model specifications.
- Experimental protocols.
- Results tables.
- Figure references.
- Discussion.
- Limitations.
- Statement of work distinguishing agent work and student decisions.

The final scientific interpretation remains the group's responsibility.

### Deliverable 4: SESSION.md

Complete authentic terminal coding-agent session export.

Missing any of the four deliverables triggers the grading restriction specified by the examination.

---

## 11. Agent Operating Rules

These rules are mandatory.

1. Read `AGENTS.md` and `docs/project_spec.md` before implementation.
2. Identify the current user's authorized question before editing code.
3. For this user's session, default to Q1 only.
4. Never write outside the repository root.
5. Never alter another question's code without explicit approval.
6. Never overwrite existing files without checking their contents.
7. Keep caches, generated files, and temporary artifacts inside the repository.
8. Before each milestone, state the intended modifications and acceptance criteria.
9. Work on one milestone at a time.
10. Run tests and inspect actual output before claiming completion.
11. Document failures instead of hiding them.
12. Preserve mathematical conventions consistently across modules.
13. Never fabricate data, results, tests, or Git/session history.
14. Request approval before destructive or irreversible operations.
15. Commit after completed and verified milestones.
16. Stop when critical data, mathematical, or environmental prerequisites are missing.
17. Do not proceed automatically to the next milestone.
18. The student must approve methodological decisions affecting scientific conclusions.

---

## 12. Q1 Agent Workflow

For the current user, follow this development order:

**M0 — Repository and environment**
- Preserve examination specification.
- Verify AGENTS.md.
- Prepare Git and uv.
- Configure repository-local caches.
- Validate Python version.

**M1 — Historical data feasibility**
- Verify AKShare interfaces.
- Audit historical coverage.
- Identify contract metadata gaps.
- Verify underlying prices and SHIBOR.
- Record unresolved limitations.

**M2 — Data pipeline**
- Build real-data snapshots.
- Clean quotations.
- Verify contracts and dates.
- Build curves and forward conventions.
- Implement offline fallback.

**M3 — Shared mathematical foundations**
- Black–Scholes call/put pricing.
- Brent implied volatility.
- Put–call parity.
- Fixed log-moneyness grid.
- Numerical tests.

**M4 — Classical SVI**
- Constrained least squares.
- Arbitrage diagnostics.
- Total variance interpolation.
- Repricing verification.

**M5 — Differentiable SVI**
- PyTorch parameterization.
- Adam/L-BFGS.
- Gradient verification.
- SciPy convergence comparison.

**M6 — Neural surface**
- MLP.
- Option price-space objective.
- Arbitrage penalties.
- Training and validation.

**M7 — Experiments**
- Penalty ablations.
- Missing-quote robustness.
- Cross-date comparison.

**M8 — Integration and verification**
- Complete Q1 metrics.
- Generate self-contained figures.
- Run all Q1 tests.
- Validate offline reproduction.
- Prepare report-supporting outputs.
- Coordinate shared repository integration.

Each milestone must have:

- Clear scope.
- Implementation evidence.
- Verification evidence.
- Actual test results.
- A Git commit.
- Known limitations.

Do not continue to the next milestone without user authorization.

---

## 13. Final Definition of Done

The complete group project is finished only when:

1. Q1, Q2, and Q3 each contain both required approaches.
2. Required historical datasets and verification procedures are available.
3. All six approaches execute successfully or have explicitly documented failures.
4. Experimental outputs contain real measured results.
5. Figures and metrics follow the required conventions.
6. Dependencies are pinned.
7. Offline reproduction works from a clean clone.
8. The student-authored final report is complete.
9. Authentic session exports are collected.
10. Git history contains meaningful incremental commits.
11. The final repository receives the `submission` tag after verification.

**Current-agent completion is different from group-project completion.**

The Q1 agent may declare its own Q1 implementation ready for integration, but must not declare the entire group submission complete until Q2, Q3, and the shared deliverables have also been independently verified.
