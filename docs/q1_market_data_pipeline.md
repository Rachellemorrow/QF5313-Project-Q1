# Q1 M2 — Reproducible offline option-data pipeline

## Outcome: code verified; historical data still partial

The Q1 pipeline processes only the real snapshots already committed by M1. It
performs no downloads, source discovery, new account access, option pricing,
calibration or neural training. M1/M1.5 archives and the M2A mathematical libraries
are unchanged. The earlier historical-data blocker is **not** removed.

- **Code readiness:** `VERIFIED_IMPLEMENTATION`.
- **Dataset readiness:** `PARTIAL_HISTORICAL`.
- **Examination historical requirements satisfied:** **false**.
- **Strictly usable empirical quotes/dates/expiries:** **0 / 0 / 0** under the
  conservative metadata, unit and convention gates. This is not a software failure
  and does not mean that the archived prices are invented or universally unusable.

## Changes and shared-file boundary

- `data/fetch.py`: new shared entry point with explicitly required `--question q1
  --offline`. It supports only the authorized Q1 mode; there is no live fetch mode,
  provider SDK, Q2/Q3 dispatcher implementation or automatic source refresh.
- `data/metadata.json`: new registry with a `questions.q1` namespace only. It records
  source/output hashes, original schemas, retrieval times, policy, canonical schema,
  coverage and actual test evidence. No other question's registration is altered.
- `code/q1/market_data.py`: canonical schema, normalization, dated metadata checks,
  exact-date joins, filtering, coverage, deterministic builds and offline reload.
- `code/q1/rates.py`: explicit SHIBOR interpolation/conversion and approval policy.
- `code/q1/provenance.py`: pinned manifest verification and confined local loading.
- `data/clean/q1/m2_pipeline/`: normalized, labelled partial data and diagnostic
  outputs. `data/clean/q1/.gitattributes` adds byte preservation for this directory
  only, so Windows checkout conversion cannot invalidate data SHA-256 checksums.
- `tests/q1/test_rates.py`, `test_market_data.py`, and
  `tests/q1/fixtures/synthetic/m2_pipeline_cases.json`: offline verification. The
  fixture is explicitly synthetic and stays separate from all real data outputs.

The shared-file additions were explicitly requested for M2 and are narrowly scoped
and documented here. Existing shared environment configuration, Q2/Q3, scientific
reports, session exports, source audit artifacts and mathematics were not edited.

## Source verification and provenance

Input manifests are pinned by SHA-256 in `q1.provenance.INPUT_MANIFESTS`:

| Input manifest | SHA-256 |
|---|---|
| `data/clean/q1/m1_evidence/snapshot_manifest.json` | `b34faa9acfaadc4efc70788c98017a7fae1f06ac48d6c695817ca0f52c7ea438` |
| `data/clean/q1/m15_recovery/audit_manifest.json` | `be7b08c321a1ddacba28f7b81f3d205726dba9f576b31456f7536e8435adb7de` |

The loader verifies each manifest's digest and every listed artifact's size and
digest before processing. Each consumed file is checked again when read. Relative
paths cannot escape the repository, and large audit artifacts are rejected under
the 50 MB policy. Quotes retain original source path, file digest, row/field-row
number, API probe and retrieval timestamp; raw premium text is retained alongside
its float64 normalization. Metadata/spot/rate sources retain their own hashes.

The output manifest and registry protect generated snapshots. The pipeline records
source API schemas/dtypes, original date coverage and exact cleaning policy. Original
M1 disclosures, including the incomplete initial index-directory HTTP capture and
unresolved redistribution rights, remain applicable. Reprocessing existing evidence
does not constitute a new license or permission for external publication.

Generated CSVs use UTF-8, LF, explicit missing cells and 17-digit numeric formatting.
Offline numeric reload uses round-trip float parsing. Code fingerprints normalize
source-code line endings only and are labelled `code_lf_sha256`; original data
checksums are always byte checksums. No wall-clock timestamp is invented for an
offline source retrieval. Source timestamps are retained, while deterministic
generated content has no changing build timestamp or checkout-specific paths.

## Canonical records and temporal metadata

`schema.json` lists the exact ordered fields and pandas dtypes. Identity/date,
provenance and diagnostic fields use nullable strings; numbers use nullable
`Float64`; flags use explicit nullable-boolean dtype with no missing flag values.
Missing bid/ask, volume, OI, strikes, types, expiries and multipliers remain missing.
No historical attributes are decoded from trading-code strings.

Canonical records distinguish daily closes from timestamped quote snapshots.
The latter retain the source's market clock with explicit Asia/Shanghai offset;
they are not copied back into historical liquidity fields. Quote timestamps must
agree with observation dates. Normalization is deterministic and row-preserving.
It reports invalid/nonpositive prices, invalid numbers and inconsistent OHLC rather
than modifying market observations. Exact duplicates are marked after a stable
source ordering; conflicting duplicate prices are all rejected, never arbitrarily
selected. Source-occurrence IDs distinguish repeated inputs.

Dated metadata requires a unique covering inclusive `valid_from`/`valid_to`
version, verified source provenance, complete strike/type/underlying/expiry/unit
attributes and a consistent trading lifecycle. Overlapping versions, future-only
versions, observations before listing/after expiry and direct quote/metadata
conflicts are rejected. A current master is a labelled **candidate**, not a
historical version. Listing and expiry dates are not invented effective dates for
strike or multiplier adjustments. Dated versions take precedence over a later
current snapshot; no forward/backward fill is performed.

Two row states are supported:

- `VERIFIED_EMPIRICAL`: a real observed quote with complete verified date-effective
  required metadata, provenance and units. This does **not** imply liquidity,
  convention approval or complete multi-year coverage.
- `PARTIAL_HISTORICAL`: real evidence whose attributes/history are insufficient;
  synthetic/non-real records are also explicitly barred from empirical use.

`eligible_for_empirical` additionally requires all target, quality, alignment,
liquidity and convention gates to pass. A few verified rows would still not certify
a complete historical dataset. The current dataset-level coverage gate remains
closed without authoritative expected chain rosters and an official calendar;
this pipeline does not infer completeness merely from minimum/maximum dates or
four observed expiries.

## Underlying, interest-rate and dividend conventions

- Underlying matching is on the **same observation date, instrument family and
  underlying identifier**. Duplicate matches fail. No prior-day, future-day or
  nearest-day substitutions are made. ETF matching uses the existing explicitly
  unadjusted history; qfq/hfq prices are not substituted. Index rows remain separate
  and cannot satisfy 50ETF coverage.
- SHIBOR levels are read as annual **percent** quotes and divided by 100 for decimal
  quote interpolation. All eight tenors are required: O/N, 1W, 2W, 1M, 3M, 6M, 9M,
  1Y. Missing dates/nodes are not filled or silently dropped; extrapolation fails.
- Default tenor coordinates use **indicative, unadjusted calendar offsets**. They
  are not claimed to be verified SHIBOR spot-lag, holiday or business-day conventions.
  Option time is recorded as calendar-date difference / 365 (ACT/365F), with an
  unapproved-conventions flag. Same-date fixing time relative to quote time remains
  an empirical convention requiring review.
- Default rate interpretation is `unapproved`. The pipeline emits only
  `shibor_quote_decimal`, **not** a zero rate or discount factor. A SHIBOR deposit
  fixing is not silently relabelled as a zero curve.
- Reusable numerical modes support explicit `zero_continuous` interpretation or
  `simple_act360` deposit conversion. The latter maps each node using
  `D=1/(1+L*days/360)`, converts to continuous ACT/365F zeros, and linearly
  interpolates zeros. These alternatives are implemented, not empirically approved.
- Approval requires an explicit reference, explicit continuous decimal dividend
  yield and supplied verified tenor coordinates. Merely passing a mode name or
  relying on default calendar offsets is insufficient. The default q is **missing**,
  not zero. No dividend estimate or adjustment policy is inferred from ETF prices.
- Without approval, `zero_rate`, `discount_factor`, `forward` and `log_moneyness`
  remain missing. Under explicit approved inputs the forward is the definitional
  `S*exp(-q*T)/D`; no Black–Scholes, SVI or neural implementation is added here.

## Liquidity policy and rejection accounting

The default diagnostic policy requires positive finite premiums and volume at least
1 in **source-reported units**. Volume-unit validation is still warned about;
source quote units are a separate strict empirical gate. Available bid/ask must be
nonnegative/ordered and have relative spread `(ask-bid)/mid <= 0.25`. Missing bid/ask
and OI are explicitly warned about by default, not manufactured. Optional policies
can require bid/ask, minimum OI or a forward-log-moneyness band. A requested band
without an approved forward is rejected as unavailable, not replaced with spot.

Every applicable rejection reason is retained as a JSON array. Long-form
`rejections.csv` and `rejection_counts.csv` provide auditable counts; reasons overlap,
so their counts must not be added to infer a number of rejected observations.
`partial_diagnostic_quotes.csv` removes structural/price/alignment/liquidity errors
but can retain unresolved metadata/unit/convention flags. It is **not** an empirical
calibration dataset. `accepted_quotes.csv` is intentionally header-only for this run.

## Actual real-sample results

| Evidence category | Quotes | Dates | Distinct expiries | Interpretation |
|---|---:|---:|---:|---|
| All archived observations processed | 606 | 187 | 4 known | Includes separate index/unknown-metadata histories; not an ETF historical panel |
| Known 50ETF observations | 489 | 130 | 4 candidate | Ten current identifiers, three strikes; 2026-04-23–2026-11-04 |
| Partial diagnostic preview | 489 | 130 | 4 candidate | Not date-effective verified or empirical-ready |
| Strictly usable empirical observations | **0** | **0** | **0** | Required gates remain unresolved |

There are 604 actual daily rows plus two timestamped current quote snapshots. The
known 50ETF subset contains 261 call records and 228 put records. The 94 index
records and 23 older unknown-metadata SSE records do not extend ETF coverage.
All 604 daily rows retain missing bid/ask/OI; only the two quote snapshots carry
their actually observed liquidity fields. No row has verified effective-dated
metadata. The 130 recent ETF dates do not span two years and are not full chains.

Actual rejection counts:

| Reason | Rows |
|---|---:|
| `PRICE_UNIT_UNVERIFIED` | 606 |
| `DIVIDEND_OR_RATE_CONVENTIONS_UNAPPROVED` | 606 |
| `UNDATED_CURRENT_METADATA` | 529 |
| `INDICATIVE_UNAPPROVED_CONVENTIONS` | 489 |
| `INCOMPLETE_CONTRACT_METADATA` | 117 |
| `MISSING_DATE_OR_EXPIRY` | 117 |
| `MISSING_OR_NON_TARGET_UNDERLYING` | 117 |
| `MISSING_SPOT_DATE` | 117 |
| `NON_TARGET_INSTRUMENT_FAMILY` | 94 |
| `MISSING_EFFECTIVE_METADATA` | 77 |
| `INVALID_OR_NONPOSITIVE_PRICE` | 1 |
| `INCONSISTENT_OHLC` | 1 |

The last two reasons flag the same archived observation; it is retained, not
repaired. Current metadata values are preserved as candidates in cleaned output,
with their validity flags false. There are no empirical benchmark results here.

## Output artifacts and offline reproduction

`data/clean/q1/m2_pipeline/` contains 13 hashed artifacts plus `manifest.json`:
canonical/all quotes, accepted quotes, partial diagnostic quotes, contract metadata,
underlying spots, SHIBOR quotes, long-form rejections, rejection counts, warning
counts, date-by-expiry coverage, coverage summary, schema and provenance. Artifact
payloads total **2,645,388 bytes** (excluding the manifest), below 50 MB.

Registered output-manifest SHA-256:

`baa14155d69f3ee9e443924a740c048808594c1f30c43a371b939ccecada63a4`

Exact executed commands, from the repository root:

```powershell
py -3.11 -B scripts/project.py uv lock --check --offline
py -3.11 -B scripts/project.py uv sync --locked --check --offline
py -3.11 -B scripts/project.py uv run --locked --offline python data/fetch.py --question q1 --offline --output .tmp/q1_m2_smoke
py -3.11 -B scripts/project.py uv run --locked --offline python data/fetch.py --question q1 --offline
py -3.11 -B scripts/project.py uv run --locked --offline python data/fetch.py --question q1 --offline --verify-only
py -3.11 -B scripts/project.py uv run --locked --offline python -m pytest tests/q1/test_rates.py tests/q1/test_market_data.py -q
py -3.11 -B scripts/project.py uv run --locked --offline python -m pytest tests/q1 -q
```

An identical build into an existing directory verifies bytes and makes no changes.
A differing existing directory is refused: inspect it and choose a new repository-
local `--output`; there is no overwrite/reset flag. The shared registry is not
silently regenerated or modified on an ad-hoc build. Registered reload checks its
manifest digest, every output artifact, canonical schema and the pinned input
archive. Unregistered reload requires an explicit expected manifest digest.

All commands use the existing locked Python 3.13 environment and repository-local
caches/temporary output. No new dependencies or global software/configuration
changes were needed. The initial smoke remains ignored; final snapshots are committed.

## Actual automated verification

- New M2 suite: **53 passed, 0 failed**, 43.02 seconds.
- Full Q1 suite: **183 passed, 0 failed**, 68.44 seconds, including all 130 earlier
  environment, audit and mathematical regression tests.
- Tests cover dtypes/schema, invalid quotes, exact/conflicting duplicates,
  historical-version bounds/overlaps, metadata conflicts/missing fields,
  same-date spot/rate joins, adjustment rejection, liquidity/moneyness rules,
  explicit rate approval, negative rates, deposit conversion, missing tenors,
  corruption/path traversal, byte-identical rebuilds and exact numeric reload.
- The real-data smoke verifies the actual counts above. Unit stand-ins live only
  under the labelled synthetic test fixture. A controlled trusted-loader simulation
  exercises the `VERIFIED_EMPIRICAL` branch; it is not reported as recovered data.
- Tests deny socket access. Intentional tampering tests modify only temporary
  copies. Original audit files and their checksums are preserved.

## Remaining boundary and mathematics readiness

Complete one-month daily ETF chains, five complete sample dates spanning two years,
verified historical multipliers/strikes and an authoritative expected roster/calendar
remain absent. Curve/dividend/unit/quote-time conventions and source reuse rights
still need review. Correct code and successful normalization cannot create this
missing evidence or authorize treating partial coverage as examination-compliant.

Data-independent mathematical foundations already exist from the separately
authorized **M2A** commit and pass regression tests unchanged. Further mathematical
code work can be reviewed independently of data availability, but empirical model
calibration remains gated. No M3 work or next milestone starts automatically.
