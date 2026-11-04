# Q1 M1 — Historical option data feasibility audit

## Decision

**PARTIAL feasibility; the examination's historical ETF option dataset is NOT yet
established. Do not start model development or claim a compliant historical panel.**

Real API responses and small offline snapshots exist. Current SSE identifiers can
be joined to selected contract histories, including calls and puts across four
current expiries. The critical gap is reliable **expired-contract discovery and
date-effective metadata**, especially for complete chains in 2024/2025. Finding
one old identifier with prices does not resolve that gap.

This is Q1 only. No Q2/Q3 files, group dependency pins, scientific report, or shared
fetch/comparison modules were changed. No Black–Scholes, SVI, neural model,
discount-curve construction, or IV calculation was implemented.

## 1. Preconditions, acquisition bounds, and evidence

- Entry working tree: clean, at `f44fe4dd79b1212513719ec245a38c0f9c5cf72c`.
- Python 3.13.18; uv 0.13.7; AKShare **1.20.27**.
- Re-ran M0 verification: all **18** environment tests passed; offline lock and
  synchronization checks passed; all 50 installed distributions compatible.
- Project commands used `scripts/project.py`, retaining repository-local `.venv`,
  `.tools`, `.uv-cache`, `.cache`, `.tmp`, and pytest output. No global installation,
  system PATH/policy change, or external project output directory was requested.
- First acquisition began **2026-11-04 16:22:47 UTC**, i.e. **2026-11-05 00:22:47
  Asia/Shanghai**. Preserve these actual timestamps; do not relabel them as quote
  dates. The two current SSE spot snapshots explicitly report **2026-11-04
  15:00:00**. Metadata retrieval time is not a historical observation date.
- Three acquisition runs captured 21 + 3 + 8 = **32** HTTP responses, totaling
  **1,692,909** decompressed response bytes. IMPORTANT: this is not a complete
  whole-process request count; see the pagination capture limitation below.
- Source and derived evidence occupies approximately **2.8 MB**, well below 50 MB.
  The audit archive is deliberately committed, not hidden in regenerable raw data.

### Artifact locations

| Artifact | Location |
|---|---|
| Consolidated machine-readable API and requirement audit | `data/clean/q1/m1_evidence/api_audit_summary.json` |
| Observation-level price availability matrix | `data/clean/q1/m1_evidence/historical_coverage_matrix.csv` |
| Current SSE contract inventory; unknown prices remain unknown | `data/clean/q1/m1_evidence/current_contract_inventory.csv` |
| Five explicit target dates, not fabricated observations | `data/clean/q1/m1_evidence/target_date_coverage.csv` |
| October coverage on dates actually returned by ETF history | `data/clean/q1/m1_evidence/october_2026_coverage.csv` |
| Same-date spot/rate matching | `data/clean/q1/m1_evidence/historical_alignment.csv` |
| Every archived file's path, size and SHA-256 | `data/clean/q1/m1_evidence/snapshot_manifest.json` |
| Main API records, returned tables, HTTP bodies, source inspection | `data/clean/q1/m1_audit/` |
| Historical ETF adjustment/alignment probes | `data/clean/q1/m1_alignment/` |
| Eight matched current-expiry call/put history probes | `data/clean/q1/m1_pairs/` |

These `data/clean/q1/` files are **audit evidence**, not a final cleaned modeling
dataset. CSV serialization preserves the AKShare table's rows/order/values; original
column dtypes are separately recorded because CSV cannot preserve pandas types.
HTTP bodies are archived after transfer decompression, not as packet captures.
Every probe records parameters, installed signature/source hash, timestamp, schema,
date range, errors, and snapshot checksum. Network records identify the upstream URL.

`data/clean/q1/.gitattributes` disables newline conversion **only for the Q1 audit
directories**: otherwise Windows CRLF checkout conversion would invalidate byte
checksums. It does not affect Q2/Q3 or shared configuration.
Some original JSON records use Windows CRLF. Q1 audit whitespace checks recognize
those endings, and original HTTP bodies are excluded from whitespace lint rather
than altered to satisfy formatting rules. Their bytes and checksums are preserved.

## 2. Installed interfaces and actual responses

The exact source/docstrings/signatures are archived in each run's
`installed_api_sources.json`. No parameter below was inferred from web examples.
Original upstream sources include SSE `query.sse.com.cn`, Sina's option services,
Eastmoney historical kline services, and Jin10's SHIBOR JSON. Source URLs and
parameters are in the individual run's `api_audit.json` network records.

| Installed interface | Actual result and observed coverage | Important limits |
|---|---|---|
| `option_current_day_sse()` | 702 current metadata rows; four expiry dates, 2026-11-25 through 2027-06-23 | No observation-date parameter, historical master, market prices, volume, bid/ask, or OI |
| `option_cffex_hs300_list_sina()` | `io2611`, `io2612`, `io2703`, `io2701`, `io2709`, `io2706` | Current **INDEX** product months, not a historical contract directory or ETF data |
| `option_cffex_hs300_daily_sina(symbol='io2202P4350')` | 54 rows, 2021-11-29–2022-02-18; observed current ID `io2611C4300`: 40 rows, 2026-08-24–2026-11-04 | OHLC/volume only; no strike, expiry day, multiplier, bid/ask or OI in daily response |
| `fund_etf_hist_em(symbol='159707', period='daily', start_date='19700101', end_date='20500101', adjust='')` | Probed `510050`; five rows for 2026-10-26–30 in each adjustment mode; 505 unadjusted rows, 2024-10-08–2026-11-04 | Default parameters shown are the installed signature, not the requested product/window; actual parameters are recorded per probe |
| `index_zh_a_hist(symbol='000859', period='daily', start_date='19700101', end_date='22220101')` | Probed `000300`, 2024-10-08–11: four rows | Internally fetches an index code directory and broad history before slicing; requested short dates do not guarantee a small upstream call |
| `macro_china_shibor_all()` | 2,398 distinct dates, 2015-05-08–2026-11-04; eight tenor levels | No date bounds; fixed JSON response. Installed docstring claims a 2017-03-17 start, contradicted by the observed response |

Actual returned column schemas/dtypes:

- SSE master: `合约编码`, `合约交易代码`, `合约简称`, `标的券名称及代码`, `类型`,
  `行权价`, `合约单位`, `期权行权日`, `行权交收日`, `到期日`, `开始日期`;
  **all returned as pandas `str`**, not numerically validated contract specifications.
- SSE daily: `日期` (`object`, Python dates); `开盘`, `最高`, `最低`, `收盘`
  (`float64`), `成交量` (`int64`).
- CFFEX daily: `date` (`object`, Python dates); `open`, `high`, `low`, `close`
  (`float64`), `volume` (`int64`).
- ETF/index: `日期` (`str`); `开盘`, `收盘`, `最高`, `最低`, `成交额`, `振幅`,
  `涨跌幅`, `涨跌额`, `换手率` (`float64`); `成交量` (`int64`).
- SHIBOR: `日期` (`str`); `{O/N,1W,2W,1M,3M,6M,9M,1Y}-{定价,涨跌幅}`
  (16 `float64` columns). All tenor **levels** are populated; each change column has
  one missing value. No rate gaps were filled.

### Additional SSE/Sina probes

| Interface / exact installed signature | Finding |
|---|---|
| `option_sse_list_sina(symbol='50ETF', exchange='null')` | Current months `202611`, `202612`, `202703`, `202706`; no as-of date |
| `option_sse_codes_sina(symbol='看涨期权', trade_date='202202', underlying='510050')` | `trade_date` is **expiry month**, not observation date. Requested `202410` for both calls/puts; both failed |
| `option_sse_daily_sina(symbol='10003889')` | Known default/example ID returned 23 rows, 2022-01-17–2022-02-23. Does not return contract metadata |
| `option_sse_spot_price_sina(symbol='10003720')` | Requested expired ID `10003889`: one row `买量` with blank value. **Not a usable quote or metadata recovery**. Two observed current IDs each returned 43 fields |
| `option_sse_expire_day_sina(trade_date='202102', symbol='50ETF', exchange='null')` | Requested observed month `202611`: `('2026-11-25', 20)`. Remaining days are relative to retrieval, not a historical maturity calculation |
| `option_daily_stats_sse(date='20240626')` | Requested `20241008`: five underlying-level rows. This is aggregate activity/OI, **not a strike-level chain** |
| `option_cffex_hs300_spot_sina(symbol='io2204')` | Requested observed `io2611`: 31 strike rows, each with call/put IDs, current bid/ask/latest/OI. No explicit quote date, exact expiry day, or multiplier |

Both expired discovery errors are exactly:

```text
ValueError: Length mismatch: Expected axis has 1 elements, new values have 2 elements
```

The corresponding archived upstream bodies are empty lists encoded as assignments:
`var hq_str_OP_UP_5100502410="";` and
`var hq_str_OP_DOWN_5100502410="";`. The parser error is not evidence that all
expired data is unavailable everywhere; it demonstrates failure of these specific
public discovery requests. No ID range was brute-forced and no nonexistent API
date parameter was invented.

## 3. Coverage and contract metadata

### Current universe versus historical records

Current master counts: 96 for 50ETF (`510050`), 116 for 300ETF (`510300`), 148 for
500ETF (`510500`), 176 for 科创50 (`588000`), and 166 for 科创板50 (`588080`).

The price matrix contains **604 actual daily price rows**:

- **487** rows for ten observed current 50ETF contract IDs, 2026-04-23–2026-11-04.
  Identifiers join to current metadata; this is not proof that today's strike/unit
  was unchanged throughout each historical interval. No joined date precedes its
  source listing date, but adjustment-history validity remains **UNVERIFIED**.
- **23** rows for SSE example `10003889`. Underlying, strike, type, exact expiry and
  unit remain **null** because the expired spot service did not recover them.
- **94** CFFEX **index** rows, kept in a different instrument family. Symbol-derived
  expiry month/type/strike are placed in separately labelled inferred columns;
  missing exact expiry days and multipliers are not fabricated.

A positive finite close is recorded as a **candidate market price**, not as a
liquid, executable, or model-validated quote. All `model_ready` flags are false.
Current quotes without a source timestamp are not assigned a historical date just
because they were retrieved today. Unrequested current inventory prices are null,
not falsely labelled unavailable or zero.

### Four-expiry price pairing

At common observed strike **2.900**, the bounded eight-contract probe returned:

| Current expiry | Call ID / rows | Put ID / rows | Earliest returned date |
|---|---|---|---|
| 2026-11-25 | `10012439` / 24 | `10012448` / 24 | 2026-09-24 |
| 2026-12-23 | `10011427` / 130 | `10011436` / 130 | 2026-04-23 |
| 2027-03-24 | `10012164` / 65 | `10012173` / 69 | 2026-07-23 |
| 2027-06-23 | `10012704` / 5 | `10012713` / 5 | 2026-10-29 |

This is one paired strike per expiry, **not a complete smile/chain**. Unequal call
and put row counts are retained, not silently intersected or imputed.

October 2026 ETF history contains 17 observed dates. Sparse selected quotes cover
three current expiries on the first 15 dates and four on October 29/30. The late
appearance of the fourth current expiry illustrates why today's contract roster
cannot be projected backwards through a month containing an expiry rollover.
An official exchange trading calendar was not independently verified.

### Provisional interpretation and five target dates

For auditing only, interpret the requirement conservatively as one complete
calendar month of daily chains **plus** at least five dated sample chains spanning
at least two years, with at least four expiries on relevant observation dates.
The student must approve the final sampling interpretation; it has not been relaxed.

Targets: **2024-10-08, 2024-10-31, 2025-10-09, 2026-10-08, 2026-10-30**.
All five have spot and rate rows in the audit snapshots. The first three have no
verified ETF option panel here; October 2026 has only the sparse recent sample.
The initially considered 2025-10-08 is absent from both returned spot/rate series;
the target was moved to the actually observed 2025-10-09 without inventing a quote
or asserting an unverified holiday-calendar explanation. Target rows are explicitly
labelled as targets, not observations.

## 4. Underlying prices, rates, and adjustments

- Historical ETF `adjust=''`, `qfq`, and `hfq` were called explicitly. At
  **2024-10-08**, the reported closes are **2.978**, **2.843**, and **3.640**,
  respectively. They are not interchangeable. The recent five-day unadjusted and
  qfq samples happen to agree; this does not prove absence of historical dividends.
- Use of unadjusted contemporaneous ETF spot for contractual quote matching is a
  proposed convention, not an approved pricing methodology. Do not pair nominal
  historical strikes with an arbitrarily back-adjusted spot series.
- All **487** selected current-ID ETF price rows have a same-date positive
  unadjusted spot close and all eight SHIBOR levels. No nearest-day joins, carry
  forwards, forward fills, or curve interpolation were applied. The old unknown
  SSE identifier cannot be assigned an underlying; the four index spot rows do not
  cover either index option history window. Index-wide alignment remains partial.
- SHIBOR tenors observed: O/N, 1W, 2W, 1M, 3M, 6M, 9M, 1Y. The installed function
  calls its levels **percent**; its change fields are not level rates. A percentage
  quote is not already a decimal discount rate or a bootstrapped zero curve.
- Simple-interest ACT/360 is a convention to verify against authoritative SHIBOR
  rules, **not a convention established by this API response**. Fixing time,
  day-count, compounding, expiry day-count/calendar, and how interbank deposit
  fixings become the examination's interpolated zero curve require student review.
  The attempted official SHIBOR page fetch failed with a transport error; an SSE
  reference-page fetch returned HTTP 403. Web search was unavailable/cancelled.
- Current master: 674 contracts have unit **10000** and 28 have **10180**. The latter
  are 500ETF A-series contracts; e.g. `10011809`, trading code
  `510500C2612A07000`, reports strike **6.876** and unit **10180**. Decoding the
  numeric code as a present strike of 7.000 would be wrong. No historical effective
  adjustment dates, dividend notices, or unit histories were established.
- Daily volume exists, but its unit and unusually large values in the expired SSE
  example have not been independently corroborated. Premium currency/unit,
  settlement versus last-trade definitions, and index multipliers also need an
  authoritative contract-specification check before economic calculations.
- No zero dividend yield, dividend estimate, forward convention, or adjusted
  contract exclusion rule has been silently adopted. These require student approval.

## 5. Requirement-by-requirement evidence

`PASS` means the stated narrow fact was actually verified. `FAIL` means this audit
does not meet that requirement, not a claim of universal source impossibility.

| Requirement | Status | Evidence / gap |
|---|---|---|
| Six named installed APIs/signatures exercised | PASS | Actual calls and installed sources archived |
| One month of complete daily ETF option chains | FAIL | Only sparse selected contracts; no archived monthly roster |
| At least four priced expiries | PARTIAL | Actual paired histories across four current expiries; only one shared strike and five common recent dates |
| Five complete sample chains spanning two years | FAIL | No verified 2024/2025 ETF option chains |
| Reliable expired contract discovery | FAIL | Both 202410 discovery calls empty/parser failures |
| Retrieval for a known expired contract | PARTIAL | 23 SSE and 54 CFFEX rows, not complete contract masters |
| Historical price/metadata matching | PARTIAL | Current-ID joins work; date-effective adjustment history absent |
| Historical ETF/index spot retrieval | PASS | 505 ETF rows; four index rows |
| Adjusted/unadjusted prices distinguished | PASS | Different old-date closes observed |
| Eight SHIBOR tenor histories | PASS | 2,398 dates; all level quotes populated |
| Same-date alignment for required panel | PARTIAL | 487 recent selected ETF rows aligned; full panel absent |
| Rate units and curve conventions | PARTIAL | Percent label present; official conventions/curve method unverified |
| Corporate-action and unit history | PARTIAL | Current 10180 units observed; effective-dated archive absent |
| Historical bid/ask and OI | FAIL | Daily APIs omit them; current quote fields cannot backfill history |
| Volume availability and units | PARTIAL | Historical field present; units/quality require corroboration |
| Student-approved dividend/forward convention | UNVERIFIED | No assumption chosen in M1 |
| Small checksum-verifiable offline evidence | PASS | Committed source/derived manifests; byte-preserved Q1 files |
| Complete initial HTTP capture | PARTIAL | Index-helper pagination bypass disclosed and recorder corrected |

### HTTP capture and execution limitations

The original hook recorded `http.request` but not AKShare's
`request_with_retry(..., fresh_session=True)`. The index-code lookup displayed 17
remaining-page iterations; its installed source also fetches a first page. Thus
**18 additional page calls are inferred**, not directly captured. Their response
bytes and any retry count are unverified. The original summary's nominal budgets
must not be read as a guarantee for that initial entire process.

The recorder now instruments the retry entry point too, performs no retries, caps
each probe at six requests, caps a response at 2 MB and a run at 8 MB, uses 5/12
second connect/read timeouts, and stops repeated requests to detected rate-limited
hosts. Subsequent three- and eight-request runs were captured. The index directory
was **not downloaded again** just to repair bookkeeping. Raw initial evidence is
preserved unchanged; the consolidated machine summary explicitly carries this
correction. This is not a packet-level or whole-machine filesystem audit.

One exploratory PowerShell/stdin diagnostic lost Chinese literals and failed with
`KeyError`; subsequent Unicode processing runs from UTF-8 source files. The first
offline builder attempt failed on a relative/absolute path comparison after writing
one derived CSV. The path handling was fixed; incomplete/preliminary derived files
were preserved under ignored `.tmp/` rather than overwritten or deleted. None of
these errors was a successful data-coverage result.

## 6. Reproduction and next decisions

Executed acquisition commands (do **not** rerun into the existing directories):

```powershell
py -3.11 -B scripts/project.py uv run --locked --offline python code/q1/audit_interfaces.py --live --output data/clean/q1/m1_audit
py -3.11 -B scripts/project.py uv run --locked --offline python code/q1/audit_interfaces.py --live --supplemental --output data/clean/q1/m1_alignment
py -3.11 -B scripts/project.py uv run --locked --offline python code/q1/audit_interfaces.py --live --four-expiry-pairs --output data/clean/q1/m1_pairs
py -3.11 -B scripts/project.py uv run --locked --offline python code/q1/build_feasibility.py
```

`uv --offline` constrains **package resolution**, not Python's network access; the
audit's `--live` flag explicitly enables real HTTP requests. Live responses change
with time and the corrected six-request cap can stop an index map lookup. Ordinary
review should use committed snapshots, not repeated acquisition.

Offline verification command:

```powershell
py -3.11 -B scripts/project.py uv run --locked --offline python -m pytest tests/q1 -v
```

Tests deny socket connections while checking the M1 evidence and rebuilding the
derived CSVs into a new repository-local pytest temporary directory. Existing
evidence directories are immutable; the builder refuses to overwrite them.
Actual result: **29 passed in 19.88 seconds** (11 M1 integrity/evidence tests plus
18 existing M0 environment tests). The M1 tests made no live data requests.

**Student decisions needed before M2/model development:**

1. Approve the sampling interpretation and identify a permitted archive/provider
   containing expired identifiers, dated strikes/expiries/types/units, and EOD
   quotes, ideally with historical OI/spreads and adjustment notices. An authorized
   student-provided snapshot is an alternative, subject to provenance/licensing.
2. Approve a bounded follow-up investigation of such an archive; availability is
   not claimed here. Do not brute-force IDs or substitute present metadata.
3. Review unadjusted spot, dividends, fixing-time/day-count and curve conventions
   after authoritative documents and historical notices are available.
4. If ETF archives remain insufficient, seek student/instructor approval for any
   revised scope. CFFEX index prices are evidence of a separate source only, **not
   an authorized replacement for ETF options**.

M1 can conclude as an evidence-backed audit with a historical-data blocker. M2 is
not automatically authorized and no financial model should be started from this
partial panel.
