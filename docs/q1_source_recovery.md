# Q1 M1.5 — Historical ETF option archive recovery

## Decision: BLOCKED

**The original Q1 historical requirements are still not demonstrated as satisfiable
with the data and permissions currently available to this repository.** This is
not a finding that the required data do not exist. Several credible providers
document relevant products; an authorized export with verified dated metadata and
offline/grader-sharing rights has not been obtained.

No new qualifying ETF option rows, dated price/metadata joins, or complete chains
were recovered in M1.5. Documentation examples are **not** a downloaded research
dataset. The M1 archive remains unchanged. No account, subscription, payment, trial,
license application, provider SDK, model, or M2 pipeline was created.

Scope: Q1 only. Entry commit `0241cd29856b4e1bfa9473a63771c43061184717`; clean
working tree. Python 3.13.18 / uv 0.13.7 passed offline lock/sync checks; the existing
29 tests passed before investigation. Dependencies and Q2/Q3 are unchanged.

## Evidence terminology and storage

- **VERIFIED:** the specified document, schema statement, access condition or
  request outcome was actually inspected. This does not verify underlying rows.
- **PARTIAL:** relevant documented capability exists but authorized retrieval,
  coverage, join quality, adjustments or reuse rights remain unverified.
- **UNVERIFIED:** insufficient evidence; do not interpret as zero provider coverage.
- **BLOCKED:** the current project cannot pass the historical-data acceptance gate.

Machine-readable records are in `data/clean/q1/m15_recovery/`:

| File | Meaning |
|---|---|
| `source_evidence.json` | Source findings, documentary assertions, explicit access stops and ranked alternatives |
| `source_comparison.csv` | Comparable provider-level access/coverage/rights conclusions |
| `new_verified_coverage.csv` | Header-only: **zero new verified observations**, not invented date/expiry cells |
| `public_reference_checks.json` | Actual local request timestamps, PDF checksums and exact DNS failure |
| `audit_manifest.json` | Checksums for the audit records; explicitly **not** a market-data manifest |

Public documentation was read through `webfetch`. That tool does not expose exact
request timestamps; machine evidence records the review date, **not fabricated
retrieval timestamps**. Separately, the local public-reference requests record
their actual UTC timestamps. The three public SSE PDFs were downloaded to ignored
`.tmp/q1_m15_public_reference_inspection/` and read; **they are documentation, not
market-data samples**. Full PDFs, vendor catalog pages and illustrative market rows
are not redistributed in Git. Committed content is our audit summaries, schema
facts, citations and request/checksum records.

## 1. BigQuant — first-priority investigation

Sources actually read:

- [Daily option bars: `cn_option_bar1d`](https://bigquant.com/data/datasources/cn_option_bar1d).
- [Daily static data: `cn_option_static_data`](https://bigquant.com/data/datasources/cn_option_static_data).
- [BigQuant user agreement](https://bigquant.com/wiki/doc/yonghu-xieyi-oSyCHIHzNm).
- [Platform guide](https://bigquant.com/wiki/doc/xinshou-zhinan-htT8CrjJJh), which
  links the newer DAI data-platform documentation. The old guide identifies itself
  as historical; it is not treated as proof of the current SDK or entitlement.

### What is documented, versus what was verified in data

| Requested verification | Evidence | Result |
|---|---|---|
| Legitimate public/user-authorized API access | Catalog and documentation are public. No anonymous market-data API or authorized account/export entitlement established | PARTIAL; data access stopped |
| Authentication, subscription/payment, licensing | Agreement describes registered services, supplemental terms, restrictions on automation and third-party data copying | Restrictions VERIFIED; exact table tier/price UNVERIFIED |
| Actual 510050 historical coverage | Catalog describes SSE bars from 2015-02-09; SSE daily static metadata from 2023-08-25; both narrative end dates 2026-11-03 | Documentary claim only; actual query coverage UNVERIFIED |
| Both calls and puts | Static schema has `call_put` with C/P conventions; no retrieved 510050 call/put sample | UNVERIFIED in data |
| Four expiries on sample dates | No authorized date-by-expiry counts obtained | UNVERIFIED |
| Five sample dates spanning two years | Claimed date span is potentially sufficient, but exact target dates and completeness not queried | UNVERIFIED |
| One month of daily chains | No complete month retrieved; known gaps disclosed below | UNVERIFIED |
| Reliable price/metadata join | Both tables document `(date, instrument)` keys; uniqueness, unmatched rows and temporal validity not tested on data | PARTIAL schema evidence only |
| Historical adjustments and multipliers | Dated `strike_price`/`multiplier` fields documented; separate adjustment-change table referenced | PARTIAL; no adjustment records validated |
| Legal/technical offline snapshots | CSV export/repository/grader permissions and applicable third-party license not established | UNVERIFIED; no extraction/export attempted |

**Stop condition applied:** the agreement restricts automated extraction without
written consent, access through unapproved interfaces, and copying/redistribution
of some data; publicly shared content does not automatically include third-party
market data. No credentials were requested, discovered or supplied by the agent.
No login, token endpoint, query service, account creation, trial activation or
payment was attempted. No further BigQuant automated data access was pursued after
reviewing those restrictions. Provider-specific subscription and current pricing
remain unknown rather than assumed free or paid.

### Documented join and field semantics requiring validation

- Use `date` **and** `instrument` as the proposed daily join key, not the latest
  metadata for an expired identifier. A BigQuant ETF instrument may be
  `10011425.SHO`; the static `contract_id` carries the long exchange trading code,
  while `trading_code` is the numeric identifier in the documented SSE example.
  These names must not be joined blindly to another provider's naming scheme.
- Preserve `underlying`, `call_put`, `expire_date`, `strike_price`, `multiplier`,
  listing/delisting dates, original identifier and original trading code. Restrict
  to SSE and `510050.SH` using actual dated attributes, not inferred index options.
- The daily static table is documented as a daily pre-session state. Its as-of
  timing and date validity still need empirical checks, especially around changes.
- The separate `cn_option_basic_info` mentioned in the common documentation has
  one row per contract and no `date`; it cannot automatically repair missing
  effective-dated strikes/multipliers. `cn_option_contract_info_change` is referenced
  by the static catalog but was **not independently queried or inspected**.
- Bars document OHLC, settlement, prior close/settlement, volume and OI; the two
  specified tables do not document bid/ask history. Documented zero-volume bars can
  retain a close despite missing open/high/low: a close is not proof of liquidity.
- The catalog warns that `in_out_money` uses a back-adjusted ETF underlying close.
  Do not use that field as verified nominal-contract moneyness. The schema labels
  `amount` in ten-thousand yuan while its narrative says yuan; retain the conflict
  until an authorized sample resolves it. `pre_open_interest` is documented as
  uninformative zero. A malformed header-like static row is reported for 2024-10-09.

### Published gap warnings — not independently measured

- SSE daily bars: **2024-12-26** and **2025-03-03** reportedly missing.
- Static metadata: **2024-02-08–2024-09-24**, **2025-01-16–2025-02-05**, and
  **2025-11-28–2025-12-15** reportedly missing, plus unspecified scattered gaps.
- Therefore a valid two-year span does not imply a complete month or valid metadata
  on every requested date. Do not forward-fill these gaps from later snapshots.
- Catalog footer publication/update timestamps are not observation coverage dates.
  In particular, a catalog's 2024 creation timestamp must not be substituted for
  the narrative's historical start date—or vice versa.

## 2. Official SSE / SSE Infonet

The [official market-service page](https://www.sseinfo.com/services/assortment/market/)
links the following public PDFs, which were successfully downloaded and read:

| Document | Verified version | Relevant evidence |
|---|---|---|
| Historical data product specification | 2.0.0, 2025-06-12 | Section 2.3: option history includes snapshots, daily and minute bars |
| Historical data interface specification | 1.1.5, 2026-03-27 | Pages 22–24: option snapshot and K-line fields/layouts |
| Stock-option product specification | 1.1.1, 2025-06-12 | Section 2.1: **current trading day's** contract basics; section 2.3: closing data |

Full source URLs and SHA-256 hashes are in `public_reference_checks.json`.

The documented delivery layout is `sho/YYYYMMDD/Day.csv` and
`sho/YYYYMMDD/Snapshot.csv`, UTF-8 CSV with LF newlines. These are **file layout
names**, not discovered public download URLs. We did not construct URLs for them.

- Daily fields include `SecurityID`, `DateTime`, `PreClosePx`, `OpenPx`, `HighPx`,
  `LowPx`, `LastPx` (daily close), `Volume`, `Amount`, `LongPosition`, `AvgPx`,
  `PreSettlePx`, `SettlePx`.
- Historical snapshots document OI/volume and five-level bid/ask prices and sizes.
  Bid/ask is a snapshot product, not a promised field in the daily bar.
- The option-product specification describes underlying, call/put, strike and
  trading/exercise/expiry dates in current-day basics. It does **not establish that
  an effective-dated historical master/multiplier archive is included** with a
  historical-price purchase. That is a required vendor question, not a solved join.
- Format examples and field activation dates are not empirical coverage dates.
  The actual deliverable date range, complete 510050 roster, adjustment archive and
  sample-date expiry counts remain **UNVERIFIED**.

### Legitimate acquisition procedure and permissions

The [published authorization statement](https://www.sseinfo.com/aboutus/authstatement/)
requires permission for receipt/use/distribution of exchange information. The
market-service page publishes licensing price lists and an application route;
this does not determine the exact fee or academic exemption for this request.

The **student or authorized institution**, not this agent, should:

1. Contact SSE Infonet via the published service number **4008888400, menu 2-5**,
   or its linked [business application page](https://bsp.sseinfo.com/business/?id=1671080829401092098).
2. Request a bounded academic sample/quote for SSE 510050 option daily bars and
   archived effective-dated contract basics, units and adjustment notices; ask
   separately for historical bid/ask snapshots if required for filtering.
3. Ask for exact coverage/missing-date lists, archived expired IDs, sample schemas,
   units, identifier mappings and delivery mechanism. The public documents alone
   do not answer all these questions.
4. Obtain explicit permission for local retention, group research, inclusion in a
   private repository, and offline access by the course grader. Ask whether public
   redistribution is prohibited and whether a research/sample exemption exists.

**Stopped before any application, login, licensed download or purchase.** No free
research **market-data** sample was identified on the inspected pages; three free
technical PDFs are not such a sample. This is a bounded search, not proof that SSE
cannot supply an academic sample on request.

## 3. Other credible alternatives and open repositories

### RiceQuant RQData

The [options API documentation](https://www.ricequant.com/doc/rqdata/python/options-mod.html)
explicitly documents dated contract selection and
`options.get_contract_property(order_book_ids, start_date=None, end_date=None,
fields=None, market='cn')` with daily strike/multiplier histories. Its documented
dividend example retains a numeric contract ID while strike and multiplier change.
These are **documentation examples**, not observations downloaded by this project.
The catalog claims 50ETF daily/minute coverage since its 2015-02-09 listing.

The [access manual](https://www.ricequant.com/doc/rqdata/python/manual) describes
account initialization, trial/full licenses and quotas. It documents CSV export as
a technical operation, not permission to redistribute licensed data to a repository
or grader. **Stopped before installing `rqdatac`, initialization/login, requesting
a trial or querying data.** Exact 2024/2026 coverage and subscription rights remain
unverified. This is a strong candidate if an existing institutional entitlement can
be used with written export permission; none is assumed here.

### Tushare Pro

Official [opt_basic](https://tushare.pro/document/2?doc_id=158) documentation requires
**5,000 points**; [opt_daily](https://tushare.pro/document/2?doc_id=159) requires
**2,000 points**, with a documented 15,000-row per-call limit and token-based examples.
Daily fields include close, settlement, volume and OI. Basics document adjusted
`exercise_price` and `opt_multiplier`, but a listing-date filter is **not proof of
an effective-dated adjustment series**. Historical as-of adjustment support and
repository/grader rights must be established separately. **No token, subscription,
SDK installation or data call was attempted.** Points are an entitlement condition,
not an independently verified price quote or proof that this user has access.

### Public research repositories

- A bounded [Harvard Dataverse metadata search](https://dataverse.harvard.edu/api/search?q=%2250ETF%22&type=dataset&per_page=5)
  returned status OK and zero results for that exact query. This is not an exhaustive
  search or proof of absence.
- A bounded [Zenodo metadata search](https://zenodo.org/api/records?q=%2250ETF%22%20options&size=5)
  returned broad, partly unrelated results. One candidate, DOI
  `10.5281/zenodo.4288593`, is a 2020 replication dataset for a panel study of option
  introduction and stock-market volatility, with publisher-asserted CC-BY-4.0.
  Its relevance to contract-level chains is unverified. A local anonymous request
  for its record failed DNS resolution; **no record file or data sample was
  downloaded or inspected locally**. The exact error is retained in the request log.
  It is not promoted to a 2024/2026 option chain or to verified third-party rights.

## 4. Source comparison

Counts below are **new fully qualified sample dates verified in M1.5**, not provider
database totals. Zero means nothing qualifying was recovered here; provider totals
and expiry counts remain unknown. M1's sparse API observations remain separate.

| Provider | Access requirements | Verified range / usable dates | Expiries / C-P | Historical metadata quality | Liquidity fields | Offline reproduction and licensing | Verdict |
|---|---|---|---|---|---|---|---|
| BigQuant | Authorized account/table/interface and export permission not established; exact price unknown | UNVERIFIED / 0 new | UNVERIFIED; C/P documented | Dated static keys/fields documented; substantial gaps; no actual join/adjustment check | Daily volume/OI documented; bid/ask absent from these two schemas | Audit records only; third-party data and redistribution restrictions unresolved | PARTIAL capability; BLOCKED access |
| SSE Infonet | License/application; academic price/sample needs confirmation | UNVERIFIED / 0 new | UNVERIFIED | Current basics documented; historical master/multiplier delivery not established | Daily volume/OI/settlement; snapshots have bid/ask | Public specs checked; no licensed data or sharing permission | PARTIAL product validation; BLOCKED acquisition |
| RiceQuant | Authorized trial/full account and rights | UNVERIFIED / 0 new | UNVERIFIED; C/P selection documented | Explicit daily strike/multiplier history API; no actual query | Price API referenced; precise entitled fields not inspected here | CSV export technically documented; grader/repository rights unknown | PARTIAL; strong licensed alternative |
| Tushare Pro | Token plus 5000/2000 points | UNVERIFIED / 0 new | UNVERIFIED | Adjusted basics documented; date-effective series unverified | Daily volume/OI/settlement documented | No export or redistribution entitlement verified | PARTIAL; adjustment-history gap remains |
| Open repositories checked | Public metadata; candidate CC-BY asserted | No qualifying dataset verified / 0 new | UNVERIFIED | No contract schema inspected | UNVERIFIED | Local candidate retrieval failed; no market snapshot | UNVERIFIED suitability |

## 5. Conditional small-sample acceptance plan — not executed

After authorization, first obtain a small **single-date** complete 510050 roster
and prices, then expand only after matching and license checks pass. Retain source
contract IDs and date-effective strike, expiry, C/P, multiplier and adjustment flags.
Check unique keys and anti-joins on date+contract ID; report unmatched prices and
metadata rather than filling them. Use original quoted close/settlement definitions
and retain zero-volume/missing-price flags without inventing observations.

The existing five targets, **2024-10-08, 2024-10-31, 2025-10-09, 2026-10-08,
2026-10-30**, span more than two calendar years. They avoid the specifically listed
BigQuant gaps but **are not verified available there**. October 2024 or October
2026 is a candidate full month, subject to an official trading calendar and complete
daily chain counts. On every target/date, count distinct **exact expiries**, both
calls and puts, strike coverage, positive finite prices, and dated metadata matches.
No month is accepted merely because the minimum and maximum dates exist.

Include at least one genuine adjusted contract across its effective change date,
compare before/after strike and multiplier with an exchange notice, and avoid a
latest-master join. Save only legally permitted samples, provenance, schema, units,
retrieval timestamp and SHA-256; verify offline reload and the 50 MB policy. A
technical CSV export alone does not satisfy the permission check.

## 6. Ranked next actions — student approval required

1. **BigQuant:** ask an already authorized account holder/provider for table
   entitlement and written academic export/grader-sharing permission, plus a tiny
   2024/2026 price/static/adjustment sample and gap inventory. No permission is
   inferred from the public catalog or user task alone.
2. **Existing institutional RiceQuant license:** request approval for a small
   dated-contract-property sample; it explicitly addresses the adjustment problem.
   Trial registration is a student/provider action, not authorized for this agent.
3. **SSE Infonet official academic request:** request a sample/quote and confirmation
   of historical master/adjustment delivery and retention/sharing rights. No forms
   or payments were submitted.
4. **Already entitled Tushare account:** consider only with a separate solution for
   effective-dated adjustments and explicit storage/sharing permission.

No institution/account entitlement is assumed. A permitted student-supplied export
is acceptable for further verification only with source provenance and permission;
do not post credentials in the transcript. Do not substitute index options or
reduce the historical requirement without explicit methodological approval.

## 7. Commands, limitations, and stop

Executed bounded local reference inspection:

```powershell
py -3.11 -B scripts/project.py uv run --locked --offline python code/q1/inspect_public_archives.py --output .tmp/q1_m15_public_reference_inspection
```

The output directory must be new. `uv --offline` refers to package resolution;
this explicit reference-inspection command makes public HTTP requests. It does
not query BigQuant, RiceQuant or Tushare data endpoints. Future requests use an
anonymous session without environment-derived credentials and stop on login
redirects/access gates; tests use mocked responses and no live services.

Offline regression command:

```powershell
py -3.11 -B scripts/project.py uv run --locked --offline python -m pytest tests/q1 -q
```

Actual result: **42 passed in 16.85 seconds**: 29 existing M0/M1 tests plus 13
M1.5 audit-integrity, access-gate, anonymous-session and bounded-download checks.
The M1.5 regression tests used mocks and denied live socket connections; they do
not imply successful vendor data access. The existing M1 snapshot hashes passed.

The public catalogs and PDF schemas establish documented possibilities, not actual
complete historical coverage or a legal conclusion about a particular license.
No new price/metadata join, expiry matrix with observations, adjustment validation
or market-data reload could legitimately be completed under the current access.
The empty new-coverage CSV makes that boundary machine-readable. Prior M1 data and
their checksums are preserved; old evidence is not relabelled as newly recovered.

**Return status: BLOCKED. Stop here pending authorized access/sample and permission.
M2 and financial modeling have not started.**
