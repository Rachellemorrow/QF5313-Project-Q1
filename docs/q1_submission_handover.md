# Q1 final submission handover (2026-11-05)

SOFTWARE: PARTIAL; EMPIRICAL: BLOCKED; GROUP_SUBMISSION: NOT_READY.

Final recovery is committed in d34b0138ad9c9e76b23731e62d996f2ad7ec9b69. Entry working tree was clean. No M0-M8 experiment, training, download, or source recovery was repeated.

## Report support

Use q1_report_draft.md as the reviewed agent-assisted Q1 contribution for student revision and group integration. Its Section 10 maps every examination requirement in Sections 3.1?3.5 and distinguishes reporting coverage from unmet empirical requirements. Use q1_report_support.md as the supporting evidence packet: forward Black-Scholes and Brent inversion; SVI constraints/objectives; separately labelled examination and complete butterfly formulas; calendar/forward variance; neural price loss and penalties; M4-M7 settings, measured tables, failure coverage, figure index, provenance and limitations. Use q1_final_recovery.md for exact rejection and strike-support tables and reproduction commands. Primary evidence is q1_final_recovery_evidence.json and the tracked verified_runs bundle. These are support materials, not student-authored report text.

The measured conclusions remain 606 archived observations and zero eligible empirical quotes. Synthetic results must remain labelled synthetic. Preserve L-BFGS failures, conditional coverage and dense neural calendar violations. M8's 264-test record is historical, not rerun in this handover.

## Authentic session material for group SESSION.md

Both requested native sessions were exported with scripts/q1_session_export.py. Complete JSON, lossless message-by-message Markdown (including tool states), and original export proofs are in:

- .tmp/q1_handover_sessions_m0_m8/ (378 messages, 476 tool parts)
- .tmp/q1_handover_sessions_recovery/ (44 messages, 48 tool parts)

q1_handover_verification.json records IDs, hashes, prompt inventory and independent round-trip/hash checks. No missing calls were reconstructed or summarized. These local exports are deliberately not published in Git before sensitive-content and attachment review; .tmp is ignored and absent from a clean clone. Students must securely preserve them, review complete contents and referenced attachments, record explicit redactions, and incorporate the authentic Markdown into the group's complete SESSION.md without dropping earlier records. Collect other members' sessions and separately preserve this Codex handover chat if required; it is not part of either OpenCode export. No root SESSION.md has been fabricated. Export proofs retain their original generic live-snapshot warning; neither export certifies other machines or later activity.

## Verification and tracking

q1_handover_verification.json records executed checks. Two targeted recovery tests passed in 2.11 seconds; offline source verification passed with zero eligible quotes; restoration verified 24 snapshots and indexed 102 original artifacts without training; offline lock and no-change sync checks passed. Required Q1 entry points, fetcher, metadata, exporter, launcher, pinned dependencies and all 24 replay archives are tracked. Q1 tests, cleaned source snapshots, mathematical and experiment documentation are tracked. Raw data and original result HTML are ignored by policy; tracked cleaned evidence and compressed numerical/checkpoint bundles support offline artifact replay. Original HTML references are indexed, not included in a clean clone. No full-suite rerun or independent retraining replication is claimed.

## Remaining student and group responsibilities

Students must review, revise and take responsibility for every final report section: introduction/problem and contributions; data/provenance/rights and eligibility; model equations and assumptions; experimental protocol; results and interpretation; validation; limitations; reproducibility; conclusions; references and statement of work. report.md is absent. Review the formulas and numerical tables in the support packet rather than presenting agent prose as student authorship.

Group members must integrate Q2/Q3's four approaches, datasets, tests and metrics; coordinate shared README/environment changes; prepare the complete report and authentic SESSION.md; verify all six approaches and offline delivery; confirm source redistribution rights, pricing conventions and unresolved historical scope. README remains an early-stage document. Approve any future empirical recovery separately. No Q2/Q3 files or shared configuration were modified. No submission tag was created; the group owns final submission readiness and tagging.

Reproduction commands are in q1_final_recovery.md under Reproduction and delivery. Use new output paths. Setup alone may require network; replay uses tracked evidence and is not independent training replication.


## Final delivery checklist for the group leader

Send the existing Git repository with its history, including this final handover commit, rather than copying only the draft. Do not send local environments/caches. Shared dependency and comparator files are dependencies for integration, not permission to overwrite teammates' changes. Merge with group coordination.

### A. Repository files to retain in the group checkout

- `docs/q1_report_draft.md`: reviewed agent-assisted Q1 draft; students must revise and integrate into root `report.md`.
- `docs/q1_report_support.md`, `docs/Q1_validation.md`, `docs/q1_submission_handover.md`: support, scope and reproduction guidance.
- `docs/q1_final_recovery.md`, `docs/q1_final_recovery_evidence.json`, `docs/q1_final_verification.json`: final zero-eligibility result and exact rejection evidence.
- `docs/q1_m4_verification.json`, `docs/q1_m5_verification.json`, `docs/q1_m6_verification.json`, `docs/q1_m7_verification.json`, `docs/q1_m8_verification.json`: historical experiment/audit evidence.
- `docs/q1_handover_verification.json`, `docs/q1_final_handover_check.json`: authentic-session proofs, exact source-file inventory and final integrity checks.
- All tracked `docs/q1_*.md`, `docs/project_spec.md`, `docs/decisions/m0-environment.md` and `AGENTS.md`: methods, source disclosures, requirements and contract; preserve the documented historical initial-contract deviation in M8 rather than claiming retroactive compliance.
- `code/approach1_q1_svi.py`, `code/approach2_q1_torch.py`, every Python file in `code/q1/`, and `code/compare.py`: complete Q1 source and comparison dependencies. Exact 34-file source inventory is in `docs/q1_final_handover_check.json`.
- `data/fetch.py`, `data/metadata.json` and the complete tracked `data/clean/q1/` tree (127 files): offline data, schema, provenance and hash-verified source evidence.
- The complete tracked `tests/q1/` tree (48 files), including `fixtures/synthetic/` and `fixtures/verified_runs/manifest.json` plus all 24 `.gz` archives: tests and lossless numeric/checkpoint evidence.
- `scripts/project.py`, `scripts/q1_isolated_replay.py`, `scripts/q1_session_export.py`, `pyproject.toml`, `uv.lock`, `.python-version`, `.gitignore` and local `.gitattributes`: pinned environment, replay/export tools and byte-preservation rules.

### B. Separate local material: NOT included in a Git clone

Send each of these six files securely after reviewing sensitive contents; preserve explicit redaction records and any referenced attachments. Both requested sessions exist; no requested session is missing. Completeness is certified only for the exported native payloads, not external attachments, later activity, other machines or other group members.

- `.tmp/q1_handover_sessions_m0_m8/session.json`
- `.tmp/q1_handover_sessions_m0_m8/session.md`
- `.tmp/q1_handover_sessions_m0_m8/export_verification.json`
- `.tmp/q1_handover_sessions_recovery/session.json`
- `.tmp/q1_handover_sessions_recovery/session.md`
- `.tmp/q1_handover_sessions_recovery/export_verification.json`

Send the following four saved experiment directories with all their files (metrics, diagnostics, manifests, checkpoints and original self-contained HTML), after source/rights review. This avoids retraining to recover original figures. The exact 27 HTML paths and checksums are listed in `tests/q1/fixtures/verified_runs/manifest.json` under `original_artifact_inventory`; draft Section 4.4 provides individual figure links.

- `results/verification/approach1/q1_synthetic_svi_verification/`
- `results/m5_final/verification/approach2/q1_synthetic_svi_optimizer_comparison/`
- `results/verification/approach2/q1_synthetic_neural_verification/`
- `results/verification/approach2/q1_synthetic_missing_quotes/`

The tracked bundle is the authoritative numerical replay fallback if originals are not transferred. Regenerate `comparison.html` from restored metrics using the commands in the draft; regenerating this dashboard does not recreate every original milestone HTML figure. Do not claim the 27 ignored originals are present in a clean clone. The group leader should test the figure links after packaging and preserve the directory layout or update report references.

### C. Explicit decisions for the group leader and students

SOFTWARE: PARTIAL; EMPIRICAL: BLOCKED; GROUP_SUBMISSION: NOT_READY. Zero real observations pass the final gate; no Chinese ETF empirical fit or <2% market benchmark is demonstrated. Preserve synthetic labels, L-BFGS failures, missing target coverage, inactive butterfly ablation and dense neural calendar/forward violations. Review formula interpretation, liquidity/grid rationale, conventions and source rights. Record actual human/agent work division, including this drafting/review chat. Assemble authentic group `SESSION.md` and student-reviewed `report.md`; coordinate README and Q2/Q3 integration. No submission tag is created here.

Final handover checks inspected source syntax and tracking, snapshot and figure hashes, native-session message round trips and hashes, offline source integrity, and offline lock/no-change sync. A bounded scan of 243 tracked/proposed files (including decompressed archives and the draft) found no private-key headers, AWS AKIA IDs or long sk-prefixed keys; this is not proof of absence of every secret. No tracked file exceeded 50 MB and no tracked cache/temporary path was found. The two new handover records must also pass the same bounded scan before commit. No training, download or expensive experiment was rerun. Prior test results remain historical; tests were not rerun in this final file-delivery audit.
