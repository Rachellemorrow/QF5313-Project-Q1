# QF5313 Advanced Programming 2026
Group: Q1/Q2/Q3; this session implements Q1 only. Current milestone: M0 environment.
Windows x86-64 bootstrap prerequisite: installed Python 3.11 and the `py` launcher.
From the repository root, one-command environment reproduction and verification:
`py -3.11 -B scripts/project.py setup; if ($LASTEXITCODE -eq 0) { py -3.11 -B scripts/project.py verify }`
No activation, global installation, PATH modification, or execution-policy change.
Initial setup needs internet; verification is offline after successful setup.
See `AGENTS.md` for exact commands and `docs/decisions/m0-environment.md` for limitations.
No financial models or market-data downloads are included in M0.
