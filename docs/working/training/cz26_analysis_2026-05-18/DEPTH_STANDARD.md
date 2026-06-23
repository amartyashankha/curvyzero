# Analysis Depth Standard

Use this as the checklist for CZ26 analysis. The old `r18fresh` postmortem did
this well; CZ26 should match or exceed that standard.

## Required Shape

Every serious analysis note should include:

1. Source files and exact commands or script names.
2. A plain glossary for every shorthand.
3. Exact opponent recipe counts.
4. Coverage and missingness before conclusions.
5. Survival, training reward, and tournament rank as separate signals.
6. Matched comparisons where only one setting changes.
7. Exact-horizon readouts, especially 30k, 170k, and 300k.
8. Per-run table or a clear link to the generated per-run artifact.
9. Reward component split:
   - survival component;
   - bonus component;
   - inferred outcome residual.
10. Tournament exposure:
   - learned-only rank;
   - games;
   - battles;
   - whether rank is too sparse to trust.
11. Action-collapse warnings.
12. A plain `Read:` paragraph after every table.
13. Caveats and next questions.

## What Not To Do

- Do not use raw top tournament rows without removing `iteration 0`.
- Do not compare reward values across different reward definitions as if they
  are on one scale.
- Do not treat latest checkpoint as the best checkpoint.
- Do not rank recipes from one projection table without checking matched
  contrasts.
- Do not treat one-battle tournament ranks as precise ordering.
- Do not hide low-coverage rows.

## Current Generated Report

The current comprehensive generated report is:

```text
artifacts/local/cz26_analysis_2026-05-18/cz26_deep_report.md
artifacts/local/cz26_analysis_2026-05-18/cz26_deep_report.json
```

It contains:

- exact recipe counts;
- coverage;
- lifecycle summaries;
- exact-horizon tables;
- matched pairwise contrasts;
- top learned tournament rows;
- action-collapse watchlist;
- per-run table for all 136 runs.
