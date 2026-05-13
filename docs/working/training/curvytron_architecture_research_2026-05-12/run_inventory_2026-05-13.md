# CurvyTron Run Inventory - 2026-05-13

Snapshot time: 2026-05-13 10:43 EDT.

Purpose: catalog the current CurvyTron run surfaces plainly. This is a factual
inventory, not a learning conclusion and not a capacity diagnosis.

## Current Plain Read

There are three current large surfaces:

| Surface | Rows | Launched | Age at snapshot | Current state |
| --- | ---: | --- | --- | --- |
| `curvy-survive-bonus-large-20260513b` | 300 | 05:37 EDT | 5h 1m | current survival-plus-bonus diagnostic batch |
| `curvy-mix2-clean-20260513a` | 156 | 07:49 EDT | 2h 48m | current first clean opponent-mixture batch |
| `curvy-mix3-currentckpt-20260513a` | 300 | 09:31 EDT | 1h 6m | current fresh-checkpoint opponent-mixture batch |

There are also 65 stale or preserved rows still visible as artifacts:

| Surface | Rows | State |
| --- | ---: | --- |
| `survivaldiag-v1b-20260513h` | 50 | old preserved batch |
| `curvy-mix-recent-canary-20260513a` | 3 | failed canary |
| `curvy-mix2-canary-20260513a` | 6 | failed canary |
| `curvy-mix2-canary-20260513b` | 6 | corrected canary, historical proof |

Logical rows represented by the current catalog: `821`.

That number is the union of current rows, preserved old rows, and historical
canaries. It is not the number of useful learning claims.

## Timeline

| Batch | Manifest time | Launch time | Rows | State |
| --- | --- | --- | ---: | --- |
| `survivaldiag-v1b-20260513h` | 02:55 | local artifact only | 50 | old preserved 50-row batch |
| `curvy-survive-bonus-large-20260513a` | 04:39 | 05:02 | 300 | broken first grouped 300-row launch; run roots were deleted |
| `curvy-survive-bonus-large-20260513b` | 05:34 | 05:37 | 300 | current survival batch |
| `curvy-mix-recent-canary-20260513a` | 06:16 | 06:18 | 3 | failed first mixture canary |
| `curvy-mix2-canary-20260513a` | 06:57 | 06:59 | 6 | failed relation-mismatch canary |
| `curvy-mix2-canary-20260513b` | 07:06 | 07:06 | 6 | corrected mixture canary |
| `curvy-mix2-clean-20260513a` | 07:35 | 07:49 | 156 | current first clean mixture batch |
| `curvy-mix3-nextwave-20260513a` | 08:30 | dry-run only | 300 | stale draft, not launched |
| `curvy-mix3-currentckpt-20260513a` | 09:28 | 09:31 | 300 | current fresh-checkpoint mixture batch |

## Current Status Snapshot

Fresh status was pulled with `lightzero_curvytron_run_status.py --output json`
against the three current large manifests.

| Surface | Rows | Trainer status | Heartbeats | Train roots | Pollers running | Latest checkpoint spread | Latest eval read |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| `curvy-survive-bonus-large-20260513b` | 300 | 300 running | 300 | 300 | 300 | mostly `k105` to `k150` | 300 eval values, mean 93.936, max 155.375 |
| `curvy-mix2-clean-20260513a` | 156 | 156 running | 156 | 156 | 154 | mostly `k20` to `k70` | 121 eval values, mean 61.409, max 107.125 |
| `curvy-mix3-currentckpt-20260513a` | 300 | 38 running, 262 not yet trainer-started in status | 38 | 194 | 190 | 37 rows have checkpoints, mostly `k10`/`k20` | 37 eval values, mean 49.503, max 73.625 |

Plain interpretation:

- `curvy-survive-bonus-large-20260513b` is mature and artifact-rich.
- `curvy-mix2-clean-20260513a` is mature enough for monitoring and early
  mixture readout, but not a final recipe ranking.
- `curvy-mix3-currentckpt-20260513a` is the newest current-checkpoint mixture
  batch. It has started and has real artifacts, but most rows have not yet
  reached trainer-started status in the status reader.

## Volume Roots

Modal volume listing under
`training/lightzero-curvytron-visual-survival/` shows 718 run directories:

| Prefix | Directories | Meaning |
| --- | ---: | --- |
| `curvy-survive-bonus-*` | 300 | current survival batch |
| `curvy-mix3cur-*` | 197 | current mix3 rows that have created roots so far |
| `curvy-mix2clean-*` | 156 | current mix2 rows |
| `survivaldiag-v1b-*` | 50 | preserved old batch |
| `curvy-mix2-*` | 6 | failed canary roots |
| `curvy-mix2b-*` | 6 | corrected canary roots |
| `curvy-mix-canary-*` | 3 | failed first mixture canary roots |

The volume-root count can lag the logical launch count because some submitted
rows have not yet created trainer-owned roots.

## Manifest And Launch Artifacts

Current survival batch:

- manifest:
  `artifacts/local/curvytron_survivaldiag_manifests/curvy-survive-bonus-large-20260513b.json`
- grouped launch:
  `artifacts/local/curvytron_survivaldiag_manifests/curvy-survive-bonus-large-20260513b.grouped_submit_launch.json`

Current mix2 batch:

- manifest:
  `artifacts/local/curvytron_opponent_mixture_manifests/curvy-mix2-clean-20260513a.json`
- grouped launch:
  `artifacts/local/curvytron_opponent_mixture_manifests/curvy-mix2-clean-20260513a.grouped_submit_launch.json`

Current mix3 batch:

- manifest:
  `artifacts/local/curvytron_opponent_mixture_manifests/curvy-mix3-currentckpt-20260513a.json`
- grouped launch:
  `artifacts/local/curvytron_opponent_mixture_manifests/curvy-mix3-currentckpt-20260513a.grouped_submit_launch.json`

Stale or historical:

- `survivaldiag-v1b-20260513h.json`: preserved old 50-row batch.
- `curvy-survive-bonus-large-20260513a.json`: broken first 300-row grouped
  launch; not current.
- `curvy-mix-recent-canary-20260513a.json`: failed canary; not current.
- `curvy-mix2-canary-20260513a.json`: failed canary; not current.
- `curvy-mix2-canary-20260513b.json`: corrected canary; historical proof.
- `curvy-mix3-nextwave-20260513a.json`: dry-run draft with stale refs; not
  current.

## Shapes

`curvy-survive-bonus-large-20260513b`:

- 300 rows.
- 150 `body_circles_fast`, 150 `browser_lines`.
- Survival-plus-bonus reward, no outcome reward.
- Blank/passive diagnostic surface, current source of frozen checkpoints.

`curvy-mix2-clean-20260513a`:

- 156 rows.
- 78 `body_circles_fast`, 78 `browser_lines`.
- Recipes:
  `r50-blank50`, `r50-mid50`, `r50-old50`, `r50-scr50`,
  `r50-blank25-scr25`, `r50-mid25-old25`,
  `r50-blank20-mid15-scr15`, plus controls `recent100`, `mid100`,
  `old100`, `blank100`, `scr100`.
- Base tokens are balanced across:
  `rf/rb`, `sim8`, `collect32`, `batch32`, and `rep0/repM/repH`.

`curvy-mix3-currentckpt-20260513a`:

- 300 rows.
- 150 `body_circles_fast`, 150 `browser_lines`.
- Main recipes:
  `r25-blank75`, `r50-blank50`, `r75-blank25`, `r50-scr50`,
  `r50-mid25-old25`, `r40-blank20-mid20-scr20`.
- Controls:
  `recent100`, `blank100`, `scr100`, `mid100`, `old100`.
- Includes small compute probes:
  `sim16`, `collect64`, and `batch64`.

## What Is Stale

These are not current launch surfaces:

- `curvy-survive-bonus-large-20260513a`
- `curvy-mix-recent-canary-20260513a`
- `curvy-mix2-canary-20260513a`
- `curvy-mix3-nextwave-20260513a`

These are historical/proof artifacts:

- `survivaldiag-v1b-20260513h`
- `curvy-mix2-canary-20260513b`

Do not use stale surfaces for new learning claims.
