# Asset Registry

Last updated: 2026-05-17 11:48 EDT.

This file is the current list of important live or recently-live assets for the
r18fresh postmortem. Update this before launching, killing, purging, or sleeping.

## Current Assets

| Asset | Kind | Status | Purpose | Preserve Before Cleanup |
| --- | --- | --- | --- | --- |
| `curvy-r18fresh-allv2-20260516a` | trainer run prefix / batch | analyze-before-kill | 18-run batch under review | eval curves, checkpoint refs, learner metrics, assignment proof |
| `curvy-r18fresh-live-bounded-dsf1-20260516b` | tournament arena | preserve | tournament that saw r18fresh checkpoints | latest rating snapshot, active top rows, battle summaries, export snapshots |
| `elo-r18fresh-live-bounded-dsf1-20260516b` | rating id | preserve | Elo/rating state for current tournament | rounds, latest pointer, active/provisional/retired rows |
| `curvy-r18fresh-live-bounded-dsf1-20260516b-elo-r18fresh-live-bounded-dsf1-20260516b-training` | trainer-facing leaderboard/export | inspect | source for frozen-opponent assignments | latest generation, selected refs, assignment shas |
| `curvy-ownlatest-staticmix-20260516b` | control run prefix | keep-running | own-latest/static-mix control lane | eval snapshots and assignment-consumption proof |
| `curvyzero-checkpoint-tournament-v2` | Modal app | current | tournament worker/browser deployment | app URL and logs if debugging |
| `curvyzero-lightzero-curvytron-visual-survival-train-v2` | Modal app | current | trainer deployment | run logs and Volume artifacts |
| `curvyzero-curvytron-gif-browser-v2` | Modal app | current | run/GIF inspection | category defaults and URL filters |

## Latest Modal Inventory Check

Checked 2026-05-17 11:48 EDT:

- Stopped deployed tournament app `ap-NOZz2IQRfFlBUbz0zIRC2M` because skipped
  `round-000035` workers were still running and timing out after state marked
  that batch skipped.
- Redeployed current `curvyzero-checkpoint-tournament-v2` code; new deployed app
  id is `ap-dL1sKLPi1l5XnWx6pNXSBR`.
- Fresh drain `fc-01KRV9TXJHWSVXF2ZY046RQP4F` created active
  `round-000036`, which covers `3427` checkpoint refs and is bounded to
  `300` pairs / `6300` games.
- Preserve all v2 Volumes and Dicts. Do not purge tournament or run Volumes
  while `round-000036` is active.

Checked 2026-05-16 18:09 EDT:

- v2 Volumes still exist and were not broadly purged:
  `curvyzero-runs-v2`, `curvyzero-curvytron-tournaments-v2`, and
  `curvyzero-curvytron-control-v2`.
- Deployed current apps still exist and were kept running:
  `curvyzero-checkpoint-tournament-v2`,
  `curvyzero-lightzero-curvytron-visual-survival-train-v2`, and
  `curvyzero-curvytron-gif-browser-v2`.
- Active ephemeral `curvyzero-checkpoint-tournament-v2` apps were stopped:
  `ap-X0DUzy51Yzb3wWIBUx42jf`, `ap-M4Svi3KT1vcG5sNLgSsp8n`,
  `ap-5KP4ObfLZgGjdhL2BVHSHq`, `ap-bcT3SUJbWLOHe8Rd3pkjxy`,
  `ap-6VT00kYVP4Ucd2tcPik7KD`, and `ap-2ubSmNvhhM7UVAkQ8N2tn0`.
- Removed only transient tournament scale-probe directories:
  `curvy-scale-probe-18latest-gamefanout-20260516a`,
  `curvy-scale-probe-5latest-gamefanout-20260516a`, and
  `curvy-scale-probe-5latest-nogif-20260516a`.
- Preserved r18fresh/ bounded r18fresh tournaments, the validation arena, e2e
  proof artifacts, runs volume, and control pointers.

Checked 2026-05-16 16:23 EDT:

- v2 Volumes exist: `curvyzero-runs-v2`,
  `curvyzero-curvytron-tournaments-v2`, and
  `curvyzero-curvytron-control-v2`.
- v2 coordination objects exist: `curvyzero-curvytron-checkpoint-events-v2`,
  `curvyzero-curvytron-checkpoint-intake-v2`, and
  `curvyzero-curvytron-opponent-leaderboard-live-v2`.
- Deployed current apps exist: `curvyzero-checkpoint-tournament-v2`,
  `curvyzero-lightzero-curvytron-visual-survival-train-v2`, and
  `curvyzero-curvytron-gif-browser-v2`.
- One detached tournament app was still present:
  `ap-X0DUzy51Yzb3wWIBUx42jf`, description
  `curvyzero-checkpoint-tournament-v2`, created 2026-05-16 05:42 EDT.
  It was later stopped in the 18:09 cleanup after the relevant state had been
  recorded.

## Required Status Values

- `current`: active source of truth.
- `preserve`: do not delete until listed artifacts are copied or pinned.
- `keep-running`: intentionally left alive as a control.
- `analyze-before-kill`: likely stale or complete, but must be summarized first.
- `archive`: historical only; should not guide launches.
- `cleanup-candidate`: safe to kill/delete after a named check.

## Cleanup Rule

No asset should be killed because it "looks old". Kill only after the row above
records what was preserved, or after we explicitly decide it has no useful
signal.
