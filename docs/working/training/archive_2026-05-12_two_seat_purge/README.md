# Two-Seat Training Docs Archive - 2026-05-12

These notes were moved out of the active training docs because they contain
stale launch commands or stale trainer-lane guidance.

Current postmortem truth:

- The old custom two-seat adapter is historical/prototype evidence, not the
  trusted main learning lane.
- Do not launch CurvyTron two-seat self-play through
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`
  with `--mode two-seat-selfplay` as a learning proof unless it feeds native
  replay/targets or passes target parity.
- `src/curvyzero/infra/modal/lightzero_curvytron_two_seat_train_smoke.py` has
  been deleted.
- Start from
  `docs/working/training/curvytron_train_muzero_reconciliation_2026-05-12.md`
  or `docs/working/training/curvytron_architecture_research_2026-05-12/`
  before using anything in this archive.

Files in this folder are historical context. Do not follow their commands or
lane recommendations as current guidance.
