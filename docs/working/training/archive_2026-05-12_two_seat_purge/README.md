# Two-Seat Training Docs Archive - 2026-05-12

These notes were moved out of the active training docs because they predate the
canonical two-seat Coach launcher and contain stale launch commands or stale
trainer-lane guidance.

Current canonical truth:

- Launch CurvyTron two-seat self-play through
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`
  with `--mode two-seat-selfplay`.
- `src/curvyzero/infra/modal/lightzero_curvytron_two_seat_train_smoke.py` has
  been deleted. Historical commands in this archive must be translated through
  the canonical launcher before use.
- Stock LightZero in-training eval is off by default.
- CurvyZero checkpoint eval, inspection, and GIF generation are on by default.
- Checkpoints default every `100` iterations.

Files in this folder are historical context. Do not follow their commands or
lane recommendations without first translating them through the canonical
launcher and current eval defaults above.
