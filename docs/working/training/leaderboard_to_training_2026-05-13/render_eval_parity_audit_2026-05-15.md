# Render/Eval Parity Audit - 2026-05-15

Live-control note: this file is detailed background for the parity issue. The
active tournament-debug view is `TOURNAMENT_DEBUG.md`; the task board is
`TODO.md`.

2026-05-15 supersession: this audit describes the invalidated v2refresh18p
diagnostic lane. Do not copy it into fresh launch guidance. Current production
policy observations are CPU `cpu_oracle` `browser_lines + simple_symbols`; GPU
`browser_lines + simple_symbols` is lab/profiling-only until trainer-visible
contract parity passes. The source-state training env should reject
`body_circles_fast` as a fresh trainer policy surface.

Historical P0 finding: v2refresh18p training used `body_circles_fast` plus
`simple_symbols`. A dirty worktree at the time partially fixed tournament
policy observations by making `SourceStateGray64Stack4(body_circles_fast)`
resolve to `simple_symbols` and the direct gray64 fast renderer. Tournament
still did not explicitly recover or record `source_state_bonus_render_mode`
from checkpoint metadata, so parity depended on that implicit
`body_circles_fast` default.

Relevant source refs:

- `scripts/build_curvytron_tonight18_manifest.py` set that historical batch to
  `source_max_steps=1_048_576`, `source_state_trail_render_mode=body_circles_fast`,
  and `source_state_bonus_render_mode=simple_symbols`.
- `curvyzero_source_state_visual_survival_lightzero_env.py` resolves
  `body_circles_fast + simple_symbols` to the direct gray64 fast policy
  observation renderer.
- `curvytron_checkpoint_tournament.py` reads checkpoint
  `source_state_trail_render_mode`, but has no equivalent
  `source_state_bonus_render_mode` recovery or summary field.
- `curvytron_current_policy_selfplay_smoke.py::SourceStateGray64Stack4` accepts
  `bonus_render_mode`; when omitted, `body_circles_fast` resolves to
  `simple_symbols` and uses `render_source_state_gray64_fast_player_perspectives`.
- background eval/GIF config/spawn paths currently do not propagate
  `source_state_trail_render_mode` or `source_state_bonus_render_mode` into
  `_make_policy_and_env`.

Action selection:

- Tournament rating defaults to eval/greedy MCTS (`policy.eval_mode.forward`).
- Background GIF defaults to eval-greedy in current source; collect/noisy mode is
  still available by explicit policy mode.

Cadence:

- The current local v2refresh18p manifest records `decision_ms=16.666666666666668`
  and the default one-source-frame cadence. Tournament derives
  `decision_source_frames=1` when using that same `decision_ms` and source
  physics step.

Smallest required fix:

1. Treat bonus render mode as part of the checkpoint observation contract.
2. Thread `source_state_bonus_render_mode` through tournament checkpoint loading,
   game spec normalization, `run_checkpoint_game`, and tournament summaries.
3. Keep/lock `SourceStateGray64Stack4` using the same direct gray64 fast renderer
   for `body_circles_fast + simple_symbols` that the trainer uses, or share the
   trainer renderer helper directly.
4. Thread both trail and bonus render modes through background eval/GIF
   `_make_policy_and_env` calls.
5. Add a golden parity test with an active bonus: trainer stack vs tournament
   stack must match for both seats under `body_circles_fast + simple_symbols`.
