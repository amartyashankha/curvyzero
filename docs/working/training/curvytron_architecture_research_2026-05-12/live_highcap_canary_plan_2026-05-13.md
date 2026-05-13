# Live High-Cap Canary Plan - 2026-05-13

Purpose: clear the remaining live gates for the current survivaldiag dry-run
manifest before any large launch.

Status: strict-stop retry passed. These were tiny canaries, not overnight
training runs. The first attempt was stopped manually and did not clear the
gate; the later strict-stop retry did clear the live mechanics/readout gate.

Important finding: `max_train_iter=4` is not a strict "stop after four
checkpoints" cap in this LightZero path. LightZero checks the cap after a
collect/update block, and one block can produce many learner updates. In the
sim16 fast row, the command did contain `max_train_iter=4`, but the run saved
checkpoints through `iteration_139` before stopping on the env-step budget. For
future tiny live canaries, pass an explicit learner-call cap with
`--stop-after-learner-train-calls`; the launcher now installs that cap in train
mode too.

Also note: after a forced Modal stop, `status_heartbeat.json` and
`checkpoint_eval_poller.json` can still say `running`. Use `modal app list` to
check whether the app is actually alive.

## Why These Runs Exist

Local tests and dry-run manifest validation are green, but they do not prove
that the exact high-cap live path writes the rich artifacts we need. These
canaries test:

- stock `--mode train`;
- `source_max_steps=65536`;
- `survival_plus_bonus_no_outcome`;
- background checkpoint eval and GIF polling;
- live status fields for reward components, bonus counts, terminal causes,
  action distributions, eval health, and GIF health;
- exact first-wave lane variants in the current manifest.

## First Attempt Rows

Common shape:

- compute: `gpu-l4-t4-cpu40`
- train cap: `max_train_iter=4`, `max_env_step=8192`
- checkpoint cadence: every iteration
- collector/batch: `collector_env_num=32`, `n_episode=32`, `batch_size=32`
- stock LightZero in-loop eval: off
- CurvyZero background eval/GIF: on
- action repeat: `min=1`, `max=3`, `extra_probability=0.20`
- background eval seeds: `2`
- env telemetry stride: `1`

| Purpose | Run id | Attempt id | Render | Search | Opponent |
| --- | --- | --- | --- | ---: | --- |
| Blank high-cap gate | `curvytron-survivaldiag-highcap-blank-fast-20260513b` | `highcap-blank-fast-row001-c00-20260513b` | `body_circles_fast` | 8 | `blank_canvas_noop` |
| Blank high-cap gate | `curvytron-survivaldiag-highcap-blank-browser-20260513b` | `highcap-blank-browser-row002-c00-20260513b` | `browser_lines` | 8 | `blank_canvas_noop` |
| Passive dirty exact-lane gate | `curvytron-survivaldiag-passive-immortal-fast-20260513b` | `passive-immortal-fast-row045-c01-20260513b` | `body_circles_fast` | 8 | normal runtime, `opponent_death_mode=immortal` |
| Passive dirty exact-lane gate | `curvytron-survivaldiag-passive-immortal-browser-20260513b` | `passive-immortal-browser-row046-c01-20260513b` | `browser_lines` | 8 | normal runtime, `opponent_death_mode=immortal` |
| Sim16 exact-lane gate | `curvytron-survivaldiag-sim16-blank-fast-20260513b` | `sim16-blank-fast-row049-c01-20260513b` | `body_circles_fast` | 16 | `blank_canvas_noop` |
| Sim16 exact-lane gate | `curvytron-survivaldiag-sim16-blank-browser-20260513b` | `sim16-blank-browser-row050-c01-20260513b` | `browser_lines` | 16 | `blank_canvas_noop` |

## Pass Criteria

For each row:

- `ok=true`;
- `called_train_muzero=true`;
- retry command used `--stop-after-learner-train-calls`, and the summary proves
  the learner-call cap was honored;
- `source_max_steps=65536`;
- checkpoints saved and mirrored;
- `checkpoint_eval_poller.json` completes;
- eval count is greater than zero;
- GIF count is greater than zero;
- live status exposes reward, bonus, terminal, action, eval-health, and GIF
  fields.

Only a strict-stop retry can clear the main high-cap and live-readout gate for
the anchor lane. The manually stopped `20260513b` apps did not clear it. If the
passive dirty pair fails, remove
`b03_passive_immortal_dirty_control` from the launch manifest. If the sim16 pair
fails, remove `b04_compute_sentinel_sim16` from the launch manifest.

## Status Snapshot Command

After a strict-stop retry finishes, collect the live status snapshot with:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_run_status \
  --run-ids curvytron-survivaldiag-highcap-blank-fast-20260513b,curvytron-survivaldiag-highcap-blank-browser-20260513b,curvytron-survivaldiag-passive-immortal-fast-20260513b,curvytron-survivaldiag-passive-immortal-browser-20260513b,curvytron-survivaldiag-sim16-blank-fast-20260513b,curvytron-survivaldiag-sim16-blank-browser-20260513b \
  --attempt-ids highcap-blank-fast-row001-c00-20260513b,highcap-blank-browser-row002-c00-20260513b,passive-immortal-fast-row045-c01-20260513b,passive-immortal-browser-row046-c01-20260513b,sim16-blank-fast-row049-c01-20260513b,sim16-blank-browser-row050-c01-20260513b \
  --output eval-json
```

Save or inspect this snapshot before changing launch gates. A row should not be
called green only because `modal run` returned; it needs summary, action
observability, poller, eval, and GIF artifacts.

## Strict-Stop Retry Rows

Common shape:

- compute: `gpu-l4-t4-cpu40`
- train cap: `--stop-after-learner-train-calls 1`
- checkpoint cadence: every iteration
- collector/batch: `collector_env_num=8`, `n_episode=8`, `batch_size=8`
- stock LightZero in-loop eval: off
- CurvyZero background eval/GIF: on
- action repeat: `min=1`, `max=3`, `extra_probability=0.20`
- background eval seeds: `1`
- env cap: `source_max_steps=65536`

| Purpose | Run id | Attempt id | Render | Search | Opponent | Result |
| --- | --- | --- | --- | ---: | --- | --- |
| Passive dirty exact-lane gate | `curvytron-survivaldiag-prelaunch-passive-fast-20260513g` | `prelaunch-passive-fast-c00-20260513g` | `body_circles_fast` | 2 | normal runtime, `opponent_death_mode=immortal` | passed |
| Passive dirty exact-lane gate | `curvytron-survivaldiag-prelaunch-passive-browser-20260513g` | `prelaunch-passive-browser-c00-20260513g` | `browser_lines` | 2 | normal runtime, `opponent_death_mode=immortal` | passed |
| Sim16 exact-lane gate | `curvytron-survivaldiag-prelaunch-sim16-fast-20260513g` | `prelaunch-sim16-fast-c00-20260513g` | `body_circles_fast` | 16 | `blank_canvas_noop` | passed |
| Sim16 exact-lane gate | `curvytron-survivaldiag-prelaunch-sim16-browser-20260513g` | `prelaunch-sim16-browser-c00-20260513g` | `browser_lines` | 16 | `blank_canvas_noop` | passed |

Status readout for all four showed:

- `train_status=completed`;
- `train_stage=completed`;
- `checkpoint_count=1`;
- `background_poller_status=completed`;
- `background_poller_eval_completed_count=1`;
- `background_poller_gif_completed_count=1`;
- `eval_health=ok`;
- reward components present as `survival` plus `bonus`;
- bonus count/reward fields present;
- action histogram/entropy and terminal-cause fields present.

This clears the exact-lane mechanics gate. It does not prove learning, and the
passive-immortal rows remain dirty controls.
