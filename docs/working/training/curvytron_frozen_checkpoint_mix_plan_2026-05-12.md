# CurvyTron Frozen Checkpoint Mix Plan

Date: 2026-05-12

## Current Goal

Keep the canonical CurvyTron two-seat LightZero trainer as the main path.
Add an optional opponent mix where most env rows remain current-policy self-play,
and some rows use a frozen older checkpoint as one opponent.

The key rule: only current-policy seats create learner replay rows. Frozen
checkpoint actions are environment actions only; they must not become learner
labels.

## First Safe Primitive

- Default stays pure current-policy self-play.
- Optional knob chooses the fraction of env rows that are current policy versus
  a frozen checkpoint opponent.
- The frozen opponent is chosen per env row episode/reset, not per step.
- Mixed rows use one frozen seat and one current-policy seat.
- Replay, return targets, and learner samples only include current-policy rows.
- Progress and records must say which rows were self-play and which were
  current-vs-frozen.

Implemented in the canonical path:

- `src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py`
- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`

Current knobs:

- `--two-seat-frozen-opponent-probability`
- `--two-seat-frozen-opponent-checkpoint-ref` or the existing
  `--opponent-checkpoint-ref` fallback
- `--two-seat-frozen-opponent-player-id`
- `--two-seat-frozen-opponent-num-simulations`
- `--two-seat-frozen-opponent-use-cuda`

Safety tests now cover:

- Default self-play behavior still works.
- Mixed rows use the frozen checkpoint only for env actions.
- Frozen-controlled seats do not enter replay.
- Replay sampling rejects non-current-policy rows if they ever leak in.
- Modal payload forwards the frozen-opponent knobs.

## Reward Sanity Check

The current trainer reward is not only terminal win/loss:

- Alive step helper: default `+0.01`.
- Bonus pickup helper: default `+0.05` on the pickup step.
- Terminal outcome: env sparse outcome times `0.01 * episode_step_count`.
  Current env terminal winner rows use winner `+1` and loser `-1`, so this is
  `+0.01*T` for the winner and `-0.01*T` for the loser.

Open question: in pure current-policy self-play, symmetric policies may still
produce weak or brittle terminal signal. The frozen-checkpoint mix is partly a
diversity experiment for that reason, not just a performance tweak.

Research note from the 2026-05-12 critique:

- Self-play against the latest copy is not broken, but identical policies can
  make the terminal signal weak or cyclic in symmetric simultaneous games.
- Dense survival reward helps the agent stop dying immediately, but if it gets
  too large it can change the objective into "delay the end" instead of "win."
- Frozen checkpoint opponents are a reasonable way to break mirror-policy loops
  and test robustness against older strategies.

## Next Layer

Once the primitive is tested, add checkpoint-pool variants:

- Sweep coarsely across the important axes, not finely along one knob.
- Opponent mix ratio: include a no-frozen baseline, a small mix, a medium mix,
  and one aggressive mix.
- Checkpoint age: recent, halfway-back, and older/distinct checkpoints.
- Base config: mostly use the best clean base config, with only a small number
  of variants from learning-rate/reward/noise if current artifacts justify them.
- Use deterministic bucket counts per batch rather than noisy per-row sampling.
- Name runs clearly so the website and run list are readable.

The exact values are not settled. The batch should explore the space enough to
learn something tomorrow without making the artifact review impossible.

Current run-analysis note:

- Best clean bases from the 2026-05-12 matrix look like the `lr_1e-4` run and
  the default B64/sim8 seed family.
- The action-repeat/no-op run is interesting as an exploratory opponent, but it
  should not be the main base because that variant has extra accounting caveats.

## Recommended Batch After Canary

Launch one waited Modal canary first. Continue only if the canary shows strict
checkpoint load, nonzero mixed-row counts, no frozen-controlled replay rows, and
normal current-policy action/reward telemetry.

### Canary Result

Latest waited canary: `mixpast-canary-p30-lr1e4opp-i100-20260512c`.

Status: passed as a plumbing canary.

What it proved:

- The canonical Modal launcher can accept a frozen checkpoint ref and resolve it
  inside the remote container where `/runs` is mounted.
- The frozen opponent checkpoint loaded from explicit row18
  `iteration_100.pth.tar`.
- Training reached `ok=true`, saved `iteration_0`, `iteration_5`, and
  `iteration_6` checkpoints, and wrote a completed summary.
- `lightzero_policy_model_device` was `cuda:0`.
- Learner updates ran and changed model parameters.
- Opponent mix telemetry was present with both
  `current_policy_selfplay` and `current_policy_vs_frozen_checkpoint` rows.
- Replay stayed learner-side only: the canary's replay row count tracked current
  policy rows, and frozen-controlled seats were only env actions.
- Fresh policy actions were not collapsed in this short canary; both players had
  nontrivial action entropy in the training collection path.

Two launch bugs were found and fixed during the canary loop:

- Do not check Modal Volume checkpoint refs in the local entrypoint. Resolve and
  verify frozen checkpoint refs inside `_run_two_seat_selfplay_payload`, where
  `/runs` is mounted.
- Summary array stripping must handle string metadata arrays such as rollout
  labels. Numeric array summaries still include min/max/mean; string arrays now
  keep shape/dtype plus a short sample.

Verification checks passed after the fixes:

- `uv run ruff check ...`
- `uv run pytest tests/test_curvytron_two_seat_render_mode.py tests/test_curvytron_live_checkpoint_eval_plumbing.py -q`
  -> `43 passed, 1 skipped`.

Remaining caveat: this is a correctness/plumbing canary, not learning evidence.
The next decision should be based on a small number of longer mixed runs plus
the existing overnight40 signal, not on this six-iteration run.

Canary:

| run id | base | frozen ratio | opponent checkpoint | purpose |
|---|---|---:|---|---|
| `mixpast-canary-r18-mid25-sim8-20260512` | row18 `lr_1e-4` | `0.25` | halfway-back checkpoint from row18 | plumbing plus first signal |

If the canary passes, use this 14-run batch:

| run id | base config | frozen ratio | opponent checkpoint age/source | variant |
|---|---|---:|---|---|
| `mixpast-r18-recent10-a-20260512` | row18 `lr_1e-4` | `0.10` | recent row18 checkpoint | clean |
| `mixpast-r18-recent25-a-20260512` | row18 `lr_1e-4` | `0.25` | recent row18 checkpoint | clean |
| `mixpast-r18-recent50-a-20260512` | row18 `lr_1e-4` | `0.50` | recent row18 checkpoint | clean |
| `mixpast-r18-mid10-a-20260512` | row18 `lr_1e-4` | `0.10` | halfway-back row18 checkpoint | clean |
| `mixpast-r18-mid25-a-20260512` | row18 `lr_1e-4` | `0.25` | halfway-back row18 checkpoint | clean |
| `mixpast-r18-mid50-a-20260512` | row18 `lr_1e-4` | `0.50` | halfway-back row18 checkpoint | clean |
| `mixpast-r18-old25-a-20260512` | row18 `lr_1e-4` | `0.25` | older/distinct row18 checkpoint | clean |
| `mixpast-b64s8-r04-recent25-a-20260512` | default B64/sim8 row04 | `0.25` | recent row04 checkpoint | clean |
| `mixpast-b64s8-r04-mid25-a-20260512` | default B64/sim8 row04 | `0.25` | halfway-back row04 checkpoint | clean |
| `mixpast-b64s8-r04-old25-a-20260512` | default B64/sim8 row04 | `0.25` | older row04 checkpoint | clean |
| `mixpast-b64s8-r05-recent25-a-20260512` | default B64/sim8 row05 | `0.25` | recent row05 checkpoint | clean seed replicate |
| `mixpast-b64s8-r05-mid50-a-20260512` | default B64/sim8 row05 | `0.50` | halfway-back row05 checkpoint | clean seed replicate |
| `mixpast-r18-mid25-noobsnoise-a-20260512` | row18 `lr_1e-4` | `0.25` | halfway-back row18 checkpoint | no observation noise |
| `mixpast-r18-vs-r30-mid25-a-20260512` | row18 `lr_1e-4` | `0.25` | row30 action-repeat checkpoint | exploratory opponent only |

Readout:

- Treat `0.25` as the primary ratio unless `0.10` or `0.50` clearly wins.
- Prefer row18 if the lr run wins both same-source recent and halfway-back
  comparisons; otherwise prefer the default B64/sim8 family if row04/row05 agree.
- Use row30 only as a robustness probe. Do not promote it to main-base evidence
  unless clean rows already look healthy.

## Open Checks

- Literature/research check: opponent pools, fictitious self-play, league
  training, and AlphaZero-style checkpoint opponents.
- Run-analysis check: pick 1-2 strong base configs from the current matrix before
  launching frozen-opponent variants.
- Code check: extend from one frozen checkpoint to a checkpoint pool once the
  first primitive is validated.
