# LightZero Official Atari Collapse Investigation - 2026-05-09

Scope: official LightZero Atari Pong GPU4096/sim10 collapse only. No training
was run for this investigation. No pytest was run. I only read local docs/code
and upstream LightZero/ALE sources.

## Incident

The 4096/sim10 official visual Atari rung ran stock ALE
`PongNoFrameskip-v4` through LightZero's Atari env and `MuZeroPolicy`.
Infrastructure passed:

- Modal used an NVIDIA L4 and CUDA was available during training.
- Checkpoints mirrored through `iteration_8`.
- Eval loaded checkpoints strictly with no model fallback.
- Eval acted through `MuZeroPolicy.eval_mode.forward`.

Quality failed:

| Checkpoint | 256-step no-fallback eval actions | Return | Nonzero rewards |
| --- | --- | ---: | --- |
| `iteration_0` | `{0:238, 1:18}` | `-6.0` | six `-1`s |
| `iteration_4` | `{5:256}` | `-6.0` | six `-1`s |
| `iteration_8` | `{5:256}` | `-6.0` | six `-1`s |

This is a real collapse: by `iteration_4`, eval deterministically selected
action `5` on every step and the capped eval lost every point available inside
the 256-step window.

## Stock Config Recheck

There are two relevant upstream stock references:

- LightZero `v0.2.0` non-segment config, matching our installed package
  version and wrapper import, uses `zoo.atari.config.atari_muzero_config`,
  `PongNoFrameskip-v4`, `collector_env_num=8`, `n_episode=8`,
  `evaluator_env_num=3`, `num_simulations=50`,
  `update_per_collect=None`, `replay_ratio=0.25`, `batch_size=256`,
  `max_env_step=200000`, `observation_shape=(4,64,64)`,
  `game_segment_length=400`, `learning_rate=0.2`,
  `target_update_freq=100`, `eval_freq=2000`, and replay buffer `1e6`.
- LightZero `v0.2.0` segment config, the Pong quick-start-like lane, uses
  `train_muzero_segment`, `collector_env_num=8`, `num_segments=8`,
  `game_segment_length=20`, `evaluator_env_num=3`, `num_simulations=50`,
  `update_per_collect=None`, `replay_ratio=0.25`, `batch_size=256`,
  `max_env_step=500000`, `observation_shape=(4,96,96)`,
  `train_start_after_envsteps=2000`, and `eval_freq=5000`.

Our 4096/sim10 run was closer than GPU2048, but still far from either stock
recipe:

| Knob | Our 4096/sim10 | v0.2.0 non-segment stock | v0.2.0 segment stock |
| --- | ---: | ---: | ---: |
| env steps | `4096` | `200000` | `500000` |
| train entry | `train_muzero` | `train_muzero` | `train_muzero_segment` |
| observation | `(4,64,64)` | `(4,64,64)` | `(4,96,96)` |
| collector envs | `2` | `8` | `8` |
| collect units | `n_episode=2` | `n_episode=8` | `num_segments=8` |
| evaluator envs/episodes | `1 / 1` | `3 / 3` | `3 / 3` |
| simulations | `10` | `50` | `50` |
| batch size | `32` | `256` | `256` |
| update rule | `update_per_collect=2` | `None`, replay ratio `0.25` | `None`, replay ratio `0.25` |
| game segment length | `64` | `400` | `20` |
| train-start gate | none | none | `2000` env steps |
| eval cadence | `1`, ckpt every iter | `2000` | `5000` |
| episode/eval cap | `512` env cap, `256` outer eval cap | default `108000` | default `108000` |

Sources:

- `https://raw.githubusercontent.com/opendilab/LightZero/v0.2.0/zoo/atari/config/atari_muzero_config.py`
- `https://raw.githubusercontent.com/opendilab/LightZero/v0.2.0/zoo/atari/config/atari_muzero_segment_config.py`
- `https://raw.githubusercontent.com/opendilab/LightZero/v0.2.0/zoo/atari/envs/atari_lightzero_env.py`
- `https://raw.githubusercontent.com/opendilab/LightZero/v0.2.0/zoo/atari/envs/atari_wrappers.py`
- `https://ale.farama.org/environments/pong/`

## Suspect Read

### 1. Too Few Updates And Checkpoints - Most Likely

This is the strongest explanation for the learned collapse. The run reached
only `iteration_8`, even after raising env steps to 4096. Stock non-segment
Atari trains to hundreds of thousands of env steps, with 50 simulations, 8
collectors, batch 256, and auto update count from replay ratio. Our run had
2 collectors, 10 simulations, batch 32, and exactly 2 updates per collect.

The collapse appearing between `iteration_0` and `iteration_4` is compatible
with noisy early policy/value updates, not with a mature Atari learner.
`target_update_freq=100` also means the whole run ends before even one full
official target-network cadence matters.

Verdict: likely primary cause of action collapse.

### 2. Eval Cap / `max_episode_steps` - Explains `-6`, Not Collapse

The manual eval stopped after 256 steps. The observed losses were exactly
at steps `60, 95, 130, 165, 200, 235`, so `-6.0` is mostly a property of the
outer eval cap. The 4096 trainer-side 512-step evals saw about `-13` or `-14`,
which is what we would expect if the same bad policy kept playing longer.

Stock evaluator cfg defaults to `eval_max_episode_steps=108000` unless the
config overrides it. Our eval cap is useful for cheap comparisons, but it is
not stock Atari scoring.

Verdict: high confidence explanation for the flat return value. It does not
explain why action `5` became deterministic.

### 3. Stock LightZero Expects A Different Eval Protocol - Likely Important

Our eval manually reconstructs policy/env, manually maintains a four-frame
stack, and manually loops one evaluator env. That is a good no-fallback smoke,
but stock training uses DI-engine/LightZero evaluator machinery with
`evaluator_env_num=3`, `n_evaluator_episode=3`, evaluator cfg
`episode_life=False`, unclipped rewards, and the library's normal env/policy
interaction path.

The manual eval probably uses the correct policy API, but it is not identical
to stock evaluator protocol. The biggest concrete gap is not temperature; it is
manual frame stacking plus one seed/one episode/short cap outside the official
evaluator.

Verdict: likely affects measurement. Could amplify apparent collapse, but does
not by itself explain that `iteration_4` and `iteration_8` both select action
`5` every step.

### 4. `game_segment_length` / `update_per_collect` Effects - Likely Training Suspect

The 4096 run changed two core learning-balance knobs:

- `game_segment_length=64`, versus non-segment stock `400`; segment stock uses
  `20` but with `train_muzero_segment`, `num_segments=8`, and a
  `train_start_after_envsteps=2000` gate.
- `update_per_collect=2`, versus stock `None` with `replay_ratio=0.25`.

So our config is neither stock non-segment nor stock segment. Shorter segments
plus very few updates can make early roots/targets brittle. The run produced a
policy preference, but not enough replay/update diversity to correct it.

Verdict: likely contributor. Treat as part of the "undertrained and off-recipe"
cluster.

### 5. Action Mapping / NOOP / FIRE - Possible But Lower

ALE Pong's reduced action space is:

```text
0 NOOP, 1 FIRE, 2 RIGHT, 3 LEFT, 4 RIGHTFIRE, 5 LEFTFIRE
```

So action `5` is a valid Pong action, not an out-of-range id. Our eval passes
the integer action directly into the LightZero Atari env, which uses the same
action space size `6` and an all-ones action mask.

The important nuance: LightZero `wrap_lightzero` in `v0.2.0` applies
`NoopResetWrapper`, `MaxAndSkipWrapper`, optional `EpisodicLifeWrapper`,
`TimeLimit`, warp/scale/reward wrappers, but it does not apply
`FireResetWrapper` in the `wrap_lightzero` path. That means stock LightZero
Atari Pong also lets the policy choose FIRE/FIRE-combo actions rather than
forcing a fire reset in this wrapper.

Action mapping is therefore not the leading suspect. Still, a tiny action
meaning probe would be cheap: verify that `env.unwrapped.get_action_meanings()`
in the actual Modal image reports the six ALE Pong meanings above.

Verdict: possible sanity check, but not the main explanation.

### 6. Eval Exploration Temperature - Unlikely

Upstream `MuZeroPolicy._forward_eval` uses MCTS with no root noise, then calls
`select_action(..., temperature=1, deterministic=True)`. The comment says eval
chooses the highest-value/highest-visit action rather than sampling. That means
there is no eval exploration temperature knob expected to keep action support
broad.

For our failure, deterministic eval is doing what stock eval mode intends:
revealing that the root search prefers action `5` everywhere.

Verdict: unlikely bug. Broad action support in eval should come from the
policy/search being state-dependent, not eval sampling.

### 7. Checkpoint Loading / Config Mismatch - Lower, But Keep A Probe

Existing loader/eval evidence is good:

- state dict key `model`;
- strict load into `MuZeroPolicy` model;
- expected visual conv shapes;
- no fallback in the failed evals;
- train/eval both use non-segment `atari_muzero_config`, 64x64 visual model,
  action space size 6, and matching sim/batch/update/game segment knobs for
  the 4096 eval.

Known differences are intentional: eval forces CPU and uses manual single-env
policy/env construction. Those should not change loaded weights or action
space, but they are enough reason to run one stock-evaluator comparison before
spending on more training.

Verdict: unlikely as primary cause, but cheap to rule down further.

## Ranked Hypotheses

1. **Undertrained, noisy early learner on an off-recipe config.** Only
   `iteration_8`, 4096 env steps, 10 sims, batch 32, UPC 2, 2 collectors, and
   target update cadence never really activates. This best explains the learned
   deterministic action `5` preference.
2. **The `-6.0` value is an eval-window artifact.** The policy is bad, but the
   exact flat `-6.0` is caused by the 256-step cap, not a full Pong episode.
3. **Manual eval protocol differs from stock evaluator enough to require a
   parity check.** Manual frame stack and one capped episode are fine for smoke,
   but not a final stock Atari quality read.
4. **Mixed non-segment/segment ideas changed target geometry.** We use
   non-segment trainer with `game_segment_length=64` and explicit UPC 2, while
   stock non-segment is length 400 and stock segment is a different entrypoint.
5. **Action mapping/FIRE semantics are probably correct but should be logged
   once from the actual runtime.** Action `5` is `LEFTFIRE`; stock LightZero's
   `wrap_lightzero` does not force `FireResetWrapper`.
6. **Checkpoint/config mismatch is not supported by current evidence.** Strict
   load and matching surface passed, but stock-evaluator parity would be cleaner.
7. **Eval temperature is not a bug.** Upstream eval mode is deterministic by
   design.

## Next Cheapest Check

Do one eval-only parity probe, not training:

1. Load the existing 4096/sim10 `iteration_4` or `iteration_8` checkpoint.
2. In the same Modal image, create the compiled LightZero evaluator env through
   `get_vec_env_setting(cfg.env)`.
3. Log:
   - `env.unwrapped.get_action_meanings()` or equivalent wrapper-chain action
     meanings;
   - first reset observation shape before any manual stacking;
   - the first 32 eval-mode outputs: action, visit distribution, predicted
     policy logits, searched value;
   - the same checkpoint under the stock DI-engine evaluator if easy, with
     `n_evaluator_episode=1` and no training.
4. Compare manual eval versus stock evaluator on the same seed and cap.

Pass condition: both paths choose action `5` constantly and report the same
action meanings. Then stop blaming eval plumbing and classify the run as
undertrained/off-recipe.

Fail condition: stock evaluator uses different frame stacking, different
actions, or different root visits. Then fix eval parity before any new Atari
training rung.

Do not run another train until this parity probe is done.
