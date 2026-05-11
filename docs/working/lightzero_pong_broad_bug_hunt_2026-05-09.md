# LightZero Pong Broad Bug Hunt - 2026-05-09

Scope: bounded critic pass over the dummy Pong/LightZero lane for simple
reasons training shows no survival gain and collapses actions. No pytest. I
read the project env/config/eval code, recent working notes, and the local
LightZero source under `/tmp/lightzero-src`.

## Short Read

I did not find a simple action-id inversion or illegal-action mask bug.
`ACTION_LABELS = ("up", "stay", "down")`, `PongEnv._move_paddle()` maps
`0/1/2` to `-1/0/+1`, LightZero uses `Discrete(3)`, and the MCTS adapter uses
an all-ones action mask.

The highest-likelihood bugs are protocol/config scale bugs:

1. `max_env_step` is overloaded as both total LightZero training budget and
   Pong episode horizon/step-feature denominator.
2. The learner likely gets too little useful data and too few updates for a
   sparse terminal-reward Pong task.
3. Survival/loss-delay is telemetry only, not a training reward.
4. Trainer-side sidecar rows are not final-checkpoint quality; independent
   eval is probably mostly right, but official load parity still needs one
   small falsifier.

The post deep-seed-fix run reduces seed dominance suspicion: it had 131 unique
seeds across 148 trainer-side rows, but the independent strict MCTS checkpoint
still chose zero `down` actions.

## Ranked Suspects

### 1. `max_env_step` mixes training budget with episode horizon

Likelihood: high.

In the wrappers, `max_env_step` is passed to `train_muzero(...,
max_env_step=...)`, but also becomes `DummyPongLightZeroEnv.max_steps`,
`PongConfig(max_steps=...)`, and the denominator for the tabular feature
`step / max_steps`.

Official LightZero examples treat `max_env_step` as total collected env
interactions. The env episode horizon is separate: CartPole uses the stock
CartPole horizon; Atari Pong config has a separate env shape/wrapper and
`game_segment_length=400`.

Why it can hurt: scaling "training budget" silently changes the task, the
normalized observation, truncation behavior, and independent eval horizon. It
also makes a `1024` env-step run not a long run; it is only about 1024 total
interactions.

Simplest test: split knobs into `train_env_step_budget` and
`pong_episode_max_steps`. Run the same seed with `pong_episode_max_steps=120`
fixed and budgets `1024/4096`, then assert train config, checkpoint metadata,
and scorecard all report the same named horizon profile.

### 2. Too little data and too few learner updates

Likelihood: high.

Current Pong attempts are tiny compared with official defaults:

- Official CartPole config: `collector_env_num=8`, `n_episode=8`,
  `num_simulations=25`, `batch_size=256`, `update_per_collect=100`.
- Official Atari/Pong-like config: `num_simulations=50`, `batch_size=256`,
  `game_segment_length=400`, large env-step budgets.
- Current trusted Pong run: `collector_env_num=1`, `n_episode=1`,
  `num_simulations=8`, `batch_size=32`, `update_per_collect=1`,
  `max_env_step=1024`, `max_train_iter=16`.

The local CartPole progression smoke that worked used `update_per_collect=4`.
Pong kept `update_per_collect=1` despite sparse rewards.

Simplest test: rerun the post-seed-fix shape with only
`update_per_collect=4` changed, plus summary fields for replay transitions,
actual learner updates, skipped updates due to insufficient replay, and
checkpoint tensor deltas. Pass condition should include nonzero `down` on
states where `ball_dy_from_ego_center > 0`, not just win count.

### 3. Sparse score reward is too weak for the current tiny MuZero setup

Likelihood: high.

`DummyPongLightZeroEnv.step()` returns raw ego score reward: `+1`, `-1`, or
`0`. `shaped_loss_delay_return` and `survival_fraction` are written only to
telemetry. So "no survival gain" may simply mean survival is not being trained,
except indirectly through eventual terminal score.

CartPole gets dense alive-step reward. Dummy Pong gives mostly zeros until a
score/miss, and the opponent action is hidden inside the transition.

Simplest test: do a labeled training-only reward ablation. Keep eval on raw
score reward, but compare raw sparse reward against a temporary loss-delay or
contact/curriculum reward. If `down` appears only with denser reward, the
control signal, not the action plumbing, is the blocker.

### 4. MCTS eval is probably not "wrong", but official load parity is unproven

Likelihood: medium.

The independent adapter reconstructs `MuZeroPolicy`, strict-loads
`checkpoint["model"]` into `policy._model`, and calls
`policy.eval_mode.forward(obs, action_mask, to_play=[-1], ready_env_id=[0])`.
LightZero source shows `_init_eval` uses `self._eval_model = self._model`, and
eval mode performs no-root-noise MCTS followed by deterministic visit-count
selection. That matches our adapter closely.

Remaining caveat: the official train entry loads pretrained checkpoints through
`policy.learn_mode.load_state_dict(torch.load(...))`, not by manually loading
only the model key.

Simplest test: on the same checkpoint and same saved observation batch, compare:

- current adapter strict load of `checkpoint["model"]`;
- official `policy.learn_mode.load_state_dict(full_checkpoint)`;
- diagnostic `checkpoint["target_model"]` control.

Then diff logits, visit distributions, selected actions, and `_eval_model`
parameter hashes.

### 5. Trainer-side telemetry can still mislead

Likelihood: medium-high.

Earlier trainer sidecars were dominated by repeated seeds and mixed collection
and evaluation activity. The deep seed fix made seed diversity much better, but
the sidecar is still not a clean held-out final-checkpoint scorecard.

Simplest test: write separate sidecar streams or row fields for `collector`,
`evaluator`, random collect, checkpoint/eval iteration, and action source.
Summarize final evaluator rows separately from all rows. Keep independent MCTS
scorecard as the quality gate.

### 6. Opponent distribution is too narrow

Likelihood: medium.

Training defaults to `random_uniform` only. Independent scorecards test
`random_uniform`, `lagged_track_ball_1`, and `track_ball`. Losing to
`track_ball` alone would not prove a bug, but the checkpoint also loses to
`random_uniform` and still chooses zero `down`.

Simplest test: train a deterministic-opponent control against
`lagged_track_ball_1`, then first score player0-only against the same opponent.
If this learns vertical action diversity, random-opponent stochasticity plus
sparse reward is the main issue.

### 7. Observation/frame contract is easy to overstate

Likelihood: medium-low for current `tabular_ego`, higher for `raster_flat`.

Current trusted runs use `tabular_ego`, which includes ball velocity and
ego-frame ball direction. That is probably observable enough with
`frame_stack_num=1`. But the env name `dummy_pong_lag1` is metadata in the
LightZero wrapper; the wrapper does not branch on `curvyzero_env` to change
observation or dynamics.

For `raster_flat`, the observation is one current grid flattened to 135 values.
Unlike official Atari Pong, it has no frame stack by default, so velocity would
be unobservable.

Simplest test: add a config-surface assertion that `curvyzero_env` names a real
profile, not only a label. For any `raster_flat` run, either set
`frame_stack_num > 1` with matching `observation_shape`, or mark it
non-velocity-observable and keep it out of learning claims.

### 8. Search/exploration defaults are small for Pong

Likelihood: medium-low.

Official examples use `num_simulations=25` for CartPole and `50` for Atari
Pong-like configs. The Pong lane defaults to `2` in smokes and used `8` in
recent attempts. Collection uses root noise, eval does not; collection samples
from visit counts with fixed temperature `0.25` unless epsilon mode is enabled.

This likely does not explain zero `down` by itself, because the learned prior
also suppresses it, but low search budget can amplify bad early priors.

Simplest test: fixed checkpoint MCTS eval sweep with `num_simulations=2,8,16,25`
and first-N debug rows containing logits, visit counts, selected action, and
ball vertical offset. If higher simulations still never choose `down`, search
budget is not the main cause.

### 9. `to_play=-1` and reward sign look low suspicion

Likelihood: low.

The wrapper exposes a single ego-controlled environment with the scripted
opponent folded into `step()`. LightZero examples use `to_play=-1` for
non-board-game environments. The reward returned is the ego reward, and board
game sign-flip logic should not apply because `env_type='not_board_games'`.

Simplest test: one diagnostic-only eval with `to_play=[0]` and one with
`to_play=[-1]` on the same observations/checkpoint. Treat any difference as a
sign to inspect LightZero MCTS value handling, not as a proposed main config.

## Answering The Prompt Directly

Wrong LightZero defaults? Probably yes for budget/update/reward/search scale,
and definitely dangerous for `max_env_step` semantics. Action space and model
input shape are compatible for `tabular_ego`; collection/replay/exploration are
under-audited; segment length is not an obvious bug but is inherited from
CartPole (`50`) rather than designed for Pong.

Could the policy be trained but evaluated wrong? Possible but no longer the
leading read. The independent MCTS adapter matches LightZero eval source well
enough that official load parity is the right remaining falsifier.

Could action labels or frames be wrong? Action labels look correct. Current
`tabular_ego` has velocity, so frame stacking is less suspicious. `raster_flat`
would be suspicious without frame stacking. The `dummy_pong_lag1` env label is
not enforced as a real wrapper profile.

Too narrow seed/opponent distribution? Seed dominance was a real earlier bug
and looks much improved in the latest run. Opponent distribution is still
narrow: train is random-only, while eval includes stronger scripted opponents.
That does not explain failure vs random, but it is a useful control axis.

Too little data or too few updates? Yes, very likely. The current runs are
small in both total interactions and learner work, and the reward is sparse.

## Files Read

- `src/curvyzero/training/dummy_pong.py`
- `src/curvyzero/training/lightzero_dummy_pong_env.py`
- `src/curvyzero/training/lightzero_dummy_pong_features.py`
- `src/curvyzero/training/lightzero_dummy_pong_policy.py`
- `src/curvyzero/training/dummy_pong_eval.py`
- `src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py`
- `src/curvyzero/infra/modal/lightzero_dummy_pong_tiny_train_smoke.py`
- `src/curvyzero/infra/modal/lightzero_dummy_pong_train_attempt.py`
- `src/curvyzero/infra/modal/lightzero_dummy_pong_mcts_scoreboard_attempt.py`
- `scripts/run_dummy_pong_lightzero_mcts_checkpoint_scoreboard.py`
- `/tmp/lightzero-src/zoo/classic_control/cartpole/config/cartpole_muzero_config.py`
- `/tmp/lightzero-src/zoo/atari/config/atari_muzero_config.py`
- `/tmp/lightzero-src/lzero/agent/config/muzero/gym_pongnoframeskip_v4.py`
- `/tmp/lightzero-src/lzero/policy/muzero.py`
- `/tmp/lightzero-src/lzero/worker/muzero_collector.py`
- `/tmp/lightzero-src/lzero/worker/muzero_evaluator.py`
- `/tmp/lightzero-src/lzero/entry/train_muzero.py`
- `/tmp/lightzero-src/lzero/mcts/buffer/game_segment.py`
- Recent working/experiment notes for LightZero Pong action collapse, config
  bugs, checkpoint diagnostics, seed fixes, CartPole comparison, and scale
  critique.

## Verification

No pytest. No training or scorecard jobs were run for this note. I did not run
`py_compile` because the change is documentation only.

## Files Changed

- `docs/working/lightzero_pong_broad_bug_hunt_2026-05-09.md`

---

# Addendum: Broad Bug-Hunt Critic Pass - 2026-05-09

Scope: current-pass review of the LightZero dummy Pong setup from the simple
failure angles the user named: reward sign, shaped telemetry, action mapping,
paddle-angle bounce, horizon/budget split, `to_play`, feature mode, scorecard
reconstruction, eval-vs-collect behavior, Modal Volume checkpoint refs, sparse
settings, and the reminder that Connect4/TicTacToe controls are not Pong or
CurvyTron controls. No pytest.

## Short Read

I do not see a simple reward-sign, action-map, or paddle-bounce inversion.
Local canaries showed:

```text
action 0 = up    -> paddle y delta -1
action 1 = stay  -> paddle y delta 0
action 2 = down  -> paddle y delta +1

player_0 scores right wall: rewards {'player_0': 1.0, 'player_1': -1.0}

paddle hits at top/center/bottom offsets produce outgoing vy -1/0/+1
```

The bigger risks are still protocol and signal risks. The setup can run, but
the checkpoint behavior is weak and unstable. The newest MCTS diagnostic
changes the read: at 8 simulations, exact root ties explain the no-`down`
scorecard symptom; at 16/32/64 simulations, the exact max ties disappear but
the first-N actions collapse to `down`. So low eval sim count is not the fix.
The deeper problem is weak learned policy/value/search signal, objective/config
fit, or train/eval reconstruction mismatch.

One important correction: Connect4/TicTacToe are useful only for sparse
terminal reward patterns. Their controls are not a close match for Pong or
CurvyTron. Board games have legal-move masks and turn/outcome structure.
Pong/CurvyTron need step-based control, velocity or visual frame history, and
eventually visual observations. Do not copy board-game `action_type` or
training conclusions directly into Pong.

## Top 5 Bug Risks

### 1. The learned policy/value signal is weak, not just under-searched

Risk: high.

Evidence for:

- The sparse probe simulation sweep over the same first 24 observations showed:
  `8` sims -> actions `[5, 19, 0]` with max-root ties on `24/24` rows;
  `16/32/64` sims -> actions `[0, 0, 24]`, with max-root ties gone.
- Higher eval search removed the exact tie artifact, but it did not produce a
  state-sensitive controller. It simply changed the collapsed action to `down`.
- The root visit distributions stayed broad and close to uniform. That points
  at weak priors/value estimates, not a clean search-budget bug.
- Collapse direction changes by run: all `up`, all `stay`, all `down`, or
  no-`down`. That is more consistent with weak learned signal than with one
  illegal action or one bad action id.

Evidence against:

- The 8-sim tie artifact is still real and can distort compact scorecards.
  Action histograms at low sim counts must be interpreted with root-tie
  telemetry.
- Baselines can use all three actions in the same scorecard machinery.

Next smallest check:

- For every candidate checkpoint, score first-N debug rows at the planned eval
  sim count and log logits, root visits, top-2 margins, selected action,
  value estimate, ball vertical offset, and terminal outcome. Gate training
  claims on better held-out score/survival plus non-degenerate state-dependent
  actions, not on "more sims removed exact ties."

### 2. Sparse reward/target settings may not teach Pong control

Risk: high.

Evidence for:

- Env reward is honest sparse score reward: `+1/-1/0`. Shaped
  loss-delay/survival is telemetry only.
- The sparse-settings probe used the intended fixed horizon and terminal knobs:
  `pong_episode_max_steps=120`, `td_steps=120`, `discount_factor=1.0`,
  small integer reward/value supports, and `update_per_collect=8`.
- That probe ran cleanly, but independent MCTS eval still chose aggregate
  learned actions `[103, 355, 0]` and did not produce a useful controller.
- Connect4/TicTacToe sparse examples prove LightZero can run delayed terminal
  rewards, not that a tiny Pong visual/control problem is solved.
- Pong and CurvyTron need step-by-step control. A final score target alone may
  be too delayed for the current tiny MLP/search/replay setup, even if the
  target is mathematically honest.

Evidence against:

- `tabular_ego` includes ball velocity and relative position, so the current
  main lane is not blind in the way a single-frame raster Pong policy would be.
- The sparse probe was small; it refutes "settings crash" more than it refutes
  all sparse Pong training.

Next smallest check:

- Run one explicit objective/config A/B at the same fixed horizon. Keep eval on
  raw score. Compare current sparse terminal reward against a clearly labeled
  auxiliary/curriculum target such as loss-delay/survival or contact/angle
  curriculum. Log reward/value target distributions, replay size, update
  counts, and held-out score/survival.

### 3. Feature mode and visual-observation assumptions can drift

Risk: high.

Evidence for:

- The main LightZero lane uses `tabular_ego`, a hand-built 10-float state. This
  is a useful probe, but not the real Pong/CurvyTron bridge.
- The real bridge needs visual observations with velocity/history or dynamics
  context. A single `raster_flat` frame is not equivalent to `tabular_ego`.
- Scorecard wrappers default to `feature_mode="tabular_ego"`. If a raster
  checkpoint is scored without explicitly passing the training feature mode,
  reconstruction can be wrong or fail in a way that looks like policy quality.
- Older probes and docs were tabular-only. Some checkpoint probes still
  explicitly support only `tabular_ego`.

Evidence against:

- Current code is better than before: the feature encoder has
  `encode_lightzero_observation(...)`, config patching uses
  `lightzero_observation_shape(...)`, and the policy-head/MCTS wrappers pass
  `feature_mode` through the adapter.
- The current bad checkpoints were trained and evaluated as `tabular_ego`, so
  visual mismatch is not the only explanation for the current failure.

Next smallest check:

- Make scorecards infer or require the training `feature_mode` from a train
  `summary_ref`. Assert checkpoint reconstruction feature mode and
  observation shape match the train summary before loading the model. Add an
  intentional wrong-feature canary that fails loudly instead of silently
  scoring a checkpoint under the wrong observation contract.

### 4. Horizon, budget, and scorecard reconstruction can still drift

Risk: medium-high.

Evidence for:

- The code now separates the LightZero budget role from Pong horizon in
  summaries: `max_env_step_role = lightzero_training_budget` and
  `pong_episode_max_steps` / `effective_pong_episode_max_steps`.
- But the legacy default still exists: if `pong_episode_max_steps` is omitted,
  the Pong horizon falls back to `max_env_step`.
- The MCTS scorecard wrapper still names its eval horizon `--max-env-step` and
  defaults it to `64`, while many train runs used other budgets/horizons.
- Prior scorecards were already found with mismatched training/eval horizon
  reconstruction.

Evidence against:

- Recent corrected scorecards and the sparse probe did pass explicit horizons
  (`120` or matching run values), and the failure persisted. So this is likely
  a recurring footgun, not the whole current root cause.

Next smallest check:

- Make every LightZero scorecard require either a train `summary_ref` or an
  explicit `--pong-episode-max-steps`, then assert both:
  `scorecard.config.max_steps == train.effective_pong_episode_max_steps` and
  `scorecard.lightzero_eval_config.feature_mode == train.feature_mode`.
  Rename or duplicate the scorecard arg to `--eval-episode-max-steps` so it
  stops looking like the training budget.

### 5. Checkpoint identity, state-key, and Modal refs can mislead reports

Risk: medium.

Evidence for:

- Modal wrappers accept `ref:`/`volume:` checkpoint inputs and resolve them to
  mounted paths. They record source refs and file summaries, which is good.
- Training mirrors LightZero checkpoints into
  `training/lightzero-dummy-pong/<run>/checkpoints/lightzero/...` by filename.
  It is easy in docs or commands to score `ckpt_best` or `iteration_N` from the
  wrong run/attempt if the ref is copied manually.
- The independent adapter strict-loads the checkpoint's model state into a
  reconstructed `MuZeroPolicy`, which is close to LightZero eval behavior but
  still not the exact trainer load path.
- Config surfaces record `opponent_checkpoint_state_key`, but the checkpoint
  loader currently finds a state dict through `_find_state_dict(...)` rather
  than taking an explicit state-key argument. That means a command can appear
  to request `model` or `target_model` while the actual loader chooses by its
  own heuristic.

Evidence against:

- Strict full-model MCTS loading has passed on current checkpoints.
- Scorecard result payloads include checkpoint input refs and SHA summaries.
- `_find_state_dict(...)` searches `model` before `target_model`, so it likely
  picks the intended key for current LightZero checkpoints.
- The collapse behavior survived multiple checkpoints/runs, so a single bad
  ref is unlikely to explain all failures.

Next smallest check:

- In each scorecard summary, include the checkpoint SHA, checkpoint ref, train
  `summary_ref`, train command surface, and train run/attempt ids in one block.
  Then run one loader-parity diagnostic on the same observation batch:
  manual `checkpoint["model"]` load versus official
  `policy.learn_mode.load_state_dict(full_checkpoint)`. Add an explicit
  `state_key` parameter to the LightZero checkpoint loader or remove the CLI
  state-key knob until it is honored.

## Lower-Suspicion Contract Items

- Reward sign: low suspicion. `PongEnv.step()` returns `+1` for player_0 when
  the ball exits right and `-1` when it exits left. The LightZero wrapper
  returns the ego agent reward.
- Shaped telemetry: low suspicion. `shaped_loss_delay_return` is computed in
  terminal info and summaries; it is not returned as env reward.
- Action mapping: low suspicion. `ACTION_LABELS = ("up", "stay", "down")` and
  `_move_paddle()` uses `action - 1`.
- Paddle-angle bounce: low suspicion as an inversion bug. Top/center/bottom
  impacts set outgoing `vy` to `-1/0/+1`. It may still shape the difficulty of
  scoring and should stay in target-ladder probes.
- `to_play`: low suspicion for this wrapper. The env exposes one ego action
  with the opponent folded into `step()`, so `to_play=-1`,
  `env_type='not_board_games'`, fixed action space, and all-ones action mask
  are coherent. Do not switch to board-game mode unless the env contract
  really becomes a board-game/alternating-player contract.

## Verification This Pass

No pytest. I ran two tiny local canaries:

- `uv run python -c ...` for action mapping, reward sign, and paddle-hit
  offset bounce.
- A local config-surface import attempt failed because this workspace does not
  currently have `easydict` installed. I did not install dependencies for this
  documentation pass; Modal LightZero jobs install LightZero and have passed
  the config/training paths before.
