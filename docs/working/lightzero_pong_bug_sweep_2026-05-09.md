# LightZero Pong Bug Sweep - 2026-05-09

Scope: side-lane inspection only. No pytest. Wrote only this note.

Start facts:

- North Star: LightZero-first MuZero, Modal whole-job, Pong telemetry beyond wins, docs as memory.
- Env reward stays true: `+1/-1/0`. Survival/loss-delay is telemetry/tie-breaker, not env reward.
- Completed 512/8 run: `lz-dpong-20260509T144635Z-eb5a0ed35de0`, attempt `attempt-20260509T144635Z-ece79bad80d0`.
- Direct policy-head strict loader works, but direct greedy eval is still constant-up.
- MCTS loader smoke passes with `res_connection_in_dynamics=True`; the full
  MCTS scorecard has now run and mostly chooses up.

## Likely Bugs Or Mismatches

### 1. Direct policy-head argmax turns weak/tied logits into always `up`

`LightZeroPolicyHeadGreedyPolicy.action()` calls `model.initial_inference(...)`,
reads `policy_logits`, and uses `torch.argmax`. If logits are tied or nearly
tied, action `0` wins, and action `0` is `up`.

Evidence already recorded elsewhere:

- 512/8 policy-head rows are all `[N, 0, 0]`.
- Strict-config rerun still all-up.
- MCTS one-call smoke had nonzero but tiny logits and still chose action `0`.

Why this matters: direct policy-head eval is a good loader canary, but it is not
LightZero's eval action selection. It can explain the direct greedy control
path, but not the full issue: the MCTS scorecard also mostly chooses up, with
combined LightZero actions `[2060,7,0]` and no down actions.

Files:

- `src/curvyzero/training/lightzero_dummy_pong_policy.py:74`
- `src/curvyzero/training/lightzero_dummy_pong_policy.py:81`
- `src/curvyzero/training/lightzero_dummy_pong_policy.py:85`

### 2. 512/8 checkpoints can be scored with a 64-step reconstruction by default

The train wrapper's scaled default is `max_env_step=512`, but both policy-head
and MCTS scoreboard Modal wrappers default to `max_env_step=64`. That value is
used to reconstruct the LightZero config and to scale the `step` feature in the
policy adapter.

This may not be the only cause because the strict-config rerun still stayed
constant-up. But it is a real investigation item: a 512-step checkpoint should
not be silently scored as a 64-step checkpoint when the step feature is part of
the observation.

Files:

- Train wrapper default: `src/curvyzero/infra/modal/lightzero_dummy_pong_train_attempt.py:33`
- Policy-head scoreboard default: `src/curvyzero/infra/modal/lightzero_dummy_pong_policy_head_scoreboard_attempt.py:200`
- MCTS scoreboard default: `src/curvyzero/infra/modal/lightzero_dummy_pong_mcts_scoreboard_attempt.py:198`
- Config patch passes `max_steps`: `src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py:229`
- Adapter encoder uses `PongConfig(max_steps=max_env_step)`: `src/curvyzero/training/lightzero_dummy_pong_policy.py:203`

### 3. Independent eval environment still uses `PongConfig()` default `max_steps=120`

`run_dummy_pong_eval()` creates `config = PongConfig()` and never threads
`lightzero_max_env_step` into the actual eval environment. So there are two
different horizon knobs:

- policy/model reconstruction and step-feature scaling: `lightzero_max_env_step`
- scorecard game horizon: default `PongConfig.max_steps=120`

For a 512-step training run, this means the scorecard is not exactly the same
task horizon even when the adapter is reconstructed correctly. Most games end
early, so this may only affect truncation/survival rows, but it should be made
explicit before using survival/truncation comparisons for checkpoint selection.

Files:

- `src/curvyzero/training/dummy_pong_eval.py:253`
- `src/curvyzero/training/dummy_pong_eval.py:488`
- `src/curvyzero/training/dummy_pong_eval.py:516`

### 4. Trainer-side wins are not independent checkpoint learning proof

The 512/8 trainer summary reports strong env/evaluator wins, but that is
trainer-side telemetry from LightZero's loop. It does not prove that the
mirrored checkpoint's greedy policy head is useful.

Concrete mismatch:

- Trainer telemetry: 42 episodes, 37 wins, 5 losses.
- Independent direct scorecard: both `ckpt_best` and `iteration_8` are all-up.

Most likely read: the job proved plumbing and produced checkpoints. The
faithful MCTS/eval-mode scorecard now runs, but it also mostly chooses up, so
visible learning still needs checkpoint progression checks and better training
signal.

Files:

- `src/curvyzero/infra/modal/lightzero_dummy_pong_tiny_train_smoke.py:369`
- `src/curvyzero/infra/modal/lightzero_dummy_pong_tiny_train_smoke.py:401`
- `src/curvyzero/infra/modal/lightzero_dummy_pong_tiny_train_smoke.py:417`

### 5. `DummyPongLightZeroEnv.random_action()` is deterministic within an episode

`random_action()` creates a new RNG from `self._episode_seed` on every call.
Repeated calls in the same episode return the same action. If LightZero uses
this helper for any collection/exploration path, action diversity can be much
worse than intended.

This probably does not explain the current independent policy-head result by
itself because the 512/8 trainer-side action counts include all three actions.
Still, if LightZero uses this helper for exploration, it is a real small bug to
fix.

File:

- `src/curvyzero/training/lightzero_dummy_pong_env.py:216`

### 6. Training wrapper timestep patch is local, not part of the env contract

The scaled train wrapper monkey-patches `_lightzero_observation()` to add
`timestep`. That fixed a real LightZero evaluator crash. The base env still does
not emit `timestep`.

This is probably not causing direct policy-head constant-up because that path
does not use the LightZero env wrapper. Keep the read narrow: `timestep`
compatibility is wrapper-local, not part of the base env observation contract,
and that mismatch can reappear in entrypoints that skip the scaled wrapper
patch.

## Current Investigation Read

Treat these as investigation items, not proven root causes:

- Direct policy-head argmax can collapse weak/tied logits to action `0`, but
  MCTS also mostly chooses up, so this explains only the control path.
- Horizon/config mismatch risk remains: training used `max_env_step=512`,
  scorecard reconstruction can default to `64`, and independent eval currently
  uses `PongConfig()` default `max_steps=120`.
- `DummyPongLightZeroEnv.random_action()` reseeds every call and can repeat the
  same helper action inside an episode.
- `timestep` compatibility is wrapper-local, not in the base env observation.

Next blockers: fix the tiny env/helper footguns, rerun the MCTS scorecard with
explicit matching `max_env_step`, then decide whether longer training is worth
running.

Files:

- Base env observation: `src/curvyzero/training/lightzero_dummy_pong_env.py:243`
- Local patch: `src/curvyzero/infra/modal/lightzero_dummy_pong_train_attempt.py:41`

## Ruled Out Or Lower Suspicion

- Reward shaping leak: wrapper returns raw `pong_step.rewards[self.ego_agent]`.
  Shaped loss-delay is only stored in `info` and summaries.
- Action mapping: `ACTION_LABELS = ("up", "stay", "down")`; env motion uses
  `delta = action - 1`, so `0=up`, `1=stay`, `2=down` is consistent.
- Tabular encoder duplication: training env and policy adapters use the same
  `encode_tabular_ego_observation()` for `tabular_ego`.
- Action mask: direct policy-head ignores masks by design; MCTS adapter uses
  all-ones mask with shape `[1,3]`. All dummy Pong actions are legal.
- Loader strictness: current 512/8 loader path handles the split residual
  dynamics variant with `res_connection_in_dynamics=True`. This is no longer
  the blocker for `iteration_8`.
- Scoreboard action telemetry: compact rows now include
  `action_histogram_by_policy`, so all-up is visible without raw JSONL digging.
- Opponent policy in checkpoint reconstruction: `opponent_policy` is part of
  the LightZero config surface but direct independent eval actually plays the
  fixed baseline ladder. That is a labeling/config detail, not the all-up cause.

## Exact Code Inspected

- `src/curvyzero/training/dummy_pong.py`
  - `ACTION_LABELS`
  - `PongConfig`
  - `PongEnv.reset`
  - `PongEnv.step`
  - `PongEnv.observation`
  - `_move_paddle`
- `src/curvyzero/training/dummy_pong_eval.py`
  - `run_dummy_pong_eval`
  - `_policy_pairs`
  - `_run_matchup`
  - `_run_episode`
  - `_make_policy`
  - `_summarize_records`
  - `_summarize_pair_groups`
  - `_load_lightzero_checkpoint_policies`
  - `_load_lightzero_mcts_checkpoint_policies`
- `src/curvyzero/training/lightzero_dummy_pong_env.py`
  - `DummyPongLightZeroEnv.__init__`
  - `reset`
  - `step`
  - `seed`
  - `random_action`
  - `_lightzero_observation`
  - `_encode_observation`
  - `_info`
  - `summarize_episode_rows`
- `src/curvyzero/training/lightzero_dummy_pong_features.py`
  - `encode_tabular_ego_observation`
- `src/curvyzero/training/lightzero_dummy_pong_policy.py`
  - `LightZeroPolicyHeadGreedyPolicy.action`
  - `LightZeroMCTSEvalModePolicy.action`
  - `load_lightzero_policy_head_greedy_checkpoint`
  - `load_lightzero_mcts_eval_mode_checkpoint`
  - `load_state_dict_policy_head_safe`
  - `load_state_dict_strict_full_model`
  - `_model_cfg_for_checkpoint_state_dict`
  - `_extract_eval_mode_action`
- `src/curvyzero/training/lightzero_dummy_pong_checkpoint_probe.py`
  - `_find_state_dict`
  - `_try_direct_model_action`
  - `_model_cfg_for_checkpoint_state_dict`
- `src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py`
  - `patched_dummy_pong_configs`
  - `validate_dummy_pong_surface`
- `src/curvyzero/infra/modal/lightzero_dummy_pong_tiny_train_smoke.py`
  - `_run_lightzero_dummy_pong_tiny_train_smoke`
  - `_parse_training_signals`
  - `_mirror_lightzero_checkpoints`
- `src/curvyzero/infra/modal/lightzero_dummy_pong_train_attempt.py`
  - `_install_lightzero_timestep_compat_patch`
  - `lightzero_dummy_pong_train_attempt`
- `src/curvyzero/infra/modal/lightzero_dummy_pong_policy_head_scoreboard_attempt.py`
  - `score_lightzero_dummy_pong_policy_head`
- `src/curvyzero/infra/modal/lightzero_dummy_pong_mcts_scoreboard_attempt.py`
  - `score_lightzero_dummy_pong_mcts`
- `src/curvyzero/infra/modal/lightzero_dummy_pong_mcts_loader_smoke.py`
  - `_strict_load_attempts`
  - `_policy_head_diagnostic`
  - `_eval_mode_forward_attempt`
- `scripts/run_dummy_pong_lightzero_checkpoint_scoreboard.py`
  - `_as_lightzero_scoreboard_summary`
  - `_scoreboard_rows`
  - `_action_histograms_by_pair_group`
- `scripts/run_dummy_pong_lightzero_mcts_checkpoint_scoreboard.py`
  - `_as_lightzero_mcts_scoreboard_summary`
  - `_scoreboard_rows`

Docs read:

- `docs/experiments/2026-05-09-lightzero-dummy-pong-policy-head-scoreboard.md`
- `docs/experiments/2026-05-09-lightzero-dummy-pong-bug-telemetry-audit.md`
- `docs/experiments/2026-05-09-lightzero-dummy-pong-longer-run.md`
- `docs/working/lightzero_checkpoint_loader_probe_2026-05-09.md`
- `docs/working/training_coach_handoff_2026-05-09.md`

## Verification Run

No pytest.

Compile smoke passed:

```sh
uv run python -m py_compile \
  src/curvyzero/training/dummy_pong.py \
  src/curvyzero/training/dummy_pong_eval.py \
  src/curvyzero/training/lightzero_dummy_pong_env.py \
  src/curvyzero/training/lightzero_dummy_pong_features.py \
  src/curvyzero/training/lightzero_dummy_pong_policy.py \
  src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py \
  src/curvyzero/infra/modal/lightzero_dummy_pong_train_attempt.py \
  src/curvyzero/infra/modal/lightzero_dummy_pong_policy_head_scoreboard_attempt.py \
  src/curvyzero/infra/modal/lightzero_dummy_pong_mcts_scoreboard_attempt.py \
  scripts/run_dummy_pong_lightzero_checkpoint_scoreboard.py \
  scripts/run_dummy_pong_lightzero_mcts_checkpoint_scoreboard.py
```

## Highest-Value Next Checks

1. Run a real-observation logits grid for `iteration_0`, `iteration_8`, and
   `ckpt_best`.
   - Use actual reset/rollout observations.
   - Record logits, softmax, argmax, margin, entropy, and action histogram.
   - This separates "policy head is dead" from "scorecard task is wrong".

2. Make scorecard config explicit for the 512/8 run.
   - Pass `max_env_step=512`.
   - Record it in the row.
   - Ideally let `run_dummy_pong_eval()` accept an eval `PongConfig(max_steps=512)`
     or clearly label that game horizon is still 120.

3. Rerun the full MCTS/eval-mode scorecard with explicit matching config.
   - Use the passing loader path with `res_connection_in_dynamics=True`.
   - Match the 512/8 training horizon explicitly with `max_env_step=512`.
   - Record the eval game horizon too; today independent eval uses
     `PongConfig()` default `max_steps=120`.
   - The first full MCTS scorecard already showed mostly-up behavior, so the
     issue is deeper than greedy policy-head alone.

4. Compare checkpoint progression, not just best/latest.
   - Score/logit-probe `iteration_0`, `iteration_8`, and `ckpt_best`.
   - Check whether policy-head tensors changed and whether action entropy moved.

5. Add an untrained-model control.
   - Same reconstructed config, random weights, same direct argmax path.
   - If trained and untrained both look all-up, suspect reconstruction/eval
     path or insufficient training signal.

6. Check replay/update signal, not just wins.
   - Extract learner losses, replay size, batch count, value/policy loss trend,
     target reward/value stats, and evaluator action entropy if available.
   - The completed 512/8 job may simply be too small to show visible learning.

7. After permission, fix the tiny obvious code issues.
   - Make `random_action()` use a persistent RNG instead of reseeding per call.
   - Consider moving `timestep` into the base LightZero env observation contract.
   - Consider changing scoreboard defaults or requiring `--max-env-step` when
     scoring a checkpoint from a known run.
