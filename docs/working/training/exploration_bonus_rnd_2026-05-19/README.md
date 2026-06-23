# Exploration Bonus / RND For CurvyTron Training

Created: 2026-05-19.

Status: active working wiki. This is an investigation trail, not a final
contract. Promote the distilled version to `docs/research/` or
`docs/design/training/` only after the experiment thesis hardens.

Concrete integration-prep interfaces live in
[`INTEGRATION_PREP.md`](INTEGRATION_PREP.md).

Planning packet:

- [`TODO.md`](TODO.md) is the live task board.
- [`ORCHESTRATION.md`](ORCHESTRATION.md) pins the simple same-process trainer
  pattern and the anti-patterns to avoid.
- [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) names the gates from baseline to
  meter-only RND to the first positive-weight canary.
- [`CRITIQUES.md`](CRITIQUES.md) distills the parallel red-team passes into
  concrete plan changes.

Current planning bias after critique: do not build a general training extension
framework before integration truth. Meter-only RND has enough proof to run
diagnostic smokes. Positive `rnd_replay_target_v0` is not recommended yet and
remains blocked on the intrinsic-normalization decision; resume, support
metadata, seed robustness, and retained extrinsic quality are still open.

2026-05-20 optimizer update: speed/profile smokes with RND need a real learner
batch and a real predictor cadence. A batch-size-1 CPU-oracle RND smoke reached
`train_muzero_with_reward_model` but failed inside training with
`Expected more than 1 value per channel...`. Use ordinary small learner batches
(`batch_size >= 32`, preferably the run's normal batch) for RND proof smokes.

The previous Curvy default `rnd_update_per_collect=1` was a diagnostic smoke
value, not a serious RND training value. Current code defaults to
`rnd_update_per_collect=100`, matching the LightZero/DI-engine scale more
closely. Serious positive-RND tests should sweep `50`, `100`, and policy
`update_per_collect` parity before claiming the intrinsic signal is weak.

Current Curvy RND also logs raw MSE stats before the LightZero-style per-batch
min/max normalization. That normalization is useful as a compatibility canary,
but it is batch-relative: it ranks samples inside the current estimate batch
instead of proving globally decaying novelty across the whole run.

## Short Answer

The current CurvyZero problem is not simply "the agent never learns." CZ26 shows
useful intermediate checkpoints followed by latest-checkpoint regression. An
exploration bonus should therefore be judged on retained extrinsic policy
quality, not on total training reward or novelty counters.

My current recommendation:

1. Add a named, training-only exploration axis, but keep patch one to
   `exploration_bonus_mode=none|rnd_meter_v0`.
2. Start with a meter-only true-RND arm (`weight=0.0`) before any native count
   bonus, nonzero RND reward, or broad sweep.
3. Keep source/game reward, eval reward, tournament Elo, and promotion scoring
   extrinsic-only.
4. Treat full positive-weight RND as stateful training machinery, not as a
   hidden reward variant. Predictor, target, observation normalizer, reward
   normalizer, update counters, and config must be checkpointed or explicitly
   marked non-resumable.
5. Do not blindly write `extrinsic + intrinsic` into MuZero reward targets unless
   we are comfortable with MCTS planning for novelty as if novelty were game
   dynamics.
6. For any positive-weight RND probe, require telemetry that proves the
   predictor is actually training often enough: `train_cnt_rnd`,
   `estimate_cnt_rnd`, train/estimate ratio, small-buffer skip count, raw MSE
   p50/p95, normalized intrinsic p50/p95, and intrinsic/extrinsic ratio.

There are two research tracks hiding under "exploration bonus"; only the second
is patch-one relevant:

- Cheap Curvy-native novelty canary: env/source-state count or coverage bonus.
  This is easiest to test and diagnose, but it is not true RND.
- True RND canary: replay-target augmentation through a LightZero-style reward
  model, with Curvy-specific image shape support and full state checkpointing.
  This is closer to RND, but touches learner/replay/search semantics.

## Repo Worldview

CurvyZero is a source-fidelity and training-control repo around a stock
LightZero MuZero loop, not a small custom trainer.

Current trusted training path:

- `src/curvyzero/contracts/curvytron.py` is the current defaults/control-plane
  source for CZ26 names, cadence, and training objects.
- `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py`
  is the main single-ego LightZero env against env-owned opponents. It emits
  `(4,64,64)` gray64 policy observations and scalar trainer reward.
- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`
  builds the patched LightZero config, calls `lzero.entry.train_muzero`, writes
  sidecars, and installs checkpoint/progress/metrics hooks.
- `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py` and
  `src/curvyzero/tournament/curvytron_checkpoint_tournament.py` handle
  checkpoint intake, rating, leaderboard snapshots, and trainer assignments.

Reward boundaries matter:

- Source/public game reward should stay sparse game outcome. Do not place
  intrinsic reward in `VectorMultiplayerEnv`.
- Trainer reward is assembled in
  `CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv.step()` and
  `_reward_components_for_player()`.
- Existing "bonus reward" is same-step CurvyTron item pickup reward, not
  intrinsic novelty.
- Checkpoint metadata sidecars currently record observation surface, env variant,
  reward variant, action cadence, source max steps, and learner seat mode. Any
  intrinsic training axis belongs in that provenance surface too.

Relevant local docs:

- `docs/working/training/cz26_analysis_2026-05-18/DEEP_ANALYSIS.md`
- `docs/working/training/curvytron_feedback_loop/TRAINING_LOOP_MODULARITY_PLAN.md`
- `docs/design/training/curvytron_learning_gates.md`
- `docs/research/curvytron_survival_reward_design_2026-05-11.md`
- `docs/research/reward_shaping_for_pong_curvy_muzero.md`
- `docs/research/lightzero_feature_fit_for_curvyzero.md`

## What RND Actually Is

Random Network Distillation uses two networks over observations:

- a fixed random target network;
- a trainable predictor network.

The intrinsic reward is prediction error on the target features. Novel
observations usually have higher error; familiar observations decay as the
predictor trains.

The original RND paper and blog make several implementation details central,
not cosmetic:

- RND was designed to avoid the "noisy-TV" trap of next-state prediction by
  making the prediction target deterministic and representable.
- Their strongest PPO version used separate intrinsic and extrinsic value heads,
  different discount factors, and combined advantages.
- They normalized RND observations, clipped normalized inputs, normalized
  intrinsic rewards by running statistics, and controlled predictor update rate.
- Their blog explicitly warns that small implementation details made the
  difference between never leaving the first room and solving hard exploration.

Sources:

- Paper: https://arxiv.org/abs/1810.12894
- PDF: https://arxiv.org/pdf/1810.12894
- OpenAI blog: https://openai.com/index/reinforcement-learning-with-prediction-based-rewards/
- Reference code: https://github.com/openai/random-network-distillation

## OpenAI RND Code Notes

Local clone:

```text
/private/tmp/random-network-distillation-20260519
commit f75c0f1efa473d5109d487062fd8ed49ddce6634
```

This is an archived TF1/OpenAI Baselines/MPI codebase. Reuse the ideas, not the
code.

Important mechanics:

- `run_atari.py` builds Atari envs and drives `PpoAgent`.
- `ppo_agent.py` stores separate buffers for extrinsic reward, intrinsic reward,
  intrinsic value predictions, and extrinsic value predictions.
- `policies/cnn_gru_policy_dynamics.py` and
  `policies/cnn_policy_param_matched.py` define the RND target and predictor.
- RND reward is computed after rollout collection over `obs[0:n] + last_obs`,
  not by the environment.
- The predictor uses only the newest grayscale frame from the stack, normalized
  by observation RMS and clipped to `[-5, 5]`.
- Intrinsic rewards are normalized by a running standard deviation of discounted
  intrinsic returns; extrinsic rewards are not normalized.
- PPO combines `int_coeff * adv_int + ext_coeff * adv_ext`.

Reusable for CurvyZero:

- separate intrinsic/extrinsic telemetry;
- observation RMS warmup and clipped normalized RND inputs;
- intrinsic reward normalization;
- explicit coefficients and schedules;
- stateful side-network checkpoint/resume discipline.

Not reusable:

- TF1 graph code;
- Atari-specific preprocessing and RAM room tracking;
- PPO advantage plumbing;
- the assumption that intrinsic reward can simply be merged into a scalar reward
  without considering MuZero search.

## LightZero RND Code Notes

Local clone:

```text
/private/tmp/lightzero-exploration-audit-20260519
commit de74055298068f53b70e07bc38c41101fce51766
```

LightZero already has an RND-shaped MuZero reward-model entry:

- `lzero/entry/train_muzero_with_reward_model.py`
- `lzero/reward_model/rnd_reward_model.py`
- example configs:
  `zoo/minigrid/config/minigrid_muzero_rnd_config.py` and
  `zoo/memory/config/memory_muzero_rnd_config.py`

How it works:

- collect normal MuZero game segments;
- feed collected observations into `RNDRewardModel`;
- train predictor after collection;
- sample replay;
- call `reward_model.estimate(train_data)`;
- replace sampled target rewards with augmented rewards before learner train.

Useful knobs:

- `reward_model.type = "rnd_muzero"`
- `input_type = "obs" | "latent_state" | "obs_latent_state"`
- `intrinsic_reward_type = "add" | "new" | "assign"`
- `intrinsic_reward_weight`
- `input_norm`, clamp min/max
- `extrinsic_reward_norm`
- `rnd_buffer_size`, `update_per_collect`, `learning_rate`
- `policy.use_rnd_model = True`
- target representation update type/frequency/theta

Risks for CurvyZero:

- The current `estimate()` reshapes observations as `(batch_size, obs_shape, 6)`
  and then `(batch_size * 6, obs_shape)`. This fits the flat MiniGrid/Memory
  examples but is not obviously compatible with CurvyZero `(4,64,64)` image
  stacks without adaptation.
- The code uses min-max normalization of current sampled RND errors to `[0, 1]`,
  which can make reward scale batch-relative and unstable.
- It modifies sampled reward targets, not env telemetry. Without added sidecars
  and metrics, the intrinsic component can disappear into generic reward loss.
- It does not by itself solve checkpointing of reward-model optimizer,
  normalizers, and update counters in the CurvyZero Modal artifact contract.
- It risks teaching MuZero's reward model and MCTS that novelty is part of the
  environment reward.

If we do a true RND canary, the cleaner starting point is replay-target
augmentation rather than env-side RND:

- env-side RND would require every collector/env worker to own or fetch a moving
  predictor/target/normalizer state;
- auxiliary-only RND is safe but does not create a direct exploration incentive;
- replay-target augmentation matches LightZero's existing shape: collect real
  env rewards, train RND from collected observations, then augment sampled
  reward targets before learner training.

This still needs a Curvy-specific adapter. The upstream reward model hardcodes
flat example assumptions and a six-step reshape; a Curvy adapter should derive
unroll length from `target_reward.shape`, support `(4,64,64)` observations, and
avoid mutating replay data in place.

Cadence note: LightZero's reward-model pattern trains the predictor after each
collection wave, then estimates intrinsic reward inside every learner update.
That means a huge collector batch with only a few collection waves can easily
produce a low train-count/high-estimate-count ratio. The current Curvy default
is `update_per_collect=100`; use telemetry rather than assumption to verify the
ratio in every RND run.

## Literature Notes

RND is one point in a family of intrinsic reward methods:

- Count/pseudo-count exploration generalizes tabular count bonuses with density
  models and helped hard Atari games including Montezuma's Revenge:
  https://arxiv.org/abs/1606.01868
- ICM uses prediction error in an inverse-dynamics learned feature space, partly
  to ignore uncontrollable visual factors:
  https://arxiv.org/abs/1705.05363
- Large-scale curiosity found random features are often enough, learned features
  can generalize better, and stochastic setups expose prediction-error limits:
  https://arxiv.org/abs/1808.04355
- Episodic curiosity/NGU add episodic memory or nearest-neighbor novelty so the
  agent repeatedly explores states within an episode:
  https://www.nsavinov.com/publication/2019-ec/
  https://arxiv.org/abs/2002.06038
- RIDE rewards controllable impact in learned representation space and argues
  many novelty methods fail in procedurally generated settings where exact
  revisit is unlikely:
  https://arxiv.org/abs/2002.12292
- Plan2Explore plans toward expected model disagreement/novelty in a learned
  world model rather than retrospectively rewarding surprise:
  https://arxiv.org/abs/2005.05960
- Bonus-based exploration benchmark work is a warning: bonuses can underperform
  simpler exploration baselines and hurt easy-exploration games:
  https://arxiv.org/abs/1908.02388
  https://arxiv.org/abs/2109.11052
- MuZero learns a model that predicts reward, policy, and value for tree search;
  this is why mixing novelty into reward targets changes what search plans for:
  https://arxiv.org/abs/1911.08265
- EfficientZero and related MuZero variants are relevant sample-efficient visual
  model-based RL context, but they do not remove the intrinsic reward/search
  contamination question:
  https://arxiv.org/abs/2111.00210
- Epistemic MCTS is a more planning-native future direction: propagate learned
  model uncertainty inside AlphaZero/MuZero-style search instead of reward
  shaping:
  https://openreview.net/forum?id=Tb8RiXOc3N

Takeaway for CurvyTron: RND is plausible because the policy input is visual and
rewards are sparse/noisy, but bonus methods often fail by changing the objective
instead of improving retained task performance. The CZ26 issue makes retention
and extrinsic eval the primary gates.

## Candidate Designs

### A. Meter-Only RND

Compute RND error but add zero reward.

Purpose: prove instrumentation, state, logging, checkpointing, and replay
plumbing without changing learning.

This should be first. If meter-only changes behavior, the harness changed.

### B. Native Env-Side Count/Coverage Bonus

Use source/vector state to compute cheap novelty:

- quantized ego position and heading;
- wall clearance bins;
- active trail/body counts;
- coarse trail bbox or hash;
- terminal cause/event counters.

Pros:

- easiest to inspect and test;
- can be logged in `env_steps.jsonl`;
- no side network needed.

Cons:

- hand-coded;
- may reward wandering, stalling, or dangerous trail growth;
- not policy-perception faithful.

### C. RND On Gray64 Observation

Use the actual `(4,64,64)` policy surface or latest gray64 frame as RND input.

Pros:

- aligns with what the model sees;
- closest to original RND spirit;
- can detect visual novelty from trail geometry.

Cons:

- can reward visual artifacts, trail density, bonus sprites, or opponent
  nonstationarity;
- stateful side network and normalizers need durable checkpoint/resume;
- LightZero's existing RND path likely needs shape adaptation.

Recommended true-RND implementation form: replay-target augmentation, not
env-side reward. Keep env telemetry extrinsic and add learner-side RND metrics
next to learner metrics.

### D. RND On MuZero Latent State

Use representation-network latent features as RND input.

Pros:

- may focus novelty on learned task features;
- avoids raw visual aliasing.

Cons:

- target moves if representation changes unless carefully frozen/momentum
  managed;
- harder to reason about and checkpoint;
- can entangle representation learning, novelty, and value/reward losses.

### E. Auxiliary-Only RND

Train RND and log novelty/auxiliary loss without using it as reward.

Pros:

- safest stateful side-network proof;
- exercises checkpoint/resume without objective drift.

Cons:

- does not directly change exploration unless later connected to collection or
  targets.

## Contract Requirements

Do not hide this inside `reward_variant`. Keep the extrinsic reward recipe and
the exploration axis separate.

Research-era proposed config surface. The patch-one subset is intentionally
smaller: `mode=none|rnd_meter_v0`, `weight=0.0`,
`feature_source=policy_gray64_latest/v0`, plus the RND meter's batch/update/
buffer/learning-rate/input-normalization knobs.

```text
exploration_bonus_mode = none | rnd_meter_v0 | rnd_replay_target_v0 later | count_coverage_v0 later | rnd_latent_v0 research
exploration_bonus_weight = float
exploration_bonus_cap = float
exploration_bonus_schedule = constant | linear_decay_to_zero
exploration_bonus_decay_until_train_iter = int
exploration_bonus_feature_source = policy_gray64_latest/v0 | policy_gray64_stack4/v0 | source_state_compact/v0 | muzero_latent
exploration_bonus_schema_id = string
exploration_bonus_config_hash = string
exploration_bonus_training_only = true
exploration_bonus_target_reward_effect = unchanged for rnd_meter_v0
exploration_bonus_trainer_effect = uses_reward_model_entrypoint_and_trains_rnd_meter
```

For stateful RND:

```text
rnd_predictor_state_ref/hash
rnd_target_state_ref/hash
rnd_optimizer_state_ref/hash
rnd_obs_rms_state_ref/hash
rnd_reward_rms_state_ref/hash
rnd_update_count
rnd_train_data_count
```

Telemetry additions:

```text
extrinsic_trainer_reward_for_ego
intrinsic_exploration_bonus_for_ego
trainer_reward = extrinsic + weighted_intrinsic
intrinsic_raw
intrinsic_normalized
intrinsic_weight
intrinsic_cap
intrinsic_schedule_value
intrinsic_feature_source
```

Tournament/rating metadata should include the exploration config hash for
stratified analysis and compatibility. Tournament scoring itself must ignore
intrinsic reward.

## Implementation Map

Likely native hook locations:

- Env reward accumulation:
  `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py`
  `step()`
- Existing extrinsic components:
  `_reward_components_for_player()`
- Reward space:
  `_make_reward_space()`
- Info/telemetry:
  `_step_info()`, `_base_info()`, `_write_telemetry_row()`
- Modal config builder:
  `_build_visual_survival_configs()`
- Reward policy and support math:
  `_reward_policy_for_variant()`, `_lightzero_target_config_for_reward()`
- Checkpoint metadata:
  `_build_checkpoint_policy_metadata_sidecar_payload()`
- Tournament pool hash later, not for meter-only:
  `src/curvyzero/tournament/curvytron/contracts.py`
  `rating_pool_hash()`

Likely LightZero RND spike locations:

- switch entrypoint from `lzero.entry.train_muzero` to
  `lzero.entry.train_muzero_with_reward_model`;
- patch `main_config.reward_model`;
- set `policy.use_rnd_model = True`;
- validate/adapt observation shape handling for `(4,64,64)`;
- add CurvyZero checkpoint sidecars for reward-model state and normalizers.

For production-quality RND, add resume hooks for:

- predictor and target state;
- RND optimizer state;
- observation RMS and intrinsic reward normalizer state;
- RND update counters and train-data counters;
- exploration config hash.

Without that, a resumed run restarts its novelty model and cannot be compared
to an uninterrupted run.

## Experiment Plan

Operational gates now live in [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md). The
matrix below is retained as historical research context, not as the first run
plan.

Use matched blocks, not a giant first sweep.

Base blocks:

| Block | Purpose |
| --- | --- |
| A | `out67`, `n10`, `imm0`, mixed blank/wall/rank1: clean mid-run survival signal |
| B | `out67`, `n20`, `imm0`, mixed blank/wall/rank1: tests higher action diversity |
| C | `out100`, `n20`, `imm0`, promising mixed recipe: tournament-rank direction |
| D | Grid B retention recipe such as `b20w05r1` or `b25w25r1`: latest-survival stress |

Arms:

| Arm | Meaning |
| --- | --- |
| E0 | no intrinsic reward |
| E0m | compute novelty/RND metrics, weight `0` |
| E1 | native count/coverage, low weight, decay to 0 by about 170k |
| E2 | native count/coverage, constant low weight |
| E3 | RND gray64, low weight, decay |
| E4 | RND gray64, constant low weight |

Full matrix: `4 blocks x 6 arms x 3 seeds = 72 runs`.

Historical canary if compute is tight: blocks A and D, arms E0/E0m/E1/E3, three
seeds: `24 runs`. Current plan is narrower: E0 baseline, then E0m meter-only
RND, then hold positive-weight RND until the normalization contract is resolved.

Primary acceptance metrics:

- latest/best survival ratio;
- latest survival paired against control;
- best survival must not drop materially;
- post-peak regression slope;
- sparse outcome curve;
- learned tournament rank with enough exposure, excluding iteration 0;
- action collapse rate;
- extrinsic reward components separate from intrinsic.

Failure signatures:

- total training reward rises while extrinsic eval/tournament does not;
- unique coverage rises while deaths rise;
- best checkpoint improves but latest/best remains in the CZ26 `0.55-0.60`
  band;
- RND error stays high forever or collapses immediately;
- value/reward support clipping increases;
- tournament/promotion accidentally scores intrinsic reward.

## Open Questions

- Patch one is a meter-only RND module and is explicitly
  diagnostic/non-resumable until the existing full-resume hook captures reward
  model state.
- Should intrinsic return be episodic or non-episodic in this game?
- Should RND see the latest gray64 frame, all four stacked frames, source
  features, or MuZero latent state?
- What is the right decay schedule if the central problem is retention?
- Can we add RND telemetry without changing training determinism?
- If using LightZero's reward-model path, how do we adapt shape handling and
  checkpoint the reward model state?
- Should intrinsic reward ever influence MuZero reward targets used by search,
  or should it be a separate learner-side signal only?

## Current Bias

Start with `E0m`: same trainer process, LightZero reward-model entrypoint,
latest gray64 input, `weight=0.0`, target rewards unchanged exactly. Keep any
low-weight replay-target canary parked until the normalization contract is
resolved. Keep native count/coverage, stack4 input, latent RND, and generic
extension work in the parking lot.

The crisp success condition is not "more novelty." It is:

```text
matched latest/best survival improves, sparse outcome is flat-or-better,
action collapse does not worsen, and extrinsic tournament evidence improves
without relying on low-exposure ranks.
```

## 2026-05-22 Optimizer Update

`rnd_update_per_collect=100` is still the right diagnostic scale for the
meter-only path, but the first implementation accidentally paid for heavy
predictor/target hashing inside that 100-update loop.

Current code hashes predictor/target state once before the RND update batch and
once after it. That keeps the proof we care about:

```text
predictor changed
target stayed frozen
```

but avoids hundreds of full model hashes per collect.

Fresh H100 C64/sim16/3-learner meter profiles after the fix:

| row | steps/sec | `rnd_train_with_data` | `rnd_state_hash` |
| --- | ---: | ---: | ---: |
| stock `rnd_meter_v0` | `351.02` | `0.590s` | `0.131s` |
| direct search hook + `rnd_meter_v0` | `448.52` | `0.603s` | `0.140s` |

Before the fix, the same RND area was about `3.5s` for training and about
`3.0s` for hashing in these short profiles. Treat that old overhead as fixed,
not as evidence against RND itself.
