# Glossary

Last updated: 2026-05-19

Use plain meanings here. Refine as scouts return findings.

## Collector Env

One environment instance used by LightZero collectors to gather experience.

## `collector_env_num`

How many collector env instances LightZero runs for collection. This is not automatically the same as learner batch size.

## `n_episode`

How many completed episodes LightZero tries to collect per collector call in the policy config. Current default is 256. With `collector_env_num=256`, the usual intent is one episode per collector env per collect call, but learner updates still sample replay transitions later.

## Learner Batch Size

How many sampled replay items the learner uses for one update. Current default is 64.

This is not the same as collector env count and not the same as opponent slot count.

## Episodes Per Collect

How many completed episodes LightZero tries to gather in one collector call. Current default is 256.

## Opponent Mixture

A validated list of possible opponent entries. Current code selects an entry at episode reset unless deterministic collector-env splitting is used.

## Opponent Slot

A human-facing way to describe one mixture component, such as blank-canvas noop, hard-coded wall avoidant, or frozen checkpoint. This term should become more precise after the batch investigation.

## Opponent Slot Count Total

The sum of explicit integer slot counts in a deterministic collector-env split. In tonight18-style recipes this was 64. It is not the learner batch size.

## `opponent_immortal`

Explicit boolean in opponent mixture entries. It is separate from the policy kind. Runtime `opponent_death_mode` should be derived from this.

## Policy Observation Perspective

The view of the game state passed to the policy network. This needs one clear contract across training and tournament eval.

## Reward Contract

The set of reward variant name, scalar reward calculation, component metadata, reward bounds, schema id/hash, and LightZero support-range assumptions.

## `VisualSurvivalConfigSpec`

The grouped typed input contract for building the LightZero visual-survival config. It is normalized internal config, not the experiment-facing knob list. Current groups are run/runtime, training scale, timing, observation, behavior, reward/target, and opponent.

## `VisualSurvivalExperimentSpec`

Compact experiment-facing input that expands to `VisualSurvivalConfigSpec`. It currently contains deliberate experiment choices only: seed/paths, reward strength, opponent source, action noise, and scale preset. It rejects source physics cadence, render defaults, policy-observation backend, learner seat mode, support caps, learner batch size, and LightZero patch internals as ordinary knobs.

## `FrozenOpponentConfig`

Nested typed part of `VisualSurvivalConfigSpec` for frozen checkpoint opponents: checkpoint metadata, snapshot ref, checkpoint state key, and whether the opponent policy should use CUDA.

## `VisualSurvivalConfigResult`

The typed result from `build_visual_survival_config(...)`. It carries `template_module`, `main_config`, `create_config`, `surface`, and `patches`, with convenience access to `env_config` and `lightzero_target_config`.

## Hook Bundle

Proposed boundary for installing trainer-side hooks around stock LightZero behavior: checkpoint publishing, progress writing, resume state, opponent refresh, and future extension hooks.
