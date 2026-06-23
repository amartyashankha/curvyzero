# Batch Construction Plan

Last updated: 2026-05-19

## Why This Exists

Opponent slots are currently described as percentages or mixtures, but training uses real batch sizes. Before changing sampling, we need to know where the batch is built and what contract is actually meaningful.

## Questions

- What is the active training batch size? Answer: `learner_batch_size=64` by current default.
- What does the `64` mean in the current training code? Answer: it can mean learner batch size, opponent slot count, or background eval batch size depending on context. These must be named separately.
- What does the `256` mean, if it is present? Answer: current default collector env count and episodes-per-collect.
- Where are opponent slots sampled? Answer: raw mixture is selected per env reset; assignment-refresh mode can preassign singleton mixtures per collector env.
- Is sampling per environment, per episode, per collector worker, per learner batch, or somewhere else? Answer: raw mixture is per reset/episode; deterministic split is per collector env; learner batch is replay-sampled transitions and not directly slot-controlled.
- Can slot weights be converted cleanly into deterministic counts? Answer: yes for collector env assignment when using `deterministic_collector_env_mixture_plan`.
- If deterministic counts are used, what must sum to a power of two and why? Answer: current helper requires the opponent slot count total to be a power of two and divide `collector_env_num`; this is a local split-contract rule, not a LightZero requirement.

## Current Caution

Do not claim deterministic split values control the learner replay mini-batch. They control collector env assignment. The replay batch can still skew because it is sampled from stored transitions.

Plain version: `256` is the current collector env / episodes-per-collect scale, while `64` is the learner update batch size. A separate 64-slot opponent recipe can exist, but that is a collector assignment recipe. It is not the same thing as saying each learner gradient update contains exactly that mix.

Do not overload `64`. Use explicit names:

- `learner_batch_size`
- `collector_env_num`
- `episodes_per_collect`
- `opponent_slot_count_total`

## Current Test Guardrails

- `tests/test_opponent_mixture.py::test_opponent_mixture_deterministic_collector_plan_repeats_slot_count_bag` proves the 64-slot recipe expands to 256 collector env assignments with the expected counts.
- `tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_opponent_assignment_refresh_reset_param_uses_exact_collector_slot_split` proves assignment refresh sends singleton mixtures and split metadata to collector reset params.
- `tests/test_curvyzero_source_state_visual_survival_lightzero_env.py::test_opponent_mixture_selects_once_per_reset_not_per_step` now also proves raw inline mixtures stay `episode_reset` sampling and do not silently set deterministic split fields.
- `tests/test_lightzero_config_builder.py::test_public_visual_survival_builder_builds_source_state_contract` uses `batch_size=32` with `collector_env_num=256`, making the learner-batch/collector-env distinction explicit in the public builder gate.

## Possible Target Shape

If the investigation supports it, replace vague percentages with explicit slot weights or counts:

- values are validated,
- the total has a documented relationship to `collector_env_num`, not learner `batch_size`,
- remainder behavior is explicit,
- sampling is still randomized where randomness is useful,
- the split is stable enough that gradient signal is not dominated by accidental mixture drift.

## Done Criteria

- We can explain how one learner update gets its opponent mix.
- The batch-size terms are defined in plain language.
- Any new split contract has tests for validation and generated assignments.
