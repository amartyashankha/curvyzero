# Repo-Native Actor-Loop Next Step

Date: 2026-05-09
Status: proposal plus one dry-run scaffold

## Short Read

Build the first repo-native PPO/CleanRL-style CurvyTron smoke as an
instrumented actor-loop diagnostic, not as a replacement for LightZero
replication.

The current repo-native file is only a dry-run scaffold:
`scripts/repo_native_ppo_actor_loop_dry_run.py`. It samples a masked uniform
policy and writes PPO-shaped artifacts. It is not a learner and makes no
learning claim.

The useful first target is 1v1/no-bonus with all-live-player wrapper `[B,P]`
collection:

```text
obs[B,P,106], mask[B,P,3], live[B,P]
  -> compact live ego rows
  -> shared masked policy/value forward
  -> scatter action ids back to wrapper joint_action[B,P]
  -> trainer env step once per row over held source controls
  -> stage PPO rollout buffer and terminal/final-observation data
  -> run a tiny PPO update and scorecard
```

LightZero stays active as a replication/control lane in parallel. The
repo-native lane should expose timing, buffer, reset/autoreset, and all-player
wrapper action metadata plainly; the LightZero lane should keep proving
external trainer behavior, MuZero/search target semantics, checkpoint/eval
plumbing, and replication controls. Neither lane cancels the other.

## Inputs Read

- `src/curvyzero/env/trainer_contract.py`: pins trainer-facing ray observation
  schema `curvyzero_egocentric_rays/v0`, flat observation width `106`, action
  order `left, straight, right`, sparse round reward, final-observation and
  terminal-info expectations.
- `src/curvyzero/env/trainer_observation.py`: current scalar 1v1 ray
  observation/reward adapter; useful for the first smoke, not vector ray
  generation or full source fidelity.
- `src/curvyzero/training/policy_row_mapping.py`: pure compact/scatter helpers
  for all-live-player wrapper `[B,P] -> [R] -> [B,P]`.
- `src/curvyzero/training/replay_chunk_v0.py`: validated 1v1/no-bonus `.npz`
  chunk contract for observation, reward, action, action weights, root value,
  done flags, seed/source, final observation, and final reward map.
- Optimizer docs: repo-owned PPO/CleanRL-style runner is the next transparent
  speed/debug prototype, while LightZero remains a serious replication/control
  lane in parallel.
- Environment active lanes: first milestone is 1v1/no-bonus source-faithful
  reset/step/observation/reward/replay and speed checks; the current fast path
  is still contract-scoped.

## First Smoke

Use a repo-native PPO-style runner with the smallest honest scope:

- Backend: current scalar `CurvyTronEnv` rows behind a fixed `[B,P]` actor-loop
  facade until the production vector `step_many` kernel is extracted.
- Observation: `observe_1v1_egocentric_rays_v0(...)`, stored as
  `float32[T,B,2,106]`.
- Action: shared masked categorical over three actions, one policy row per live
  player, then `policy_rows_to_joint_action(...)`.
- Opponent/self-play: shared current policy controls both players for the main
  actor smoke; eval also runs frozen random and simple scripted controls.
- Reward: true sparse round outcome only. Survival/contact/clearance remain
  telemetry, not env reward.
- PPO buffer: `observation`, `legal_action_mask`, `live_mask`, `action`,
  `action_logprob`, `action_prob`, `value`, `reward`, `done`, `terminated`,
  `truncated`, `episode_id`, `reset_seed`, `final_observation`,
  `final_reward_map`, and `policy_version`.
- Learner: tiny MLP policy/value, masked categorical, GAE/returns, PPO clipped
  objective, value loss, entropy, grad norm, one or two update epochs.
- Stop condition: one local run writes a checkpoint, rollout artifact, profile
  report, and independent scorecard without hidden autoreset or masked-action
  violations. Learning quality is a later gate.

## Measurement Buckets

The first report should preserve these buckets so it compares cleanly with the
vector bridge, PPO runner, and LightZero control lane:

| Bucket | Scope |
| --- | --- |
| `reset_autoreset_sec` | Explicit row reset after terminal/final data has been staged. |
| `observation_packing_sec` | Ray observation, masks, rewards, final observations. |
| `row_compaction_sec` | `[B,P]` live rows to policy rows. |
| `policy_forward_sec` | PPO model forward and masked sampling. |
| `action_scatter_sec` | Policy rows back to wrapper `joint_action[B,P]`. |
| `env_step_sec` | Wrapper environment step. |
| `replay_or_rollout_stage_sec` | PPO buffer writes and final reward/observation staging. |
| `target_construction_sec` | Returns/GAE and masks. |
| `learner_update_sec` | PPO minibatches, optimizer step, grad stats. |
| `checkpoint_publish_sec` | Checkpoint write. |
| `eval_scorecard_sec` | Scorecard eval. |

Top-line metrics: completed games/min, env transitions/sec, ego decisions/sec,
p50/p95/p99 action latency, replay/rollout bytes per chunk, timeout rate,
masked-action violation count, entropy, KL, clip fraction, value explained
variance, actor idle, learner idle, and policy staleness once actors split.

## Artifacts

Each run should write one artifact directory:

```text
run_config.json
contract_manifest.json
profile_report.json
rollout_buffer.npz
ppo_metrics.jsonl
scorecard.json
checkpoint_step_*.npz or .pt
sample_terminal_trace.json
```

`replay_chunk_v0.py` should remain the MuZero/search-compatible replay export
contract. PPO needs a rollout buffer with logprobs, values, advantages, and
returns; those can be exported separately rather than overloading replay v0.
If a bridge export to replay v0 is added, `action_weights` can carry policy
probabilities and `root_value` can carry PPO value estimates, but PPO-specific
fields still need their own artifact.

## LightZero Parallel Lane

This repo-native smoke runs beside LightZero replication/control work.

LightZero should continue to answer:

- can the pinned LightZero setup reproduce stock/control behavior;
- can the custom-env bridge preserve CurvyZero observation/action/reward
  metadata;
- are MuZero/search targets, support scales, checkpoints, and eval scorecards
  sane;
- what artifact and trainer lifecycle patterns should CurvyZero reuse.

The repo-native PPO smoke should answer:

- does the CurvyTron `[B,P]` all-player wrapper actor loop work without hidden
  framework semantics;
- are reset/autoreset, final observations, masks, and sparse rewards coherent;
- what does a transparent actor-loop timing profile look like before search;
- can a simple baseline expose environment/reward bugs before MuZero claims.

Interface lessons should flow both ways:

| From repo-native PPO to LightZero | From LightZero to repo-native PPO |
| --- | --- |
| Exact `[B,P]` wrapper action metadata and ego-row mapping. | Checkpoint, eval, manifest, and run-directory conventions. |
| Final-observation and autoreset policy that does not hide terminal transitions. | Search target/action-weight diagnostics and support-scale lessons. |
| Shared measurement bucket names and scorecard fields. | Trainer lifecycle expectations: collectors, learners, evaluators, resumes. |
| Masked-action and sparse-reward bug checks. | Config hygiene and artifact copying patterns from replication runs. |
| PPO baseline scorecards to contextualize MuZero/LightZero learning claims. | MuZero target quality signals that may later inform replay v0 exports. |

Decision rule: do not use PPO success to retire LightZero, and do not use
LightZero plumbing success to skip the repo-native timing/buffer smoke. They are
complementary controls until one lane proves a production-quality path.

## Scaffold Added

Tiny dry-run scaffold:
`scripts/repo_native_ppo_actor_loop_dry_run.py`.

It currently:

- runs scalar `CurvyTronEnv` rows in an all-player `[B,P]` wrapper-loop shape;
- uses trainer-facing ray observations and action masks;
- compacts live ego rows with `build_policy_row_mapping(...)`;
- samples a masked uniform policy, then scatters wrapper joint actions;
- stages a PPO-shaped rollout buffer and writes a JSON report plus `.npz`.

It intentionally does not train, use Torch, claim source fidelity, or replace
the LightZero lane.

Local no-pytest smoke run:

```sh
python3 scripts/repo_native_ppo_actor_loop_dry_run.py \
  --batch-size 4 \
  --rollout-steps 8 \
  --artifact-root /private/tmp/curvy-repo-native-ppo-actor-loop-dry-run-smoke \
  --format plain
```

Observed output:

```text
repo_native_ppo_actor_loop_dry_run B=4 T=8 env_transitions_per_sec=303.8 ego_decisions_per_sec=607.6 games_per_min=2847.95 policy_action_p95_ms=0.599 artifact_root=/private/tmp/curvy-repo-native-ppo-actor-loop-dry-run-smoke
```

Read this as a contract smoke only. The toy env terminates very quickly under
random actions, so games/min is not a training-quality or production-speed
number.

## Next Implementation Slice

1. Promote the scaffold from masked-uniform dry run to a tiny `src/curvyzero/training`
   PPO runner only after choosing whether to add a PyTorch dependency or keep
   the first learner as a script-local optional import.
2. Keep the same artifact/report shape and add `target_construction_sec`,
   `learner_update_sec`, `checkpoint_publish_sec`, PPO metrics, and scorecards.
3. Mirror the report fields in the LightZero control lane so both lanes can be
   compared by interface behavior, artifact completeness, and scorecard results.
4. Swap scalar env rows for source-faithful/vector `step_many` only when the
   environment lane exposes the real reset/autoreset/final-observation boundary.
5. Gate any learning claim on held-out eval versus random/scripted baselines,
   not collector return alone.
