# Optimizer Profile Report Contract

Date: 2026-05-09

Status: first shared report shape for optimizer-owned measurement. This is a
contract for timing and metadata, not for learning claims.

## Purpose

Expose enough timing, schema, denominator, and caveat information to compare
repo-native `[B, P]` actor-loop runs and LightZero replication/control runs
without pretending they are the same architecture.

The report should answer:

- what loop shape ran;
- what data contracts crossed the loop;
- where wall time went;
- what latency and throughput looked like;
- whether replay/rollout, checkpoint, eval, and policy-version metadata were
  visible;
- which denominator each throughput number uses;
- what the report does not prove.

Coach owns learning claims. Optimizer owns this measurement surface.

## Top-Level Shape

```json
{
  "optimizer_profile_schema": "curvyzero_optimizer_profile_report/v0",
  "schema_id": "lane_specific_report_schema/v0",
  "status": "ok",
  "run": {},
  "lane": "repo_native_actor_loop",
  "contracts": {},
  "denominators": {},
  "loop_shape": {},
  "timing_sec": {},
  "latency_sec": {},
  "throughput": {},
  "integrity": {},
  "replay_or_rollout": {},
  "policy_search": {},
  "learner": {},
  "eval": {},
  "policy_staleness": {},
  "artifacts": {},
  "caveats": []
}
```

Keep absent work explicit without inventing taxonomy. If a lane did not run a
learner, eval, checkpoint publish, or async actors, either report the small
`ran: false` marker or put the absence in `caveats`.

## Run Metadata

Required fields:

- `run_id`, `created_at_utc`, `git_ref`, `git_dirty`, and `status_entry_count`
  when available;
- `command` or `entrypoint`;
- `local_or_modal`;
- `host`, `python`, dependency versions;
- `device_backend`: `cpu`, `cuda`, `mps`, `jax`, `lightzero`, or mixed label;
- `seed`;
- `debug_event_mode`: `debug-event`, `sampled-event`, `no-event`, or framework
  equivalent;
- `lane`: short plain label, for example `repo_native_actor_loop`,
  `source_trainer_actor_loop_profile`, `lightzero_control`,
  `lightzero_bridge`, `mctx_probe`, or `synthetic_probe`.

## Contracts

Required fields where applicable:

- `environment_impl_id`;
- `observation_schema_id`, shape, dtype, and whether history/frame stack exists;
- visual surface identity, for example `debug_visual_tensor` with
  `curvyzero_debug_occupancy_gray64/v0`, plus raw dtype/shape and
  LightZero-facing dtype/shape when they differ;
- `action_space_id`, action count, action order, and legal-mask dtype;
- `reward_schema_id`, reward semantics, shaping status, and discount;
- `terminal_semantics`: `done`, `terminated`, `truncated`, timeout policy;
- `player_mapping`: player id order, ego-row mapping, seat policy;
- `batch_layout`: `B`, `P`, time dimension, and whether arrays are `[B,P,...]`;
- `reset_seed_policy`, `episode_id_policy`, `reset_source_policy`;
- `checkpoint_id_policy` and policy-version semantics.

For LightZero lanes, also record:

- stock config path or custom config patch source;
- compiled support-scale fields;
- env/action wrapper boundary;
- whether evaluator is stock DI-engine, manual strict-load, or fallback.

## Denominators

Required where applicable:

- env transitions;
- player ticks;
- ego decisions;
- policy rows;
- MCTS roots and simulations;
- rollout or replay rows;
- learner samples and updates;
- completed games.

Do not compare two throughput numbers unless their denominators match or the
report explains the mismatch.

## Timing Buckets

Use zero or `null` for buckets that do not apply, but keep the key visible.

- `reset_autoreset_sec`;
- `env_step_sec`;
- `render_sec`;
- `stack_normalize_sec`;
- `observation_packing_sec`;
- `row_compaction_sec`;
- `policy_forward_sec`;
- `search_sec`;
- `cpu_gpu_transfer_sec`;
- `action_select_sec`;
- `action_scatter_sec`;
- `replay_or_rollout_stage_sec`;
- `replay_write_or_learner_handoff_sec`;
- `target_construction_sec`;
- `learner_sample_sec`;
- `learner_update_sec`;
- `checkpoint_publish_sec`;
- `eval_scorecard_sec`;
- `actor_idle_sec`;
- `learner_idle_sec`;
- `loop_elapsed_sec`;
- `loop_overhead_sec`;
- `wall_elapsed_sec`.

## Latency

Report count, mean, p50, p95, p99, and max where the lane can expose them:

- actor step total;
- env step;
- observation packing;
- policy/search decision;
- replay/rollout stage;
- replay write or handoff;
- learner update;
- checkpoint publish;
- eval episode.

## Throughput

Required where applicable:

- env transitions/sec;
- player ticks/sec;
- ego decisions/sec;
- policy rows/sec;
- replay or rollout rows/sec;
- completed games/minute;
- learner updates/sec;
- samples/sec;
- checkpoint publishes/hour or per run;
- eval episodes/sec.

## Replay Or Rollout

Required fields:

- writer kind: in-memory, local `.npz`, Modal Volume, learner queue, framework
  replay, or none;
- schema id and schema hash;
- bytes per chunk or buffer;
- rows per chunk or buffer;
- field specs: shape, dtype, and checksum or lightweight integrity hash for
  each array field;
- write latency summary;
- final observation included: yes/no;
- final reward map included: yes/no;
- replay age or queue depth if available.

For PPO rollout buffers, include logprob/value/advantage/return field presence
separately from MuZero replay-v0 fields.

## Policy/Search

Required fields:

- policy kind: random, scripted, PPO, LightZero MuZero, Mctx, synthetic;
- policy version;
- checkpoint id or `none`;
- opponent/checkpoint assignment policy;
- action histogram;
- action-mask contract status;
- root action weights / visits if search is present;
- root value and searched value summaries if search is present.

## Integrity

Required where applicable:

- masked-action violations;
- no-legal-action rows;
- NaN/Inf count by tensor group;
- done equals terminated-or-truncated invariant failures;
- final-observation count and final-reward-map count;
- reset/autoreset-after-staging check result;
- row scatter/gather mismatch count;
- schema rejection count.

## Learner

Required if a learner ran:

- update count;
- batch size;
- minibatch size;
- update epochs;
- optimizer kind;
- loss fields present;
- grad norm summary if available;
- GPU utilization if available;
- learner idle and actor idle.

Coach owns interpretation of loss curves and checkpoint quality.

## Policy Staleness

Required once actors and learners are decoupled:

- actor policy version;
- learner latest policy version;
- steps or seconds behind;
- checkpoint refresh interval;
- staleness p50/p95/p99;
- dropped or rejected rollout count due to staleness.

For single-process synchronous runs, report `mode: synchronous` and
`max_version_lag: 0`.

## Caveats

Every optimizer report should include plain caveats for fake or incomplete work:
synthetic policy/search, debug payloads, toy env rows, no learner update, no
checkpoint publish, manual eval, no production replay stream, smoke-scale
budget, no source-fidelity claim, and no training-quality claim.

Every speed readout must also state whether env step, render,
stack/normalize, policy/search, replay, and reset are inside the number. For
the current CurvyTron visual path, `debug_visual_tensor` means debug/profiling
occupancy only, not source-faithful visual truth.

Do not include return, win rate, Elo, or best-checkpoint summaries as optimizer
evidence. Coach owns those learning claims.
