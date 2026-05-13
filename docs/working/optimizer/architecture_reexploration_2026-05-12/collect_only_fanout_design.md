# Collect-Only Fanout Prototype Design

Date: 2026-05-12

Scope: smallest prototype for launching multiple searched CurvyTron collection
jobs from the same frozen checkpoint, writing trajectory chunks, and merging
them for later learner work. This is an Optimizer architecture note, not a
learning claim.

Local note: `uv run python -c "import lzero"` fails in this checkout with
`ModuleNotFoundError: No module named 'lzero'`. The repo's Modal launcher
installs `LightZero==0.2.0`, so the design below uses repo code, current docs,
and the LightZero object names already used by the launcher/audit hooks. Exact
constructor signatures should be verified inside the Modal LightZero image
before implementation.

## Recommendation

Do not try to turn stock `lzero.entry.train_muzero` itself into collect-only for
the first fanout prototype.

Use a small repo-owned collector entrypoint that reuses the same CurvyTron
LightZero config patching, loads one frozen checkpoint into a LightZero
`MuZeroPolicy`, drives LightZero collection through `policy.collect_mode`, and
writes native LightZero `GameSegment` chunks plus a strict JSON manifest. A
separate merge/import smoke should read those chunks in the same LightZero
runtime and push them into a `MuZeroGameBuffer` with `push_game_segments`.

This keeps the important contracts stock:

- policy/search target production comes from LightZero `collect_mode`;
- trajectory shape remains LightZero `GameSegment`;
- replay import is proven through `MuZeroGameBuffer.push_game_segments`;
- learner training is not touched.

It avoids claiming that a direct `collect_mode.forward` loop is equivalent to
stock training, and it avoids using the old custom two-seat path as evidence.

## Why Not Stock `train_muzero` Collect-Only?

The trusted stock lane is valuable precisely because `train_muzero` owns the
whole loop:

```text
build env managers
  -> build policy
  -> build learner
  -> build MuZeroGameBuffer
  -> maybe eval
  -> collect
  -> push to replay
  -> sample
  -> learn
  -> checkpoint
```

The current launcher wraps this loop for profiling and audit, but it still calls
`train_muzero` and lets LightZero run synchronously. There is no repo-visible
collect-only mode. The existing `stop_after_learner_train_calls` profile cap
still enters learner work before stopping. Setting train or env step caps would
be brittle unless verified against LightZero internals, and monkeypatching
learner methods to no-op would make the prototype more about fighting
`train_muzero` than testing fanout.

So the smallest clean split is:

```text
stock control path:
  train_muzero owns collect/replay/learn/checkpoint

fanout prototype:
  repo collector owns job orchestration and chunk writes
  LightZero owns policy.collect_mode/search and GameSegment shape
  LightZero MuZeroGameBuffer owns import compatibility smoke
```

This is not a replacement training loop yet. It is an artifact and throughput
probe.

## Prototype Shape

Add later, in a separate code change, one Modal function such as
`lightzero_curvytron_collect_only_fanout.py` or a narrow mode on the existing
Modal module if that is less plumbing. It should not change live Coach training
defaults.

Per actor job:

```text
checkpoint K
  -> build the same source_state_fixed_opponent LightZero config surface
  -> load K into a LightZero MuZeroPolicy
  -> construct collector env manager with N local envs
  -> run MuZeroCollector.collect(policy.collect_mode, n_episode=M)
  -> write one chunk directory
```

Merge/import smoke:

```text
read fanout manifest
  -> validate all chunks share checkpoint/env/search schema
  -> read native GameSegment payloads
  -> create MuZeroGameBuffer from the same config
  -> push_game_segments(all_segments)
  -> optionally sample one batch and record target shapes
```

The first merge smoke may stop after `push_game_segments` and `sample`. It
should not call `BaseLearner.train`.

## Chunk Contract

Use a directory per actor output, not a single loose file:

```text
collect_chunks/
  checkpoint_<id>/
    actor_000/
      manifest.json
      segments.cloudpickle
      target_audit.json
      env_steps.jsonl
      metrics.json
```

`segments.cloudpickle` is an implementation payload, not a stable storage
standard. It is acceptable for the first prototype because it will be produced
and consumed in the same Modal image with the same `LightZero==0.2.0`. The
manifest must say this clearly. A later durable replay format can convert the
same fields to arrays once import compatibility is proven.

Each chunk manifest should contain:

- `schema_id`: e.g. `curvyzero_lightzero_collect_only_chunk/v0`.
- `producer`: module/function name and git commit if available.
- `created_at`, `run_id`, `attempt_id`, `fanout_group_id`, `actor_id`.
- `claim_label`: `collect_only_throughput_probe_not_learning`.
- `training_behavior_changed`: `true` relative to stock `train_muzero`, because
  the learner is absent.
- `checkpoint`: immutable checkpoint ref/path, filename, size, mtime, optional
  sha256, LightZero state key used, iteration label if known.
- `opponent_checkpoint`: same fields for the frozen opponent checkpoint, if
  different from the collecting policy checkpoint.
- `packages`: LightZero, DI-engine, torch, numpy, gym versions.
- `command`: seed, actor seed offset, collector env count, `n_episode`,
  `max_env_step` or local cap, `source_max_steps`, `decision_ms`,
  `num_simulations`, `batch_size`, `env_manager_type`, compute/device, and any
  profile flags.
- `policy_surface`: `muzero`, model type, observation shape, action space size,
  `policy.cuda`, `policy.multi_gpu`, discount, `td_steps`, support sizes, and
  `reanalyze_ratio` if visible.
- `env_surface`: `env_variant=source_state_fixed_opponent`, env id/type/import
  names, runtime topology, underlying env class, ruleset/rules hash, reward
  variant/schema/hash, observation schema/hash, raw observation schema/hash,
  action mask dtype/meaning, `to_play` meaning, `timestep` meaning, dynamic seed
  strategy, death mode, natural bonus flag, control noise profile, policy action
  repeat settings, and ego override settings.
- `render_surface`: `source_state_trail_render_mode`, default/supported trail
  modes, visual surface, truth level, source-state-backed flag, browser pixel
  fidelity claim, raw RGB frame shape, gray64 frame shape, stack owner.
- `opponent_surface`: opponent kind, training relation, checkpoint ref, state
  key, `opponent_use_cuda`, opponent seed, opponent simulations, opponent batch
  size.
- `segment_payload`: file name, serialization kind, LightZero version lock,
  segment count, per-segment lengths, total env decisions, terminal segment
  count, compressed/uncompressed bytes if available.
- `compatibility_keys`: values that must match before merge: checkpoint id,
  env schema hash, rules hash, observation schema hash, reward schema hash,
  action space size, search sim count, support/target config, render mode,
  LightZero version, and policy architecture.

The serialized `GameSegment` objects must preserve, at minimum, the fields the
current target audit already inspects:

- `obs_segment`: LightZero obs dicts with `observation`, `action_mask`,
  `to_play`, and `timestep`.
- `action_segment`: scalar ego action selected by collect search.
- `reward_segment`: scalar ego reward returned by the env.
- `to_play_segment`: currently single-agent/fixed-opponent stock semantics.
- `child_visit_segment`: policy/search target visit distribution.
- `root_value_segment`: search root value target.
- `action_mask_segment`: legal ego action mask at each step.
- any LightZero metadata required by `MuZeroGameBuffer.push_game_segments`.

The sidecar `env_steps.jsonl` should retain the existing env telemetry rows,
including requested/executed ego action, opponent action, joint action, reward
components, done/terminated/truncated, terminal reason, episode return, trace
hash, and optional per-step timing. This sidecar is not a substitute for
`GameSegment`; it is for audit and debugging.

## Minimal First Experiment

Goal: prove coarse fanout searched collection throughput and chunk/import
plumbing. Do not train. Do not claim learning.

Run a ladder from the same immutable checkpoint K:

```text
N actors: 1, 2, 4
per actor: collector_env_num=1 or 2, n_episode=1 or 2
fixed: env_variant=source_state_fixed_opponent
fixed: opponent_policy_kind=frozen_lightzero_checkpoint
fixed: same opponent checkpoint ref
fixed: same render mode
fixed: same num_simulations, source_max_steps, decision_ms, batch_size
```

For each N:

1. Launch N collect-only jobs with disjoint actor seeds.
2. Each actor writes one chunk directory.
3. A merge job validates manifests and concatenates native segments in actor id
   order.
4. The merge job creates a `MuZeroGameBuffer`, calls `push_game_segments`, and
   samples one batch if enough data exists.
5. The result writes one `fanout_summary.json`.

Success means:

- all actors load checkpoint K and finish collection;
- each actor writes non-empty `GameSegment` payloads and telemetry;
- merge validates compatible chunk metadata;
- `MuZeroGameBuffer.push_game_segments` accepts the merged segments;
- optional `sample(batch_size, policy)` returns current/target batch shapes;
- aggregate decisions/sec improves from N=1 to N=2 or N=4 enough to justify the
  next ladder.

Non-goals:

- no `BaseLearner.train`;
- no checkpoint publishing from the prototype;
- no background eval/GIF;
- no Coach run mutation;
- no old custom two-seat replay path.

## Likely Later Edits

Narrow code additions:

- A new collect-only Modal entrypoint under `src/curvyzero/infra/modal/`, or a
  clearly isolated mode in the existing LightZero CurvyTron Modal module.
- A small helper near the existing LightZero config builder to share the
  source-state fixed-opponent config patching without duplicating the surface.
- A collect-chunk writer/reader module under `src/curvyzero/training/`, probably
  separate from the older `replay_chunk_v0.py` because this prototype is native
  LightZero `GameSegment`, not the legacy 1v1 no-bonus array contract.
- Tests that validate manifest compatibility logic without requiring LightZero.
- Modal-only smoke that imports LightZero, writes one tiny chunk, reads it, and
  pushes into `MuZeroGameBuffer`.

Avoid touching:

- Live Coach training launch defaults.
- Stock `train_muzero` training behavior.
- The old custom `two-seat-selfplay` trainer.
- Reward semantics, `to_play` semantics, target calculation, or GameSegment
  field meanings.
- Render fidelity defaults except as explicit experiment parameters.
- Frozen opponent checkpoint selection in existing training runs.

## Prototype Measurements

Each actor should write:

- wall time: total, setup/import, checkpoint load, env manager init, policy init,
  collect, chunk serialization, chunk write;
- counts: episodes, game segments, env decisions, terminal episodes, resets,
  MCTS calls, root batch sizes, simulation budget, model initial/recurrent
  inference calls if hooks are available;
- rates: episodes/sec, segments/sec, decisions/sec, simulated nodes/sec or
  decisions * simulations/sec, chunk bytes/sec;
- env timing: render/gray64/stack, vector step, opponent action, observation
  pack, telemetry write, if `profile_env_timing_enabled` is used;
- resource: requested compute, CUDA availability/device, CPU count, peak RSS if
  easy, GPU samples if the existing sampler can be reused;
- data: payload bytes, telemetry bytes, manifest bytes, segments per chunk,
  decision count per segment, terminal rate;
- seeds: base seed, actor seed, dynamic reset seed strategy, opponent seed,
  override/repeat seeds if active.

The merge job should write:

- chunks discovered/accepted/rejected and rejection reasons;
- compatibility key table across chunks;
- read/deserialization time and bytes/sec;
- `push_game_segments` time and accepted segment count;
- optional sample time, requested batch size, and current/target batch shapes;
- aggregate actor wall time, max actor wall time, total decisions, aggregate
  decisions/sec by max actor wall time, and estimated fanout efficiency versus
  the N=1 baseline;
- data age: checkpoint K timestamp to chunk creation, chunk creation to merge.

## Open Questions Before Implementation

- Does `MuZeroCollector.collect` return exactly the same tuple/list shape in
  `LightZero==0.2.0` as the current target audit assumes for the installed
  Modal image?
- Can native `GameSegment` objects be reliably cloudpickled and read back inside
  the same Modal image, or do they carry non-portable RNG/state that requires a
  custom field-level serializer immediately?
- What is the smallest LightZero policy/env-manager construction path that
  avoids instantiating `BaseLearner` while preserving collect-mode behavior?
- Does `MuZeroGameBuffer.sample(batch_size, policy)` require a learn-mode policy
  object even for the import smoke, or can the first proof stop at
  `push_game_segments`?

Answer these in a Modal-only smoke before scaling beyond 1 actor.
