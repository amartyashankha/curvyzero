# MuZero On Modal Architecture

Status: Design map
Last updated: 2026-05-09

## Short Answer

Start LightZero-first for the immediate dummy Pong MuZero spike. The first
implementation gate and two Modal jobs should be:

0. a LightZero feature-fit audit that exposes reset/step, observation shape,
   legal action handling, custom reward/info telemetry, trainer entrypoint fit,
   checkpoint discovery, and independent CurvyZero scorecard path before
   training;
1. a LightZero custom-env config/import smoke that imports, configures,
   creates, resets, and steps the dummy Pong ego-view env without training;
2. a brutally capped LightZero dummy Pong MuZero train smoke that calls
   LightZero's real trainer, writes checkpoint/log evidence, and mirrors a
   CurvyZero scorecard summary from an independent scorecard path.

Project-owned JAX/Mctx is third: fallback/comparison only if LightZero cannot
handle the custom env, hides required telemetry, or grows into framework
surgery.

The first useful system is not a model server, a web endpoint, a live actor
fleet, or a multi-node cluster. It is a boring resumable train attempt that
writes checkpoints, replay chunks, metrics, and eval summaries to a Volume.

## Current Truth / No More Pretending

- We validated stock LightZero CartPole MuZero progression.
- We validated an Mctx search benchmark.
- We validated a CEM-v2 Pong baseline.
- We validated a raster-only MLP Pong baseline.
- We have not run an actual project-owned MuZero/Mctx train loop for Pong.
- We have not run an actual project-owned MuZero/Mctx train loop for Curvy.
- CEM-v2 and the MLP are baselines and scaffolding only. They are not MuZero
  progress.
- The next main lane is LightZero-first on a custom dummy Pong env: config/import
  smoke first, tiny trainer smoke second.
- Project-owned Mctx is fallback/comparison, not the next lane.

Prevention rules:

- Prove the target is scoreable before scaling it.
- Keep baselines separate from MuZero.
- Name the algorithm in every experiment title, command, and summary.
- Distinguish stock LightZero MuZero from project-owned MuZero/Mctx.
- Do not describe CEM, imitation, or MLP results as MuZero progress.

## Current Code Reality

The repo does not yet contain a full MuZero trainer. Current pieces are:

- `src/curvyzero/env/core.py`: deterministic 1v1 `curvyzero-v0` toy
  environment with fixed actions, terminal rewards, and simple state
  observations.
- `src/curvyzero/training/dummy_pong.py`: small two-player Pong-like visual toy
  with simultaneous actions, tabular ego observations, and tiny raster grids.
- `src/curvyzero/training/dummy_pong_selfplay_replay.py`: self-play replay
  generation that emits one row per ego agent at each step.
- `src/curvyzero/training/dummy_pong_selfplay_train.py`: tiny NumPy
  policy/value trainer. This is not MuZero search.
- `src/curvyzero/infra/modal/run_management.py`: run ids, attempt manifests,
  checkpoint pointers, Volume refs, and file summaries.
- `src/curvyzero/infra/modal/dummy_pong_train_attempt.py`: one coarse CPU Modal
  train-attempt wrapper for the current Pong scaffold.
- `src/curvyzero/infra/modal/dummy_pong_cem_train_attempt.py`: one coarse CPU
  Modal wrapper for the CEM-v2 Pong baseline. It is not MuZero.
- `src/curvyzero/infra/modal/dummy_pong_imitation_train_attempt.py`: one
  coarse CPU Modal wrapper for the raster-only MLP imitation baseline. It is
  not MuZero.
- `src/curvyzero/infra/modal/dummy_pong_scoreboard_attempt.py`: one coarse CPU
  Modal eval/scoreboard wrapper.
- `src/curvyzero/infra/modal/mctx_dependency_smoke.py` and
  `mctx_gpu_dependency_smoke.py`: tiny synthetic Mctx/JAX import and search
  smokes, not trainers.
- `src/curvyzero/infra/modal/mctx_synthetic_benchmark.py`: one synthetic GPU
  Mctx benchmark lane. It is not connected to Pong training yet.
- `src/curvyzero/infra/modal/lightzero_cartpole_tiny_train_smoke.py`: the only
  already-run actual MuZero trainer lane. It uses stock LightZero CartPole,
  not CurvyZero Pong, and its `--mode progression` run is the smallest real
  MuZero training smoke we can rerun next.
- `src/curvyzero/infra/modal/lightzero_pong_env_smoke.py`: stock LightZero
  Pong environment creation smoke, not training. It currently reaches the ALE
  ROM/license gate; after explicit license approval, unblock with an
  AutoROM-enabled image before any stock Pong `train_muzero_segment` attempt.

That means the next architecture work should add the LightZero custom dummy
Pong adapter smokes, not a project-owned Mctx trainer first and not more
baseline work mislabeled as progress.

Near-term decision: keep stock LightZero CartPole as reference evidence, then
try LightZero on a CurvyZero-owned dummy Pong env. The order is config/import,
then tiny train. Do not route CEM, supervised MLP, synthetic Mctx benchmark, or
project-owned trainer scaffolding through the LightZero MuZero evidence bucket.

## Simplest MuZero Pieces

A minimal MuZero-style system has these parts:

| Piece | Job | v0 placement |
| --- | --- | --- |
| Real environment | Produces observations, rewards, dones, and legal action masks. | Inside the training container. |
| Self-play loop | Runs many episodes, asks the policy/search for actions, and writes replay rows. | Inside the training container. |
| Model | `representation(obs)`, `prediction(hidden)`, and `dynamics(hidden, action)`. | GPU process for JAX/Mctx; CPU is acceptable for tiny smoke toys. |
| Search wrapper | Calls Mctx over learned dynamics and returns action, visit/action weights, root value, and diagnostics. | Same process as the model. |
| Replay buffer | Holds recent chunks for sampling and writes durable chunks. | In-process ring buffer plus chunked files on Volume. |
| Trainer | Samples replay, builds unroll targets, computes losses, updates model and optimizer. | Same container in v0. |
| Checkpointer | Writes model, optimizer, trainer cursor, RNG/cursors, config hashes, and latest pointer. | Volume writes from the training Function. |
| Eval | Scores checkpoints against fixed seeds and fixed opponents. | Same Function first; separate coarse eval Functions later. |
| Launcher | Starts train/eval jobs and returns refs. | Local CLI or deployed Modal Function lookup. |

The important boundary is simple: Modal starts and stores whole jobs; the
container owns the fast loop.

## One-Container v0

The first MuZero train attempt should run like this:

```text
Modal Function: muzero_train_attempt(config)
  reload runs Volume
  load checkpoint or initialize model/trainer state
  create env batch and in-memory replay ring
  while not done:
    collect self-play episodes or rollout fragments
      build fixed-size observation batches
      run representation/prediction/search on GPU
      step real envs on CPU
      append replay rows in memory
    train for N minibatches on replay
    periodically write replay chunks, checkpoint payloads, and metrics
    publish latest pointer after payload files exist
    commit Volume after material checkpoints/chunks
  run small final eval or enqueue/launch coarse eval later
  write summary and commit
  return compact refs
```

This layout is good enough until measurement proves one part is too slow.

Do not split v0 into separate self-play actors, a model server, a trainer
consumer, and live queues. That split makes sense only after the one-container
run shows a concrete bottleneck.

## CPU And GPU Placement

Likely CPU-heavy work:

- Real environment stepping, collision/trail updates, and observation building.
- Replay serialization, compression, and manifest writing.
- Lightweight scripted or frozen policy opponents.
- Small CPU eval scoreboards and toy Pong reproduction runs.
- The current Pong learner and scoreboard, because they are NumPy/scripted
  CPU jobs.

Likely GPU-heavy work:

- Representation/prediction/dynamics forward passes.
- Mctx tree search over many ego rows.
- Model unroll training and optimizer updates.
- Large batched inference for eval once checkpoints are nontrivial.

Do not assume too early. The environment performance notes show that trail and
collision work can dominate even before the real source-fidelity backend is
complete. The Mctx notes show hidden-state size and number of simulations can
dominate GPU memory. The first benchmark should report both env steps/sec and
search/train throughput.

Current placement is blunt: the existing Pong Modal train/eval path is real
Modal execution, but it is CPU. GPU should start only with the first JAX/Mctx
or GPU-model learner, not by moving the current NumPy trainer.

## Modal Primitives

| Modal primitive | Use it for | Do not use it for |
| --- | --- | --- |
| Function | Whole train attempts, self-play shards, eval shards, replay conversion, dependency smokes, benchmarks. | One action, one env step, one MCTS node, one optimizer batch. |
| Volume | Checkpoints, replay chunks, summaries, metrics, videos, run manifests, latest/best pointers. | Millions of tiny files or concurrent writers touching the same file. |
| Cache Volume | JAX/torch/HF/xdg caches when dependencies benefit from it. | Irreplaceable experiment records. |
| Queue | Coarse work items such as "eval checkpoint X on seed shard Y" or "convert replay chunk Z". | Replay storage, per-step actions, per-node search, hot inference calls. |
| Dict | Tiny coordination state: latest checkpoint ref, run heartbeat, lease records, compact status. | Model weights, replay buffers, batch data, counters updated in hot loops. |
| CloudBucketMount or object storage | Long-lived replay archives and large immutable exports after Volume storage is too small. | Append-heavy hot logs or random-write checkpoint files. |
| Web endpoint | Demo, human inspection, remote policy API, or stable external consumer after a checkpoint is worth serving. | v0 self-play/training action selection. |

Volume write rule: write payloads first, then pointer manifests. Existing
containers must reload before reading files committed by other containers.

## What Goes In The Volume

Recommended active-run layout:

```text
/runs/training/muzero/<run_id>/
  run.json
  latest_attempt.json
  attempts/<attempt_id>/
    attempt.json
    train/
      summary.json
      metrics.jsonl
      replay/
        shard-000000/
          part-000000.npz
          manifest.json
      logs/
  checkpoints/
    iteration-000001000/
      model.safetensors or checkpoint.npz
      optimizer.npz
      trainer_state.json
      metadata.json
    latest.json
    best.json
  eval/<eval_id>/
    summary.json
    episodes.jsonl
    videos/
```

Checkpoint state should include enough to resume: model weights, optimizer
state, completed training step, replay cursor or replay chunk manifest, RNG
state or deterministic seed cursors, config, rules hash, observation schema
hash, reward schema hash, action schema, search config hash, code/dependency
fingerprint, and parent checkpoint refs.

Replay chunks should be chunky immutable files. Store data rows or arrays, not
one file per step.

## Queues And Dicts

Queues and Dicts are coordination tools. They should carry refs and small
status, not bulk data.

Good queue items:

```json
{"kind": "eval", "checkpoint_ref": "...", "seed_start": 1000, "seed_count": 128}
{"kind": "convert_replay", "chunk_ref": "..."}
```

Good Dict entries:

```json
{"run_id": "...", "latest_checkpoint_ref": "...", "updated_at": "..."}
{"actor_id": "selfplay-003", "heartbeat_at": "...", "last_chunk_ref": "..."}
```

Bad uses:

- `queue.put(action)` every tick.
- `dict["weights"] = checkpoint_payload`.
- `dict["replay_row_123"] = row`.
- one Function call per environment step.
- one endpoint call per MCTS expansion.

## Why No Model Server In v0

A model server helps when there are many separate clients that need low-latency
inference from the same checkpoint: demos, evaluation services, human play,
external bots, or many CPU self-play actors sharing one GPU.

It is not v0 because self-play plus search needs tight batching and repeated
calls between model, search, and environment. Crossing a web endpoint or Modal
Function boundary for action selection would add network latency and make
replay/search debugging harder. In v0, put the model beside the search and env
batch in one container.

Add a server only after:

- a checkpoint is worth demonstrating or evaluating interactively;
- CPU self-play actors are clearly starved on GPU inference;
- batching through a dedicated inference service beats local in-container
  batching in measured throughput;
- checkpoint loading/warm serving latency matters more than training
  simplicity.

## Simplest Pong And CurvyTron Toy Layout

For current Pong:

- Use the existing CPU Modal `dummy_pong_train_attempt` and
  `dummy_pong_scoreboard_attempt` only as artifact and scoreboard scaffolds.
- Use CEM-v2 and raster-only MLP only as baselines. Do not call them MuZero.
- Do not call the current Pong scaffold MuZero. It creates replay, trains tiny
  NumPy policies or baselines, writes checkpoint refs, and lets the scoreboard
  judge policy quality.
- The next useful Pong step is learner quality and run-health observability, not
  GPU scale.
- Wins against default `track_ball` are not the current hard gate, because the
  current default geometry appears to make scoring from normal resets
  impossible. Treat default `track_ball` as a survival/tie floor. Use scoreable
  ladder opponents such as `lagged_track_ball_1` for win pressure.
- While wins are still zero, do not treat 0/64 by itself as the whole diagnosis.
  The shaped-learning signal must include survival/loss delay: mean episode
  length, p90/max survival, survival variance, truncation rate, score return,
  and shaped return proxy against `track_ball`, plus no collapse against
  `random_uniform`.
- Variance is not environment reward. It can be a small early exploration or
  checkpoint-selection tie-breaker when mean score/survival is similar.

For first dummy Pong MuZero:

- Use LightZero first, not project-owned Mctx.
- Add a custom ego-view dummy Pong env adapter with fixed action count `A=3`.
- Run the config/import smoke before training: import framework and adapter,
  patch config, create/reset/step the env, and return JSON diagnostics.
- Then run the tiny LightZero trainer smoke on Modal with CPU caps:
  `max_env_step=64`, `max_train_iter=2`, `num_simulations=2`, `batch_size=8`.
- Keep LightZero artifacts as LightZero artifacts, but mirror a CurvyZero
  summary into `curvyzero-runs`.
- Eval/report against `random_uniform`, default `track_ball` as a survival/tie
  floor, and `lagged_track_ball_1` as the scoreable ladder opponent when the
  smoke reaches eval.
- Keep the same shaped-learning telemetry. LightZero MuZero does not get a
  pass because it uses search; it still needs wins plus survival/loss-delay
  movement.
- Keep the algorithm label honest: this is `LightZero MuZero custom dummy
  Pong`, not `project-owned MuZero/Mctx`.
- Use project-owned Mctx only if LightZero cannot preserve the custom env
  contract, trace metadata, scorecard telemetry, or artifact mapping.

For first CurvyTron toy:

- Use `curvyzero-v0`, not source-faithful CurvyTron, and label it honestly.
- Start 1v1, fixed `A=3`: left, straight, right.
- Use ego-perspective observations and one replay row per live ego player.
- Keep opponents policy-only or same-checkpoint policy-only at first.
- Search either one focal ego or both players as independent ego rows. Do not
  start with full joint-action search.

## Deterministic First

The first LightZero custom dummy Pong smoke should use deterministic rules:
no bonuses, no random trail holes, no random item spawns, fixed action shape,
explicit seeds, and a policy-only opponent such as `random_uniform` or
`lagged_track_ball_1`.

If LightZero fails and we fall back to a project-owned MuZero/Mctx trainer,
that fallback should also start deterministic and use standard deterministic
MuZero or Gumbel MuZero.

Stochastic MuZero is a later branch, not v0. It becomes relevant when planning
must branch over chance events: random bonuses/items, random boosts/hazards,
random trail gaps, hidden/noisy transitions, or opponent action uncertainty that
cannot be handled by opponent metadata and policy-only rollout. The branch adds
afterstates, chance outcomes, chance-node search, and extra targets, so it
should wait until deterministic MuZero has a measured failure on a stochastic
ruleset.

This does not change reward or eval policy. Keep native outcome rewards and
heldout eval metrics. If stochastic rulesets arrive, add random-stream and
chance-event logs plus enough heldout episodes to separate policy improvement
from lucky item/gap rolls.

See `docs/research/stochastic_muzero.md` for the decision note.

## Later Scaling Layout

Scale only after one-container training has clear metrics and stable artifact
schemas.

Later layout:

```text
selfplay_shard Function(s), CPU or GPU
  load latest or assigned checkpoint
  run many envs locally
  write immutable replay chunks to Volume/bucket
  publish chunk manifests

trainer Function, GPU
  reload replay manifests
  sample chunks into local replay window
  train model
  write checkpoints and latest pointer

eval_shard Function(s), CPU or GPU
  load checkpoint and opponent refs
  run seed shard
  write eval shard summary

aggregator Function
  read eval shard summaries
  write scoreboard and best pointer
```

Use `Function.map` or `starmap` for eval shards and replay conversion. A Queue
is only needed if live producers and consumers are simpler than launching
bounded shard lists.

Replay scaling path:

1. In-memory replay ring plus Volume chunks in one training Function.
2. Separate coarse self-play shards that write immutable chunks.
3. Trainer reads chunk manifests and maintains a local sampling window.
4. Promote old immutable chunks to object storage when Volume size or retention
   becomes a problem.

Checkpoint scaling path:

1. Latest pointer and periodic checkpoints.
2. Best pointer chosen by fixed eval.
3. Checkpoint pool for opponent sampling.
4. Lightweight league or population only after fixed-pool eval shows cycling or
   overfitting.

Multi-node training is later than all of this. It adds distributed rank setup,
network failure handling, and checkpoint complexity before we have a learner
worth scaling.

## Multiplayer And CurvyTron Changes

CurvyTron source behavior is real-time multiplayer control state over elapsed-ms
frames. The simplest MuZero formulation still needs an explicit wrapper
compromise.

Use a shared ego-perspective model:

- Every player uses the same network.
- Observation is from the ego player's frame.
- Value is scalar ego return.
- Replay has one row per live ego player per decision.
- Opponents are policy-only agents sampled from latest, recent, fixed, random,
  or heuristic checkpoints.

For 1v1 `A=3`, full wrapper joint action already gives `3 * 3 = 9` actions per
decision.
For more players it becomes `3^N`, which grows too fast. That is why v0 should
avoid full joint-action search.

Recommended order:

1. Policy-only self-play with ego rows.
2. Focal-ego search while opponents act from policy-only checkpoints.
3. Search both 1v1 players as independent ego rows for data collection.
4. Try joint-action or all-live-player search only as a bounded research
   experiment.
5. Add checkpoint pools before any heavy league machinery.

Multiplayer replay needs more metadata than Pong:

- `ego_player_id`
- `num_players`
- `joint_action`
- `alive_mask`
- `death_tick`
- `death_cause`
- opponent policy versions
- terminal rank/payoff
- rules, observation, reward, action, and search config hashes

## Open Questions

- After the LightZero tabular dummy Pong smoke, should the next observation
  step be flattened Pong raster, tiny conv raster, `curvyzero-v0`, or a
  smaller single-player toy?
- Which observation should the first CurvyTron MuZero model use: compact rays,
  local raster, or both behind separate schema ids?
- What fixed Mctx profile is affordable on an L4: batch size, hidden shape,
  simulations, and max depth?
- How fast is environment stepping after source-fidelity mechanics and
  observation generation are included?
- Should v0 replay store full observations, compressed raster chunks, or refs
  into episode arrays?
- What is the first resume contract: latest checkpoint only, or checkpoint plus
  replay window and trainer RNG/cursors?
- How should fixed eval split ids be shared across Pong, `curvyzero-v0`, and
  future multiplayer CurvyTron?
- When does a model server beat in-container inference for self-play actors?

## Next Experiments

1. Add the LightZero feature-fit audit gate, either as
   `lightzero_dummy_pong_feature_fit_smoke` or as
   `lightzero_dummy_pong_config_import_smoke --mode feature-fit`. It must report
   reset/step, observation shape/schema, legal action mask or all-actions-legal
   note, reward/info telemetry, trainer entrypoint fit, checkpoint discovery
   plan, and independent CurvyZero scorecard path.
2. Add `lightzero_dummy_pong_config_import_smoke`: pinned LightZero image,
   custom dummy Pong adapter import, tiny MuZero config patch, create/reset/step,
   feature-fit report, and JSON diagnostics. It must not train.
3. Add `lightzero_dummy_pong_tiny_train_smoke`: same adapter, real LightZero
   MuZero trainer call, tiny caps, LightZero checkpoint/log scan, and CurvyZero
   scorecard summary.
4. Use project-owned `mctx_known_env_tiny_train_smoke` only as fallback or
   comparison after the two LightZero probes.
5. Keep the GPU `mctx_synthetic_benchmark(config)` Function as benchmark
   support only. It is search/runtime evidence, not training evidence.
6. Define larger replay/checkpoint resume schemas only after the LightZero
   custom-env smoke writes a useful tiny checkpoint/eval result or fails with a
   clear reason.

## Local References

- `docs/research/muzero_architecture_deep_dive.md`
- `docs/research/mctx_integration.md`
- `docs/research/stochastic_muzero.md`
- `docs/research/multiplayer_selfplay_muzero.md`
- `docs/research/modal_patterns.md`
- `docs/research/modal_training_execution_plan.md`
- `docs/design/training_architecture.md`
- `docs/design/modal_architecture.md`
- `docs/research/environment/performance_vectorization_plan.md`
