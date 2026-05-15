# Holistic Bottleneck Risk Register

Date: 2026-05-15

Purpose: keep the optimizer lane from becoming render-only. Current production
policy observations are CPU `cpu_oracle`:

```text
browser_lines + simple_symbols -> stacked [4,64,64]
```

The lab GPU renderer is promising after stronger frame parity work, but it has
not passed trainer-contract parity or proven a stock `train_muzero` speedup.
The risks below are ranked by expected ability to invalidate an optimization
recommendation.

## Ranked Risks And Falsification Experiments

1. **The stock LightZero loop is synchronous enough that render wins can vanish
   behind collector/search orchestration.**

   Evidence: `train_muzero` alternates collect, replay push/sample, and learner
   updates in one process. Recent full-loop profiles show collection, policy
   collect, and MCTS remain large wall buckets after env work is subprocessed.

   Experiment: run profile-only C32/C64/C128 rows with `cpu_oracle`,
   `browser_lines + simple_symbols`, `sim8`, and telemetry stride `32`, then
   repeat with an observation-stub/mock boundary that preserves tensor shape but
   bypasses render. If the stub row gains less than expected from worker render
   time, chase collector/MCTS/IPC before more renderer kernels.

2. **Frozen checkpoint opponents are scalar, per-env policy/search workers,
   not a shared batched opponent service.**

   Evidence: each source-state env owns/caches an opponent policy and calls
   `select_actions` for a single opponent row during `env.step`. With
   subprocess envs and `opponent_use_cuda=false`, this cost is mostly hidden
   inside worker env timing, not parent policy/MCTS timers.

   Experiment: matched profile rows at C64 and C128:
   `fixed_straight`, `proactive_wall_avoidant`, and one frozen checkpoint
   opponent, all with identical render/reward/death settings. Compare
   `opponent_action_sec`, `env_step_sec`, parent `mcts_search_sec`, and env
   steps/sec. If frozen is materially slower, prototype an opponent-inference
   microbatch/service before widening more collectors.

3. **The GPU renderer integration boundary is the optimization, not just the
   kernel.**

   Evidence: the scalar `jax_gpu` env hook reaches the stock trainer but is
   slower than CPU and fails in subprocess worker shape. The fast H100 rows are
   batched renderer economics, not stock trainer economics.

   Experiment: build a local/profile-only mock collector boundary: many source
   env rows step CPU physics, gather compact state, render both player views in
   a batched backend, update `[B,P,4,64,64]` stacks, run a policy-forward stub,
   and copy a replay-like payload. Compare against CPU oracle for B64/B256.
   Promote only if this boundary wins, not if the isolated kernel wins.

4. **Replay/GameBuffer invariants may be a hidden fragility at large width or
   long horizons.**

   Evidence: previous rows hit LightZero `np.random.choice(..., p=probs)` size
   mismatches. Current fixed-opponent code no longer patches `td_steps` to
   `source_max_steps`, but stale notes and long/no-death rows show this class
   of failure can masquerade as speed or run-health noise.

   Experiment: passive invariant hook around `push_game_segments` and `sample`:
   record transition count, priority vector length, lookup length, segment
   lengths, terminal flags, `game_segment_length`, `num_unroll_steps`, and
   `td_steps` before and after every replay call, including exceptions. Run
   profile-only short/long horizon rows; fail the cell on the first mismatch.

5. **`[4,64,64]` may be an information bottleneck, while `96x96` may be the
   first plausible larger grid.**

   Evidence: 64x64 keeps source-canvas downsample semantics but gives trails
   less than one final-grid cell of width. Renderer-only H100 rows make 96x96
   plausible, but model/root inference, IPC, replay, and LightZero branch
   support are unmeasured.

   Experiment: local-only shape branch for 64 and 96 on the same adversarial
   visual corpus: head/trail separation, parallel trails, bonus-on-trail, wall
   clipping. Pair it with root initial-inference and replay-copy profiling.
   Only then run a tiny matched learning sentinel.

6. **The inherited Atari conv/SSL model may be paying for representation
   machinery CurvyTron does not need.**

   Evidence: the launcher patches Atari MuZero to `model_type=conv`,
   `image_channel=4`, `frame_stack_num=1`, `observation_shape=[4,64,64]`, and
   keeps `self_supervised_learning_loss=True`. Learner time is small in short
   rows, but root initial inference sits inside collect/search and scales with
   observation shape.

   Experiment: profile-only config variants that keep checkpoint compatibility
   separate: SSL on/off, smaller support/head settings where compatible, and a
   96 branch if added. Measure model initial/recurrent inference, learner time,
   GPU memory, and root batch. Treat quality as unknown until paired learning
   rows exist.

7. **Subprocess IPC and Python object copying can erase observation-backend
   wins, especially as width or resolution grows.**

   Evidence: the env returns dict observations with copied float32 stacks;
   subprocess env manager serializes these across process boundaries. Bigger
   observations multiply env payload, replay storage, and learner batch bytes.

   Experiment: base vs subprocess profile rows with a render-stub observation,
   then CPU oracle, then batched GPU mock. Record env-manager wait/step,
   stack-copy time, payload byte size, replay push/sample, and root batch size.
   If subprocess overhead dominates, prioritize parent-side vector facade or
   shared-memory/tensor handoff before higher resolution.

8. **Tournament/checkpoint metadata can silently make speed results
   incomparable.**

   Evidence: string checkpoint specs default to the current policy surface;
   richer specs and checkpoint/ref metadata try to recover trail/bonus surface,
   model env variant, reward variant, and state key. Past tournament notes
   already found timing/render mismatches.

   Experiment: a no-training parity audit over sampled checkpoints: load via
   trainer eval, tournament game, and GIF/eval path, then compare policy
   surface, learner seat/perspective, reward/model variant, action mask,
   `decision_ms`, `source_physics_step_ms`, state key, and observation hashes
   on the same fixed state. Refuse optimizer comparisons missing these fields.

9. **Higher collector width can improve steps/sec while making the policy
   update older and less useful.**

   Evidence: stock `train_muzero` collection and learning are not concurrent.
   Very wide `n_episode=collector_env_num` rows collect many complete episodes
   before learner updates, so throughput can rise while policy freshness falls.

   Experiment: matched wall-clock rows at C64/C128/C256/C384 with identical
   seeds and evaluation cadence. Record policy-version age per collected
   transition, replay age at sample time, episodes per collect, learner updates
   per env step, and held-out survival. Keep wide rows as speed-only until this
   curve is known.

10. **Closed-loop tournament scaling can become the practical bottleneck even
    if training gets faster.**

    Evidence: all-pairs grows quadratically, volume scans have already been
    operationally painful, and checkpoint intake/rating metadata must stay
    consistent with training surfaces.

    Experiment: replay existing checkpoint metadata through synthetic rating
    schedules: all-pairs, adaptive top-band, random-near-rating, and fixed
    anchor sentinels. Compare wall estimate, coverage, rating stability, and
    ability to pick the same top-K. Optimize this before making checkpoint
    cadence much more aggressive.

11. **Telemetry, checkpoint cadence, and volume inode pressure can distort both
    profiles and long runs.**

    Evidence: profile docs saw `/runs` inode pressure, and env telemetry writes
    JSONL from every worker unless stride is raised. Checkpoint/GIF/eval work
    is correctly disabled in profiles but still part of real closed-loop cost.

    Experiment: matched C64/C128 profile rows with telemetry stride `1`, `32`,
    `128`, and telemetry disabled if supported; then checkpoint cadence
    `10k/50k/off-profile`. Record wall, env telemetry write time, artifact
    scan time, inode counts, and checkpoint mirror time.

## Working Verdict

Keep CPU `browser_lines + simple_symbols` as the production contract until a
candidate passes trainer-visible parity. The next highest-value optimizer work
is not another isolated renderer timing row; it is a boundary test that counts
opponent policy cost, stack/final-observation semantics, subprocess transport,
policy/root inference, replay copying, and metadata comparability in one
profile-only harness.
