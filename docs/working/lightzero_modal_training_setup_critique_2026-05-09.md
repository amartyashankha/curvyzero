# LightZero Modal Training Setup Critique - 2026-05-09

Scope: high-level training-coach critique of the current LightZero/MuZero dummy
Pong lane, Modal wrappers, Volume/checkpoint layout, eval ladder, self-play
claims, and reward shaping docs. No pytest was run.

## Short Answer

Yes, the current minimal patterns are mostly the right ones.

The strongest thing in the setup is the boundary discipline: Modal owns whole
jobs, the training/eval hot loop stays inside one container process, and
artifacts land in `curvyzero-runs`. That is the right pattern for this stage.
Do not split env steps, MCTS calls, replay rows, or opponent inference across
Modal Functions, Queues, Dicts, or endpoints.

The weakest thing is not Modal. It is policy quality and promotion discipline.
Independent LightZero MCTS scorecards now work, strict checkpoint loading works,
and they are showing a useful negative result: current checkpoints still lose
to random or barely reach lagged tracking, never choose `down`, and do not beat
`track_ball`. Trainer-side wins and sidecar rows are useful run-health evidence,
but they are not a checkpoint-quality claim.

## Minimal Pattern Status

| Area | Critique |
| --- | --- |
| Whole-job Modal | Correct. `lightzero_dummy_pong_tiny_train_smoke.py`, `lightzero_dummy_pong_train_attempt.py`, and MCTS scorecard wrappers run coarse jobs and return compact refs. Keep this. |
| Volumes/checkpoints | Mostly correct. Training writes run/attempt manifests, summaries, sidecar episodes, logs, and mirrored `.pth.tar` payloads into `curvyzero-runs`. Missing piece: canonical LightZero `checkpoints/latest.json` and `best.json` pointers. |
| CPU/GPU | Correct for now: CPU LightZero dummy Pong is enough. Current jobs set `cuda=false` and do not request GPUs. GPU smokes are runtime evidence, not Pong training evidence. Add GPU only after CPU is compute-bound and eval is trusted. |
| Self-play truth | Be precise. Current LightZero training is ego learner vs scripted opponent, or potentially ego learner vs frozen checkpoint opponent. That is self-play-ish, not full simultaneous shared-policy self-play. |
| Frozen checkpoint | The code already has the right minimal branch: `lightzero_policy_head_checkpoint` and `lightzero_mcts_checkpoint` load once inside the env and act locally. Use MCTS eval-mode for a bounded smoke, but do not scale frozen self-play around a bad frozen opponent. |
| Evals | The current evals are not too complicated. They are exactly what exposed trainer/eval mismatch and action collapse. Keep fixed baselines plus action histograms; avoid leagues/Elo/full round robins. |
| Reward shaping | Keep env reward true: `+1/-1/0` score reward. Keep shaped loss-delay as telemetry, curriculum, and tie-breaker only. For LightZero/MuZero, do not train reward targets on shaped survival. |

## Ranked Recommendations

1. Keep the whole-job Modal pattern unchanged.
   One Modal Function should own one train attempt or one scorecard. Do not add
   actor fleets, model servers, Queues, Dicts, or per-step remotes until a
   single-container loop has an actual measured bottleneck.

2. Add LightZero checkpoint pointers before more run complexity.
   Mirrored payloads under `checkpoints/lightzero/*.pth.tar` are good, but
   downstream jobs should read `checkpoints/latest.json` and later
   `checkpoints/best.json`. Pointer files should include checkpoint ref, sha256,
   iteration/name, run/attempt ids, config ref, feature/action/reward schemas,
   eval status, and adapter expectation.

3. Treat independent MCTS scorecards as required, not optional.
   Trainer-side sidecar rows prove the training loop ran. The external MCTS
   scorecard proves whether the checkpoint acts usefully under the same
   inference boundary we intend to trust. The action histogram requirement is
   essential because current checkpoints look up/stay collapsed and never choose
   action index 2.

4. Do not spend GPU or multi-node effort on this Pong state.
   The current problem is not throughput. The current problem is learner signal,
   action symmetry/collapse, and train/eval alignment. CPU Modal is the right
   place to keep debugging this tiny MLP/tabular lane.

5. Use frozen-checkpoint opponents only as a bounded bridge.
   The env and wrapper shape are right: resolve the Volume ref before training,
   load the frozen policy once in-container, and call it inside `env.step`
   without Modal calls. First run should be a small MCTS-adapter smoke against a
   known checkpoint, mainly to verify telemetry and cost. Do not call it full
   self-play, and do not build a checkpoint pool until a selected checkpoint
   beats at least `random_uniform` and holds up against `lagged_track_ball_1`.

6. Keep the eval ladder small but honest.
   The minimum useful table is candidate vs `random_uniform`,
   `lagged_track_ball_1`, and `track_ball`, with paired seats where available,
   plus baseline sanity rows. Report wins, score return, survival steps,
   truncations, shaped loss-delay, and action histograms. Learned-vs-learned is
   secondary and only useful after fixed-baseline rows are sane.

7. Resolve the reward-shaping docs in favor of true MuZero rewards.
   `reward_shaping_for_pong_curvy_muzero.md` is the rule to carry forward for
   LightZero/MuZero: sparse env score reward feeds replay and reward targets;
   shaped loss-delay is telemetry/curriculum/tie-breaker. Older notes allowing a
   shaped training target should be read as non-MuZero toy baseline ablations,
   not the LightZero default.

8. Make the next quality investigation narrower than another scale run.
   Before longer training, inspect why independent MCTS policies do not choose
   `down`: action mapping, ego-seat mirroring, target construction, LightZero
   config patching from CartPole, exploration/temperature, and whether the
   training opponent distribution makes `down` unrewarded. More env steps will
   mostly produce larger bad checkpoints until that is understood.

## Notes By Focus Area

### Whole-Job Modal

The Modal wrappers are using the right shape:

- build a small pinned LightZero image;
- copy `src/` into `/repo/src`;
- mount `curvyzero-runs` at `/runs`;
- run the whole trainer or scorecard inside one function;
- write artifacts and summaries;
- commit the Volume once outputs are complete.

This is the same pattern recommended by the Modal architecture docs. It also
matches the hot-loop-locality decision: Modal is orchestration/storage, not the
inner RL runtime.

### Volumes And Checkpoints

The current durable payload pattern is good enough:

```text
/runs/training/lightzero-dummy-pong/<run_id>/
  run.json
  latest_attempt.json
  attempts/<attempt_id>/
    attempt.json
    config.json
    command.json
    train/summary.json
    train/episodes.jsonl
    train/lightzero_training_signals.json
    train/lightzero_artifacts_manifest.json
  checkpoints/lightzero/
    iteration_*.pth.tar
    ckpt_best.pth.tar
    manifest.json
```

The gap is pointer discipline, not storage architecture. Add:

```text
checkpoints/latest.json
checkpoints/best.json
```

Do not enable retries as real resume until those pointers correspond to a
reentrant trainer state. For now, a failed attempt should stay a failed
attempt.

### CPU And GPU

CPU is the right default for current dummy Pong because:

- model is tiny MLP/tabular;
- env is tiny;
- eval revealed action collapse and poor policy quality;
- current GPU evidence is mostly Mctx/JAX dependency or synthetic benchmark
  evidence, not LightZero Pong training evidence.

GPU becomes sensible when one of these is true:

- raster/conv LightZero training is clearly model-bound;
- MCTS simulations dominate wall time on a trusted eval path;
- one-container CPU profile shows the training loop is compute-bound;
- checkpoint/eval quality is good enough that more throughput matters.

### Self-Play Truth

Current LightZero dummy Pong is a single-ego wrapper around simultaneous Pong.
LightZero chooses one action. The env supplies the other paddle from a scripted
or checkpoint policy. That is the correct minimal adaptation to LightZero's
single-agent-ish API, but it is not full multiplayer self-play.

Use plain names:

- `ego_vs_random_uniform`
- `ego_vs_lagged_track_ball_1`
- `ego_vs_frozen_lightzero_checkpoint`
- `LightZero learner vs frozen checkpoint opponent`

Avoid names like "full self-play" or "league" for this lane.

### Frozen-Checkpoint Next Step

The code now appears to implement the planned support:

- `DummyPongLightZeroEnv` accepts opponent checkpoint fields;
- `_make_opponent_policy()` supports policy-head and MCTS checkpoint opponents;
- Modal training resolves `ref:`/`volume:` checkpoint inputs before calling the
  LightZero config patcher;
- env telemetry records opponent checkpoint refs, hashes, adapter, and
  simulations.

That is the right minimal shape. The next frozen-checkpoint step should be a
small verification run, not a scale run:

```text
learner: LightZero train_muzero
opponent: frozen LightZero MCTS checkpoint
opponent simulations: 2
caps: small CPU caps
purpose: telemetry, load/cost, no per-step Modal calls
claim allowed: frozen-opponent plumbing works
claim not allowed: self-play improved the policy
```

The stronger learning next step is still to understand why current checkpoints
cannot choose all actions and do not beat random reliably.

### Evals

The eval setup is appropriately "boring but not naive." It needs a few rows
because each catches a different failure:

- `random_uniform`: floor and stochastic sanity;
- `lagged_track_ball_1`: current scoreable pressure target;
- `track_ball`: survival/tie floor under current geometry;
- baseline-vs-baseline rows: environment drift and geometry sanity;
- action histograms: collapse detector;
- shaped loss-delay: zero-win diagnostic, not progress by itself.

Do not add leagues, Elo, or large checkpoint pools yet. The current compact
scorecard is already enough to reject bad checkpoints.

### Reward Shaping

The current env reward is right:

```text
ego scores:       +1
opponent scores:  -1
no score event:    0
```

Keep shaped loss-delay separate:

```text
win:      +1.0
loss:     -1.0 + 0.5 * episode_steps / max_steps
timeout:   0.0
```

For LightZero/MuZero, this should stay telemetry, checkpoint-selection
tie-breaker, curriculum signal, and debugging readout. It should not replace
the environment reward stream or the MuZero reward target.

## Sources Read

- `docs/working/modal_lightzero_training_pattern_2026-05-09.md`
- `docs/working/modal_training_setup_critique_2026-05-09.md`
- `docs/working/lightzero_pong_frozen_checkpoint_selfplay_plan_2026-05-09.md`
- `docs/working/lightzero_dummy_pong_eval_ladder_2026-05-09.md`
- `docs/research/reward_shaping_for_pong_curvy_muzero.md`
- `docs/research/pong_reward_design.md`
- `docs/design/modal_training_run_management.md`
- `docs/research/modal_training_patterns.md`
- `docs/design/muzero_modal_architecture.md`
- `docs/research/lightzero_integration.md`
- `docs/research/training_evaluation.md`
- `docs/experiments/2026-05-09-lightzero-dummy-pong-implementation-inspection.md`
- `docs/experiments/2026-05-09-lightzero-dummy-pong-mcts-scorecard.md`
- `docs/experiments/2026-05-09-lightzero-dummy-pong-post-deep-seed-fix-run.md`
- `src/curvyzero/training/lightzero_dummy_pong_env.py`
- `src/curvyzero/training/lightzero_dummy_pong_policy.py`
- `src/curvyzero/training/dummy_pong.py`
- `src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py`
- `src/curvyzero/infra/modal/lightzero_dummy_pong_tiny_train_smoke.py`
- `src/curvyzero/infra/modal/lightzero_dummy_pong_train_attempt.py`
- `src/curvyzero/infra/modal/lightzero_dummy_pong_mcts_scoreboard_attempt.py`
- `src/curvyzero/infra/modal/run_management.py`
