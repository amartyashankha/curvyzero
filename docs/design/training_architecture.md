# Training Architecture

Status: Draft

Training should wait on the simulator until the environment has deterministic
tests, source-based rule notes, and baseline evidence that learning is possible.

Current Pong status, 2026-05-09: the local self-play loop is an inspectable
scaffold, not a proven strategy. Generation 2 lost to its parent and won 0
games against `track_ball`. The next architecture decision is whether to repair
that crude trainer or switch to a known simple baseline/curriculum, with
fixed-baseline evals first.

Execution target: local runs are for tiny debug only. Serious train/eval jobs
should run on Modal as whole jobs and write durable artifacts.

## v0 Sequence

1. Random-agent stress test.
2. Heuristic agent versus random.
3. Dummy single-player loop with checkpoint and eval output files.
4. Dummy simultaneous 1v1 loop with ego-perspective replay rows.
5. Fixed baseline eval tables for dummy tasks.
6. Learned-checkpoint eval against those fixed tables.
7. Synthetic MCTS/MuZero benchmark.
8. Minimal searched self-play only after the earlier gates are stable.

Current runnable scripts and Modal entry points:

- `scripts/run_dummy_survival_train.py`
- `scripts/run_dummy_survival_eval.py`
- `scripts/run_dummy_survival_checkpoint_sweep.py`
- `scripts/run_dummy_survival_selection_holdout.py`
- `scripts/run_dummy_line_duel_train.py`
- `scripts/run_dummy_line_duel_eval.py`
- `scripts/run_dummy_pong_eval.py`
- `scripts/run_dummy_pong_observability.py`
- `scripts/build_dummy_pong_imitation_replay.py`
- `scripts/build_dummy_pong_scoring_replay.py`
- `scripts/inspect_dummy_pong_artifacts.py`
- `scripts/train_dummy_pong_imitation.py`
- `scripts/train_dummy_pong_value.py`
- `curvyzero.infra.modal.dummy_survival`
- `curvyzero.infra.modal.dummy_line_duel`

Both eval CLIs can now load explicit dummy `checkpoint.npz` policies via
`--checkpoint-policy learned:path/to/checkpoint.npz`. Current checkpoints are
early test outputs, not quality baselines.

## Saved Outputs

Every training or eval job should produce a compact machine-readable summary
and enough row-level data to debug surprises later.

Training job minimum:

- `summary.json`: config, seed, schemas, final eval, model summary, saved-file refs.
- `checkpoint.npz` or later checkpoint directory: model state plus metadata.
- `iteration_metrics.jsonl`: one row per training iteration.
- replay rows/chunks when the job creates replay data.

Eval job minimum:

- `summary.json`: eval config, seed set, policy/opponent specs, aggregate table.
- `episodes.jsonl`: one row per episode with outcome, length, actions, and
  failure/death causes when available.
- New Pong training runs should also emit run-health output: iteration metrics,
  action histograms by seat, entropy/collapse metrics, terminal causes, failure
  examples, and heldout results after selection.

Selection/eval rule additions:

- `eval_split`: split id/role, seed count, seed-list hash, paired-seat flag.
- Planner-only or untrained-model-same-planner baselines for learned-checkpoint
  claims.
- `selection_record.json` for best-checkpoint sweeps.
- `latest_checkpoint` and `selected_checkpoint` visible together.
- Heldout confirmation before treating a selected checkpoint as a quality result.

Visual-observation debug files:

- Pong sidecar traces write `frames.jsonl` with tiny raster grids. Those frames
  are the intended MuZero-facing Pong observation path; tabular Pong fields are
  only debug/eval helpers.
- Pong imitation replay writes learner-ready raster rows. The first supervised
  checkpoint can copy `track_ball` in replay and can now run in eval. It is
  still weaker than scripted `track_ball`, so score-bearing replay and stronger
  learned eval are still needed before any reward-learning claim.
- Pong scoring replay can emit all ego rows with positive and negative terminal
  rewards. That makes it useful for value-target checks. Random-policy rows
  should not be treated as expert policy targets.
- Pong value-target training now backs up score-delta rewards into scalar
  returns and writes a reloadable value checkpoint. Treat this as target
  plumbing until a policy or search loop uses it to choose better actions.
- Pong paddle contacts expose a small strategy signal: top, center, and bottom
  hits send the ball up, straight, or down. The next Pong eval should measure
  whether a policy can use off-center hits to score against `track_ball`.
- The first angle-control probe can force off-center contacts and beat random,
  but it still only times out against `track_ball`. The useful next design is
  a simple checkpoint scoreboard before full MuZero-style search. The
  contact-outcome dataset is observability for explaining scoreboard failures.
- Pong self-play replay/train/eval exists locally, but gen2 failed the parent
  and `track_ball` gates. Keep generation and promotion code as guardrails only;
  do not add leagues or scale-out until a simple learner improves for a clear
  reason.

Modal jobs should return the compact summary and saved-file references. Durable
remote storage exists as a tiny `curvyzero-runs` Volume smoke for dummy survival.
A Modal `Volume` is Modal's persistent shared file storage for remote jobs. Real
training storage still needs run/attempt ids, latest pointers, and resume
behavior. Short-lived remote output paths remain acceptable for small
import/summary smokes.

## Observation And Reward Defaults

- v0 observation target: ego-centric ray features for heuristic, imitation, and PPO experiments.
- v1 observation target: ego-centered heading-aligned local raster for CNN/MuZero experiments.
- v0 reward: sparse terminal 1v1 win/loss/draw.
- Same-tick 1v1 deaths are draws unless a later source-derived rule changes this.
- Avoid shaping rewards until a concrete failure mode justifies a new reward schema.

## Multiplayer Formulation

- Shared model trained from ego perspective.
- 1v1 first.
- Scalar payoff first: win/loss for 1v1, centered rank payoff for multiplayer.
- Rotate ego seat for multiplayer data.
- Use checkpoint-pool or policy-only opponents before all-player MCTS.

## MuZero Components To Design Later

- Representation network: observation to hidden state.
- Dynamics network: hidden state plus action to next hidden state and reward.
- Prediction network: hidden state to policy logits and value.
- Search wrapper: MCTS over latent dynamics.
- Replay buffer: observations, actions, rewards, search policies, values, ego ids, model/checkpoint ids, rules hashes.
- Trainer: target construction, losses, checkpointing, evaluation.

## Library Spike Order

- Preferred search spike: JAX/Mctx with `gumbel_muzero_policy` and fixed-shape synthetic benchmarks.
- Fallback spike: PyTorch/LightZero with an ego-agent wrapper and scripted/frozen opponents.
- Stable boundary: both libraries adapt to `curvyzero.env`; neither defines the simulator core.

## Open Questions

- Exact observation format for the first PPO baseline.
- Whether MuZero v0 should use compact ray features or raster observations.
- How many simulations per decision are feasible after the Mctx benchmark.
- How to batch MCTS across many envs while keeping opponents cheap.
