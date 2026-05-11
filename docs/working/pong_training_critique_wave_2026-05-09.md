# Pong Training Critique Wave - 2026-05-09

Status: active critique wave.

## Why This Exists

The current Pong loop may be too narrow. We proved a local self-play path, but
we may be overfitting our thinking to "generations" and "promotion gates"
before proving that this is the best way to get a useful toy training signal.

Treat the current setup as a hypothesis, not a decision.

Plain correction: generation 2 lost to its parent and won 0 games against
`track_ball`, and the later beatability probe shows default `track_ball` was
an impossible hard target from normal resets. Generations and promotions are
guardrails, not the main strategy.

## Current Hypothesis Under Review

The current local hypothesis is:

```text
self-play replay -> shaped return -> policy/value update -> checkpoint eval
```

This loop is real locally, but the first trainer is weak:

- first generation only tied random and lost to `track_ball`;
- second generation had one small random-baseline bump but lost to its parent;
- action use narrowed badly in the trainer outputs;
- shaping and promotion gates may be distracting from a simpler training setup.

## Parallel Review Lanes

- Process/strategy: are we choosing the right loop, scaling pattern, and docs
  rhythm?
- Algorithm/research: should we switch to a simpler known baseline such as PPO,
  policy gradient, DQN-style Pong, CEM, or scripted curriculum before MuZero?
- Measurement: are we using the right evals and artifacts to kill or debug runs?
- Trainer/math: is the current update rule too crude or simply mis-tuned?
- Docs coherence: where do docs overstate confidence or hide uncertainty?

## Questions To Answer Before Scaling

- Is self-play-from-scratch the right first Pong run, or should we train against
  scripted/random curricula first?
- Is the shaped return helping, or making curves easier to misread?
- Is the policy/value trainer worth improving, or should it be replaced by a
  known small RL baseline?
- If we switch, which simple baseline/curriculum comes first: PPO,
  actor-critic, CEM, or another small policy-gradient path?
- What fixed-baseline eval rows run before any selection?
- What is the smallest progress signal that would justify Modal scaling?
- What weaker or changed Pong target ladder should replace default
  `track_ball` before any new win-pressure run?
- What artifacts should every run emit so failures are obvious without reading
  giant logs?

## Temporary Working Rule

Do not scale the old self-play trainer. The 512-game fresh-replay audit got
worse than repair ckpt25, which remains the best old survival baseline, and the
default `track_ball` win target is now known to be impossible from normal
resets. The next Pong learner or curriculum is a separate lane with a
weaker/changed target ladder, not blind generation scaling or a longer version
of the same loop. This is not permission to keep training local. Any run beyond
tiny debugging should run as a whole Modal job with durable Volume artifacts.

Every next run should emit run health, not only a scoreboard: iteration
metrics, action histograms by seat, entropy/collapse metrics, terminal causes,
failure examples, and heldout after selection.

## First Shaped-Reward Run Decision

If the next Pong run intentionally uses shaped training reward, use terminal
loss-delay shaping, not a positive per-step survival bonus. The training target
should keep wins at `+1`, make losses less bad only in proportion to
`episode_steps / max_steps`, and give no positive timeout bonus:

```text
survival_fraction = episode_steps / max_steps

win:      +1.0
loss:     -1.0 + alpha * survival_fraction
timeout:   0.0 or small negative timeout guardrail
```

Use `alpha <= 0.5` for the first run so a long loss is still worse than a win.
Do not use the current survival-curriculum shape
`score_return + survival_weight * survival_fraction + truncation_bonus` as-is
for this decision, because it rewards time survived even on non-loss outcomes
and can make max-step truncation attractive. Eval and promotion must still rank
true score/win rows first, with survival, truncation rate, action entropy, and
terminal causes shown beside them to catch timeout farming or action collapse.

## Modal Execution Correction

The prior coaching stance treated Modal Pong train/eval wrapping as later. That
is wrong for the current execution plan.

- Local work is only for tiny debugger loops and import/shape checks.
- CPU Modal should run the existing NumPy Pong replay, training, and scoreboard
  jobs because they are whole jobs and need durable artifacts.
- GPU Modal should be used immediately for dependency/runtime smokes and Mctx
  synthetic benchmarks; real MuZero training should start on Modal GPU after the
  JAX/Mctx path is proven.
- Volumes are part of the training contract now: replay chunks, checkpoints,
  eval summaries, health metrics, and JSON pointer files belong under
  run/attempt paths in `curvyzero-runs`.
- The existing Pong scoreboard wrapper already proves CPU Modal eval plumbing.
  The Pong train wrapper also exists now. A tiny remote smoke trained one
  checkpoint, saved it to the Volume, and scored it by Volume ref. This proves
  remote plumbing, not policy quality.
- `src/curvyzero/infra/modal/dummy_pong_train_attempt.py` now provides that
  minimal CPU Modal reproduction wrapper; it proves Volume artifact discipline,
  not Pong learning quality.

Concrete plan: see `docs/research/modal_training_execution_plan.md`.

## Stock MuZero Reference Smoke

Lane result: blocked locally, but the reference is clear. The closest stock
visual MuZero path is LightZero's Atari Pong config:
`zoo/atari/config/atari_muzero_segment_config.py --env PongNoFrameskip-v4`.
This repo cannot run it today because `mctx`, `lzero`, `ding`, `jax`, `torch`,
`gymnasium`, and `ale_py` are all absent from the current environment.

New contained lane: `src/curvyzero/infra/modal/mctx_dependency_smoke.py` checks
the ADR-0004 Mctx-first path inside a CPU Modal image. The matching
`src/curvyzero/infra/modal/mctx_gpu_dependency_smoke.py` checks the same path
inside a cheap GPU image. Both use pinned JAX/Mctx deps and one tiny synthetic
`gumbel_muzero_policy` search. Run CPU first, then optional cheap GPU:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.mctx_dependency_smoke --kind cpu
uv run --extra modal modal run -m curvyzero.infra.modal.mctx_gpu_dependency_smoke
```

The GPU smoke requests L4/T4 only. Passing it proves dependency/runtime shape,
not Pong learning.

Plain critique: our dummy Pong self-play loop is not a stock MuZero baseline.
It should not keep accumulating generation/promotion machinery until a basic
known-good MuZero-style example has been installed and smoked in a contained
lane. See `docs/research/muzero_reference_examples.md`.
