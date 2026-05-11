# 2026-05-09 Dummy Survival Safety Epsilon Smoke

## Question

Does filtering epsilon exploration to actions with positive current clearance
reduce crash-heavy replay enough to make later dummy survival checkpoints
degrade less?

## Setup

- Code change: `DummyPlanner` epsilon exploration can optionally sample only
  actions with positive immediate clearance when any exist, falling back to all
  actions only when every clearance is zero. This is now gated by the
  `--safety-filter-epsilon` flag because the smoke was negative/mixed.
- Existing planner/eval behavior for `epsilon=0` is otherwise unchanged.
- Trainer: `scripts/run_dummy_survival_train.py`
- Sweep: `scripts/run_dummy_survival_checkpoint_sweep.py`
- Training: 8 iterations, 20 episodes per iteration, seed 0
- Periodic checkpoint cadence: every 2 completed iterations
- Eval: 10 episodes, seed 123

## Command

```sh
python3 -m py_compile \
  src/curvyzero/training/dummy_survival.py \
  scripts/run_dummy_survival_train.py \
  scripts/run_dummy_survival_eval.py \
  scripts/run_dummy_survival_checkpoint_sweep.py

uv run python scripts/run_dummy_survival_train.py \
  --iterations 8 \
  --episodes-per-iter 20 \
  --seed 0 \
  --output-dir artifacts/local/dummy_survival_safety_epsilon_i8_e20_seed0_c2 \
  --checkpoint-every-iterations 2 \
  --safety-filter-epsilon

uv run python scripts/run_dummy_survival_checkpoint_sweep.py \
  --checkpoint-dir artifacts/local/dummy_survival_safety_epsilon_i8_e20_seed0_c2/checkpoints \
  --episodes 10 \
  --seed 123 \
  --output-dir artifacts/local/dummy_survival_safety_epsilon_i8_e20_seed0_c2_eval_seed123_e10
```

## Results

- Smoke completed successfully.
- Fixed baselines remained stable:
  - `random_uniform`: crash rate 1.0, survival rate 0.0, mean steps 8.8.
  - `one_step_safe`: crash rate 0.0, survival rate 1.0, mean steps 80.0.
- Best learned checkpoint: `iteration-0004`.
- `iteration-0002`: crash rate 1.0, survival rate 0.0, mean steps 78.6.
- `iteration-0004`: crash rate 0.8, survival rate 0.2, mean steps 74.4.
- `iteration-0006`: crash rate 0.8, survival rate 0.2, mean steps 49.6.
- `iteration-0008`: crash rate 1.0, survival rate 0.0, mean steps 37.0.

Compared with the safety-planner smoke, where `iteration-0002` and
`iteration-0004` survived all 10 fixed-seed eval episodes and later
checkpoints degraded to 0.1 survival rate, this run was worse overall.
`iteration-0006` degraded slightly less than before, but `iteration-0008`
degraded more.

Training-run internal eval was also unstable: survival rates by completed
iteration were 1.0, 0.0, 1.0, 0.25, 0.75, 0.35, 0.0, 0.0. Training rollouts
remained crash-heavy despite filtering immediate zero-clearance epsilon moves.

## Interpretation

The minimal epsilon safety filter is behaviorally correct but not sufficient to
fix dummy survival collection by itself. It prevents raw random immediate
zero-clearance moves during epsilon exploration, but replay can still be
dominated by trajectories that enter traps and crash later.

On this small smoke, later checkpoints do not degrade less in a useful way.
The best-checkpoint signal also regressed versus the planner-only smoke, so the
result should be treated as a negative or mixed collection-fix result rather
than an improvement.

## Artifacts

- `artifacts/local/dummy_survival_safety_epsilon_i8_e20_seed0_c2/summary.json`
- `artifacts/local/dummy_survival_safety_epsilon_i8_e20_seed0_c2/checkpoint.npz`
- `artifacts/local/dummy_survival_safety_epsilon_i8_e20_seed0_c2/iteration_metrics.jsonl`
- `artifacts/local/dummy_survival_safety_epsilon_i8_e20_seed0_c2/checkpoints/iteration-0002.npz`
- `artifacts/local/dummy_survival_safety_epsilon_i8_e20_seed0_c2/checkpoints/iteration-0004.npz`
- `artifacts/local/dummy_survival_safety_epsilon_i8_e20_seed0_c2/checkpoints/iteration-0006.npz`
- `artifacts/local/dummy_survival_safety_epsilon_i8_e20_seed0_c2/checkpoints/iteration-0008.npz`
- `artifacts/local/dummy_survival_safety_epsilon_i8_e20_seed0_c2_eval_seed123_e10/summary.json`
- `artifacts/local/dummy_survival_safety_epsilon_i8_e20_seed0_c2_eval_seed123_e10/checkpoint_eval.jsonl`
- `artifacts/local/dummy_survival_safety_epsilon_i8_e20_seed0_c2_eval_seed123_e10/best_checkpoint.json`
- `artifacts/local/dummy_survival_safety_epsilon_i8_e20_seed0_c2_eval_seed123_e10/best_checkpoint_path.txt`

## Follow-ups

- Keep checkpoint selection in place; final checkpoints are still unreliable.
- If improving collection further, test a stronger rollout policy such as
  one-step-safe epsilon fallback or safety-biased exploration among positive
  clearance actions.
- Re-run with more seeds before drawing conclusions about the mixed
  `iteration-0006` result.
