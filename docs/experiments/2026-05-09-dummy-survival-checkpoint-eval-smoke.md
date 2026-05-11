# 2026-05-09 Dummy Survival Checkpoint Eval Smoke

## Question

Can EVAL1 load a dummy survival `checkpoint.npz` and compare the learned
checkpoint policy against the fixed random/scripted baselines?

## Setup

- Evaluator: `scripts/run_dummy_survival_eval.py`
- Checkpoint: `artifacts/local/dummy_survival_smoke/checkpoint.npz`
- Baselines: `random_uniform`, `one_step_safe`
- Episodes: 10 per policy
- Seed: 123

## Command

```sh
uv run python scripts/run_dummy_survival_eval.py \
  --episodes 10 \
  --seed 123 \
  --policies random_uniform one_step_safe \
  --checkpoint-policy learned:artifacts/local/dummy_survival_smoke/checkpoint.npz \
  --output-dir artifacts/local/dummy_survival_checkpoint_eval_smoke
```

## Results

- Smoke completed successfully.
- Checkpoint loaded as `learned:dummy_survival_smoke/checkpoint`.
- Loaded checkpoint metadata: 5 iterations, 20 episodes per iteration, seed 0.
- Loaded model summary: 268 states, 521 learned dynamics edges.
- `random_uniform`: crash rate 1.0, survival rate 0.0, mean steps 8.8.
- `one_step_safe`: crash rate 0.0, survival rate 1.0, mean steps 80.0.
- `learned:dummy_survival_smoke/checkpoint`: crash rate 1.0, survival rate
  0.0, mean steps 25.5.

## Interpretation

Checkpoint loading works and the evaluator can now distinguish "loaded learned
policy" from "useful learned policy." This checkpoint is still below the
scripted safety policy and does not survive, though it lasts longer than random
on this small seed set.

## Artifacts

- `artifacts/local/dummy_survival_checkpoint_eval_smoke/summary.json`
- `artifacts/local/dummy_survival_checkpoint_eval_smoke/episodes.jsonl`

## Follow-ups

- Use this command as the immediate regression check after changing the dummy
  survival trainer.
- Add checkpoint resume/loading for Modal artifacts once storage moves beyond
  remote ephemeral files.
