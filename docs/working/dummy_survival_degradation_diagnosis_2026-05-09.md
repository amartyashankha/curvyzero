# Dummy Survival Degradation Diagnosis - 2026-05-09

## Question

Why do later dummy survival checkpoints degrade after early checkpoints and
planner-only baselines survive?

## Commands

```sh
PYTHONPATH=src python3 -m py_compile scripts/analyze_dummy_survival_checkpoints.py

PYTHONPATH=src python3 scripts/run_dummy_survival_checkpoint_sweep.py \
  --checkpoint-dir artifacts/local/dummy_survival_safety_planner_i8_e20_seed0_c2/checkpoints \
  --episodes 20 \
  --seed 123 \
  --output-dir artifacts/local/dummy_survival_degradation_probe_eval_seed123_e20

PYTHONPATH=src python3 scripts/analyze_dummy_survival_checkpoints.py \
  --checkpoint-dir artifacts/local/dummy_survival_safety_planner_i8_e20_seed0_c2/checkpoints \
  --episodes 20 \
  --seed 123 \
  --output-dir artifacts/local/dummy_survival_degradation_diagnosis_seed123_e20
```

No pytest was run.

## Results

On the shared 20-seed monitor split, both safety-prior baselines survived every
episode:

| Policy | Survival | Mean steps | Actions [L,S,R] |
| --- | ---: | ---: | --- |
| `one_step_safe` | 1.0 | 80.0 | `[320, 1280, 0]` |
| `untrained_model_same_planner` | 1.0 | 80.0 | `[320, 1280, 0]` |
| `learned:iteration-0002` | 1.0 | 80.0 | `[279, 1247, 74]` |
| `learned:iteration-0004` | 1.0 | 80.0 | `[146, 1186, 268]` |
| `learned:iteration-0006` | 0.2 | 50.4 | `[230, 537, 241]` |
| `learned:iteration-0008` | 0.2 | 68.0 | `[302, 841, 217]` |

The learned checkpoints increasingly override the planner-only safety action:

| Policy | Dynamics edges | Q range | Safety overrides | Lower-clearance overrides | Zero-clearance selections |
| --- | ---: | --- | ---: | ---: | ---: |
| `untrained_model_same_planner` | 0 | none | 0 | 0 | 0 |
| `learned:iteration-0002` | 487 | `[-0.988, 0.000]` | 78 | 45 | 0 |
| `learned:iteration-0004` | 1176 | `[-0.996, 0.000]` | 123 | 76 | 0 |
| `learned:iteration-0006` | 1717 | `[-0.999, 0.000]` | 215 | 184 | 16 |
| `learned:iteration-0008` | 2156 | `[-0.999, 0.000]` | 291 | 244 | 16 |

All checkpoint Q values are non-positive. The training metrics for this run
also show 100% crash rate during collection at every iteration, so the replay
stream is dominated by crash returns. Early checkpoint success is therefore the
planner prior surviving despite the model, not a learned positive policy.

## Concrete Failure

For env seed `33158374`, the untrained planner and `iteration-0002` both reach
80 steps. `iteration-0008` crashes at step 65. Its first bad override is at
step 2:

- State key: `[1, 1, 3, 4, 4, 5]`
- Clearances `[left, straight, right]`: `[4, 4, 6]`
- Safety action: `right`, clearance `6`
- Learned action: `straight`, clearance `4`
- Scores: `[-0.725, -0.712, -0.897]`
- Selected next Q: `[-0.714, -0.770, -0.818]`
- Safety next Q: `[-0.918, -0.939, -0.900]`

The learned model chooses the lower-clearance action because its next-state
value is less negative than the safer action's next-state value. This is enough
to override the safety-prior tie-break even though there is no positive value
signal.

The top repeated late-checkpoint override has the same shape. In
`iteration-0006`, state `[0, 1, 0, 3, 5, 1]` chooses `right` with clearance `1`
over `straight` with clearance `6` because scores are
`[-0.238, -0.758, 0.000]`; the selected action points to an unknown or
unpenalized next state.

## Diagnosis

Later checkpoints degrade because learned Q/dynamics increasingly override the
planner-only safety prior with a crash-only, non-positive value landscape.
Known safer routes accumulate negative value estimates from crash-heavy replay,
while unknown or under-updated next states remain at `0.0`. Since the planner
scores `reward + discount * next_value` before the clearance tie-break, those
less-negative or unknown actions can beat the maximum-clearance action.

This also explains why the planner-only baseline survives: with no dynamics
edges and no Q values, all model scores are tied at zero, so the planner falls
back to clearance and straight-preference ordering. The learned checkpoints
break that tie in the wrong direction.

## Artifacts

- `artifacts/local/dummy_survival_degradation_probe_eval_seed123_e20/summary.json`
- `artifacts/local/dummy_survival_degradation_probe_eval_seed123_e20/checkpoint_eval.jsonl`
- `artifacts/local/dummy_survival_degradation_diagnosis_seed123_e20/summary.json`
- `artifacts/local/dummy_survival_degradation_diagnosis_seed123_e20/episodes.jsonl`
- `artifacts/local/dummy_survival_degradation_diagnosis_seed123_e20/overrides.jsonl`

## Caveats

This is a 20-episode local monitor split, not a quality claim. The analysis
script mirrors the current planner scoring path for diagnosis only. It does not
propose a trainer rewrite.
