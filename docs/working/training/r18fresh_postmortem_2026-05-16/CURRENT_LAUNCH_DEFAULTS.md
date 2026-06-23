# Current Launch Defaults

Date: 2026-05-17

Status: current defaults for the next broad CurvyTron training lane. If older
docs disagree, treat this file and `src/curvyzero/contracts/curvytron.py` as the
source of truth.

## Current Training Lane

Use this for the next broad exploratory batch:

| Setting | Current value | Why |
| --- | --- | --- |
| compute | `gpu-l4-t4-cpu40` | Cheaper broad parallelism; fresh profiles show acceptable speed penalty. |
| collector envs | `256` | Best measured L4 row used C256. |
| episodes per collect | `256` | Keep `n_episode` matched to collector count. |
| learner batch size | `64` | Best measured L4/C256 row used batch64. |
| MCTS simulations | `8` | Current quality/speed default. |
| max env steps | `30000000` | Broad-run training budget, not a tiny smoke default. |
| max train iterations | `300000` | Broad-run learner budget, not a tiny smoke default. |
| policy trail render | `browser_lines` | Current policy-observation contract. |
| policy bonus render | `simple_symbols` | Current policy-observation contract. |
| policy observation backend | `cpu_oracle` | Reliable production backend. Scalar `jax_gpu` is not the batched backend and stays out of production. |
| checkpoint cadence | `10000` learner iterations | Sparse enough for broad runs; enough to feed the tournament loop. |
| source max steps | `1048576` | Avoid artificial caps on improving games. |
| learner seat mode | `random_per_episode` | Train from both seats. |

## Current Naming Lane

Use `NAMING_CONVENTIONS.md` for new current-code artifacts.

| Object | Current name pattern |
| --- | --- |
| Grid A batch | `cz26a` |
| Grid B batch | `cz26b` |
| canary / fast proof batch | `cz26c` |
| training run | `<batch>-r<row>-<reward>-<noise>-<imm>-<recipe>` |
| reward tags | `out0`, `out33`, `out50`, `out67`, `out100` |
| action-noise tags | `n0`, `n10`, `n20` |
| leaderboard immortality tags | `imm0`, `imm10` |

Historical names such as `tonight18`, `restart18`, `r18fresh`,
`survbonusout`, `so10rep10`, and `rank1imm` are archive/evidence labels. Do
not use them for new current-code launch IDs.

## Opponent Recipe Count Contract

Author opponent recipes as a 64-slot bag. The current collector wave has 256
environments, so repeat the bag four times and deterministically shuffle the
resulting 256 assignments.

Keep learner `batch_size=64` unchanged. The learner samples from replay, so the
64-slot recipe bag controls the data entering the buffer, not the exact makeup
of every gradient mini-batch. Exact per-gradient proportions would require
stratified replay sampling, which is not part of the current launch defaults.

## Current Automation Contract

The current lane is CZ26. Do not use r18fresh defaults for new live
operations.

| Object | Current value |
| --- | --- |
| tournament | `cz26-live-20260517a` |
| rating run | `elo-cz26-live-20260517a` |
| trainer-facing leaderboard | `cz26-live-20260517a-elo-cz26-live-20260517a-training` |
| assignment bank | `cz26-training-candidates` / `try-cz26-training-candidates` |
| refresh pointer count | `24` |

The training-candidate refresh scheduler reads this CZ26 control-volume config:

```text
control:training/lightzero-curvytron-visual-survival/cz26-control/attempts/try-cz26-control/opponents/training_candidate_refresh_config.json
```

Each new launch manifest must publish that config, plus the assignment bank and
refresh pointers it names. Historical r18fresh pointers are archive evidence
only and should not be the current default.

Canaries may explicitly lower `save_ckpt_after_iter` and
`opponent_assignment_refresh_interval_train_iter` so the loop can be observed
quickly. That must be visible in the manifest. The broad Grid A/Grid B default
stays checkpoint cadence `10000` and assignment refresh interval `2000`.

## Speed Evidence

Fresh current-surface optimizer profile:

```text
browser_lines + simple_symbols + cpu_oracle
sim8
no-death512
no eval/GIF/checkpoint I/O in the speed denominator
```

| Best row | Env steps/s |
| --- | ---: |
| L4/T4 CPU40, C256, batch64 | `713.83` |
| H100 CPU40, C256, batch32 | `1001.94` |

Plain read: L4 throughput is about `28.8%` lower than the best H100 row, and the
same amount of profile work would take about `1.40x` as long. That is acceptable
for broad cheaper experiments.

Important caveat: batch64 helped on the measured L4/C256 row, but hurt on H100.
Do not generalize "batch64 is always better"; it is the current L4 lane default,
not a universal MuZero rule.

## Historical Settings To Avoid

- `gpu-h100-cpu40` is now an explicit expensive/sentinel override, not the broad
  default.
- `batch_size=32` is historical for r18fresh/H100 and still useful for matched
  ablations, but not the broad L4 default.
- `body_circles_fast`, `fast_gray64_direct`, and `browser_sprites` are not
  current training/tournament policy surfaces.
- Scalar `policy_observation_backend=jax_gpu` is not the new batched GPU
  observation path.
