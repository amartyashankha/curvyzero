# LightZero Pong Checkpoint Diagnostics - 2026-05-09

## Bottom Line

The apparent 4096/64 mismatch is now mostly explained by telemetry semantics,
not by the independent adapter loading an obviously wrong checkpoint.

The long train's trainer-side `535/43` row is sidecar env telemetry aggregated
across LightZero train/eval activity, not a held-out final-checkpoint scorecard.
It is dominated by one repeated episode seed: 513 of 578 rows use
`episode_seed=2`. Those rows are reported as `505/8`; the non-seed-2 rows are
`30/35`. So the "trainer is strong, independent eval is bad" gap is much
smaller and less mysterious than it first looked.

Independent MCTS still says the exported final policy is bad: the 4096/64
`iteration_64:model` checkpoint chooses no `down` actions in the full
scorecard and no `down` actions in the checkpoint probe sample. That looks like
real checkpoint behavior under our adapter, not an action-extraction artifact.

## Artifacts

- 512/8 run:
  `lz-dpong-20260509T144635Z-eb5a0ed35de0`,
  attempt `attempt-20260509T144635Z-ece79bad80d0`.
- 512/8 checkpoint diagnostic:
  `training/lightzero-dummy-pong/lz-dpong-20260509T144635Z-eb5a0ed35de0/attempts/attempt-20260509T144635Z-ece79bad80d0/probe/lightzero_checkpoint_diagnostics_20260509T152525Z.json`.
- 4096/64 run:
  `lz-dpong-20260509T151212Z-b95b61de2eb0`,
  attempt `attempt-20260509T151212Z-8b9db08f8fcb`.
- 4096/64 `iteration_64.pth.tar` sha:
  `11a0cc80f797ce8e63150e0a6018efc163b7858bed9efd92b77dda8cadaf95e4`.
- 4096/64 checkpoint diagnostic:
  `training/lightzero-dummy-pong/lz-dpong-20260509T151212Z-b95b61de2eb0/attempts/attempt-20260509T151212Z-8b9db08f8fcb/probe/lightzero_checkpoint_diagnostics_20260509T152423Z.json`.
- Probe module added:
  `src/curvyzero/infra/modal/lightzero_dummy_pong_checkpoint_diagnostics.py`.

No pytest was run.

## Checkpoint Refs

512/8 mirrored refs:

| ref | last_iter | last_step | sha |
| --- | ---: | ---: | --- |
| `iteration_0.pth.tar` | 0 | 24 | `78402cb44a466e92fd7c0f510509c6270ee0289ff804955fe25fba2af80a5151` |
| `iteration_8.pth.tar` | 8 | 124 | `c67d9835ac9adfcef6b02ff919ba7abbd6dba40cafb59fa2c76f267d2148b8fd` |
| `ckpt_best.pth.tar` | 0 | 0 | `c809978f574583ceaaed6c2c8381b6ab450dccbf750ce34f6ef0c44ec474bfe2` |

4096/64 mirrored refs:

| ref | last_iter | last_step | sha |
| --- | ---: | ---: | --- |
| `iteration_64.pth.tar` | 64 | 869 | `11a0cc80f797ce8e63150e0a6018efc163b7858bed9efd92b77dda8cadaf95e4` |
| `ckpt_best.pth.tar` | not probed here | not probed here | `49c4df93dabbfc1cef0cd1f62b04e396bbc886cb88d9c01bbc8e67004f3aa7b0` |

## Model vs Target Model

The independent scorecard loads the `model` state dict. That is likely the
right eval choice. `target_model` is useful as a control, but it is not a
better explanation for trainer-side success.

4096/64 `iteration_64` control:

| variant | direct policy head on 48 real eval states | MCTS sample, 16 states |
| --- | --- | --- |
| `model` | `[48,0,0]`, mean top1 margin `0.1403` | `[8,8,0]` |
| `target_model` | `[48,0,0]`, all logits tied at zero | `[8,1,7]` |

`model` and `target_model` are very different: all 108 tensors changed,
relative total L2 delta `25.09`, and all policy tensors changed. The
`target_model` final policy output layer is zero, so its occasional MCTS
`down` actions are a tie/search control signal, not learned quality.

512/8 contrast:

| variant | direct policy head on 48 states | MCTS sample, 12 states |
| --- | --- | --- |
| `iteration_0:model` | `[0,48,0]` | `[9,3,0]` |
| `iteration_8:model` | `[48,0,0]` | `[12,0,0]` |
| `ckpt_best:model` | `[48,0,0]`, tied zero logits | `[10,2,0]` |

Read: checkpoint tensors do change, but by `iteration_8` and `iteration_64`
the loaded `model` direct policy prefers `up` everywhere in the sampled real
eval states. The 4096/64 direct logits are not merely numerical ties
(`up ~0.11`, `stay ~-0.03`, `down ~-0.13` on sampled rows).

## Mismatch Assessment

- Wrong `target_model`: unlikely. `target_model` is distinct and mostly
  untrained-looking in the policy head. It should not replace `model` for the
  main scorecard.
- Action extraction: unlikely as the primary cause. The probe extracts the
  scalar `action` returned by `eval_mode.forward`, and the same extraction sees
  `down` actions from the `target_model` control.
- Action sign/seat/perspective: still worth a focused check, but no longer the
  lead hypothesis. The encoded probe includes both ego perspectives from real
  dummy Pong states; `iteration_64:model` still has no direct `down`.
- Selection mode/temperature: possible only for explaining sidecar action
  diversity. The sidecar includes train/eval activity, exploration, and a
  repeated seed. It is not a final deterministic checkpoint scorecard.
- `max_env_step`/feature scaling: not the main 4096/64 explanation. The
  independent scorecard and probe used `max_env_step=4096`; the checkpoint
  still avoided `down`.
- Seed handling: primary fix before the next train. Repeated seed-2 sidecar
  rows made trainer-side results look much stronger than held-out checkpoint
  eval.

## Exact Seed Fix

Before another scaled train, make episode seeding explicit and auditable:

1. Add an env config knob, e.g. `dynamic_seed`, defaulting to `True` for train
   and evaluator envs.
2. In `DummyPongLightZeroEnv.__init__`, initialize `_dynamic_seed` from that
   config instead of hardcoding `False`.
3. Keep `seed(seed, dynamic_seed=...)` supported, but do not let LightZero
   evaluator resets silently stay on `base_seed` forever.
4. Use `_next_episode_seed = base_seed + episode_index` when dynamic seeding is
   enabled, with an optional env/rank offset if multiple envs are used later.
5. Add a summary guard: report seed histograms and fail or warn if any one seed
   accounts for more than a small threshold, say 10 percent, of sidecar rows.
6. Stop labeling sidecar telemetry as a scorecard. Split it as
   `train_env_sidecar`, and make final quality depend on an independent
   post-train MCTS scorecard over a fresh recorded eval-wave seed list.

## Next Diagnostics

1. Run a tiny `target_model` full scorecard control for `iteration_64` only.
   Expected: unstable/tie-like behavior, not strong quality. Purpose: prove the
   main scorecard's `model` choice is intentional.
2. Run an independent `iteration_64:model` scorecard on seed `2` repeated and
   on seeds `3..67`. This directly tests whether the learned policy is
   exploiting the repeated seed.
3. Run a player-0-only independent scorecard versus `random_uniform`, matching
   the training env's fixed `ego_agent=player_0`, then compare to the paired
   seating scorecard.
4. After the seed fix, rerun 4096/64 or a shorter 1024/16 train with:
   dynamic seeds, separate evaluator seed offset, seed histogram guard, and an
   automatic independent final-checkpoint scorecard.
