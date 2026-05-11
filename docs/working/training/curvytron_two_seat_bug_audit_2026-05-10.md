# CurvyTron two-seat bug audit - 2026-05-10

Scope:

- `src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py`
- `src/curvyzero/training/curvyzero_stacked_debug_visual_survival_profile.py`
- `src/curvyzero/infra/modal/lightzero_curvytron_two_seat_train_smoke.py`
- direct helpers: `policy_row_mapping.py`, `SourceStateGray64Stack4`, `VectorMultiplayerEnv`, `run_management.py`

I did not run pytest. I ran only local compile/introspection commands.

## Short answer

No tiny obvious code bug found in the scoped two-seat path.

The current two-seat smoke does control both active seats before `env.step`, and its metadata-bearing learner sample builds discounted survival-time value targets. The replay sample can be more than two rows, but only if the per-iteration collection has more than one live decision step or `batch_size > 1`. Reset rows are refreshed without rolling live rows. Checkpoints are written under the Modal Volume mount when `allow_optimizer_step=True`.

## Checks

### Does current policy control both seats before `env.step`?

Yes.

The collection loop builds one policy row per live/legal player slot from `[B,P,...]` observations, then calls the same `MuZeroPolicy.eval_mode.forward` object once per active policy row. For the normal two-player row this means player 0 and player 1 are both selected before the environment is stepped.

Evidence:

- `build_policy_row_mapping(...)` filters active players and emits active rows first.
- `_collect_current_policy_iteration(...)` loops over every active policy row, collects all selected actions, and only then calls `policy_rows_to_joint_action(...)`.
- `env.step(joint_action)` happens after the full `[B,P]` joint action is built.
- `VectorMultiplayerEnv._action_mask()` marks active seats using `present & alive & ~done & ~warmdown_pending`, so dead/done seats are not policy rows.

Important caveat: collection still builds one policy row per live/legal player
slot before `env.step`, but active rows are now attempted as one batched
`MuZeroPolicy.collect_mode.forward` call with `to_play=-1` and per-row
`ready_env_id`. If batched collect fails, the path can fall back and records
`policy_batch_fallback_reason`. This proves both seats are controlled by the
same current policy before the joint step, but it is still the custom bounded
two-seat adapter, not LightZero's stock collector/GameBuffer or full
distributed self-play.

### Are target values really survival-time returns?

For the two-seat path, yes when the replay sample includes the metadata fields.

The two-seat replay rows include `iteration`, `env_row_id`, `player_id`, and `decision_index`. `_sample_replay_batch(...)` carries those into `iteration_batch`, `env_row_id_batch`, `player_id_batch`, and `decision_index_batch`. `_learn_mode_batches(...)` sees those keys and uses `_target_value_batch(...)`, which calls `_discounted_survival_returns_from_sample(...)` and `_survival_return_value_targets(...)`.

The value target is therefore remaining discounted alive reward per `(iteration, env_row_id, player_id, decision_index)` in the sampled rows.

Sanity introspection:

- With five all-alive toy steps and discount `1.0`, metadata targets were `[[5,4], [4,3], [3,2], [2,1], [1,0]]`.
- The legacy no-metadata target would be flat `[[0,1], [0,1], [0,1], [0,1], [0,1]]`.

Important caveat: the standalone stacked debug visual profile still samples rows without two-seat metadata, so that older profile path falls back to the legacy immediate-reward adapter. The current two-seat train smoke avoids that by adding metadata before calling the shared learner helper.

### Is replay sampling using more than two rows?

Sometimes. It samples all rows from the current collection iteration.

Default local/Modal settings are `batch_size=1`, `collect_steps_per_iteration=None`, and `steps=4`, so the first iteration should normally produce `4 steps * 2 players = 8 replay rows` if both players stay alive for all four decisions.

But the learner update samples `iteration_replay_rows`, not global replay. So with `--collect-steps-per-iteration 1 --batch-size 1`, or with an early terminal step, the learner can still train on only two rows. This is probably fine for a smoke, but it is still a flat-learning risk if someone treats this as a training run.

### Are reset rows refreshed correctly?

Mostly yes.

Before each collection iteration and inside an iteration after terminal rows, the code calls `env.autoreset_done_rows(...)` for rows with `needs_reset`. It then calls `_refresh_reset_rows_in_visual_stack(...)`, which builds a fresh stack from the current env state and copies only reset rows into the live stack. Live rows are not rolled or overwritten.

That avoids stale terminal frames on reset rows and avoids accidental extra frame shifts on non-reset rows.

Small caveat: a reset row stack is refreshed as a new stack with one current frame and leading zero frames. That matches the first reset behavior, but it means a reset row does not get four copies of the initial frame.

### Are checkpoints saved to Modal Volume?

Yes, when `allow_optimizer_step=True`.

The Modal wrapper maps the `curvyzero-runs` volume to `/runs`. It builds `checkpoint_dir = /runs/training/<task>/<run>/checkpoints/lightzero` and passes that directory into the smoke only when optimizer steps are allowed. `_save_lightzero_policy_checkpoint(...)` writes:

- `iteration_<n>.pth.tar`
- `ckpt_best.pth.tar`
- `latest.pth.tar`

The wrapper then writes `summary.json` under the attempt train ref. It no longer explicitly calls `runs_volume.commit()` for progress or summary writes. The result also records `checkpoint_root_ref` and per-file refs when checkpoint metadata includes `checkpoint_root_ref`.

If `allow_optimizer_step=False`, checkpoints are intentionally not requested.

## Top 5 remaining risks

1. Very small learner samples can still happen. The default smoke gets about eight rows, but `collect_steps_per_iteration=1`, early deaths, or tiny batches can make the learner see only two rows.

2. The update sample is per-iteration by default, but the smoke now supports
   `--replay-scope accumulated` plus `--learner-sample-size` for coach-like
   tiny full runs. Use accumulated replay for any training smoke that is meant
   to resemble the real loop.

3. The two-seat adapter is still outside LightZero's normal collector/GameBuffer path. It proves a useful boundary, but it does not prove the upstream trainer target builder, replay priorities, or distributed refresh path.

4. The standalone debug visual profile still uses legacy immediate value targets because its sample rows do not carry two-seat trajectory metadata.

5. Both seats share one policy object and are searched as independent policy
   rows with `to_play=-1`. The batched call improves parity/performance, but
   this is still not joint-action MCTS; opponent/seat identity still only comes
   from the observation and replay metadata.

## 2026-05-11 optimizer update

- Active two-seat policy rows are batched into one LightZero
  `MuZeroPolicy.collect_mode.forward` call per decision step when possible.
- The Modal wrapper now has `--compute cpu` and `--compute gpu-l4-t4`; GPU uses
  `use_cuda=True` and reports `lightzero_policy_model_device`.
- The wrapper no longer explicitly commits the Modal Volume during progress or
  summary writes. Modal should persist the volume without paying this
  per-iteration tax.
- Default progress and checkpoint cadence is now sparse:
  `progress_every_iterations=100` and `checkpoint_every_iterations=100`.
  Final checkpoints are still written when `allow_optimizer_step=True`, even if
  the final iteration is not on the interval.
- Initial checkpoint writes are off by default; pass `--save-initial-checkpoint`
  only for checkpoint-delta debugging.
- Compact summaries now include aggregate `collect_timing_summary` and first/last
  iteration edges instead of dumping every iteration and full replay sample
  index lists.
- Two full GPU smoke runs passed with real optimizer updates:
  seeds `6` and `7`, `B=8`, `collect_steps=8`, `outer_iterations=3`,
  `updates_per_iteration=1`, `num_simulations=5`, accumulated replay, learner
  sample `128`. Both reported `ok=true`, `model_parameters_changed=true`,
  `cuda:0`, final checkpoint refs, and no problems.
- Larger GPU collect-only probes showed that increasing `batch_size` helps but
  is not enough for the 10x target by itself: `B=64` produced `5120` replay rows
  in `24.29s`, while `B=128` regressed to `10240` rows in `57.15s`.
- The Modal wrapper now has a `--parallel-runs N` path that fans out independent
  one-GPU runs with derived seeds/run ids. This is for profiling/sweeps and run
  groups today, not yet shared-replay distributed learning.
- A same-shape rerun against the old `B=8`, `16x64`, `updates=1`, `sims=2`
  heavy path measured `102.24s` on CPU and `50.64s` on GPU L4/T4, versus a
  rough old log-based baseline of `~318s` inner wall time. This confirms a real
  cleanup win, but not a 10x single-loop speedup.

## Commands run

```bash
python -m py_compile \
  src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py \
  src/curvyzero/training/curvyzero_stacked_debug_visual_survival_profile.py \
  src/curvyzero/infra/modal/lightzero_curvytron_two_seat_train_smoke.py \
  src/curvyzero/training/policy_row_mapping.py

python -c "import sys; sys.path.insert(0, 'src'); from curvyzero.training.curvyzero_stacked_debug_visual_survival_profile import toy_alive_survival_target_diagnostic; r=toy_alive_survival_target_diagnostic(steps=5, num_unroll_steps=1, discount=1.0); print(r['metadata_adapter_targets']['target_value_rows']); print(r['legacy_adapter_targets_without_metadata']['target_value_rows'])"
```
