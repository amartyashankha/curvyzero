# Checkpoint Save Path Critique - 2026-05-13

Scope: read-only trace of the stock CurvyTron Modal `train_muzero` checkpoint
path, from local launcher hooks through upstream LightZero/DI-engine behavior.
No source code was changed for this investigation.

## Short Answer

`iteration_N.pth.tar` is written only by DI-engine's learner checkpoint hooks,
not by `progress_latest.json` itself. For the stock LightZero path, the periodic
file is created when a completed `BaseLearner.train(...)` call reaches the
`after_iter` hook, the process is rank 0, and
`engine.last_iter.val % save_ckpt_after_iter == 0`. The initial
`iteration_0.pth.tar` is expected because `last_iter` starts at 0 and the hook
runs before `BaseLearner.train` increments it.

The observed stale rows are plausible without a source-code checkpoint write
failure: the local resume/progress wrapper patches `SaveCkptHook.__call__`,
calls the original hook, and then writes `progress_latest.json` unconditionally.
DI-engine calls `SaveCkptHook.__call__` on every learner `after_iter`, but the
hook internally no-ops unless the modulo condition is true. Therefore a fresh
`progress_latest.json` with high `learner_train_iter` proves the hook returned;
it does not prove a new `iteration_N.pth.tar` was created.

## Local Path

The attempt launcher is
`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train_attempt.py`.
The CPU/GPU Modal functions pass `save_ckpt_after_iter` through to
`_run_visual_survival_train(...)` (`train_attempt.py:90-152`,
`train_attempt.py:185-254`). The local entrypoint either spawns the remote train
function or waits and calls it directly (`train_attempt.py:333-417`).

Inside `_run_visual_survival_train(...)`:

- The attempt train root is
  `training/<task>/<run>/attempts/<attempt>/train`, and `exp_name` is
  `.../train/lightzero_exp` (`lightzero_curvyzero_stacked_debug_visual_survival_train.py:3187-3193`).
- The launcher writes attempt/status heartbeat records before training, including
  `checkpoint_root_ref` for the mirrored run-level checkpoint directory
  (`.../checkpoints/lightzero`) (`...train.py:3378-3407`,
  `...train.py:8098-8128`).
- `_build_visual_survival_configs(...)` starts from
  `zoo.atari.config.atari_muzero_config`, patches `exp_name`, cuda,
  env/model shape, eval frequency, and `save_ckpt_after_iter`
  (`...train.py:4427-4491`).
- `_set_save_ckpt_after_iter(...)` writes specifically to
  `policy.learn.learner.hook.save_ckpt_after_iter`
  (`...train.py:5101-5114`).
- The actual call is `lzero.entry.train_muzero([main_config, create_config],
  seed=..., max_train_iter=..., max_env_step=...)` after `os.chdir(RUNS_MOUNT)`
  (`...train.py:3527-3601`).

The final run-level mirror is post-training: after `train_muzero` returns,
`_scan_lightzero_artifacts(str(exp_name))` and `_mirror_lightzero_checkpoints`
copy visible checkpoint files into `training/<task>/<run>/checkpoints/lightzero`
(`...train.py:3672-3676`, `...train.py:5471-5513`). A still-running trainer can
therefore have newer files in the attempt `train/lightzero_exp/ckpt` directory
before the final mirror catches up. The status path chooses the first existing
directory from `[run mirror, attempt ckpt]`, so an existing run mirror with only
old files can mask a newer attempt-local ckpt directory in status output
(`lightzero_curvytron_run_status.py:809-859`).

## Upstream Conditions

Upstream LightZero `train_muzero` creates a DI-engine `BaseLearner` with
`exp_name=cfg.exp_name` after `compile_config(...)`
(`lzero/entry/train_muzero.py:71-103` in the current upstream source:
https://raw.githubusercontent.com/opendilab/LightZero/main/lzero/entry/train_muzero.py).
It then calls `learner.call_hook('before_run')`, does an initial evaluator call,
collects data, samples replay only when `replay_buffer.get_num_of_transitions() >
batch_size`, and calls `learner.train(train_data, collector.envstep)`
(`train_muzero.py:132-215`). The loop stops only after collection/learning when
`collector.envstep >= max_env_step` or `learner.train_iter >= max_train_iter`
(`train_muzero.py:220-237`).

DI-engine `BaseLearner.train(...)` calls `self.call_hook('after_iter')` and only
then increments `_last_iter` (`ding/worker/learner/base_learner.py:215-279`:
https://raw.githubusercontent.com/opendilab/DI-engine/main/ding/worker/learner/base_learner.py).
The default learner config includes `save_ckpt_after_iter` and
`save_ckpt_after_run` hooks (`base_learner.py:34-46`).

DI-engine maps `save_ckpt_after_iter` to `SaveCkptHook(...,
position='after_iter', ext_args={'freq': freq})`
(`ding/worker/learner/learner_hook.py:359-365`:
https://raw.githubusercontent.com/opendilab/DI-engine/main/ding/worker/learner/learner_hook.py).
`SaveCkptHook.__call__` writes only if:

- `engine.rank == 0`;
- `engine.last_iter.val % self._freq == 0`;
- it can create `./{engine.exp_name}/ckpt`;
- `save_file(path, state_dict)` succeeds.

When those are true, the filename is `engine.ckpt_name` if set, otherwise
`iteration_{engine.last_iter.val}.pth.tar`; the checkpoint payload includes
policy state plus `last_iter` and `last_step` (`learner_hook.py:127-174`).

`BaseLearner.save_checkpoint(ckpt_name=None)` is a separate best/final callback
path. It sets `ckpt_name` if supplied and directly invokes the registered
`save_ckpt_after_run` hook (`base_learner.py:378-397`). LightZero passes this
callback to evaluator calls (`train_muzero.py:149-150`, `train_muzero.py:176-183`).

## Why Progress Can Advance Without a New Checkpoint

There are two local `progress_latest.json` writers on the stock path:

- `_install_checkpoint_progress_writer(...)` wraps `BaseLearner.save_checkpoint`
  and writes progress after the original callback returns (`...train.py:1868-1918`).
- `_install_lightzero_full_resume_state_hooks(...)` wraps
  `ding.worker.learner.learner_hook.SaveCkptHook.__call__`; after the original
  hook returns it calls `_save_lightzero_resume_sidecar_state(...)` and then
  `_write_checkpoint_progress_latest(...)` with
  `source="SaveCkptHook.__call__"` (`...train.py:2038-2076`).

The second writer is the important one for the stale rows. The wrapper does not
inspect whether the original `SaveCkptHook` actually wrote a file. It writes
progress after every `SaveCkptHook.__call__` invocation, including modulo no-op
iterations. `_save_lightzero_resume_sidecar_state(...)` also returns
`{"saved": False, "reason": "matching_iteration_checkpoint_not_found"}` when
`exp_name/ckpt/iteration_<learner.train_iter>.pth.tar` is absent, and that return
value is ignored before writing progress (`...train.py:2085-2097`).

`_write_checkpoint_progress_latest(...)` scans only `exp_name/ckpt` for the
highest visible `iteration_*.pth.tar`; if one exists, `payload["iteration"]` is
that checkpoint iteration, while `payload["learner_train_iter"]` is read
separately from `learner.train_iter` (`...train.py:1804-1865`). So this shape is
expected and meaningful:

```json
{
  "source": "SaveCkptHook.__call__",
  "checkpoint_name": "iteration_0.pth.tar",
  "iteration": 0,
  "learner_train_iter": 175528
}
```

It means the learner counter advanced and the progress wrapper ran, but the
latest visible checkpoint file in `exp_name/ckpt` was still `iteration_0.pth.tar`.
Existing local notes found exactly this pattern in five sampled k0 rows
(`stale_config_analysis_newton_2026-05-13.md:60-74`).

## Plausible Failure Modes

Most likely:

- **Progress-wrapper false freshness.** `SaveCkptHook.__call__` is invoked every
  `after_iter`, but DI-engine writes only every `save_ckpt_after_iter`. With
  `save_ckpt_after_iter=10000`, progress can refresh thousands of times between
  durable checkpoints. This fully explains fresh `learner_train_iter` with
  stale `checkpoint_name`.
- **Checkpoint cadence is far larger than human expectation.** The active rows
  sampled in existing docs show `Save every = 10000`; `iteration_0` is the first
  save, and `iteration_10000` is the next periodic save. If the learner has not
  reached an exact modulo boundary visible to the hook, no later periodic file
  should exist.
- **Rank or hook mismatch.** DI-engine saves only on `engine.rank == 0`. If
  distributed/multi-GPU rank detection is not what the process expects, nonzero
  ranks can advance training counters without writing checkpoints.
- **Replay/learning cadence confusion.** `learner_train_iter` advances only after
  `learner.train` calls, but `collector.envstep`, wall time, and training loop
  collection can be much larger. If replay is insufficient, `train_muzero` logs
  and collects more without a learner update (`train_muzero.py:197-209`).

Worth checking when a row is truly past a save boundary:

- **Save hook throws or `save_file` fails.** A real save failure should propagate
  from `SaveCkptHook.__call__` unless swallowed by external logging/runtime
  behavior; check `learner_logger.txt` and Modal logs around exact modulo
  iterations.
- **Path mismatch.** Local status/progress scans only `exp_name/ckpt` for
  `iteration_*.pth.tar`. A checkpoint written under a different `exp_name`, a
  `ckpt_<instance>` directory, a non-`iteration_*.pth.tar` name, or only as
  `ckpt_best.pth.tar` will not count as a periodic checkpoint.
- **Mirror lag versus attempt ckpt.** The run-level
  `checkpoints/lightzero` mirror is updated at the end of `_run_visual_survival_train`;
  while training is still running, the attempt-local
  `attempts/<attempt>/train/lightzero_exp/ckpt` directory is the source of truth.
  Status currently prefers an existing run mirror over the attempt directory, so
  compare both before concluding the trainer failed to save.
- **Non-atomic progress write.** `runs.write_json(...)` opens the final JSON path
  directly in `wb` mode (`run_management.py:264-276`), so empty/partial
  `progress_latest.json` reads can happen independently of checkpoint health.

## Operational Read

Trust the actual `iteration_*.pth.tar` directory listing and mtimes over
`progress_latest.timestamp`. Interpret `progress_latest.learner_train_iter` as
"learner counter observed by the wrapper", and interpret
`progress_latest.checkpoint_name` as "newest visible checkpoint file at that
moment." If those disagree, the checkpoint file wins.

For rows with `learner_train_iter >= save_ckpt_after_iter` and still only
`iteration_0.pth.tar`, the next targeted evidence is the learner log around the
first expected modulo save (`iteration_10000`, etc.), the configured
`formatted_total_config.py` hook stanza, and a direct listing of both:

- `attempts/<attempt>/train/lightzero_exp/ckpt`
- `checkpoints/lightzero`
