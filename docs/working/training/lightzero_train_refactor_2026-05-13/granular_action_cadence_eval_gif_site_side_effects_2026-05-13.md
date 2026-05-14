# Granular Action Cadence Eval/GIF/Site Side Effects - 2026-05-13

## Scope

This audit covers side effects of the trusted stock LightZero cadence change:
one policy action per CurvyTron source physics frame, and trusted train/dry
reject stale bundled `decision_ms`.

Audited surfaces:

- background checkpoint eval;
- checkpoint self-play GIF generation;
- GIF browser and run-status summaries;
- run progress/status files;
- checkpoint artifact paths and checkpoint tournament/leaderboard paths.

No source was edited for this audit.

## Bottom Line

The trusted train lane now records cadence in the command, env config, eval row,
and env-step telemetry. That is good.

The weak point is the user-facing and checkpoint-facing layer. Several places
still show only "steps", "iteration_N", or "checkpoint" without saying whether
those steps are one-frame granular steps or old bundled decisions. Old
checkpoints can still be discovered, mirrored, rated, and displayed beside new
checkpoints without a visible cadence label.

## What Looks Covered

The trusted train config writes the new cadence into the command and env config:

- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py::_run_visual_survival_train` adds `decision_ms`, `decision_source_frames`, `source_physics_step_ms`, and `source_max_steps_semantics` to `command` at lines 3303-3316.
- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py::_build_visual_survival_configs` writes the same fields into `main_config.env` at lines 4586-4605.
- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py::_validate_trusted_source_state_action_cadence` rejects non-default trusted fixed-opponent `decision_ms` at lines 682-703.
- `_run_visual_survival_train` calls that guard for `mode in {"train", "dry"}` at lines 3196-3203.

The env exposes cadence in timestep info and telemetry:

- `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py::_info` writes `decision_ms`, `decision_source_frames`, `source_physics_step_ms`, `source_frame_decision`, `max_ticks`, and `max_source_ticks` at lines 1714-1733.
- `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py::_write_telemetry_row` persists those cadence fields at lines 2050-2067 and repeat timing at lines 2108-2120.

The eval result row now carries cadence:

- `src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py::_run_survival_episode` returns `decision_ms`, `decision_source_frames`, and `source_physics_step_ms` at lines 919-954.
- `src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py::_row_from_result` copies those fields into the eval table at lines 1038-1067.

## Main Findings

### 1. Background eval config knows cadence, but the spawned eval API does not accept it

`_background_eval_config_from_command` includes cadence:

- `decision_ms`, `decision_source_frames`, `source_physics_step_ms`, and `source_max_steps_semantics` are added at `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5686-5696`.

But `_spawn_one_checkpoint_background_eval` does not pass those fields into
`lightzero_curvytron_visual_survival_checkpoint_eval_and_inspect.spawn`:

- spawn args are built at `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5982-6022`;
- they pass `source_max_steps`, but not `decision_ms`, `decision_source_frames`, or `source_physics_step_ms`.

Today this works only because eval rebuilds the env with `DEFAULT_DECISION_MS`,
which currently means one source frame:

- `src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py::_make_policy_and_env` passes `decision_ms=DEFAULT_DECISION_MS` at lines 713-721.

Risk: a future explicit cadence run, or a non-trusted eval path, can diverge
from the training command while the config object still looks complete.

### 2. Background GIF config knows cadence, but the spawned GIF API does not accept it

`_background_gif_config_from_command` includes cadence:

- `decision_ms`, `decision_source_frames`, `source_physics_step_ms`, and `source_max_steps_semantics` are added at `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5771-5781`.

But `_spawn_one_checkpoint_background_gif` does not pass those fields into
`lightzero_curvytron_visual_survival_checkpoint_selfplay_gif.spawn`:

- spawn args are built at `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:6130-6177`;
- they pass `source_max_steps`, but not the cadence fields.

The GIF path then calls eval `_make_policy_and_env`, which again uses
`DEFAULT_DECISION_MS`:

- `_capture_checkpoint_selfplay_gif_variant` calls `eval_mod._make_policy_and_env` at `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:7351-7370`;
- `_make_policy_and_env` uses `DEFAULT_DECISION_MS` at `src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py:713-721`.

Risk: the GIF summary may imply it followed the training command, but the actual
remote function cannot receive the explicit cadence fields.

### 3. GIF action traces still hide cadence at the per-action level

`_capture_checkpoint_selfplay_gif_variant` stores scalar and joint action traces,
but the trace rows omit cadence fields:

- scalar trace rows are built at `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:7403-7416`;
- joint trace rows are built at `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:7420-7427`.

The returned variant includes the env `surface`, and the full `summary.json`
keeps `variant_surfaces`:

- variant return includes `"surface"` at `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:7561-7567`;
- summary keeps `variant_surfaces` at lines 7795-7797.

But the per-action trace itself only says scalar step, action, player, reward,
done, and policy sampling fields. A plausible-looking GIF can hide whether each
action was one source frame or a bundled decision.

### 4. Train summary has cadence, but the background eval/GIF subsection omits it

The train summary includes the full `command` and `surface`, so cadence is not
lost:

- `summary["command"]` and `summary["surface"]` are written at `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:3796-3804`.

But `summary["background_checkpoint_eval"]` repeats eval/GIF settings without
the cadence fields:

- background eval summary is built at `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:3816-3860`.

Risk: readers that use only the background section can miss the cadence. This is
easy because that section is the natural place to inspect eval/GIF settings.

### 5. `progress_latest.json` has checkpoint identity, but no cadence

`_build_checkpoint_progress_latest_payload` writes iteration, learner iter,
checkpoint ref/name, elapsed time, and source:

- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:1881-1898`.

It does not include `decision_ms`, `decision_source_frames`, or
`source_max_steps_semantics`.

`_write_checkpoint_progress_latest` writes that payload to
`attempts/<attempt_id>/train/progress_latest.json`:

- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:1901-1924`.

Risk: run pickers and status tools can show a fresh checkpoint without showing
whether it came from the new granular cadence.

### 6. `status_heartbeat.json` has the command, but status readers do not surface cadence

`_write_train_status_heartbeat` embeds the full command:

- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:8221-8252`.

But `src/curvyzero/infra/modal/lightzero_curvytron_run_status.py` does not put
cadence in its main rows:

- `_train_artifact_rollup` reads heartbeat status/stage only at lines 774-807;
- `_run_status` reads `progress_latest.json`, checkpoint summary, eval rollup,
  and train artifact rollup at lines 860-930;
- `_print_table` columns do not include cadence at lines 1071-1103;
- `_print_curve_summary` columns do not include cadence at lines 1191-1220.

Risk: a CLI or status-table user sees `mean_steps`, eval means, GIF actions, and
latest checkpoint, but not whether those steps are comparable across cadence
eras.

### 7. GIF browser has the same blind spot

`curvytron_gif_browser` summarizes GIF artifacts into rows. It extracts frames,
physical steps, scalar steps, max steps, terminal reason, checkpoint label, and
refs:

- `_summary_row` builds row fields at `src/curvyzero/infra/modal/curvytron_gif_browser.py:597-660`.

It extracts variant fields, but not cadence:

- `_gif_variant_rows` copies frame/step fields at `src/curvyzero/infra/modal/curvytron_gif_browser.py:527-594`.

The HTML card renders only frames, pixels, steps, terminal reason, GIF link, and
JSON link:

- `_render_rows` card facts are at `src/curvyzero/infra/modal/curvytron_gif_browser.py:1175-1254`.

The run picker uses `progress_latest.json` and also omits cadence:

- `_latest_training_progress_for_run` reads iteration and elapsed time at
  `src/curvyzero/infra/modal/curvytron_gif_browser.py:1000-1036`.

Risk: the website can show old and new GIFs side by side with the same "steps"
word, even when old steps meant bundled source frames.

### 8. Mirrored checkpoint artifacts have no cadence sidecar

`_mirror_lightzero_checkpoints` copies raw checkpoint files into the stable
checkpoint root:

- destination is `runs.checkpoints_root_ref(TASK_ID, run_id) / "lightzero" / source.name`
  at `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5588-5590`;
- the returned summary has file ref, path, bytes/hash, source path, and
  `copied_now` at lines 5601-5611.

No per-checkpoint JSON sidecar is written beside
`checkpoints/lightzero/iteration_N.pth.tar`.

`_publish_live_lightzero_checkpoints` writes `live_checkpoint_publish.json`, but
its checkpoint payload is also file identity only:

- payload is built at `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5619-5648`.

Risk: once a checkpoint is detached from its run summary, the stable artifact
path does not say which cadence produced it.

### 9. Checkpoint discovery can mix cadence eras

The checkpoint tournament discovery scans all `lightzero_exp*/ckpt/iteration_*.pth.tar`
under selected runs:

- `CHECKPOINT_SCAN_GLOB` is defined at `src/curvyzero/tournament/curvytron/contracts.py:28`;
- `_checkpoint_candidate_rows_for_run` reads candidates at
  `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py:256-294`;
- `_discover_checkpoint_refs` selects latest/all/iteration refs at lines 297-430.

The discovery row includes run id, attempt id, exp dir, checkpoint name,
iteration, mtime, size, ref, and path. It does not read `command.json`,
`summary.json`, or any cadence field.

Risk: discovery can feed old bundled-cadence and new granular-cadence checkpoints
into one tournament or rating run with no warning.

### 10. Tournament defaults still use the old 12-frame cadence

Checkpoint tournament contracts import `DEFAULT_DECISION_SOURCE_FRAMES` from
`curvyzero.env.vector_multiplayer_env`, not the trusted source-state survival
wrapper:

- import is at `src/curvyzero/tournament/curvytron/contracts.py:10-13`;
- `DEFAULT_DECISION_MS` is computed from that value at lines 52-53.

`vector_multiplayer_env` still defines `DEFAULT_DECISION_SOURCE_FRAMES = 12`:

- `src/curvyzero/env/vector_multiplayer_env.py:52`.

Tournament runtime settings derive `decision_source_frames` from the spec or
runtime metadata:

- `src/curvyzero/tournament/curvytron_checkpoint_tournament.py::_source_frame_runtime_settings`
  at lines 2715-2777.

Tests still assert 12-frame tournament behavior:

- `tests/test_curvytron_checkpoint_tournament.py:1010-1011`;
- `tests/test_curvytron_checkpoint_tournament.py:1112-1124`.

This may be acceptable if tournament is explicitly outside the trusted train
cadence. The blind spot is that the public tournament website does not display
that distinction.

### 11. Tournament website does not show cadence or context hash

Rating context includes cadence:

- `src/curvyzero/tournament/curvytron_checkpoint_tournament.py::rating_context_hash`
  includes `decision_ms`, `decision_source_frames`, and `source_physics_step_ms`
  at lines 152-166.

Rating config writes `context_hash` and `rating_spec`:

- `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py::_write_rating_config`
  at lines 1012-1038.

But the website ranking table shows rank, checkpoint, rating, games, W-L-D, win
rate, opponents, and failures only:

- `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py:_render_tournament_page`
  builds rows at lines 5554-5578 and headers at lines 5585-5588.

The page header says only "Checkpoint battles. Score is who dies first.":

- `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py:5855-5860`.

Risk: a public reader can compare ratings across pages or runs without seeing
that one rating run used old 12-frame cadence and another used one-frame
granular cadence.

## Checkpoint Comparability Risk

Old and new checkpoints are not automatically comparable just because both are
named `iteration_N.pth.tar`.

Reasons:

- `iteration_N` is LightZero learner iteration, not source physics frame count.
- `progress_latest.json` reports checkpoint identity but not cadence.
- stable checkpoint mirror paths omit run config/cadence sidecars.
- tournament discovery reads checkpoint files, not training command metadata.
- run-status, GIF browser, and tournament website all use compact views that
hide cadence.

The safest current source for cadence is the run command/config/summary:

- `attempts/<attempt_id>/command.json`;
- `attempts/<attempt_id>/config.json`;
- `attempts/<attempt_id>/train/summary.json`;
- `attempts/<attempt_id>/train/env_steps.jsonl`.

The unsafe source is the bare checkpoint ref alone:

- `training/lightzero-curvytron-visual-survival/<run_id>/checkpoints/lightzero/iteration_N.pth.tar`;
- `training/lightzero-curvytron-visual-survival/<run_id>/attempts/<attempt_id>/train/lightzero_exp*/ckpt/iteration_N.pth.tar`.

## Suggested Guardrails

1. Add cadence fields to `progress_latest.json`.

   Include `decision_ms`, `decision_source_frames`, `source_physics_step_ms`,
   and `source_max_steps_semantics` in
   `_build_checkpoint_progress_latest_payload`.

2. Add per-checkpoint metadata beside mirrored checkpoints.

   For each `checkpoints/lightzero/iteration_N.pth.tar`, write an adjacent
   `iteration_N.metadata.json` or a run-level checkpoint index that includes the
   command cadence and source attempt ref.

3. Pass cadence through eval/GIF remote APIs.

   Add `decision_ms`, `decision_source_frames`, and `source_physics_step_ms` to
   `lightzero_curvytron_visual_survival_checkpoint_eval_and_inspect`,
   `lightzero_curvytron_visual_survival_checkpoint_selfplay_gif`,
   `_run_checkpoint_eval_and_inspect`, `_run_checkpoint_selfplay_gif`, and
   `_make_policy_and_env`.

4. Put cadence into GIF summaries and browser rows.

   Add top-level GIF summary fields and variant fields for
   `decision_source_frames`, `decision_ms`, and `source_physics_step_ms`. Show a
   compact badge in `curvytron_gif_browser` such as `1 frame/action` or
   `12 frames/action`.

5. Put cadence into run-status output.

   Read cadence from heartbeat command, progress, or summary. Add fields to
   `_run_status`, `_print_table`, `_print_curve_summary`, and eval rollups.

6. Teach checkpoint discovery about cadence.

   When discovering by run id or prefix, read the attempt command/summary and
   attach cadence to discovery rows. If selected checkpoints have mixed cadence,
   return a `mixed_cadence_warning`.

7. Label tournament context on the website.

   Show `decision_source_frames`, `decision_ms`, and `context_hash` near the
   rating run selector and in `/api/rating-standings`.

8. Decide whether tournament defaults should remain 12-frame.

   If yes, label tournament as separate from the trusted one-frame train lane.
   If no, move tournament defaults away from
   `vector_multiplayer_env.DEFAULT_DECISION_SOURCE_FRAMES` and make one-frame
   cadence explicit.

## Tests Worth Adding

- A background eval spawn test that asserts cadence fields reach the remote eval
  call, not only the local config object.
- A background GIF spawn test that asserts cadence fields reach the remote GIF
  call.
- A GIF summary/browser test that fails if `decision_source_frames` is missing
  from summary rows and rendered card facts.
- A run-status test that renders cadence in table/curve output.
- A checkpoint discovery test with one old 12-frame run and one new one-frame
  run, expecting a mixed-cadence warning.
- A tournament website/API test that exposes rating-run cadence and
  `context_hash`.

