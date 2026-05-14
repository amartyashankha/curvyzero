# Granular Action Cadence Downstream Consumers Side Effects

Date: 2026-05-13

## Plain Question

Trusted stock LightZero training now uses one policy action per source physics
frame. Stale bundled `decision_ms` is rejected for trusted train and dry runs.

This audit checks downstream consumers: checkpoint tournaments, frozen
checkpoint opponents, leaderboard and assignment docs, GIF/policy browsers, and
manifest builders.

No source code was edited for this audit.

## Short Verdict

The trusted train and background eval/GIF path is mostly protected now, but the
downstream ecosystem is not fully protected from old-cadence policy mixing.

The biggest issue is metadata. Tournament code can carry cadence and can reject
some mixed checkpoint runtime settings, but future leaderboard and opponent
assignment schemas do not yet require cadence compatibility. Frozen opponent
mixtures can point at old-cadence checkpoints and feed them into new one-frame
training without any explicit warning.

That may be allowed as an intentional curriculum choice, but it should not be
silent. A policy trained with one action per 12 source frames is not the same
kind of opponent as a policy trained with one action per source frame.

## What Looks Good

- Trusted train config now writes `decision_source_frames=1`,
  `decision_ms=SOURCE_PHYSICS_STEP_MS`, and
  `source_max_steps_semantics=source_physics_steps` in the source-state fixed
  opponent env config (`lightzero_curvyzero_stacked_debug_visual_survival_train.py:4599`).
- Background eval and self-play GIF config now carry `decision_ms`,
  `decision_source_frames`, `source_physics_step_ms`, and
  `source_max_steps_semantics` from the training command
  (`lightzero_curvyzero_stacked_debug_visual_survival_train.py:5686`).
- Active opponent-mixture manifests now set `DECISION_MS = SOURCE_PHYSICS_STEP_MS`
  and pass that to train rows (`scripts/build_curvytron_opponent_mixture_manifest.py:39`,
  `scripts/build_curvytron_opponent_mixture_manifest.py:556`).
- Tournament discovery uses broad `train/lightzero_exp*/ckpt/iteration_*.pth.tar`
  discovery (`curvytron/contracts.py:28`). That avoids the stale fixed
  `lightzero_exp/ckpt` checkpoint path problem.
- Tournament game summaries record effective `decision_ms`,
  `decision_source_frames`, `source_physics_step_ms`, and `source_max_ticks`
  (`curvytron_checkpoint_tournament.py:3044`).

## Findings

### 1. Tournament defaults are still old vector cadence

Tournament contract defaults are based on
`vector_multiplayer_env.DEFAULT_DECISION_SOURCE_FRAMES`, which is still `12`
(`vector_multiplayer_env.py:52`, `curvytron/contracts.py:10`,
`curvytron/contracts.py:52`).

That means a tournament or rating run with no explicit cadence and no usable
checkpoint metadata defaults to about 200 ms per policy action, not the new
trusted train cadence. Current tests intentionally assert this old behavior:
`test_source_frame_runtime_settings_use_source_substeps` expects 12 frames, and
`test_checkpoint_game_uses_per_seat_policy_modes_and_full_rich_gif` expects
`decision_source_frames == 12`.

This is not automatically wrong. Tournament is a separate lane. But any
tournament result that feeds future frozen opponents must say whether it is
old-cadence or new-cadence evidence.

### 2. Tournament can infer cadence, but missing metadata can hide mixing

Tournament policy loading reads cadence from checkpoint payloads and nearby
run metadata (`curvytron_checkpoint_tournament.py:2316`,
`curvytron_checkpoint_tournament.py:2436`). It stores those values in each
loaded policy entry (`curvytron_checkpoint_tournament.py:2510`).

The runtime builder then requires all present `decision_source_frames` values
to agree (`curvytron_checkpoint_tournament.py:2688`,
`curvytron_checkpoint_tournament.py:2735`). If old and new checkpoints both
carry metadata, a mixed-cadence game should fail instead of silently rating
them together.

The gap is when metadata is absent on one side. `_consistent_runtime_value`
only compares values that exist. If a new checkpoint has `decision_source_frames=1`
and an old checkpoint has no cadence metadata, the tournament can use the new
one-frame cadence for both policies. If neither checkpoint has metadata, it
falls back to the old 12-frame default.

This affects future online intake and leaderboard snapshots because older
checkpoints may have incomplete metadata.

### 3. Tournament context hash includes cadence, but leaderboard docs do not

`rating_context_hash` includes `decision_ms`, `decision_source_frames`, and
`source_physics_step_ms` (`curvytron_checkpoint_tournament.py:157`). That is
good. It should prevent reusing old rating evidence when the evaluator cadence
changes.

The public leaderboard docs, however, do not yet make cadence a required row
or eligibility field. They warn not to mix incompatible environment, reward,
render, policy-mode, or evaluator contexts, but they do not explicitly name
action cadence as part of compatibility.

Future leaderboard snapshots should include:

- `training_decision_source_frames`
- `training_decision_ms`
- `training_source_physics_step_ms`
- `training_policy_action_repeat_min/max/extra_probability`
- `evaluator_decision_source_frames`
- `cadence_family`, for example `granular_1_frame` or `legacy_12_frame_bundle`

Default training assignment selection should exclude incompatible cadence
families unless the selector is explicitly configured for mixed-cadence
curriculum.

### 4. Frozen opponent mixtures do not carry cadence compatibility

The opponent mixture schema allows refs, paths, seeds, and provider settings,
but it does not allow cadence fields (`opponent_mixture.py:36`). The assignment
parser accepts only small top-level assignment fields plus entries
(`opponent_registry.py:33`). It rejects mutable or non-iteration checkpoint
refs, which is good, but it does not know whether those exact checkpoints were
trained under the old or new cadence.

The frozen LightZero checkpoint opponent provider loads a model from a
checkpoint and builds a policy with the current visual-survival model surface
(`lightzero_checkpoint_opponent_provider.py:195`). It does not check that the
checkpoint's training cadence matches the current env cadence.

Effect: a new one-frame train run can sample a frozen old-cadence checkpoint as
an opponent. The opponent will now act every source frame because the current
env calls the provider at the current env step cadence. That may be useful, but
it is a behavior change for that old policy.

### 5. Hard-coded opponent refs are still historical old-run refs

`scripts/build_curvytron_opponent_mixture_manifest.py` still has default
recent/mid/old refs pointing at one concrete historical run under fixed
`train/lightzero_exp/ckpt` paths (`scripts/build_curvytron_opponent_mixture_manifest.py:59`).

The script now emits one-frame training cadence for new runs, so new learners
will be one-frame. But the frozen opponents selected by those defaults may be
old-cadence policies. The manifest does not record that distinction in the
mixture entry.

This is the most likely future opponent-mixture side effect: new policies can
be trained against older policies without any visible cadence label.

### 6. GIF browser hides cadence in its card rows

Eval result rows now record `decision_ms`, `decision_source_frames`, and
`source_physics_step_ms` (`lightzero_curvytron_visual_survival_eval.py:1038`).
Self-play GIF summaries can also carry cadence through background config.

But the GIF browser summary row currently exposes frame count, steps, max
steps, checkpoint ref, and terminal reason, not cadence
(`curvytron_gif_browser.py:633`). The rendered cards show frames, pixel size,
and steps, but not `decision_source_frames` (`curvytron_gif_browser.py:1175`).

So a human can compare two GIFs that look similar while one was produced under
old bundled cadence and another under one-frame cadence.

### 7. Tournament action trace does not expose cadence per action

Tournament game summaries include top-level cadence, but each action trace row
only includes `physical_step`, `joint_action`, `done`, `truncated`, and policy
output (`curvytron_checkpoint_tournament.py:2986`).

This is probably okay for compact traces if the top-level fields are trusted.
But for debugging old/new cadence mixing, adding cadence or source tick deltas
to the trace would make failures easier to read.

## Future Opponent Mixture Impact

This cadence change does affect future opponent mixtures.

The model architecture did not change, so old and new checkpoints may load
successfully. That is the danger: successful load does not mean same behavioral
contract.

A frozen opponent trained under old bundled cadence learned to choose actions
at a slower effective rhythm. In a new one-frame env, the same model is asked
for actions much more often. That can change:

- reaction timing;
- action distribution;
- wall-avoidance behavior;
- strength ranking;
- how hard the opponent is for the learner;
- whether training overfits to old policies acting outside their native cadence.

This should be treated as either:

1. **compatible only within same cadence family**, for default leaderboard
   assignment; or
2. **explicit mixed-cadence curriculum**, with labels and weights that say so.

Do not let it happen by accident.

## Recommended Guards

1. Add cadence fields to public leaderboard rows and assignment audit records.
   Keep assignment entries small, but the audit should record the cadence family
   of each frozen checkpoint.

2. Add selector filtering:
   default selector only picks checkpoints with
   `training_decision_source_frames == 1` and repeat fields matching the target
   run, unless `allow_mixed_cadence_opponents=true`.

3. Add a trainer-side warning or fail-fast mode:
   when resolving `opponent_mixture_spec`, read checkpoint/run metadata if
   available and record `opponent_training_cadence`. In strict mode, reject
   incompatible old-cadence opponents.

4. Add tournament intake filtering:
   do not put checkpoints with missing cadence metadata into an active public
   leaderboard row. They can stay provisional until metadata is recovered or
   explicitly marked `cadence_unknown`.

5. Add tournament test coverage for mixed old/new cadence:
   one checkpoint with `decision_source_frames=1`, one with
   `decision_source_frames=12`, and assert the rating/game path blocks or
   labels the mismatch.

6. Update GIF browser rows/cards to expose `decision_source_frames` and
   `source_physics_step_ms`, at least in JSON rows and preferably in compact
   card facts.

7. Replace hard-coded recent/mid/old refs in the opponent mixture manifest
   builder with registry or leaderboard-derived refs that carry cadence
   provenance.

## Bottom Line

The local trusted training cadence is fixed. The downstream consumers are only
partly cadence-aware.

Checkpoint tournament has the best protection because it can read runtime
metadata and hashes cadence into rating context. Frozen opponent mixtures and
future leaderboard assignments are the weak spot: they validate immutable refs,
but not whether the checkpoint was trained at the same action cadence as the
new run.

Before using tournament outputs as a public opponent pool, make cadence a
first-class compatibility field.
