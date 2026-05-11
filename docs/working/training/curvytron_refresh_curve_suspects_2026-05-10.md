# CurvyTron Refresh Curve Suspects - 2026-05-10

Purpose: concise working-memory note for why the staged frozen-opponent
CurvyTron survival curves are flat, noisy, or worse. This is about the
s90/s91/s92 learner-vs-frozen-checkpoint refresh runs. It is not a source
fidelity or policy-quality claim.

## Highest-Priority Checks

1. **Fix the two-seat value target.** Latest iterative two-seat smoke passed
   mechanically but `target_value` is immediate reward, not discounted survival
   return. Target fix is in progress.
2. **Fix learner batch sizing.** The LightZero profile hard-sets
   `policy.batch_size=2` and `_learn_mode_batches` slices samples to that size,
   so larger CurvyTron collect phases may have updated on only 2 replay rows.
3. **Rerun the iterative two-seat curve after both fixes.** The current
   `curvytron-two-seat-iterative-b8-s7-4x32-u2-sim2` curve has no learning
   signal: 32-random-start mean steps for `iteration_0..4` are `181.28125`,
   `174.0625`, `170.71875`, `174.125`, and `170.71875`.
4. **Build true current-policy self-play into the main trainer path.** Current
   LightZero runs are not self-play: LightZero chooses one ego action and the
   env fills the opponent with fixed or frozen policy.
5. **Check the opponent perspective bug.** The frozen opponent path may be
   giving the opponent the same wrapper stack as ego instead of a true
   opponent-perspective observation.
6. **Confirm s92 with 32 eval seeds.** Before changing many knobs, rerun s92
   `iteration_0`, `iteration_384`, and `iteration_434` on one fixed 32-seed
   panel.
7. **Run pooled frozen-opponent eval.** Score candidates against a small pool,
   not only the matched frozen opponent used during training.
8. **Check value/reward scale.** Survival returns can be hundreds of steps on
   an Atari MuZero config; verify support scale, discount, target transform,
   and reward/value target ranges.

## Ranked Suspects

1. **Two-seat value target is wrong** `[likely code bug]`  
   The iterative two-seat run passed collect/update/checkpoint plumbing, but
   `target_value` is immediate reward rather than discounted survival return.
   That can erase the long-horizon survival signal even when replay and
   optimizer steps are healthy.  
   Small check: fix discounted survival-return targets, rerun
   `curvytron-two-seat-iterative-b8-s7-4x32-u2-sim2` or the same shape, and
   compare against same-run `iteration_0` over 32 random starts.

2. **Learner batch may be capped at 2 rows** `[likely code/config bug]`  
   The LightZero profile hard-sets `policy.batch_size=2`, and
   `_learn_mode_batches` slices sampled replay to that size. Big collect phases
   may therefore still run learner updates on only 2 replay rows, producing
   noisy or inert training despite large replay counts.  
   Small check: make learner batch size explicit for the two-seat run shape,
   verify sampled batch counts in artifacts, then rerun the 32-start curve.

3. **Missing self-play in the main LightZero lane** `[missing self-play]`  
   The strongest explanation. Static frozen opponents can produce narrow
   best-responses and regress elsewhere.  
   Small check: implement a two-seat/current-policy collector, or continue
   staged refresh only with pooled-opponent eval.

4. **Opponent perspective may be wrong** `[possible code bug]`  
   The frozen-opponent env path copies the same `[4,64,64]` stack into both
   player slots before provider inference. That may give the opponent an
   ego-perspective view.  
   Small check: render true opponent-perspective rows and compare actions,
   survival, and crash traces against the current shared-stack path.

5. **s92 signal may be real but under-sampled** `[eval issue]`  
   s92 matched-opponent eval is the best refresh signal so far, especially
   `iteration_256` through `iteration_384`, but the panel is only 8 seeds.  
   Small check: rerun s92 `0/384/434` on 32 fixed seeds, matched and pooled.

6. **Matched-only eval overstates progress** `[eval issue]`  
   Fixed-baseline curves are mostly flat while matched-opponent curves can look
   much better. That can hide opponent-specific exploitation.  
   Small check: add pooled frozen-opponent eval with the s42/s44/s46/s47
   checkpoint pool before choosing the next refresh parent.

7. **Value/reward scale mismatch** `[possible config bug]`  
   Survival-only returns can be large, while the trainer inherits Atari MuZero
   defaults. Bad support or target scaling could make learning unstable.  
   Small check: dump compiled reward/value support, discount, td steps, and
   target transform; compare predicted values to realized survival returns.

8. **Survival-only reward has weak game pressure** `[objective issue]`  
   Reward is alive-step count only, with no win bonus or death penalty. A policy
   can learn delay behavior rather than robust opponent control.  
   Small check: one short A/B with small terminal outcome shaping while still
   reporting survival first.

9. **s90 trained away from a good initial response** `[expected noise]`  
   s90 matched survival dropped from `659.5` at `iteration_0` to `253.5` at
   latest. The initial checkpoint may have been accidentally good against that
   frozen opponent.  
   Small check: compare s90 `iteration_0` and `iteration_175` step traces on
   the same seeds.

10. **Peak-chased frozen parents are brittle** `[eval issue]`  
   s90/s91/s92 parents were selected from local matched-opponent peaks. Those
   may not be robust teachers.  
   Small check: select future parents by heldout pooled score, not matched peak.

11. **Checkpoint iteration is not the right x-axis** `[eval issue]`  
   LightZero did not strictly obey `max_train_iter`; `max_env_step` is the
   effective bound. Iteration counts differ across runs.  
   Small check: graph by env steps collected or learner train calls from
   summaries.

12. **Debug visual encoding may be too lossy** `[model/input issue]`  
    The current surface is debug occupancy, not source-faithful pixels or richer
    egocentric state. It may be enough for plumbing but unstable for learning.  
    Small check: compare one frozen-opponent stage against scalar/ray rows or a
    small oracle-feature side channel.

## Claim Boundary

Current main-LightZero evidence is learner-vs-frozen-checkpoint only. Do not
call it current-policy self-play. The iterative two-seat smoke is real
current-policy plumbing, but its latest curve has no learning signal and is
blocked on two fixes before scaling: discounted survival-return value targets
and learner batch sizing above 2 replay rows.
