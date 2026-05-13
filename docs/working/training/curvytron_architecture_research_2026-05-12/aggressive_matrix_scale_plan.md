# Aggressive Matrix Scale Plan

Date: 2026-05-13

Purpose: make the next large CurvyTron batch big enough to learn from, without
turning it into an uninterpretable Cartesian product.

Status: the first 50-row survivaldiag batch from this plan is launched as
`survivaldiag-v1b-20260513h`. Strict-stop high-cap runtime canaries and real
live survivaldiag status/eval snapshots prove the rich readout fields for the
first-wave exact lanes. Scripted and random/checkpoint opponent blocks remain
placeholders until they are wired, immutable where needed, and canaried.

## Plain Rule

Large is fine. Blindly crossed is not.

The next matrix should spend rows on repeated copies around meaningful opponent
families, not on every combination of every knob.

## Gates First

These are launch blockers for the blocks that depend on them:

| Gate | Blocks blocked |
| --- | --- |
| `survival_plus_bonus_no_outcome` works in stock fixed-opponent trainer | cleared for tiny e2e; still monitor in larger runs |
| Bonus pickup count is used in trainer reward, not only logs | cleared for tiny e2e; still monitor reward components |
| `blank_canvas_noop` opponent passes tests and tiny e2e canaries | cleared for blank-canvas fast/browser tiny canaries |
| strict canary stop uses `--stop-after-learner-train-calls` and Modal app state confirms stopped | cleared for first-wave exact lanes |
| scripted wall-avoidant opponent is wired into the stock trainer and passes e2e canary | scripted opponent blocks |
| random-init or `iteration_0` checkpoint path is immutable and manifest-addressable | random learned frozen blocks |
| separate seed fields are in run names/manifests | repeat-copy blocks |
| upstream status/export carries survival, reward components, bonus, terminal cause, action histograms/entropy, and eval/GIF health | cleared for first-wave exact lanes |

## Staged Blocks

One row means one stock LightZero `train_muzero` run unless noted otherwise.
Default serious rows should use high cap, likely `65536`, and stock `--mode
train`.

| Block | Rows | Varying axes | Question | Main curves |
| --- | ---: | --- | --- | --- |
| 0. Canaries | done for tiny e2e | blank fast/browser, normal fast, reward/support/artifacts | Is the machinery honest enough to trust? | reward terms, survival sanity, artifact health |
| 1. Blank canvas core | 32 | 2 renders x 4 stochasticity levels x 4 copies | Can visual policy learn wall survival with no opponent confound? | survival, reward, action collapse |
| 2. Blank canvas scale copies | +20 to +72 | more copies on best or prechosen medium/high stochasticity levels | Are gains stable across seeds? | copy distribution of latest/best survival |
| 3. Reward ablation | 0 executable now; 4 gated specs | survival-only specs stay non-commanded until a first-class stock reward variant exists | Does bonus reward help exploration or distort survival? | survival, bonus count, reward, crash rate |
| 4. Passive immortal dirty control | 4-8 first | fixed-straight passive immortal, matched renders, few copies | Does the dirty trail-maker help or fake signal? | survival plus GIF/manual checks |
| 5. Scripted wall-avoidant core | 0 until wired; then 24-64 | 2 renders x selected stochasticity x repeat copies | Does a contained trail-maker improve survival learning? | survival, reward, death type, opponent health |
| 6. Scripted policy variants | 0 until wired; then +24 to +48 | wall-avoidant/reflection variants on a stochasticity shortlist | Which geometry creates useful pressure? | survival without collapse |
| 7. Random-init frozen opponents | 0 until immutable/canaried; then 40-80 | 4-8 immutable random checkpoints x matched renders x repeated copies | Does random learned-looking trail pressure help? | survival, variance by checkpoint/copy |
| 8. Ancestor checkpoint controls | 0 executable now; 6 gated specs | old/mid/recent frozen checkpoints, matched renders, 1 copy each after identity/canary gates clear | Are old frozen controls useful under survival reward? | survival first; outcome only secondary |
| 9. Compute sentinels | 12-24 | sim16, C64, B64 on best cells | Does more search/batch buy learning per wall-clock? | survival delta per wall-clock |
| 10. Confirmation block | 40-80 | top 4-8 cells x 5 copies | Which settings replicate? | latest/best survival, reward, collapse flags |

## Scale Levels

At about 50 runs:

- run exact preflight rows;
- run the 32-row blank-canvas core;
- add blank-canvas extra repeats;
- add a tiny passive-immortal dirty control;
- add only a minimal sim16 compute sentinel;
- keep survival-only and ancestor-checkpoint rows as gated specs unless their
  exact reward/opponent gates clear.

This answers whether the clean survival lane is alive.

At about 100 runs:

- add repeated copies around blank-canvas medium/high stochasticity rows;
- add ancestor controls as controls, not claims, only if their checkpoint
  identity and exact-lane canaries clear;
- add scripted wall-avoidant core only if its trainer lane has passed canary.

This answers whether gains are stable and whether opponent family matters.

At 200+ runs, after the relevant gates:

- add random-init checkpoint families;
- add scripted policy variants;
- add compute sentinels;
- add final confirmation copies.

This answers whether a setting works reliably, not just once.

At about 300 runs, if capacity is still fine:

- widen repeat groups on the top cells rather than adding many new knobs;
- add more random-init opponent checkpoints;
- add more reset/training-seed copies for stochastic rows;
- add a second confirmation block for any late-blooming cells;
- add a few extra scripted-opponent geometry variants only if the probe data
  says they are genuinely different.

Do not use the 300-run tier to cross every compute and reward knob. Use it to
measure variance and to avoid being fooled by one lucky run.

## Concrete 100/200-Run Skeleton

This is the current best next-wave shape if the implementation gates pass. It
is a planning skeleton, not part of the running `survivaldiag-v1b-20260513h`
manifest.

| Stage | 100-run shape | 200-run shape |
| --- | ---: | ---: |
| Blank canvas core | 32 rows: `2 renders x 4 stochasticity x 4 copies` | same first 32 |
| Blank canvas extra copies | 16 rows: best 2 stochasticity levels, `2 renders x 2 levels x 4 extra copies` | +20 rows on best 2 levels, `2 renders x 2 levels x 5 copies` |
| Reward ablation | 4-8 rows: survival-only on best blank levels, matched renders, limited copies | no broad expansion unless bonus clearly distorts behavior |
| Scripted wall-avoidant core | 0 unless wired/canaried; otherwise 24 rows: `2 renders x 3 stochasticity x 4 copies` | +20 confirmation rows on best scripted cells only after gate |
| Passive immortal dirty control | 8 rows: `2 renders x 2 stochasticity x 2 copies` | optional +8 only if GIFs show interpretable trail pressure |
| Ancestor controls | 6 rows: old/mid/recent, `2 renders x 1 copy` | +6 max if one ancestor is surprisingly diagnostic |
| Compute sentinels | 2 rows: sim16 on strongest blank cell, matched renders | +12 rows: sim16/C64/B64 on top cell, matched renders and 2 copies |
| Scripted variants | 0 in first 100 | +24 rows: 2 variants, `2 renders x 2 stochasticity x 3 copies` |
| Random-init frozen opponents | 0 in first 100 unless immutable/canaried | +24 rows: 4 immutable random checkpoints, `2 renders x 3 copies` |

This deliberately makes blank canvas the anchor. If blank canvas cannot learn
wall survival, the opponent-family blocks are much harder to interpret.

Launch order should stay adaptive when possible: exact preflight first, then
blank-canvas core. If blank canvas does not improve survival without action
collapse, do not expand opponent-family rows just because capacity exists.

## Measurement Contract

Every serious row should emit enough telemetry to avoid fooling ourselves.

Required curves/fields:

- survival: latest, best, slope, late-bloom, peak-then-crash;
- trainer reward and reward components;
- bonus pickup count;
- terminal cause;
- action histogram and action entropy;
- straight/left/right rates and repeated-action collapse;
- outcome as telemetry only;
- eval/GIF artifact health;
- wall-clock time, train iterations, env steps, and eval count.

For the blank canvas block, outcome should be irrelevant. For scripted and
passive-immortal blocks, opponent health/path state must also be visible so we
can tell real trail pressure from broken opponent physics.

## Repeat Hard vs Repeat Lightly

Repeat hard:

- blank-canvas best stochastic rows;
- scripted wall-avoidant rows;
- random-init checkpoint rows;
- any cell that might become the headline claim.
- any late-blooming cell that would be easy to discard too early.

Repeat lightly:

- passive immortal canaries;
- ancestor checkpoint controls;
- reward ablations;
- sim16/C64/B64 sentinels.

Matched fast/browser render rows are not extra evidence. They are diagnostic
pairs and should share the same logical seed/copy.

## Readout

Primary:

- survival curve;
- trainer reward curve.

Secondary:

- bonus pickup count;
- terminal cause;
- action entropy/collapse;
- late-bloom behavior;
- peak-then-crash behavior;
- wall-clock efficiency.

Outcome is telemetry only. If outcome becomes the main story again, this matrix
failed its purpose.
