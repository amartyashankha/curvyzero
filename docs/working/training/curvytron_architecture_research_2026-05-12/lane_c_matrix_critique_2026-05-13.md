# Lane C Matrix Critique - 2026-05-13

Purpose: critique the proposed next CurvyTron diagnostic matrix from first
principles, using the recovered raw instructions as the anchor. No source code
was edited for this note.

## Raw Records Checked

- `/Users/shankha/.codex/history.jsonl`: records 7064-7107 for the coach role,
  Modal training, eval, docs, and parallel orchestration pattern.
- `/Users/shankha/.codex/history.jsonl`: records 9368, 9369, 9395, 9405,
  9414, 9419, 9422, 9423, and 9431 for the large CurvyTron matrix,
  weak-opponent diagnosis, blank/no-op opponent, repeated seeds, high cap,
  stochasticity, render lanes, wall-avoidant opponents, validation, and
  aggressive-but-critical scale.
- `/Users/shankha/.codex/sessions/2026/05/08/rollout-2026-05-08T19-36-31-019e09f3-29f3-70d2-bcf6-434ef853f539.jsonl`:
  user-message records 52011, 52107, 52237, 52610, 52904, 53389, 53448,
  53720, 53883, 54057, 54101, 54300, 54408, and 54425.
- Recent raw rollout index was scanned for May 13 sessions. Relevant current
  implementation echoes were seen in
  `/Users/shankha/.codex/sessions/2026/05/13/rollout-2026-05-13T01-15-28-019e1fc2-eb26-7443-a6da-4a050de6a5f5.jsonl`
  records 17392, 17396, 17400, 17403, 17407, and 17425, mainly confirming the
  current metadata surface for `blank_canvas_noop`, immortal opponent mode,
  render mode, reward mode, and action-repeat knobs.

## Short Critique

The proposed direction is right: make the next claim about visual survival
learning, not about beating a weak opponent. The risk is still that the matrix
looks organized while quietly becoming a full product of reward, opponent,
render, stochasticity, seed, search, collector, and batch knobs.

The first matrix should not ask "which CurvyTron opponent family is best?" It
should ask a simpler question:

```text
Can stock LightZero learn longer survival from visual input when the opponent
cannot provide a cheap outcome shortcut?
```

That makes `blank_canvas_noop` the correct first anchor. Passive immortal,
ancestor checkpoints, random policies, and scripted wall-avoidant opponents are
second-order once the blank lane either learns or clearly fails.

## Recommended First Wave

Run only after the remaining manifest/readout/high-cap gates pass.

Target size: about 40 to 50 rows, not 100+ yet.

Main block:

- `blank_canvas_noop`
- `survival_plus_bonus_no_outcome`
- `source_max_steps=65536`
- matched `body_circles_fast` and `browser_lines`
- stochasticity levels `none`, `low`, `medium`, `high`
- 3 matched copies per level/render pair

That is `2 renders x 4 stochasticity levels x 3 copies = 24` rows.

Add a small reward sanity block:

- `survival_only` only on the best-looking 1 or 2 blank-canvas stochasticity
  levels, or on a prechosen medium level if this is launched all at once.
- Keep matched renders.
- 2 to 3 copies.

Add a dirty control block:

- passive immortal fixed-straight, explicitly labeled dirty;
- 2 stochasticity levels only: `none` and the best planned nonzero level;
- matched renders;
- 2 copies.

Add no random-init, no ancestor expansion, and no scripted wall-avoidant
training rows in this first wave unless their exact trainer lanes and telemetry
have passed gates. They are too easy to misread before the blank result exists.

First-wave readout should promote nothing unless survival improves on held-out
eval seeds, action entropy is not collapsed, reward moves for the right reason,
and GIF/manual checks match the metric story.

## Recommended Second Wave

Only launch if first wave shows either a credible blank-canvas survival signal
or a sharply diagnosed blank-canvas failure.

If blank canvas learns:

- expand repeats on the best 2 blank-canvas cells to about 5 to 8 copies;
- add scripted wall-avoidant rows if trainer integration and probes are
  complete: `2 renders x 2-3 stochasticity levels x 4-5 copies`;
- add small ancestor controls: old/mid/recent, matched renders, 1 copy each;
- add one passive immortal confirmation pair only if first-wave GIFs were
  interpretable;
- add sim16/C64/B64 sentinels only on the strongest cell, matched by render and
  seed.

If blank canvas fails:

- do not add richer opponents yet;
- audit observation, reward, target support, action masks, eval status, and
  action-collapse behavior;
- add only narrow diagnostic rows, for example sim16 or C64 on the same blank
  cell, to test whether search/collection is the blocker.

Random-init frozen or iteration-0 checkpoint families belong after this. They
need immutable checkpoint identities plus `opponent_policy_seed` in the
manifest, and should get about five copies per meaningful setting.

## Axes To Hold Fixed

- Trainer path: stock `train_muzero`, `--mode train`, no custom
  `two-seat-selfplay`.
- Reward for the main lane: `survival_plus_bonus_no_outcome`.
- Outcome reward: zero/off for training; outcome remains telemetry only.
- Episode cap: high, e.g. `source_max_steps=65536`; do not sweep it.
- Compute baseline: L4/T4 unless profiling later proves otherwise.
- Main search/collector/learner: sim8, C32/n32, B32.
- Action-repeat knobs: hold defaults in first wave; do not mix repeat
  stochasticity with straight-override stochasticity.
- Opponent family inside a block: one block, one question.
- Eval seed panel: keep fixed held-out eval seeds for comparable curves.

## Axes To Sweep

- Stochasticity: no, low, medium, high, but define exactly what is perturbed,
  which actor is perturbed, whether it applies to collection or eval, and how
  it is named.
- Render: matched `body_circles_fast` and `browser_lines` rows for serious
  cells, sharing `logical_pair_id`, seed fields, and copy id.
- Repeated copies: vary only the relevant seed fields for the row family.
  Blank/stochastic rows should vary training/reset seeds; random-opponent rows
  should vary `opponent_policy_seed`; noisy-opponent rows should vary
  `opponent_behavior_seed`.
- Opponent family: stage, do not cross. Blank first, passive immortal dirty
  control second, scripted wall-avoidant after trainer gate, random-init after
  checkpoint/seed gate, ancestor controls lightly.
- Compute knobs: sim16, C64, B64 as sentinels on selected winning cells only.

## Missing Gates

- High-cap live canary with `source_max_steps=65536` that exercises real
  train/eval/GIF artifact paths, not only dry config construction.
- A truncation/action-mask smoke so long-horizon timeout paths do not hit the
  known all-zero mask or probability-size failure when policies eventually
  survive.
- Matrix manifest generator must carry explicit axis fields for reward
  weights, opponent runtime/death/collision/visibility, stochasticity
  semantics, render pairing, all relevant seeds, `copy_id`, `block_id`, and
  `hypothesis_id`.
- Eval/status export must carry reward components, bonus counts, terminal
  cause histograms, action histograms/entropy, eval health, GIF health, and
  opponent path/health state.
- Blank-canvas subprocess collector canary: not just local wrapper tests or a
  tiny base-env canary.
- Fast/browser observation-pair sanity: same seed/copy should show only the
  intended renderer difference, not hidden opponent pixels or lifecycle
  artifacts.
- Passive immortal telemetry: bounds violations, death-suppression counts,
  trail body counts, owner collision counts, and opponent position health.
- Scripted wall-avoidant trainer gate: first-class policy integration,
  source-state access contract, long real-env probe, noncollapsed action mix,
  OOB count, and explicit collision semantics.
- Random-init checkpoint gate: immutable checkpoint creation, strict-load proof,
  manifest-addressable opponent identity, and `opponent_policy_seed`.
- Pre-registered stop/promote rules so a 200-row expansion cannot be justified
  after the fact by whichever metric happened to move.
