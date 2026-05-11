# LightZero Pong Eval Action Collapse Debug - 2026-05-09

## Short Read

The leading explanation is not an action-map bug. It is a collect/eval action
selection mismatch amplified by weak MCTS roots.

LightZero collection is exploratory:

- collect MCTS adds root Dirichlet noise;
- normal collect samples from visit counts with temperature;
- optional epsilon can add more random actions.

LightZero eval is deterministic:

- eval MCTS uses no root noise;
- eval selects `argmax(visit_counts)`;
- `numpy.argmax` picks the lowest index on exact ties.

So training telemetry can show varied actions even when independent eval
collapses to one action. Recent checkpoints have small logits and low-count,
often tied visit distributions. That is enough for eval to settle on one action
for long stretches: `up`, `stay`, or `down` depending on checkpoint/run.

## Evidence From Recent Runs

Post deep-seed run, `iteration_16`:

- train `player_0` actions: `[999, 853, 91]`;
- paired MCTS eval aggregate: `[7285, 4781, 0]`;
- player0-only MCTS eval aggregate: `[3353, 2073, 0]`.

Lag1 shaped knob run, `iteration_64`:

- train `player_0` actions: `[97, 85, 292]`;
- small paired MCTS eval rows all chose only action `2`: aggregate
  `[0, 0, 335]`.

Lagged-opponent smoke, `iteration_8`:

- train `player_0` actions: `[806, 4083, 46]`;
- paired MCTS eval aggregate: `[0, 10269, 0]`;
- player0-only MCTS eval aggregate: `[0, 2296, 0]`.

Frozen checkpoint iter16 smoke:

- train live learner actions: `[19, 19, 78]`;
- frozen opponent actions: `[49, 67, 0]`;
- tiny independent MCTS rows still showed no `down` for LightZero checkpoints.

The collapse direction changes by run, which points away from "down is illegal"
or "action ids are inverted". Baselines emit all three actions, and the env
mapping remains `0=up`, `1=stay`, `2=down`.

## Code Read

Project MCTS scorecard path:

- `scripts/run_dummy_pong_lightzero_mcts_checkpoint_scoreboard.py` labels the
  adapter as `MuZeroPolicy.eval_mode.forward`, all-ones action mask,
  `to_play=[-1]`, `ready_env_id=[0]`.
- `src/curvyzero/training/lightzero_dummy_pong_policy.py` encodes
  `tabular_ego`, builds `action_mask = ones[1,3]`, calls
  `policy.eval_mode.forward(...)`, and extracts `output[0]["action"]`.
- `src/curvyzero/training/dummy_pong_eval.py` uses
  `PongConfig(max_steps=lightzero_max_env_step)` when a LightZero checkpoint is
  present, then runs baseline/checkpoint pairings.
- `src/curvyzero/training/lightzero_dummy_pong_env.py` returns
  `action_mask=[1,1,1]`, `to_play=-1`, and `timestep`.

LightZero source read under `/tmp/lightzero-src`:

- `lzero/policy/muzero.py` collect mode prepares root noise, searches, then
  either samples from visit counts or does epsilon-greedy exploration.
- `lzero/policy/muzero.py` eval mode calls `roots.prepare_no_noise(...)`,
  searches, then selects with `deterministic=True`.
- `lzero/policy/utils.py` uses `np.argmax(visit_counts)` when deterministic.

That explains why train-side telemetry and eval-side actions can disagree even
with the same model and observation contract.

## Tiny Debug Run

Local blockers first:

- `/runs` was not mounted locally.
- plain `uv run` did not have `lzero`.

I downloaded one checkpoint from the Modal Volume:

```sh
modal volume get curvyzero-runs \
  training/lightzero-dummy-pong/lz-dpong-20260509T154530Z-b049f29edb64/checkpoints/lightzero/iteration_16.pth.tar \
  /private/tmp/curvy-lz-iter16.pth.tar --force
```

I patched `scripts/summarize_lightzero_pong_scorecards.py debug-mcts` to read
LightZero's actual `predicted_policy_logits` key.

Then I ran:

```sh
PYTHONPATH=src uv run --with LightZero==0.2.0 \
  python scripts/summarize_lightzero_pong_scorecards.py debug-mcts \
  --checkpoint lightzero:iter16=/private/tmp/curvy-lz-iter16.pth.tar \
  --rows 12 \
  --seed 1701 \
  --opponent-policy random_uniform \
  --max-env-step 1024 \
  --num-simulations 8 \
  --format md
```

Selected result:

```text
row 0 dy=2   logits=[-0.02102, 0.02843, 0.01249] visits=[2,3,3] action=stay
row 1 dy=1   logits=[-0.02110, 0.02878, 0.01245] visits=[2,3,3] action=stay
row 2 dy=0   logits=[-0.02134, 0.02923, 0.01255] visits=[3,3,2] action=up
row 5 dy=-2  logits=[-0.02176, 0.03007, 0.01271] visits=[2,3,3] action=stay
row 10 dy=-2 logits=[-0.01353, 0.02395, 0.00703] visits=[3,3,2] action=up
```

Read: the policy prior prefers `stay` slightly, but visit counts are tiny and
often tied. Eval tie-breaking then matters. This is enough to create apparent
single-action behavior in small-simulation evals.

## Ranked Hypotheses

1. Collect/eval action selection mismatch is the main immediate cause.
   Training uses noise and sampling; eval uses no noise and deterministic
   argmax over visit counts. This directly explains varied train actions with
   collapsed eval actions.

2. The learned policy/search is weak, so deterministic eval has little real
   signal to work with. The debug rows show small logits and low visit-count
   margins. More simulations may smooth this, but the existing 8-sim rows still
   tie often.

3. Low simulation count amplifies tie-breaking. With 2 or 8 simulations and
   three legal actions, `[0,1,1]`, `[1,1,0]`, `[2,3,3]`, and `[3,3,2]` are
   common. `argmax` picks the first max, so ties bias toward lower action ids.

4. Training-side telemetry is not final checkpoint quality. It mixes live
   collection/evaluation behavior and exploratory action selection. It should
   not be compared directly against independent deterministic MCTS action
   histograms.

5. Official load parity remains worth one small falsifier. Current adapter
   strict-loads `checkpoint["model"]` into `policy._model`, which matches the
   LightZero eval source shape, but one official `learn_mode.load_state_dict`
   comparison would close this caveat.

6. `to_play`, action mask, and action mapping are low suspicion. The wrapper
   uses single-agent-style `to_play=-1`, all actions are legal, and baselines
   emit `up/stay/down`.

## Smallest Next Diagnostics

First, repeat first-N debug rows for the collapse direction you care about:

```sh
modal volume get curvyzero-runs \
  training/lightzero-dummy-pong/lz-dpong-lag1-shaped-s7/checkpoints/lightzero/iteration_64.pth.tar \
  /private/tmp/curvy-lz-lag1-shaped-iter64.pth.tar --force

PYTHONPATH=src uv run --with LightZero==0.2.0 \
  python scripts/summarize_lightzero_pong_scorecards.py debug-mcts \
  --checkpoint lightzero:iter64=/private/tmp/curvy-lz-lag1-shaped-iter64.pth.tar \
  --rows 24 \
  --seed 77 \
  --opponent-policy lagged_track_ball_1 \
  --max-env-step 4096 \
  --num-simulations 8 \
  --format md
```

Then run a tiny simulation-count sweep, still first-N only:

```sh
for sims in 2 8 16 25; do
  PYTHONPATH=src uv run --with LightZero==0.2.0 \
    python scripts/summarize_lightzero_pong_scorecards.py debug-mcts \
    --checkpoint lightzero:iter16=/private/tmp/curvy-lz-iter16.pth.tar \
    --rows 24 \
    --seed 1701 \
    --opponent-policy random_uniform \
    --max-env-step 1024 \
    --num-simulations "$sims" \
    --format md
done
```

If those show persistent ties or one-action visits, do not run more scorecards
yet. Add a one-step collect-vs-eval diagnostic on the same observations:

```text
For each saved observation:
  call policy.collect_mode.forward(..., temperature=0.25, epsilon=0.0)
  call policy.eval_mode.forward(...)
  log logits, visits, collect action, eval action
```

The expected result is collect samples varied actions from similar visit
counts, while eval picks a deterministic max/tie-break action.

Finally, run official-load parity only once:

```text
Compare on the same 24 observations:
  current adapter: policy._model.load_state_dict(checkpoint["model"])
  official path:  policy.learn_mode.load_state_dict(torch.load(checkpoint))
  control only:   checkpoint["target_model"]

Log parameter hashes, logits, visit counts, and selected actions.
```

## Commands Run

```sh
rg -n "post-deep-seed|lag1|lagged-opponent|frozen checkpoint|iter16|scorecard|debug-mcts|LightZero|Pong" docs scripts src -S
rg --files docs scripts src | rg "lightzero|pong|scorecard|eval|mcts|working"
ls /tmp/lightzero-src
git status --short
sed -n ... docs/experiments/2026-05-09-lightzero-dummy-pong-post-deep-seed-fix-run.md
sed -n ... docs/experiments/2026-05-09-lightzero-dummy-pong-lag1-shaped-knob-run.md
sed -n ... docs/experiments/2026-05-09-lightzero-dummy-pong-lagged-opponent-smoke.md
sed -n ... docs/experiments/2026-05-09-lightzero-dummy-pong-frozen-checkpoint-selfplay-iter16.md
sed -n ... docs/experiments/2026-05-09-lightzero-dummy-pong-mcts-scorecard.md
sed -n ... scripts/summarize_lightzero_pong_scorecards.py
sed -n ... scripts/run_dummy_pong_lightzero_mcts_checkpoint_scoreboard.py
sed -n ... src/curvyzero/training/lightzero_dummy_pong_policy.py
sed -n ... src/curvyzero/training/dummy_pong_eval.py
sed -n ... src/curvyzero/training/lightzero_dummy_pong_env.py
sed -n ... src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py
sed -n ... /tmp/lightzero-src/lzero/policy/muzero.py
sed -n ... /tmp/lightzero-src/lzero/policy/utils.py
sed -n ... /tmp/lightzero-src/lzero/entry/train_muzero.py
sed -n ... /tmp/lightzero-src/lzero/worker/muzero_collector.py
modal volume get curvyzero-runs training/lightzero-dummy-pong/lz-dpong-20260509T154530Z-b049f29edb64/checkpoints/lightzero/iteration_16.pth.tar /private/tmp/curvy-lz-iter16.pth.tar --force
PYTHONPATH=src uv run --with LightZero==0.2.0 python scripts/summarize_lightzero_pong_scorecards.py debug-mcts --checkpoint lightzero:iter16=/private/tmp/curvy-lz-iter16.pth.tar --rows 12 --seed 1701 --opponent-policy random_uniform --max-env-step 1024 --num-simulations 2 --format md
PYTHONPATH=src uv run --with LightZero==0.2.0 python scripts/summarize_lightzero_pong_scorecards.py debug-mcts --checkpoint lightzero:iter16=/private/tmp/curvy-lz-iter16.pth.tar --rows 12 --seed 1701 --opponent-policy random_uniform --max-env-step 1024 --num-simulations 8 --format md
uv run python -m py_compile scripts/summarize_lightzero_pong_scorecards.py
```

No pytest.

## Files Changed

- `scripts/summarize_lightzero_pong_scorecards.py`
  - debug mode now recognizes `predicted_policy_logits` and `pred_value`.
- `docs/working/lightzero_pong_eval_action_collapse_debug_2026-05-09.md`
  - this note.

## Follow-Up Addendum - 2026-05-09

Completed the requested first-N debug rows for the lagged-opponent smoke
checkpoint:

```sh
modal volume get curvyzero-runs \
  training/lightzero-dummy-pong/lz-dpong-20260509T161735Z-be3728357aad/checkpoints/lightzero/iteration_8.pth.tar \
  /private/tmp/curvy-lz-lagged-smoke-iter8.pth.tar --force

PYTHONPATH=src uv run --with LightZero==0.2.0 \
  python scripts/summarize_lightzero_pong_scorecards.py debug-mcts \
  --checkpoint lightzero:iter8-lagged-smoke=/private/tmp/curvy-lz-lagged-smoke-iter8.pth.tar \
  --rows 24 \
  --seed 1701 \
  --opponent-policy lagged_track_ball_1 \
  --max-env-step 512 \
  --num-simulations 8 \
  --format md \
  --output /private/tmp/curvy-lz-debug-iter8-lagged-smoke.md
```

Result: all 24 rows chose `1:stay`. Every row had visits `[2,3,3]`,
so the stay-only collapse is explained by exact visit ties between stay and
down plus deterministic lowest-index tie-breaking. The policy logits weakly
preferred `down` on every row, roughly `up=-0.046..-0.036`,
`stay=-0.018..-0.013`, `down=0.033..0.035`, so logits alone do not explain
the selected `stay` direction.

The lag1 shaped down-only probe did not run before status was requested. Exact
next commands:

```sh
modal volume get curvyzero-runs \
  training/lightzero-dummy-pong/lz-dpong-lag1-shaped-s7/checkpoints/lightzero/iteration_64.pth.tar \
  /private/tmp/curvy-lz-lag1-shaped-iter64.pth.tar --force

PYTHONPATH=src uv run --with LightZero==0.2.0 \
  python scripts/summarize_lightzero_pong_scorecards.py debug-mcts \
  --checkpoint lightzero:iter64-lag1-shaped=/private/tmp/curvy-lz-lag1-shaped-iter64.pth.tar \
  --rows 24 \
  --seed 77 \
  --opponent-policy lagged_track_ball_1 \
  --max-env-step 4096 \
  --num-simulations 8 \
  --format md \
  --output /private/tmp/curvy-lz-debug-iter64-lag1-shaped.md
```

## Sparse Probe Simulation-Count Sweep - 2026-05-09

Question: using the existing sparse-settings probe checkpoint, does more eval
search break the tied MCTS roots and restore non-collapsed actions?

Checkpoint source from
`docs/experiments/2026-05-09-lightzero-dummy-pong-sparse-settings-probe.md`:

```text
training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s9/checkpoints/lightzero/iteration_16.pth.tar
sha256: df463af5728cd78672f69870c56860d55ffdb3b9cc6e34d09ac628f48e7a2283
```

Local checkpoint used:

```text
/private/tmp/curvy-lz-sparse-h120-iter16.pth.tar
```

Debug setup:

```text
PYTHONPATH=src uv run --with LightZero==0.2.0 \
  python scripts/summarize_lightzero_pong_scorecards.py debug-mcts \
  --checkpoint lightzero:iter16-sparse-h120=/private/tmp/curvy-lz-sparse-h120-iter16.pth.tar \
  --rows 24 \
  --seed 1701 \
  --opponent-policy random_uniform \
  --max-env-step 120 \
  --num-simulations {8,16,32,64} \
  --format jsonl \
  --output /private/tmp/curvy-lz-sparse-h120-sims{N}-debug24.jsonl
```

No wrapper fix was needed. The local `debug-mcts` command in
`scripts/summarize_lightzero_pong_scorecards.py` already accepts
`--num-simulations`; the Modal scorecard wrapper also accepts a single
`--num-simulations` value, though it does not provide a one-command sweep mode.

Results over the first 24 player-0 decisions:

| Sims | Actions `[up, stay, down]` | Selected-action entropy | Max-root ties | Any root ties | Mean top-2 visit margin | Example visits |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 8 | `[5, 19, 0]` | `0.738` bits | `24/24` | `24/24` | `0.00` | `[2,3,3]`, `[3,2,3]` |
| 16 | `[0, 0, 24]` | `0.000` bits | `0/24` | `24/24` | `1.00` | `[5,5,6]` |
| 32 | `[0, 0, 24]` | `0.000` bits | `0/24` | `0/24` | `3.42` | `[9,10,13]`, `[8,11,13]` |
| 64 | `[0, 0, 24]` | `0.000` bits | `0/24` | `0/24` | `6.83` | `[17,21,26]`, `[16,20,28]` |

Read:

- More search does break the exact max-root ties for this checkpoint.
- More search does not improve selected-action entropy. It makes the first-N
  eval policy fully deterministic, choosing `down` on all 24 rows at 16, 32,
  and 64 simulations.
- The root visit distribution remains broad, close to uniform, even when the
  top action is no longer tied. The mean visit-distribution entropy stays near
  `1.55` bits versus the three-action maximum of `log2(3)=1.585`.
- This means the 8-simulation zero-`down` scorecard was partly a tie-breaking
  artifact, but the deeper problem is still weak search/policy signal. Higher
  eval sims choose a different collapsed action rather than producing a useful
  controller.

Implication for next training runs:

- Do not treat a higher eval simulation count as a fix for action collapse.
- Use higher-sim debug rows as a diagnostic to reduce pure tie artifacts, but
  keep reporting action histograms and root tie/margin telemetry.
- Next sparse Pong training should target stronger policy/value signal before
  scaling: longer budget or more updates may be reasonable only if paired with
  fixed heldout scorecards, first-N MCTS root summaries, and survival/loss-delay
  telemetry. The acceptance condition should be mixed eval actions plus better
  survival/score, not just fewer exact MCTS ties.

No pytest was run.
