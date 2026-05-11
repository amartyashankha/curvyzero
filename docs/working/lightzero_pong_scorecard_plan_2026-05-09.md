# LightZero Dummy Pong Scorecard Plan - 2026-05-09

Worker lane: independent scorecard plan only. This note does not implement a
LightZero adapter, exporter, or evaluator.

## Goal

Score LightZero `.pth.tar` dummy Pong checkpoints with CurvyZero-owned
evaluation telemetry, independent of LightZero's native training/eval summary.
The first narrow policy-head scorecard exists. A 512/8 `iteration_8` MCTS
loader smoke passed strict full-model load and one eval-mode forward call. Full
MCTS/eval-mode scorecards have now run on Modal, including corrected horizon
and longer-training reads. The scorecard should answer a narrow question:

```text
Given this LightZero checkpoint, how does the ego policy perform in the
project-owned dummy Pong environment against fixed baselines and peers?
```

It must not claim full Curvy self-play. The planned first case is
LightZero-controlled `player_0` versus a scripted `player_1`.

## Current State

- Tiny LightZero MuZero dummy Pong Modal train passed: run
  `lz-dpong-20260509T141607Z-3696aa333028`, attempt
  `attempt-20260509T141607Z-98662e4917b4`.
- Mirrored checkpoints: `ckpt_best.pth.tar`, `iteration_0.pth.tar`,
  `iteration_2.pth.tar`.
- Early probe passed policy-head access but strict full model load failed
  because the dynamics keys in the tiny checkpoint did not match the
  instantiated dynamics module names.
- Probe ref:
  `training/lightzero-dummy-pong/lz-dpong-20260509T141607Z-3696aa333028/attempts/attempt-20260509T141607Z-98662e4917b4/probe/lightzero_checkpoint_probe_20260509T143137Z.json`.
- Current 512/8 MCTS loader smoke passes. Probe ref:
  `training/lightzero-dummy-pong/lz-dpong-20260509T144635Z-eb5a0ed35de0/attempts/attempt-20260509T144635Z-ece79bad80d0/probe/lightzero_mcts_loader_smoke_20260509T145607Z.json`.
  Details: `ok: true`, `mcts_eval_status: ok`,
  `strict_full_model_load_ok: true`, strict-load variant
  `res_connection_in_dynamics_true`. Call shape was `data [1,10]`,
  `action_mask [[1,1,1]]`, `to_play [-1]`, `ready_env_id [0]`.
  Eval-mode forward returned action `0`, visit counts `[2,1,1]`,
  policy logits `[0.0170983, 0.00644484, 0.0132326]`, predicted value about
  `0.0000259`, and searched value about `0.000114`.
- Full 512/8 MCTS/eval-mode scorecard ran on Modal:
  `https://modal.com/apps/modal-labs/shankha-dev/ap-Ou59sqrdljB295FFBpyIUP`.
  Summary ref:
  `eval/lightzero-dummy-pong/mcts-scoreboard-512x8-iter8-20260509T150000Z-20260509T150243Z/summary.json`.
  Episodes ref:
  `eval/lightzero-dummy-pong/mcts-scoreboard-512x8-iter8-20260509T150000Z-20260509T150243Z/episodes.jsonl`.
  Checkpoint: 512/8 `iteration_8.pth.tar`; 16 episodes per seating;
  `num_simulations=8`; strict full model load OK with
  `strict_full_model_load_variant=res_connection_in_dynamics_true`.
- MCTS rows: vs `lagged_track_ball_1`, LightZero won 13 and opponent won 15,
  mean survival 25.09, LightZero shaped -0.0397, LightZero reward -0.0625,
  actions `[801,2,0]`. Vs `random_uniform`, LightZero won 17 and opponent won
  15, mean survival 13.84, LightZero shaped 0.0953, LightZero reward 0.0625,
  actions `[443,0,0]`. Vs `track_ball`, LightZero won 0 and opponent won 30,
  mean survival 25.66, LightZero shaped -0.8618, LightZero reward -0.9375,
  actions `[816,5,0]`.
- Read: MCTS eval-mode is no longer just a loader smoke. Full episode eval
  works. But checkpoint behavior is still effectively up-only: combined
  LightZero action histogram `[2060,7,0]`; it never chose down in this
  scorecard.
- Config bug fixes landed after the first MCTS scorecard:
  `DummyPongLightZeroEnv.random_action()` now uses a persistent per-episode
  RNG, the base observation includes `timestep`, and LightZero scorecards use
  `PongConfig(max_steps=lightzero_max_env_step)` for LightZero checkpoint eval.
  Baseline-only eval stays on the default 120-step horizon.
- Corrected 512/8 `iteration_8` scorecard ran with `max_env_step=512`:
  Modal URL
  `https://modal.com/apps/modal-labs/shankha-dev/ap-zUwRanuyB0OHCA8NdpOHVQ`,
  summary
  `eval/lightzero-dummy-pong/mcts-scoreboard-512x8-iter8-maxstep512-20260509T151544Z/summary.json`.
  Config check: `config.max_steps=512`,
  `lightzero_eval_config.max_env_step=512`, `num_simulations=8`.
  Aggregate LZ actions `[4849,19,0]`, still effectively up-only, 0 down.
  Vs lagged: LZ 14, lagged 14, mean steps 74.44, actions `[2373,9,0]`.
  Vs random: LZ 14, random 18, mean steps 12.81, actions `[407,3,0]`.
  Vs track: LZ 0, track 29, mean steps 64.88, actions `[2069,7,0]`.
  Horizon mismatch was real and fixed for future evals, but it is not the root
  cause of up-only.
- Long CPU train succeeded after scaled wrapper caps were intentionally changed
  to `8192/64`: Modal URL
  `https://modal.com/apps/modal-labs/shankha-dev/ap-Uj3XVYgEnzr9oSVan3NILH`,
  run `lz-dpong-20260509T151212Z-b95b61de2eb0`, attempt
  `attempt-20260509T151212Z-8b9db08f8fcb`, train summary
  `training/lightzero-dummy-pong/lz-dpong-20260509T151212Z-b95b61de2eb0/attempts/attempt-20260509T151212Z-8b9db08f8fcb/train/summary.json`.
  `iteration_64` is
  `training/lightzero-dummy-pong/lz-dpong-20260509T151212Z-b95b61de2eb0/checkpoints/lightzero/iteration_64.pth.tar`,
  sha256 `11a0cc80f797ce8e63150e0a6018efc163b7858bed9efd92b77dda8cadaf95e4`.
  Trainer-side telemetry: 578 episodes, 535 wins, 43 losses, no timeouts,
  survival mean 18.18, median/p90 19/19, max 52, score mean 0.8512, shaped
  mean 0.8513, player_0 actions `[9539,800,170]`.
- Post-train independent MCTS scorecard for `iteration_64` used
  `max_env_step=4096`, `num_simulations=8`: Modal URL
  `https://modal.com/apps/modal-labs/shankha-dev/ap-G8BlfW9uUBtT7jTKxgtx0U`,
  summary
  `training/lightzero-dummy-pong/lz-dpong-20260509T151212Z-b95b61de2eb0/attempts/attempt-20260509T151212Z-8b9db08f8fcb/eval/mcts-scoreboard-4096x64-iter64-maxstep4096/summary.json`.
  Vs random: LZ 13, random 19, mean steps 15.91, shaped -0.1862, actions
  `[290,219,0]`. Vs lagged: LZ 11, lagged 19, mean steps 266.25, shaped
  -0.2492, actions `[6475,2045,0]`. Vs track: LZ 0, track 31, mean steps
  144.34, shaped -0.9668, actions `[3335,1284,0]`.
  Read: longer training moved from almost pure up to up+stay, but still 0 down
  and does not beat random/scripted baselines. This points away from
  simply-too-short and toward a trainer/eval mismatch or remaining
  objective/wiring issue.
- Greedy policy-head scoreboard refs:
  `eval/lightzero-dummy-pong/policy-head-scoreboard-20260509T143955Z/summary.json`
  and
  `eval/lightzero-dummy-pong/policy-head-scoreboard-20260509T143955Z/episodes.jsonl`.

Policy-head scoreboard rows:

| Opponent | LightZero wins | Opponent wins | Truncations | Mean score | Shaped mean | Mean steps | p90 steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `random_uniform` | 21/40 | 19/40 | 0 | 0.05 | 0.081875 | 13.775 | 20.1 |
| `lagged_track_ball_1` | 16/40 | 20/40 | 4 | -0.1 | -0.0764583 | 21.95 | 29.1 |
| `track_ball` | 0/40 | 38/40 | 2 | -0.95 | -0.8736458 | 24.325 | 41.0 |

This is greedy policy-head eval only. It is not MCTS eval and not proof of a
good learned policy. Raw matchups show `lightzero_best` used constant up in
every LightZero row: `[N, 0, 0]`. Do not treat the 21/40 row versus
`random_uniform` as learning proof.

After the MCTS loader fix, a strict-config direct policy-head rerun also
completed:

```text
eval_id: policy-head-scoreboard-512x8-strictcfg
modal_url: https://modal.com/apps/modal-labs/shankha-dev/ap-Q7sPmscebJQWisowuweBxV
summary: eval/lightzero-dummy-pong/policy-head-scoreboard-512x8-strictcfg-20260509T145841Z/summary.json
episodes: eval/lightzero-dummy-pong/policy-head-scoreboard-512x8-strictcfg-20260509T145841Z/episodes.jsonl
```

`load_state_dict` is strict true through the direct policy-head path, so the
split residual dynamics config fix is working. The scorecard is still
constant-up and still not MCTS proof:

| Opponent | LightZero wins | Opponent wins | Truncations | Action histogram | Shaped mean | Mean steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `lagged_track_ball_1` | 13/32 | 17/32 | 2 | `[590,0,0]` | -0.0987 | 18.4375 |
| `random_uniform` | 12/32 | 20/32 | 0 | `[388,0,0]` | -0.2134 | 12.125 |
| `track_ball` | 0/32 | 28/32 | 4 | `[968,0,0]` | -0.8115 | 30.25 |

## Minimum Next Path

1. Produce a real LightZero dummy Pong checkpoint.
   - Source format: LightZero `.pth.tar`.
   - Required provenance: run id, attempt id, config, seed, feature mode,
     observation schema, action schema, opponent policy, and checkpoint hash.
   - Current status: done at tiny smoke scale for run
     `lz-dpong-20260509T141607Z-3696aa333028`, attempt
     `attempt-20260509T141607Z-98662e4917b4`.
   - Mirrored checkpoints: `ckpt_best.pth.tar`, `iteration_0.pth.tar`,
     `iteration_2.pth.tar`.
   - The smoke also wrote 5 env-side terminal rows: 4 wins, 1 loss,
     0 truncations against `random_uniform`. This is not an independent
     scorecard.

2. Harden the LightZero policy loading boundary.
   - A direct policy-head `.pth.tar` path exists for greedy scoring.
   - Strict full-model loading and one MCTS/eval-mode forward now pass for the
     512/8 `iteration_8` checkpoint. The stale missing-`cfg.policy.device` and
     action-mask issues are fixed.
   - The full MCTS/eval-mode scorecard now runs. Another policy-head scoreboard
     is not the decisive artifact.
   - The small env/helper footguns and explicit LightZero eval horizon are
     fixed. Corrected MCTS eval still shows 0 down actions.
   - The adapter/exporter must map dummy Pong observations to exactly the
     feature shape used during LightZero training, likely initial
     `tabular_ego` shape `(10,)`.
   - It must return one ego action in the existing action schema:
     `0=up`, `1=stay`, `2=down`.
   - Treat the current policy head as constant-up until proven otherwise.

3. Reuse the independent dummy Pong scoreboard shape.
   - Existing local entry point:
     `scripts/run_dummy_pong_checkpoint_scoreboard.py`.
   - Existing evaluator:
     `src/curvyzero/training/dummy_pong_eval.py`.
   - Existing Modal wrapper:
     `src/curvyzero/infra/modal/dummy_pong_scoreboard_attempt.py`.
   - Today these load CurvyZero supervised raster `.npz` checkpoints and the
   first LightZero policy-head path. Full LightZero MCTS/eval-mode scoring
     across episodes/opponents now exists for the 512/8 `iteration_8`
     checkpoint.

4. Run the scorecard outside the training process.
   - Inputs: checkpoint label/path or Volume ref, eval split id, seed,
     episode count, feature mode, opponent ladder, and adapter/exporter
     version.
   - Outputs: `summary.json`, `episodes.jsonl`, checkpoint input summaries,
     Modal run/attempt manifests, and compact scoreboard rows.
   - The scorecard should be repeatable without importing the LightZero
     trainer.

## Required Telemetry Fields

Every LightZero checkpoint row must preserve at least:

- `checkpoint_label`, `checkpoint_path_or_ref`, resolved path, byte size, and
  `sha256`.
- `checkpoint_schema_id` or LightZero checkpoint format/version when known.
- `adapter_schema_id` or export schema id.
- LightZero run id, attempt id, config ref, and checkpoint iteration/name.
- `feature_mode`, observation schema id, action schema id, reward schema id,
  ruleset id, and opponent policy id.
- base seed, per-episode seed, eval split id, split role, paired-seat flag,
  and seed-generation rule.
- pair group id, match id, player-policy mapping, episodes, and seatings.
- wins/losses by policy.
- truncations and truncation rate.
- mean, median, p90, min, max, and std survival steps.
- score return stats by policy.
- shaped loss-delay return stats by policy.
- action histograms by player and policy.
- final winner, final score reward, episode steps, max steps, truncated flag,
  last hit, and a joint-action trace hash or full trace ref.

The shaped loss-delay return is diagnostic, not the environment reward:

```text
win:     +1.0
loss:    -1.0 + 0.5 * (episode_steps / max_steps)
timeout:  0.0
```

The honest dummy Pong environment reward remains:

```text
ego scores:       +1
opponent scores:  -1
no score event:    0
```

## Baseline Ladder

Use the current fixed baseline ladder unless the main training coach changes
it:

- `random_uniform`: sanity floor.
- `lagged_track_ball_1`: scoreable scripted target.
- `track_ball`: survival/tie floor.
- learned-vs-learned rows only after two or more LightZero checkpoints can be
  loaded through the same adapter/export path.

## What Is Not Implemented

- No exporter from LightZero checkpoint format to CurvyZero policy format
  exists.
- The current mirrored `ckpt_best.pth.tar` has Torch keys `last_iter`,
  `last_step`, `model`, `optimizer`, and `target_model`; its policy head shape
  is `(3, 32)`.
- The current greedy policy-head scoreboard proves remote artifact plumbing and
  policy-head access, not LightZero policy quality. Its raw matchups expose
  constant-up behavior. The MCTS scorecard is better eval plumbing, but it
  still shows effectively up-only behavior.
- Telemetry cleanup is required: `scoreboard_rows` omit
  `action_histogram_by_policy`, but raw matchups include it. Summary rows
  should surface that field so constant policies are visible immediately.
- Trace hashing is required by the LightZero plan, but the current
  `EpisodeRecord` stores action counts rather than a full joint-action trace
  or trace hash.

## Biggest Unknown

The biggest remaining unknown is not strict load, whether MCTS eval can run
full episodes, the fixed horizon mismatch, or simple run length. MCTS eval can
run full episodes; matching-horizon 512-step eval still chose 0 down; longer
training moved to up+stay but still chose 0 down and failed the independent
baseline ladder. The next blocker is likely a trainer/eval mismatch or a
remaining objective/wiring issue.
