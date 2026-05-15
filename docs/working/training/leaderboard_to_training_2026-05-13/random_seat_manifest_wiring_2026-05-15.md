# Random Learner Seat Manifest Wiring - 2026-05-15

Current truth: fresh Coach training runs must default to
`learner_seat_mode=random_per_episode`. `fixed_player_0` and `fixed_player_1`
exist only as explicit diagnostics. There is no hidden compatibility default
for new manifests.

## Contract

- Public field: `learner_seat_mode`.
- Current default: `random_per_episode`.
- Supported values:
  - `random_per_episode`: choose learner seat at reset/episode from deterministic
    reset context.
  - `fixed_player_0`: explicit diagnostic only.
  - `fixed_player_1`: explicit diagnostic only.
- Removed field: `ego_player_index`. The env now rejects this config key and
  tells callers to use `learner_seat_mode`.
- Tournament eval has its own seat-order contract. The current tournament
  default is balanced/random physical seating so ratings do not silently favor
  one side.

## Why

The old real18 lane effectively trained from one physical seat. That makes
leaderboard feedback suspect because CurvyTron is a two-seat simultaneous game.
The restart path must train the learner from both perspectives unless a row is
clearly marked as a diagnostic.

## Wiring

The field is carried through:

- `scripts/build_curvytron_tonight18_manifest.py`
- `scripts/submit_curvytron_survivaldiag_manifest.py`
- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`
- `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py`

Perspective semantics live in
`policy_observation_perspective_contract_2026-05-15.md`: the manifest chooses
the learner seat with `learner_seat_mode`; the renderer emits the requested
controlled-player view and does not choose or randomize seats itself.

Fresh tonight18/v2 manifests must include `learner_seat_mode` in every row and
inside each row's `train_kwargs`. Poller/eval kwargs should not inherit this
field accidentally; eval seating is controlled by the tournament/eval contract,
not by the trainer's self-play setting.

## Env Behavior

At reset, the env sets:

- `ego_player_index`
- `opponent_player_index`
- `learner_player_index`
- `learner_player_id`
- `opponent_player_id`
- `learner_seat_mode`

For `random_per_episode`, both seats must occur across deterministic reset
seeds. Step, reward, action mask, observation stack, and opponent policy input
must all use the selected learner/opponent seats for that episode.

## Opponent Immortality

Opponent policy kind and immortality are separate ideas.

- `opponent_policy_kind` says what policy acts.
- `opponent_runtime_mode` says whether the opponent is normal runtime or a
  special blank-canvas/no-op runtime.
- `opponent_immortal` says whether that opponent can die.

The lower-level env still consumes `opponent_death_mode` because that is the
runtime switch it already had. New manifests and slot recipes should express
intent with `opponent_immortal`; normalization may derive
`opponent_death_mode` as an implementation detail.

## Current Manifest Shape

The next real manifest should use:

- `learner_seat_mode=random_per_episode`
- `source_max_steps=1048576`
- `decision_ms=16.666666666666668`
- `source_state_trail_render_mode=browser_lines`
- `source_state_bonus_render_mode=simple_symbols`
- at least about `20%` combined blank/immortal pressure in every row
- frozen leaderboard checkpoint slots with `opponent_immortal=false`
- blank and proactive wall-avoidant sentinel slots with
  `opponent_immortal=true`

Explicit fixed-seat rows are allowed only for a named diagnostic, smoke, or old
artifact replay.

## Tests

Required focused tests:

- manifest builder emits `random_per_episode` by default for all real rows;
- explicit fixed-seat diagnostic manifests can still be emitted on purpose;
- submitter requires the field for grouped fresh training rows;
- env rejects old `ego_player_index` config;
- env random mode uses both seats deterministically;
- env fixed-seat diagnostics preserve the requested seat;
- tournament eval balances physical seats and rates wins by checkpoint identity,
  not by physical seat.

## Launch Gate

Do not launch a new real training batch until:

- the random-seat/env tests pass;
- the tonight18 manifest tests pass;
- the opponent mixture/immortality tests pass;
- tournament balanced-seat tests pass;
- docs name the current defaults without stale compatibility wording.
