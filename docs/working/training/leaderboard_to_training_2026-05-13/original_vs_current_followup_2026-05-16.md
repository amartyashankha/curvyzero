# Original Vs Current Follow-Up - 2026-05-16

Status: docs-only follow-up. Production code was not edited.

## Plain Conclusion

The current system is no longer obviously broken at the plumbing level. The
trainer -> checkpoint -> intake -> tournament -> controller -> refreshed frozen
opponents loop has been proven on the current `r18fresh` lane, and the most
obvious seat/perspective tournament bug is not present in the current code.

The learning problem is different: current runs often find a better mid-run
checkpoint and then drift or regress. The strongest suspects are training
dynamics and objective scaling, not checkpoint ingestion.

The closest empirical baseline is still the 2026-05-10 static matched frozen
opponent lane, especially `s92`, not broad live tournament self-play. That
baseline improved against its matched frozen opponent from roughly
`151.8 -> 417.0 -> 500.4` in the stronger 32-seed read. Current `r18fresh`
improved at least once in `18/18` rows, but latest improved in only `10/18`,
with mean first/best/latest survival `159.9 / 246.0 / 175.4`.

## Baseline Compared

I did not find `mu0light0pat` as a literal local identifier. The useful baseline
I found in docs/git history has two parts:

- Empirical CurvyTron baseline: staged learner-vs-static-frozen-checkpoint
  training from 2026-05-10, with `s92` as the cleanest survival-improvement
  example.
- Code/config baseline: LightZero Atari MuZero config, which the current trainer
  still imports and patches.

Git-history note: at the repo baseline (`23c721b`), the source-state survival
env effectively supported only physical player 0. The current code now rejects
old `ego_player_index` config and uses `learner_seat_mode`, defaulting fresh
runs to `random_per_episode`.

## Current Loop And Historical r18fresh Setup

Current v2 names and runtime contracts live in
`src/curvyzero/contracts/curvytron.py`. Current broad launch defaults are in
`../r18fresh_postmortem_2026-05-16/CURRENT_LAUNCH_DEFAULTS.md`; those are now
L4/C256/N256/batch64/sim8. The completed r18fresh batch analyzed below was an
H100/batch32 case-study batch.

Current v2 names:

- Trainer app: `curvyzero-lightzero-curvytron-visual-survival-train-v2`
- Tournament app: `curvyzero-checkpoint-tournament-v2`
- Current arena: `curvy-r18fresh-live-bounded-dsf1-20260516b`
- Current rating run: `elo-r18fresh-live-bounded-dsf1-20260516b`
- `source_max_steps=1_048_576`
- `decision_source_frames=1`
- `save_ckpt_after_iter=10_000`
- assignment refresh interval: `2_000` train iterations
- learner seat default: `random_per_episode`
- policy surface default: `browser_lines + simple_symbols`

Historical r18fresh 18-row run shape:

- 3 reward variants: `sparse_outcome`,
  `survival_plus_bonus_no_outcome`, and
  `survival_plus_bonus_plus_outcome`.
- 3 opponent recipes with blank/wall-avoidant/frozen checkpoint slots.
- 2 noise lanes: clean and straight-override/action-repeat stochasticity.
- Historical practical knobs for that batch: `collector_env_num=256`,
  `n_episode=256`, `num_simulations=8`, `batch_size=32`.

## Top Suspected Regressions

1. Reward/value support saturation.
   Dense survival variants can imply million-scale returns because
   `source_max_steps=1_048_576`, but the LightZero model support is capped at
   `300` unless overridden. This is the highest-signal training bug risk.

2. Too little search for the current task.
   Current `num_simulations=8` is far below stock Atari MuZero's `50`. With only
   three actions this may be workable, but survival is horizon-sensitive and root
   noise can dominate shallow search.

3. Large stale collect waves plus small batches.
   Historical r18fresh `collector_env_num=256`, `n_episode=256`,
   `batch_size=32` is very far
   from the stock `8/8/256` LightZero Atari shape. That can produce noisy
   gradients and stale data chunks even if replay-ratio math is technically
   inherited.

4. Live refreshed opponents may be too nonstationary.
   The old survival signal was static matched frozen-opponent training. The
   current live loop changes opponent assignments while replay still contains old
   opponent distributions. That can explain "best checkpoint improves, latest
   drifts."

5. Stochastic rows can mismatch selected action and executed action.
   In noise rows, the policy-selected action can be overridden to straight or
   repeated. LightZero replay still mainly stores the requested policy action.
   Treat this as a risky diagnostic lane, not the clean baseline lane.

6. Sparse outcome still has weak temporal credit assignment.
   `sparse_outcome` uses terminal win/loss signal over long games while stock
   `td_steps=5` remains unless overridden. Dense reward avoids some of this but
   runs into the support-saturation problem above.

## Already Fixed Or Exonerated

- Player perspective/role randomization: fixed in current code. Fresh manifests
  default to `learner_seat_mode=random_per_episode`; fixed-seat modes are
  diagnostics.
- Tournament seating: current tournament default is balanced/random seating.
  Each physical seat gets its own player-perspective observation slice and
  controls that same physical seat.
- Tournament policy mode: ratings default to eval/greedy mode. Noisy collect mode
  is opt-in and changes the rating context.
- No-op/action handling: the canonical action space is
  `left, straight, right`; `straight` is the explicit no-turn/no-op-like action
  and is legal in tournament/training.
- Opponent immortality intent: public recipes now use `opponent_immortal`; the
  lower-level `opponent_death_mode` is derived at the env boundary. Blank canvas
  no-op entries must be immortal.
- Observation metadata: checkpoints now carry policy surface metadata, including
  trail and bonus render modes, and tournament loading reads these sidecars.
- Tournament GIF defaults: GIF saving defaults to on, samples default to 5 games,
  and new tournament GIFs use the faster playback timing.
- Feedback loop: current docs record proof through generations 9, 10, and 12,
  including env telemetry rows with refreshed assignment shas and provider-load
  success.

## Still Unvalidated

- Whether a support-cap/target-scale change fixes survival retention.
- Whether higher search, e.g. `num_simulations=25` or `50`, improves survival
  without unacceptable throughput loss.
- Whether smaller collect waves and larger batches stabilize latest checkpoint
  quality.
- Whether static no-tournament rows keep latest close to best. The current
  `curvy-r18nofb-staticmix-20260516a` three-row control is the right test, but
  early results are mixed and not enough.
- Whether own-latest no-tournament control is useful. Local code now has a
  same-run assignment producer and manifest-builder flag, but it is not deployed
  or launched yet and should not be treated as working loop proof.
- Golden render parity for every current policy surface under fresh long runs.
  The current production surface is `browser_lines + simple_symbols`; historical
  `body_circles_fast` lanes are diagnostic/invalidated for production claims.
- Whether straight as a first-class legal action encourages no-op collapse. The
  mapping itself is correct, but the learning consequence is still open.

## Recommended Next Control

Run a small, boring static control before another broad launch:

- no tournament feedback into the trainer;
- one immutable opponent assignment or inline mixture;
- clean lane first, no straight override/action repeat;
- `random_per_episode`;
- checkpoint every `10_000`;
- test one support/cadence change at a time.

The first high-value diagnostic shape is:

- keep reward fixed to `survival_plus_bonus_no_outcome` or
  `survival_plus_bonus_plus_outcome`;
- use static opponents;
- try `num_simulations=25`;
- raise batch size toward `128` or `256`;
- explicitly set `model_support_cap` high enough to avoid immediate saturation,
  or scale the dense reward instead of asking the model to represent million-step
  returns directly.

If that control keeps latest near best, the live tournament curriculum is the
main regression. If it still finds mid-run best then regresses, focus on support
scale, search budget, and collect/learn cadence before touching the tournament.
