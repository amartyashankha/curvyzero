# Cleanup Targets

Purpose: list stale front doors that can send future agents back into the wrong
lane.

## P0

| Target | Risk | Cleanup |
| --- | --- | --- |
| `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py` default mode and run id | No-arg/default launch still implies custom two-seat is canonical | Done: default mode is now `dry`, default ids are stock-loop-neutral, and two-seat command status says experimental adapter |
| `scripts/launch_curvytron_overnight40_20260512.zsh` | Restarts failed custom two-seat matrix | Done: launcher now hard-fails unless `ALLOW_HISTORICAL_CUSTOM_TWO_SEAT_RERUN=1` |
| `scripts/launch_curvytron_mixpast_20260512.zsh` | Conceptually useful frozen mix, but implemented on custom two-seat path | Done: launcher now hard-fails unless `ALLOW_HISTORICAL_CUSTOM_TWO_SEAT_RERUN=1`; rewrite later only for stock-loop recent-opponent route |
| `docs/working/training/curvytron_canonical_two_seat_handoff_2026-05-12.md` | Body still reads like main launch guidance | Rename/rewrite as custom adapter historical handoff |
| `docs/working/optimizer/coach_next_training_run_recommendations_2026-05-12.md` | Tells Coach to launch the now-superseded matrix | Add superseded banner and link to research folder/gates |
| Optimizer front doors | Still call two-seat Coach baseline in places | Add correction banners or handoff to Optimizer owner |

## P1

| Target | Risk | Cleanup |
| --- | --- | --- |
| `docs/working/training_coach_active_board_2026-05-10.md` | Makes fixed/frozen sound merely secondary | Clarify fixed/frozen can be stock-loop control/curriculum, not live self-play |
| `curvytron_train_muzero_reconciliation_2026-05-12.md` | Mostly right but undersells frozen route | Add "control/curriculum candidate" wording |
| `source_state_turn_commit` metadata | Can claim current-policy self-play while train is blocked | Mark as plumbing/profile only with reward-credit blocker |
| `lightzero_curvytron_visual_survival_eval.py` defaults | Old two-seat/default checkpoint assumptions | Require explicit run/checkpoint or use neutral stock-loop defaults |
| `lightzero_curvytron_run_status.py` | Polls stale live two-seat runs | Add postmortem banner or require explicit run ids |
| old detached Modal pollers | They keep their launch-time code snapshot and will not pick up local GIF/eval fixes | Use explicit backfill/current-code poller jobs; do not expect relaunch-free behavior |
| custom two-seat `to_play` handling | Player ids `0/1` appear to flow into non-board-game MuZero fields | Add assertions before any future custom/native two-seat training, or avoid this path |

## P2

| Target | Risk | Cleanup |
| --- | --- | --- |
| `curvytron_two_seat_reward_contract_2026-05-12.md` | Sounds like active trainer reward contract | Rename framing to custom two-seat adapter reward contract |
| `curvytron_overnight40_launch_2026-05-12.md` | Historical doc still says "Use only" | Add superseded banner |
| `scripts/summarize_curvytron_lightzero_profiles.py` labels | Names two-seat as normal mode | Label custom adapter clearly |
