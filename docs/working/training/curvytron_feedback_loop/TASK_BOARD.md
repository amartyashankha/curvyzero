# Task Board

Last updated: 2026-05-16.

| ID | Priority | Lane | Status | Next Action | Done When |
| --- | --- | --- | --- | --- | --- |
| GEN-OBS-1 | P0 | lineage events | lifecycle-proven-local-reran-20260516-1623 | Run a deployed current-code canary and compare Volume/Dict artifacts to the local proof. | One deployed checkpoint has ordered events from write through trainer apply. |
| GEN-OBS-2 | P0 | operator health | ready | Add CLI/UI health readout for arena, rounds, row counts, export, and trainer consumption. | One readout explains current state without manual artifact joins. |
| GEN-POL-1 | P0 | policy observation | verified-local | Include in next synthetic loop proof. | Fixed-seat, random-seat, tournament-seat, tournament loader, frozen-provider, and trainer telemetry focused tests prove controlled-player `self`/`other` semantics for the gray64 model tensor. |
| GEN-POL-2 | P0 | raw observation uniformity | verified-local | Decide whether raw RGB artifact accessors should be renamed, rejected, or made perspective-aware. | A fixed source state has documented pixel/channel parity across both controlled-player views, trainer/tournament paths use that contract, and incompatible/contradictory checkpoint metadata fails closed. |
| GEN-POL-3 | P0 | provider contract | verified-local | Include in next broader focused test bundle. | Trainer frozen-opponent provider rejects missing/wrong/contradictory observation metadata, training-candidate refresh refuses untyped source checkpoints, and control-volume checkpoint copies get clean policy metadata sidecars. |
| GEN-TEL-1 | P1 | trainer telemetry | verified-local | Surface these fields in operator health/readout if needed. | Step telemetry records learner seat, opponent seat, learner player id, and controlled-player perspective fields. |
| GEN-TEST-1 | P0 | synthetic loop | lifecycle-proven-local | Keep `test_checkpoint_intake_rating_leaderboard_assignment_trainer_lineage_chain` in the focused gate. | Local test proves synthetic checkpoint write -> intake -> rating -> leaderboard publish -> assignment -> trainer load/apply with matching refs and assignment SHA. |
| GEN-LIVE-1 | P1 | current-code canary | remote-gap-found | Fix launch/refresh wiring so the training-candidate refresh rewrites the exact pointer watched by the canary trainer; rerun deployed proof. | Current deployed code proves checkpoint -> tournament -> assignment -> same-trainer provider-load OK. |
| GEN-OPS-1 | P1 | asset hygiene | active-cleaned-20260516-1809 | Keep current asset registry before kill/launch/sleep. | Every live arena/rating/app/run has status and preserve rule. |
| GEN-DOC-1 | P1 | docs cleanup | active | Demote stale `r18fresh` wording from architecture/default language to case-study/evidence language. | General contracts point at this folder; batch docs clearly mark live-instance IDs as historical/current-instance evidence only. |
| GEN-SCHED-1 | P0 | tournament scheduling research | active-local-gate-added | Use the 18-checkpoint fanout result as the current remote gate; next probe should raise roster size, not change scheduler semantics. | We can name a bounded scheduler candidate and its known failure modes before changing production behavior. |
| GEN-SCHED-2 | P0 | top-100 protection | active-local-gate-added | Keep `test_toy_weak_new_batch_is_placed_then_retired_from_top100` and scale probes green while changing scheduler behavior. | Strong new policies are not dropped before enough evidence, or the scheduler reports that risk loudly. |
| GEN-SCHED-3 | P1 | scheduler observability | ready | Define per-wave readouts: budget expansion, distinct opponents, anchor concentration, zero-appearance rows, repeat share, and graph health. | Operator can tell why a round was scheduled and whether the evidence is broad enough. |
| GEN-REF-1 | P1 | tournament cleanup | active | Use `TOURNAMENT_REFACTOR_BRIDGE.md` to plan behavior-preserving extraction of pure scheduler/rating/artifact helpers. | Large files shrink through tested cuts without changing tournament behavior. |
| GEN-MODAL-1 | P0 | Modal game-worker scaling | complete-18-remote | Scale probes may use `save_gif=false`; live tournament intake must repair GIFs back on. | Game workers have no hidden `max_containers` ceiling, `games_per_shard` defaults to 1, warm-pool settings are visible, fanout timing is measured, and live tournament battles keep GIF samples. |
| GEN-MODAL-2 | P0 | Volume pressure | active-small-patch-landed | Add aggregate commit/reload timing to progress or latest snapshots. | Game-level fanout is fast without making live status/recovery muddy. |
| GEN-BATCH-1 | P0 | trainer slot batches | verified-local | Include in next deployed canary before launching a new broad batch. | Opponent recipes are explicit power-of-two slot-count bags; the normal assignment refresh path expands them deterministically over the collector env count and records split telemetry. |

Batch-specific analysis remains in postmortem folders. This board is for
general system contracts.

## 2026-05-16 18:09 EDT Cleanup Note

- Stopped active ephemeral `curvyzero-checkpoint-tournament-v2` apps from stale
  detached runs.
- Kept deployed `curvyzero-checkpoint-tournament-v2`,
  `curvyzero-lightzero-curvytron-visual-survival-train-v2`, and
  `curvyzero-curvytron-gif-browser-v2` running.
- Deleted only transient `curvy-scale-probe-*` tournament directories from the
  tournament v2 volume.
- Preserved r18fresh, bounded r18fresh, e2e proof artifacts, runs volume, and
  control pointers.

## 2026-05-16 18:28 EDT Tournament Volume Trim

- Cleaned only stale tournament battle detail from
  `curvyzero-curvytron-tournaments-v2`.
- Removed `battles/` and `battle_index.json` from:
  - `curvy-r18fresh-live-20260516a`
  - `curvy-r18fresh-live-bounded-20260516a`
- Removed `battles/` from:
  - `curvy-r18fresh-validate-all205-20260516a`
- Hid those three stale arenas from the tournament browser.
- Preserved `ratings/`, `intake/`, `checkpoints/`, `tournament.json`,
  `leaderboards/`, `curvyzero-runs-v2`, and `curvyzero-curvytron-control-v2`.
