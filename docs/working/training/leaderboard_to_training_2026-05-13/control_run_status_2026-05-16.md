# Control Run Status, 2026-05-16

Poll window: 2026-05-16 11:15-11:18 EDT. Read-only status pass; no jobs were
killed and no manifests/code were edited.

## Plain Answer

The two already-launched controls are the right controls for the user's current
ask, with one important caveat for own-latest.

- `curvy-r18nofb-staticmix-20260516a` is the clean no-tournament/static
  opponent control. It uses inline mixture opponents, no assignment bank, no
  refresh pointer, and selected rows `r007`, `r009`, `r011`. It answers the
  "fixed mortal/blank/frozen opponents, no tournament feedback" part.
- `curvy-ownlatest-staticmix-20260516a` is the intended no-tournament
  own-recent-checkpoint control. It uses inline static mixtures plus
  `own_checkpoint_opponent_refresh_enabled=true` and refresh interval `2000`.
  It is not true current-policy self-play; it should become "learner versus a
  frozen own previous checkpoint" only after a nonzero own checkpoint is written,
  materialized as an assignment, and applied by refresh.

Current own-latest answer: it has not reached nonzero checkpoints yet; no own
checkpoint assignment has been applied; survival trend is not measurable beyond
the iteration-0 baseline.

## Static No-Feedback Selected3

Manifest/run summary:

- Matrix: `curvy-r18nofb-staticmix-20260516a`
- Launch record:
  `artifacts/local/curvytron_tonight18_manifests/curvy-r18nofb-staticmix-20260516a/selected3.launch.json`
- Selected rows: `r007`, `r009`, `r011`
- Manifest proof: `opponent_source=mixture`,
  `fixed_knobs.assignment_refresh_interval_train_iter=0`,
  `assignment_bank=null`, selected rows have no
  `opponent_assignment_refresh_ref`.
- Runtime proof: `assignment_refresh_event_count=0` and
  `assignment_refresh_applied_count=0` for all three rows.

Latest eval-summary poll:

| Row | Checkpoints | Latest checkpoint | Eval points | First mean | Best mean | Latest mean | Latest vs first |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `r007` | 4 | `iteration_30000` | 3 | `112.625` | `138.125` @ `20000` | `138.125` @ `20000` | `+25.500` |
| `r009` | 3 | `iteration_20000` | 3 | `189.750` | `189.750` @ `0` | `168.250` @ `20000` | `-21.500` |
| `r011` | 5 | `iteration_40000` | 5 | `162.875` | `171.500` @ `20000` | `167.875` @ `40000` | `+5.000` |

Aggregate selected3 survival:

- Mean first/best/latest: `155.083 / 166.458 / 158.083`.
- Latest mean is only `+3.000` over first mean.
- Best mean is `+11.375` over first mean.
- `2/3` rows are improved at latest; `2/3` rows improved at best.
- Best-to-latest aggregate drop is `8.375` steps.
- Latest eval action-collapse flag is `false` for all three rows.

Interpretation: static no-feedback is behaving as a useful control. It is not a
clean learning win yet, but it is also not showing the severe late-regression
shape from the broader live-feedback batch at this early window. Keep watching;
the current signal is "mixed/slightly up", not "solved".

## Own-Latest Selected3

Manifest/run summary:

- Matrix: `curvy-ownlatest-staticmix-20260516a`
- Launch record:
  `artifacts/local/curvytron_tonight18_manifests/curvy-ownlatest-staticmix-20260516a/selected3.launch.json`
- Selected rows: `r007`, `r009`, `r011`
- Manifest proof: `opponent_source=mixture`, `assignment_bank=null`,
  `own_checkpoint_opponent_refresh_enabled=true`, and train refresh interval
  `2000`. Selected rows have no shared/tournament
  `opponent_assignment_refresh_ref`.

Latest status/eval-summary poll:

| Row | Checkpoints | Latest checkpoint | Eval points | Iter-0 mean | Refresh events | Applied refreshes |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `r007` | 1 | `iteration_0` | 1 | `171.000` | 1 | 0 |
| `r009` | 1 | `iteration_0` | 1 | `168.500` | 1 | 0 |
| `r011` | 1 | `iteration_0` | 1 | `186.125` | 1 | 0 |

Own-latest proof gates:

- Nonzero checkpoint reached: no. All three latest checkpoints remain
  `iteration_0`.
- Assignment refresh consumed own checkpoints: no. All three rows have
  `assignment_refresh_applied_count=0`, no latest applied assignment ref, and
  no assignment-env rows in the scanned tail.
- Survival improving: not answerable yet. There is only one eval point per row,
  so the aggregate `iteration_0` mean is `175.208` with no nonzero comparison.

Observed wrinkle: each own-latest row emitted one `kept_previous` refresh event
at train iter `0`. The latest reason in all three rows was a failed Volume reload
because an open TensorBoard event file prevented reload during opponent
assignment refresh. This did not apply a stale assignment, but it means the
own-checkpoint consumption path is not proven yet. The next useful proof is
after the first nonzero checkpoint: expect a run-local assignment/ref and then a
later `decision=applied` event plus env rows showing the applied assignment.

## Bottom Line

Use static no-feedback selected3 as the fixed-opponent/blank no-tournament
control. Use own-latest selected3 as the own-recent-checkpoint no-tournament
control, but keep it marked "not yet through first proof gate" until it writes a
nonzero checkpoint and applies the corresponding run-local assignment.
