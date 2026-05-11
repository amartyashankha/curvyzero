# No-Bonus Multiplayer Source Fixture Gaps

Status: concise source-fixture checklist
Date: 2026-05-11

Scope: CurvyTron JS source facts for 2P/3P/4P multiplayer. This is a
source-spec checklist only. It does not claim fast env, public env, replay,
learned observation, visual, bonus, browser wire, or pixel coverage.
Current lifecycle source hash: `SOURCE_LIFECYCLE_RULES_HASH=source-lifecycle-v25`.
There are 28 registered source lifecycle fixtures.

## Covered Source Fixture Slice

- 2P lifecycle: warmup/PrintManager start, next round, heading rejection retry,
  max-score match end, and one active mid-round `removeAvatar` immediate
  round-end leave proof.
- 3P lifecycle: spawn order, warmup/PrintManager start, all-dead next round,
  survivor scoring/next round, present/absent first round, present/absent
  survivor scoring, present/absent next round, present/absent tied max-score
  continuation, unique max-score match end, tied max-score continuation, one
  all-present multi-round match end, and one active mid-round `removeAvatar`
  continuation through later round end. It also has one focused source proof
  for `removeAvatar` during warmdown after `round:end` and before `game:stop`,
  and one active leave-edge source proof where removing avatar 2 leaves only
  avatar 1 alive.
- 4P lifecycle: spawn order, all-present all-dead next round, survivor
  scoring/next round, one active mid-round `removeAvatar` continuation through
  later round end, present/absent first round, present/absent survivor
  scoring, present/absent next round, present/absent tied max-score
  continuation, tied max-score continuation, unique-leader match end, and one
  all-present multi-round match end. These are promoted under
  `source-lifecycle-v25`: JS oracle plus Python source-runner. The focused
  public metadata proofs now cover unique-leader, tied max-score,
  all-present multi-round, and the three 4P present/absent reset/survivor/
  next-round fixtures. One focused 3P staged match-mode warmdown leave
  metadata proof is also green. Broad public warmdown leave, broader leave
  edge variants, trainer, replay, visual, and bonus claims are still open. The
  4P mid-round leave fixture has focused public continuation parity, but no
  broad leave claim.
- 3P/4P direct wall canaries: seeded no-bonus wall scoring/death-order checks
  exist, but they are not lifecycle/reset/warmdown coverage.

## Recently Promoted Source Proofs

| Fixture | Narrow source claim |
| --- | --- |
| `source_lifecycle_present_absent_3p_tie_at_max_score.json` | A focused 3P present/absent tied max-score leader set continues to the next round under `source-lifecycle-v25`. |
| `source_lifecycle_present_absent_4p_tie_at_max_score.json` | A focused 4P present/absent tied max-score leader set continues to the next round under `source-lifecycle-v25`. |
| `source_lifecycle_remove_avatar_during_warmdown_3p.json` | A focused 3P warmdown leave after `round:end` does not re-score, does not emit another `round:end`, and carries the leaver as non-present into the next round. |
| `source_lifecycle_remove_avatar_to_single_present_3p.json` | A focused active 3P leave edge: avatar 3 dies first and enters `deaths=[3]`; removing live avatar 2 emits `die` then `player:leave`, sets avatar 2 non-present/non-alive, does not add avatar 2 to current deaths, immediately ends the round because only avatar 1 remains alive, gives avatar 1 `roundScore=2` from total avatar count, and continues to next round because avatar 3 is still present. Focused public metadata parity is green. |

## Exact Source Fixture Gaps Left

| Gap | Suggested fixture id | Rule to isolate |
| --- | --- | --- |
| Broader leave before warmdown stop | 4P and zero-present variants | The focused 3P warmdown proof and focused 3P single-present active leave-edge proof are promoted. Add broader variants only if they isolate a new source rule, such as match end after present count collapses further. |
| Broader present/absent match variants | Add only if a new rule appears beyond the promoted 3P/4P present/absent tie-at-max fixtures. | Keep public parity and broader source variants separate. Do not list the promoted tie-at-max fixtures as missing. |
| Broader public leave edges | 4P and zero-present variants | The focused 2P immediate round-end, 3P/4P continuation, 3P warmdown leave, and 3P single-present active leave proofs are promoted narrowly. Add broader variants only when they isolate a new rule. |

## Runnability

- JS oracle command for the 3P mid-round leave fixture:
  `node tools/reference_oracle/lifecycle_oracle.js scenarios/environment/source_lifecycle_mid_round_remove_avatar_3p_continue_round_end.json`
- JS oracle command for the 4P mid-round leave fixture:
  `node tools/reference_oracle/lifecycle_oracle.js scenarios/environment/source_lifecycle_mid_round_remove_avatar_4p_continue_round_end.json`
- Focused pytest guard for the 3P and 4P mid-round leave fixtures:
  `uv run pytest tests/test_lifecycle_oracle.py tests/test_source_env.py -q -k mid_round_remove_avatar`
- JS oracle command for the 4P tie fixture:
  `node tools/reference_oracle/lifecycle_oracle.js scenarios/environment/source_lifecycle_tie_at_max_score_4p.json`
- Focused pytest guard for the 4P tie fixture:
  `uv run pytest tests/test_lifecycle_oracle.py tests/test_source_lifecycle_runner.py tests/test_vector_multiplayer_env.py -q -k 4p_tie_at_max`
- Focused pytest guard for the 4P unique-leader match-end source fixture:
  `uv run pytest tests/test_lifecycle_oracle.py tests/test_source_lifecycle_runner.py -q -k match_end_at_max_score_4p`
- Focused pytest guard for the 4P multi-round match-end source fixture:
  `uv run pytest tests/test_lifecycle_oracle.py tests/test_source_lifecycle_runner.py -q -k "4p_multi_round_match_end or multi_round_match_end_4p"`
- Focused pytest guard for the 4P unique-leader and tie public metadata proofs:
  `uv run pytest tests/test_vector_multiplayer_env.py -q -k "4p_public_unique_max_score_leader or 4p_public_tied_max_score"`
- Focused pytest guard for the single-present leave-edge source fixture:
  `uv run pytest tests/test_lifecycle_oracle.py tests/test_source_lifecycle_runner.py -q -k remove_avatar_to_single_present`
- Focused pytest guard for the single-present leave-edge public metadata proof:
  `uv run pytest tests/test_vector_multiplayer_env.py -q -k active_round_leave_to_single_present`

Promotion rule: add a source fixture only when it isolates one rule above. Keep
Python source-runner parity, fast/runtime parity, public metadata parity, and
replay/observation claims as separate promotion steps.

Public metadata note: `VectorMultiplayerEnv.remove_player(...)` now has
narrow active-round support for 3P/4P continuation, 2P immediate round end, one
4P source-rule canary, and the focused 3P single-present active leave edge. It
is metadata-only: public ids are zero-based, source ids are public id plus one,
the leaver becomes `present=false` and `alive=false`, and the leaver is not
added to `death_player`. It still rejects unsupported warmdown, terminal,
absent-player, dead-player, and bad-shape calls. Do not promote this to broad
public leave, broad warmdown leave, replay, trainer, visual, or bonus support.
