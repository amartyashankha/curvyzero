# Naming Conventions

Created: 2026-05-16.

This is the current naming cleanup for the next CurvyTron launch family. The
goal is not cute names. The goal is that a tired operator can look at a run,
row, assignment, tournament, or GIF prefix and know what it is without opening
five stale docs.

Code helper: `src/curvyzero/contracts/curvytron_naming.py` holds the current
`cz26*` batch constants and short-tag helpers. Manifest builders should import
that helper instead of copying the strings.

## Main Critique

The historical names mixed too many different ideas:

- `survbonusout` hid the actual reward setting. The real knob is terminal
  outcome strength.
- `so10rep10` described implementation details, but not that it was the action
  noise arm.
- `rank1imm` made immortality look like a different policy. It is not. The
  policy is rank1; immortality is a separate flag/probability.
- `tonight18`, `restart18`, and `r18fresh` are historical batch names. They
  should not appear in new launch IDs except when pointing at old artifacts.
- Long recipe strings like
  `blank20-wall5-ladder75-rank1_30-rank2_20-rank3_15-rank4_10` are useful as
  exact prose, but too long for dashboard-facing run names.

## Current Prefix

Use this launch-family prefix for the next current-code experiments:

```text
cz26
```

Current visible batch names:

| Visible name | Meaning |
| --- | --- |
| `cz26a` | Grid A: broad mixed-recipe robustness grid. |
| `cz26b` | Grid B: slot-population grid. |
| `cz26c` | Canary / fast proof lane attached to a separate arena. |

Put exact date, code commit, seed checkpoint ref, volume names, arena id, rating
id, and launch command in manifest metadata. Do not stuff all of that into the
run name.

## Run Name Shape

Use this shape for operator-visible training run IDs:

```text
<batch>-r<row>-<reward>-<noise>-<imm>-<recipe>
```

Example:

```text
cz26a-r017-out33-n10-imm0-b20w05r1
```

Plain read: Grid A row 17, outcome alpha 0.33, 10% action noise, leaderboard
opponents mortal, recipe with about 20% blank, 5% wall-avoidant, rest rank1.

Attempt IDs can be shorter:

```text
try-<same-run-id>
```

If the platform name-length limit forces shortening, keep the prefix, row id,
and recipe code. The manifest row must always retain the full structured
settings.

## Axis Tags

These are the short tags for the knobs we are actively fiddling with.

| Axis | Short tag | Structured field |
| --- | --- | --- |
| reward outcome alpha 0.0 | `out0` | `reward_outcome_alpha: 0.0` |
| reward outcome alpha 0.33 | `out33` | `reward_outcome_alpha: 0.33` |
| reward outcome alpha 0.5 | `out50` | `reward_outcome_alpha: 0.5` |
| reward outcome alpha 0.67 | `out67` | `reward_outcome_alpha: 0.67` |
| reward outcome alpha 1.0 | `out100` | `reward_outcome_alpha: 1.0` |
| no action noise | `n0` | straight override `0.0`, repeat extra `0.0` |
| 10% action noise | `n10` | straight override `0.10`, repeat extra `0.10` |
| 20% action noise | `n20` | straight override `0.20`, repeat extra `0.20` |
| leaderboard opponents never immortal | `imm0` | `leaderboard_immortal_probability: 0.0` |
| leaderboard opponents immortal 10% of sampled slots | `imm10` | `leaderboard_immortal_probability: 0.10` |

Hard-coded blank and wall-avoidant opponents are always immortal. The `imm`
tag only applies to leaderboard checkpoint opponents.

## Recipe Codes

Recipe codes should be short but meaningful. Exact slot counts stay in
structured manifest fields.

Code alphabet:

- `b`: blank no-op opponent, always immortal.
- `w`: wall-avoidant hard-coded opponent, always immortal.
- `r1`, `r2`, `r3`, `r4`: leaderboard ranks.
- `top2`: leaderboard rank1 plus rank2 split.
- `lad4`: rank1/rank2/rank3/rank4 ladder split.

Current Grid A recipe codes:

| Code | Exact 64-slot bag | Plain meaning |
| --- | --- | --- |
| `b20w05r1` | blank 13, wall 3, rank1 48 | Anchor: about 20% blank, 5% wall, rest rank1. |
| `b10w05r1` | blank 7, wall 3, rank1 54 | Lower blank dose, same wall dose. |
| `b20w10r1` | blank 13, wall 6, rank1 45 | Higher wall dose, same blank dose. |
| `b20w05top2` | blank 13, wall 3, rank1 32, rank2 16 | Same blank/wall as anchor, but top-2 leaderboard pressure. |

Current Grid B recipe codes:

| Code | Exact 64-slot bag | Plain meaning |
| --- | --- | --- |
| `b100` | blank 64 | Pure blank control. |
| `w100` | wall 64 | Pure wall-avoidant control. |
| `r1` | rank1 64 | Pure current-best leaderboard control. |
| `b50r1` | blank 32, rank1 32 | Half blank, half rank1. |
| `b25w25r1` | blank 16, wall 16, rank1 32 | Half hard-coded, half rank1. |
| `b20w20lad4s` | blank 13, wall 13, rank1 19, rank2 13, rank3 3, rank4 3 | User complex split, small tail on ranks 3 and 4. |
| `b20w05r1` | blank 13, wall 3, rank1 48 | Shared Grid A/Grid B anchor. |
| `b30w05r1` | blank 19, wall 3, rank1 42 | Higher blank dose than anchor. |
| `b20w05top2` | blank 13, wall 3, rank1 32, rank2 16 | Top-2 diversity with anchor blank/wall. |
| `b20w05lad4` | blank 13, wall 3, rank1 19, rank2 13, rank3 10, rank4 6 | Ladder split with anchor blank/wall. |

## Names To Retire

Do not use these names for new current-code launches:

| Retired name | Replacement |
| --- | --- |
| `survbonusout` | `out100` plus `reward_variant: survival_plus_bonus_plus_outcome` |
| `survbonusnoout` | `out0` plus the same survival+bonus reward family |
| `so10rep10` | `n10` plus explicit noise fields |
| `rank1imm5`, `rank1_immortal` in recipe IDs | `imm10` or another explicit `imm*` axis tag, with policy still named `r1` |
| `tonight18` | Historical only; next grids are `cz26a` and `cz26b` |
| `restart18`, `r18v2`, `r18fresh` | Historical only; do not use in new current-code run IDs |

## Naming Priority

Fix names in this order:

1. Manifest rows and run IDs for Grid A, Grid B, and canaries.
2. Slot recipe IDs and assignment IDs.
3. Tournament id, rating run id, and GIF run prefix.
4. Public website labels and dropdown labels.
5. Historical docs only when they are promoted into current launch docs.

Do not rename old artifacts in place. Old artifact names are evidence. New
current-code artifacts must use the cleaned convention.
