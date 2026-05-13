# Scripted Wall-Avoidant Opponent Baseline

Purpose: fold the concrete desired opponent into a scoped design/prototype
without touching trainer launch code.

## Desired Semantics

Opponent family:

- standard policy-style object with stable `policy_id`, `policy_version`, seed,
  action sidecar, and action ids in the existing left/straight/right space;
- immortal opponent: opponent collisions do not mark the opponent dead;
- trail-writing opponent: keep normal printing/body insertion enabled;
- trail-phasing opponent movement: if the opponent hits learner trails, the
  opponent survives and the learner is not killed as the passive trail owner;
- wall avoidant: go straight when safely away from walls; near walls, choose
  legal left/right actions by whichever turn rotates heading toward the arena
  interior fastest.
- no teleport/bounce semantics: the policy must not clamp position, flip
  heading instantly, or bounce off walls. It only presses valid CurvyTron
  actions early enough to veer away.

Current runtime fit:

- `VectorMultiplayerEnv(..., death_immunity_player_ids=(opponent_id,))` already
  suppresses death for the immune moving opponent. It still leaves trails.
- In source-compatible body collisions, the moving avatar is the one that dies.
  Therefore an immune opponent moving through learner trails should not kill
  either party for that collision.
- This does not make the learner phase through opponent trails. If the desired
  rule is "any contact with the opponent or its trail cannot kill the learner,"
  that needs a separate collision-ignore rule for bodies owned by the opponent.

## Policy Logic

Inputs needed from source state per row/player:

- `pos[x, y]`, `heading`, `radius`, `map_size`, `angular_velocity_per_ms`;
- legal action mask.

World clearances:

```text
left   = x - radius
right  = map_size - (x + radius)
top    = y - radius
bottom = map_size - (y + radius)
```

Danger field with margin `M`:

```text
w_left   = max(0, M - left)
w_right  = max(0, M - right)
w_top    = max(0, M - top)
w_bottom = max(0, M - bottom)
away     = normalize([w_left - w_right, w_top - w_bottom])
```

Action rule:

- if `min(left, right, top, bottom) > M`, choose straight;
- otherwise score only left/right:
  `score(action) = dot(unit_heading_after_action, away)`;
- choose the legal turn with the larger score.

Prototype tuning from local probes: `M=20.0` is the recommended starting point
for the force-field policy. It preserves mostly-straight behavior while keeping
wall clearance positive in the tested starts. Reflection-like policies need much
larger lead margins and still have edge-case failures unless they become a
short-horizon arc planner.

## Standard Interface Shape

The existing `MultiplayerOpponentPolicy` interface returns an
`OpponentPolicySelection`, which is the right sidecar/action shape. The missing
piece is a privileged source-state observation contract for geometry policies.

Recommended integration shape:

```python
@dataclass(frozen=True, slots=True)
class SourceStateWallAvoidantOpponentPolicy:
    policy_id: str = "curvyzero_source_state_wall_avoidant_opponent"
    policy_version: str = "v0.2026-05-13-design"
    seed: int = 0
    safe_margin: float = 20.0

    def select_actions(
        self,
        legal_action_mask: np.ndarray,
        opponent_mask: np.ndarray,
        *,
        decision_index: int = 0,
        observation: np.ndarray | None = None,
        source_state: Mapping[str, np.ndarray],
    ) -> OpponentPolicySelection:
        ...
```

If we want to avoid changing the public protocol, the source-state env wrapper
can be the adapter: it owns `self._env.state`, computes the scripted action
there, and emits the same policy id/version/sidecar fields as standard opponent
policies. That is less general, but it is the smallest implementation path.

## Local Probe Results

Helper added:

- `scripts/probe_curvytron_wall_avoidant_opponent.py`
- The helper now compares contact-only, reflected-heading, predictive
  reflection, short rollout, and force-field policies. It records action mix,
  OOB rows/steps, death rows, first-turn timing, first-OOB snapshots, body-write
  cursors, and accumulated runtime counters.

Probe setup:

- real `VectorMultiplayerEnv`;
- opponent leaves trails through normal env printing/body insertion;
- both players death-immune in the probe so long wall geometry can be measured
  without early round termination;
- ego action fixed straight unless noted;
- natural bonuses disabled.
- reported long probes used `death_immunity_player_ids=(0, 1)` so wall geometry
  can be measured after failures; deaths are therefore expected to stay at 0 in
  the table. This is diagnostic-only, not a trainer change.

Results:

| Policy variant | Settings | Steps | Starts | OOB/death results | Action mix | Recommended read |
| --- | --- | ---: | ---: | --- | --- | --- |
| reactive reflection proxy | `M=20`, `trigger=0`, contact-only | 1024 | 128 | OOB 128/128; deaths 0/0 due immunity; max outside 10.86 | L 0.080 / S 0.838 / R 0.083 | Reject. It waits until too late. |
| pure reflected heading | `M=20`, no normal bias | 256 | 64 | OOB 59/64; deaths 0/0; max outside 129.61 | L 0.232 / S 0.499 / R 0.270 | Reject. Specular heading is not enough. |
| inward-biased reflected heading | `M=40`, `normal_bias=0.75`, seeds 0-1 | 1024 each | 256 | OOB 11/256; deaths 0/0; worst outside 72.97 | L 0.476 / S 0.125 / R 0.399 | Not reliable enough; also turn-heavy. |
| inward-biased reflected heading | `M=50`, `normal_bias=0.75`, seeds 0-1 | 1024 each | 256 | OOB 1/256; deaths 0/0; worst outside 8.66 | L 0.469 / S 0.142 / R 0.390 | Fixable-ish, but still not clean and too turn-heavy. |
| rollout clearance | `M=20`, `lookahead=6`, seeds 0-1 | 1024 each | 256 | OOB 0/256; deaths 0/0; min clearance 4.78 | L 0.502 / S 0.125 / R 0.373 | Good diagnostic/stress policy, not the natural baseline. |
| proactive force field | `M=20`, seeds 0-2 | 1024 each | 384 | OOB 0/384; deaths 0/0; min clearance 7.10 | L 0.136 / S 0.729 / R 0.136 | Recommended integration candidate. |

The 128-row, 1024-step full-fidelity trail-writing runs each scanned about
1.07B body slots (`body_scan_slots`). That is useful evidence that trails remain
enabled, and it is also why longer collision-preserving sweeps should be batched
sparingly or run with shorter horizons during iteration.

## Why The Reflection Probes Are Not The Main Policy

The contact-only reflection proxy fails for a simple timing reason. The avatar
moves 4.8 world units per trainer decision (`16 units/s * 300 ms`). In the
1024-step contact-only run, first-OOB snapshots show the opponent was still
choosing straight one decision before crossing, with only about 0.23-4.22 units
of clearance in the sampled failures and a negative inward dot, meaning it was
still aimed outward. Once contact is the trigger, the legal turn rate cannot save
the position.

Pure reflected heading is not the desired policy abstraction. A real bounce
would change physics. The probe only used reflected headings as target
directions for legal left/right/straight actions, and even that often stayed too
shallow near walls.

Reflection can be partially rescued by adding a strong inward normal bias and a
large lead margin (`M=50`), but that still failed one 1024-step row across two
128-start seeds and chose turns about 86% of the time. A short legal-action
rollout stayed in-bounds in the tested starts, but it is similarly turn-heavy.
For integration, the force-field policy is the cleanest baseline: it is simple,
uses only legal left/right/straight actions, starts turning before contact,
survived 384 1024-step starts with normal trail writing, and still goes straight
about 73% of the time.

## Risks

- Current death immunity does not reflect or clamp positions. The proactive
  policy is needed; raw immortal straight movement can go out of bounds.
- Current death immunity only protects the immune moving player. Learner deaths
  on opponent-owned trails remain possible unless we add owner-based collision
  phasing for the learner.
- Geometry policy needs privileged source state. The existing generic opponent
  protocol can carry actions/sidecars, but not source state without an adapter or
  small interface extension.
- Long full-fidelity probes with normal trail writing get expensive because body
  history and collision scanning grow with horizon.
