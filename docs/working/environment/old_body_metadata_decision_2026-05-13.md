# Old-Body Death Metadata Decision - 2026-05-13

Status: promoted through public env, trainer surface, and trainer replay.

Target:
`source_body_old_opponent_overlap_kills_step.json`.

Decision: carry source old-body death metadata explicitly. Source CurvyTron
reports `old:true` on a body-hit death when the hit body is at least 2000 ms
old. That flag does not change collision physics, but it is part of the source
death event and should not be silently dropped.

Implemented contract:

- fixture seeding preserves `initial_state.world_bodies[].age_ms` as
  `body_birth_ms`;
- normal runtime body inserts set `body_birth_ms` from row `elapsed_ms`;
- body-hit selection returns the exact source-order body slot, not just owner;
- public info exposes `death_hit_old` with the same death-list shape as
  `death_hit_owner`;
- `death_hit_old` uses `-1` for no body-hit death, `0` for source-new, and `1`
  for source-old;
- debug die events now use the same source-order old flag;
- trainer replay whitelists and copies `death_hit_old`.

Proof:

- `tests/test_multiplayer_collision_breadth_fidelity.py`
- Focused run on 2026-05-13: `14 passed`.

Remaining caution:

This proves one old seeded opponent-body death at the source boundary
(`age_ms=2000`). Add young/old boundary pairs only if we need broader metadata
coverage later.
