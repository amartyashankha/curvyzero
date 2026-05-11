# Multiplayer Fidelity Risks

Status: short risk note for the first canaries.

- These canaries use forced positions and headings. They do not prove random
  spawn fidelity yet.
- The current headless JS path can test game state, events, death order, and
  scoring. It should not claim browser protocol or rendering fidelity.
- Same-frame checks depend on JS server update order: avatars are stepped from
  the end of the player list to the start.
- The 2P and 3P canaries cover same-frame wall deaths. They do not yet cover
  head-to-head collision or trail collision.
- The 4P canary covers ordered death scoring and final survivor score. A later
  canary should cover two prior deaths followed by two same-frame deaths.
- Scenario JSON files are intentionally untouched; these are oracle fixtures
  for headless JS first, with Python comparison added after trace emission is
  stable.
