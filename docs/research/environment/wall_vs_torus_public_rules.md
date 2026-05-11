# Wall vs Torus Public Rules

Access date: 2026-05-08

Scope: public/reference check for whether the playfield boundary is a killing wall
or a wraparound torus. This note keeps original CurvyTron source-v1 and public
Curvytron 2 separate.

## Short Answer

- source-v1: normal borders are walls. Crossing them kills the avatar. The
  `BonusGameBorderless` bonus changes the game to borderless mode for its active
  duration, and borderless mode wraps the avatar to the opposite side.
- public Curvytron 2: the public rules say players die on the walls around the
  playfield. The public `Portal` bonus says the player can cross from one border
  to the opposite side.

So the base rule should be modeled as wall-death, not torus. A torus/wrap rule is
a bonus or mode effect, not the default game.

## source-v1: Original CurvyTron Source

This section is based on the public `Curvytron/curvytron` source at local commit
`8fec14cb5e953b5d3d60b3373e8d2ae1874e327f`.

Normal boundary handling is a wall kill:

- `Game.update()` asks `world.getBoundIntersect()` for a border hit with
  `avatar.radius` as the margin when `this.borderless` is false. If a border is
  hit and `this.borderless` is false, it calls `this.kill(avatar, null, score)`.
- `BaseGame.prototype.borderless` defaults to `false`.

Borderless handling is wraparound:

- The same `Game.update()` branch uses margin `0` when `this.borderless` is true.
  If a border is hit in that state, it sets the avatar position to
  `world.getOposite(border[0], border[1])`.
- `World.getOposite()` maps left to right, right to left, top to bottom, and
  bottom to top.
- `BonusGameBorderless` has duration `10000`, probability `0.8`, and returns the
  effect `['borderless', true]`.
- `GameBonusStack` applies that effect with `target.setBorderless(...)`; bonus
  removal resolves the stack back to the default value.
- `BonusGameBorderless` is in the default room bonus set.

Important source-v1 detail: normal wall checks use the avatar radius as margin,
while borderless checks use margin `0`. That means normal death happens when the
circle crosses the wall, while borderless wrap happens when the center crosses the
edge.

## Public Curvytron 2 Rules

The public Curvytron 2 "How to Play" page states the base rule directly:

- Players are always moving forward and can only go left or right.
- Players die if they touch another player's trace, their own trace, or the walls
  around the playfield.
- Players can catch bonuses on the playfield.

The same public page lists the `Portal` bonus:

- `Portal`: "You can now cross from one border of the playfield to the opposite
  side."

Other public Curvytron 2 bonuses listed on that page do not say that they change
wall behavior:

- `Crown of invincibility` says the player can break and pass through traces, not
  walls.
- `Eraser` changes traces.
- reverse controls, speed-up, slow-down, shrink/grow, and right-angle bonuses
  change controls, speed, size, or turning.

The Curvytron 2 About page identifies Curvytron 2 as the official sequel to
Curvytron and has its own 2026 changelog. Its v1.2.0 entry mentions new
self-targeted bonuses in solo games, but that changelog does not state a new
wall/wrap rule.

## Implementation Note

Keep these as separate target references:

- `curvytron-v1-reference`: wall-death by default; `BonusGameBorderless` creates
  temporary wraparound.
- `curvytron2-reference`: public rules say wall-death by default; public `Portal`
  bonus creates border-to-opposite-border crossing.

Do not use source-v1 `BonusGameBorderless` internals as evidence for Curvytron 2
implementation details. They only support the source-v1 rules.

## Sources

- Curvytron 2 "How to Play": https://curvytron2.com/how-to-play/
- Curvytron 2 "About": https://curvytron2.com/about/
- Original CurvyTron public source repo: https://github.com/Curvytron/curvytron
- source-v1 `Game.update()` wall vs borderless branch:
  https://github.com/Curvytron/curvytron/blob/8fec14cb5e953b5d3d60b3373e8d2ae1874e327f/src/server/model/Game.js#L51-L58
- source-v1 `World.getBoundIntersect()` and `World.getOposite()`:
  https://github.com/Curvytron/curvytron/blob/8fec14cb5e953b5d3d60b3373e8d2ae1874e327f/src/server/core/World.js#L276-L323
- source-v1 `BonusGameBorderless` duration/probability/effect:
  https://github.com/Curvytron/curvytron/blob/8fec14cb5e953b5d3d60b3373e8d2ae1874e327f/src/server/model/Bonus/BonusGameBorderless.js#L20-L40
- source-v1 default room bonuses:
  https://github.com/Curvytron/curvytron/blob/8fec14cb5e953b5d3d60b3373e8d2ae1874e327f/src/shared/model/BaseRoomConfig.js#L17-L29
- source-v1 borderless default:
  https://github.com/Curvytron/curvytron/blob/8fec14cb5e953b5d3d60b3373e8d2ae1874e327f/src/shared/model/BaseGame.js#L77-L82
- source-v1 game bonus stack application:
  https://github.com/Curvytron/curvytron/blob/8fec14cb5e953b5d3d60b3373e8d2ae1874e327f/src/server/model/GameBonusStack.js#L40-L45
