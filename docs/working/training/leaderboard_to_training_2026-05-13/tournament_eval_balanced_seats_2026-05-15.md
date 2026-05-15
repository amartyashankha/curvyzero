# Tournament eval balanced seats - 2026-05-15

Tournament checkpoint eval now defaults to balanced randomized physical seats.
For each pair, game specs derive a deterministic shuffled swap schedule from the
pair seed. With an odd `games_per_pair`, the seat count differs by at most one.

Artifacts expose the mapping explicitly:

- Pair specs keep `players` as the canonical battle matchup.
- Game specs and game summaries include per-game `players`, `battle_players`,
  `seat_order`, and `seat_order_mode`.
- `seat_order.seat_to_checkpoint_id` names which checkpoint occupied physical
  seat 0 and seat 1 for that game.

Rating and standings no longer infer checkpoint wins from fixed player order.
When game summaries are present, winner seats are resolved through that game's
`players` mapping. Tally-only summaries must carry `wins_by_checkpoint`; using
`wins_by_seat` as a checkpoint proxy is intentionally rejected for rating.

Compatibility boundary: `seat_order_mode="fixed"` remains available as an
explicit diagnostic override. The default mode is `balanced_random`, and the old
implicit `randomize_seat_order` boolean path is not honored in tournament/eval
normalization.
