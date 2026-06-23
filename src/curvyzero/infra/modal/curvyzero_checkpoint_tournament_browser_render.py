"""Pure HTML rendering helpers for the CurvyTron tournament browser."""

from __future__ import annotations

import html
import math
from typing import Any, Mapping, Sequence
from urllib.parse import urlencode

from curvyzero.infra.modal.curvyzero_checkpoint_tournament_settings import (
    DEFAULT_BATTLE_GAME_LIMIT,
)
from curvyzero.tournament import curvytron_checkpoint_tournament as arena


def _rating_row_by_checkpoint(
    rating_snapshot: Mapping[str, Any],
    checkpoint_id: str,
) -> dict[str, Any]:
    rows = rating_snapshot.get("ratings")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return {}
    for row in rows:
        if isinstance(row, Mapping) and str(row.get("checkpoint_id") or "") == checkpoint_id:
            return dict(row)
    return {}


def _rating_rows(rating_snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = rating_snapshot.get("ratings")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _rating_rank_by_checkpoint(rating_snapshot: Mapping[str, Any]) -> dict[str, int]:
    ranks = {}
    for row in _rating_rows(rating_snapshot):
        checkpoint_id = str(row.get("checkpoint_id") or "")
        if not checkpoint_id:
            continue
        try:
            ranks[checkpoint_id] = int(row.get("rank") or 0)
        except (TypeError, ValueError):
            continue
    return ranks


def _battle_player_for_checkpoint(
    row: Mapping[str, Any],
    checkpoint_id: str,
) -> dict[str, Any]:
    players = row.get("players")
    if not isinstance(players, Sequence) or isinstance(players, (str, bytes)):
        return {}
    for player in players:
        if (
            isinstance(player, Mapping)
            and str(player.get("checkpoint_id") or "") == checkpoint_id
        ):
            return dict(player)
    return {}


def _battle_opponent_for_checkpoint(
    row: Mapping[str, Any],
    checkpoint_id: str,
) -> dict[str, Any]:
    players = row.get("players")
    if not isinstance(players, Sequence) or isinstance(players, (str, bytes)):
        return {}
    for player in players:
        if (
            isinstance(player, Mapping)
            and str(player.get("checkpoint_id") or "") != checkpoint_id
        ):
            return dict(player)
    return {}


def _checkpoint_battle_sort_key(
    row: Mapping[str, Any],
    checkpoint_id: str,
    rank_by_checkpoint: Mapping[str, int],
) -> tuple[int, int, str]:
    opponent = _battle_opponent_for_checkpoint(row, checkpoint_id)
    opponent_id = str(opponent.get("checkpoint_id") or "")
    try:
        opponent_rank = int(rank_by_checkpoint.get(opponent_id) or 1_000_000)
    except (TypeError, ValueError):
        opponent_rank = 1_000_000
    try:
        pair_index = int(row.get("pair_index", 0) or 0)
    except (TypeError, ValueError):
        pair_index = 0
    return opponent_rank, pair_index, str(row.get("battle_id") or "")


def _sort_checkpoint_battle_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    checkpoint_id: str,
    rank_by_checkpoint: Mapping[str, int],
) -> list[dict[str, Any]]:
    return sorted(
        [dict(row) for row in rows if isinstance(row, Mapping)],
        key=lambda row: _checkpoint_battle_sort_key(
            row,
            checkpoint_id,
            rank_by_checkpoint,
        ),
    )


def _wins_for_checkpoint(row: Mapping[str, Any], checkpoint_id: str) -> int:
    tally = row.get("tally") if isinstance(row.get("tally"), Mapping) else {}
    wins_by_checkpoint = (
        tally.get("wins_by_checkpoint")
        if isinstance(tally.get("wins_by_checkpoint"), Mapping)
        else {}
    )
    return int(wins_by_checkpoint.get(checkpoint_id) or 0)


def _review_battle_row(
    row: Mapping[str, Any],
    checkpoint_id: str,
    *,
    rank_by_checkpoint: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    battle = dict(row)
    tally = battle.get("tally") if isinstance(battle.get("tally"), Mapping) else {}
    opponent = _battle_opponent_for_checkpoint(battle, checkpoint_id)
    opponent_id = str(opponent.get("checkpoint_id") or "")
    rank_by_checkpoint = rank_by_checkpoint or {}
    checkpoint_wins = _wins_for_checkpoint(battle, checkpoint_id)
    opponent_wins = _wins_for_checkpoint(battle, opponent_id) if opponent_id else 0
    draws = int(tally.get("draw_count") or 0)
    battle.update(
        {
            "checkpoint_id": checkpoint_id,
            "opponent": opponent,
            "opponent_rank": rank_by_checkpoint.get(opponent_id),
            "checkpoint_wins": checkpoint_wins,
            "opponent_wins": opponent_wins,
            "draws": draws,
            "completed_count": int(tally.get("completed_count") or 0),
            "failure_count": int(tally.get("failure_count") or 0),
            "average_physical_steps": tally.get("average_physical_steps"),
        }
    )
    return arena._to_plain(battle)


def _href(path: str, **params: Any) -> str:
    clean = {
        key: str(value)
        for key, value in params.items()
        if value not in (None, "")
    }
    query = urlencode(clean)
    return f"{path}?{query}" if query else path


def _page_href(**params: Any) -> str:
    return _href("/", **params)


def _battle_href(**params: Any) -> str:
    return _href("/battle", **params)


def _friendly_progress_label(progress: Mapping[str, Any]) -> str:
    status = str(progress.get("status") or "")
    phase = str(progress.get("phase") or "")
    if status == "complete":
        return "rankings ready"
    if phase in {"game_map_started", "games_running", "all_games_seen"}:
        return "running games"
    if phase in {"reduced", "ratings_written"}:
        return "finalizing rankings"
    if status == "pending":
        return "starting"
    return (status or phase or "starting").replace("_", " ")


def _short_battle_label(battle_id: str, pair_index: Any = None) -> str:
    if pair_index is not None:
        return f"pair {pair_index}"
    marker = "-pair-"
    if marker in battle_id:
        tail = battle_id.split(marker, 1)[1]
        return f"pair {tail.split('-', 1)[0]}"
    return battle_id


def _render_battle_detail_section(
    *,
    payload: Mapping[str, Any],
    rating_run_id: str = "latest",
    checkpoint_id: str = "",
) -> str:
    def fmt(value: Any, *, digits: int = 2) -> str:
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return ""

    battle_id = str(payload.get("battle_id") or "")
    if not payload:
        return '<div class="empty">No battle detail found.</div>'
    summary = payload.get("summary") if isinstance(payload.get("summary"), Mapping) else {}
    battle = payload.get("battle") if isinstance(payload.get("battle"), Mapping) else {}
    players = summary.get("players") or battle.get("players") or []
    if isinstance(players, Sequence) and not isinstance(players, (str, bytes)):
        matchup = " vs ".join(
            html.escape(str(player.get("label") or player.get("checkpoint_id") or ""))
            for player in players
            if isinstance(player, Mapping)
        )
    else:
        matchup = ""
    tally = summary.get("tally") if isinstance(summary.get("tally"), Mapping) else {}
    summary_ref = html.escape(str(summary.get("summary_ref") or battle.get("summary_ref") or ""))
    summary_link = f'<a href="/meta?ref={summary_ref}">JSON</a>' if summary_ref else ""
    samples = payload.get("sample_gifs") if isinstance(payload.get("sample_gifs"), list) else []
    games = payload.get("games") if isinstance(payload.get("games"), list) else []
    tournament_id = str(payload.get("selected_tournament_id") or "")
    game_count = int(payload.get("game_count") or 0)
    game_limit = int(payload.get("game_limit") or DEFAULT_BATTLE_GAME_LIMIT)
    game_offset = int(payload.get("game_offset") or 0)
    first_row = game_offset + 1 if games else 0
    last_row = game_offset + len(games)
    prev_href = _battle_href(
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        checkpoint_id=checkpoint_id,
        battle_id=battle_id,
        game_limit=game_limit,
        game_offset=max(0, game_offset - game_limit),
    )
    next_href = _battle_href(
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        checkpoint_id=checkpoint_id,
        battle_id=battle_id,
        game_limit=game_limit,
        game_offset=game_offset + game_limit,
    )

    sample_cards = []
    for sample in samples:
        if not isinstance(sample, Mapping) or not sample.get("gif_ref"):
            continue
        gif_ref = html.escape(str(sample["gif_ref"]))
        caption = " ".join(
            item
            for item in (
                str(sample.get("game_id") or ""),
                str(sample.get("outcome") or ""),
            )
            if item
        )
        sample_cards.append(
            f"""
            <a class="gif-card" href="/gif?ref={gif_ref}">
              <img src="/gif?ref={gif_ref}" alt="{html.escape(caption)}" loading="lazy" decoding="async">
              <span>{html.escape(caption or "Sample")}</span>
            </a>
            """
        )
    sample_html = (
        f"""
        <section class="panel">
          <div class="panel-head"><h2>GIF Samples</h2><span>{len(sample_cards)} shown</span></div>
          <div class="gif-grid">{"".join(sample_cards)}</div>
        </section>
        """
        if sample_cards
        else '<p class="summary">No GIF samples were captured for this battle.</p>'
    )

    game_rows = []
    for game in games:
        if not isinstance(game, Mapping):
            continue
        game_summary_ref = html.escape(str(game.get("summary_ref") or ""))
        gif_ref = html.escape(str(game.get("gif_ref") or ""))
        gif_link = f'<a href="/gif?ref={gif_ref}">GIF</a>' if gif_ref else ""
        json_link = f'<a href="/meta?ref={game_summary_ref}">JSON</a>' if game_summary_ref else ""
        game_rows.append(
            "<tr>"
            f"<td>{html.escape(str(game.get('game_index') if game.get('game_index') is not None else ''))}</td>"
            f"<td>{html.escape(str(game.get('game_id') or ''))}</td>"
            f"<td>{html.escape(str(game.get('outcome') or ''))}</td>"
            f"<td>{html.escape(str(game.get('seed') or ''))}</td>"
            f"<td>{html.escape(str(game.get('physical_steps') or ''))}</td>"
            f"<td>{html.escape(str(game.get('ok')))}</td>"
            f"<td>{gif_link}</td>"
            f"<td>{json_link}</td>"
            "</tr>"
        )
    games_html = (
        f"""
        <section class="panel">
          <div class="panel-head">
            <h2>Games</h2>
            <span>{first_row}-{last_row} of {html.escape(str(game_count))}</span>
          </div>
          <table>
            <thead><tr><th>#</th><th>Game</th><th>Outcome</th><th>Seed</th><th>Steps</th><th>OK</th><th>GIF</th><th>JSON</th></tr></thead>
            <tbody>{"".join(game_rows)}</tbody>
          </table>
          <div class="pager">
            {'<a href="' + html.escape(prev_href) + '#battle-detail">Previous games</a>' if payload.get("has_newer_games") else '<span></span>'}
            {'<a href="' + html.escape(next_href) + '#battle-detail">Next games</a>' if payload.get("has_older_games") else '<span></span>'}
          </div>
        </section>
        """
        if game_rows
        else '<div class="empty">No game summaries found for this battle.</div>'
    )

    return f"""
    <section class="panel selected-battle" id="battle-detail">
      <div class="panel-head">
        <h2>{html.escape(battle_id)}</h2>
        <span>{matchup}</span>
      </div>
      <div class="progress-body">
        <div><strong>{html.escape(str(tally.get("completed_count", 0)))}</strong><span>games</span></div>
        <div><strong>{html.escape(str(tally.get("failure_count", 0)))}</strong><span>failures</span></div>
        <div><strong>{html.escape(str(tally.get("draw_count", 0)))}</strong><span>draws</span></div>
        <div><strong>{fmt(tally.get("average_physical_steps"))}</strong><span>avg steps</span></div>
        <div><strong>{summary_link}</strong><span>battle JSON</span></div>
      </div>
    </section>
    {sample_html}
    {games_html}
    """


def _render_page(
    *,
    tournaments: list[dict[str, Any]],
    selected_tournament_id: str,
    selected_rating_run_id: str,
    selected_checkpoint_id: str,
    rating_runs: list[dict[str, Any]],
    rating_snapshot: dict[str, Any],
    rating_progress: dict[str, Any],
    battles: dict[str, Any],
    selected_battle_id: str = "",
    battle_detail: Mapping[str, Any] | None = None,
    volume_reload_error: str = "",
) -> str:
    def fmt_number(value: Any, *, digits: int = 1) -> str:
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return ""

    def sort_number_attr(value: Any) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return ""
        if not math.isfinite(number):
            return ""
        if number.is_integer():
            return str(int(number))
        return f"{number:.6f}".rstrip("0").rstrip(".")

    options = "\n".join(
        f'<option value="{html.escape(row["tournament_id"])}" '
        f'{"selected" if row["tournament_id"] == selected_tournament_id else ""}>'
        f'{html.escape(row["tournament_id"])}'
        f'{html.escape(" (current)" if row.get("is_current") else "")}</option>'
        for row in tournaments
    )
    rating_options = "\n".join(
        f'<option value="{html.escape(row["rating_run_id"])}" '
        f'{"selected" if row["rating_run_id"] == selected_rating_run_id else ""}>'
        f'{html.escape(row["rating_run_id"])}'
        f'{html.escape(" (current)" if row.get("is_current") else "")}'
        f'{html.escape(" (" + str(row.get("status")) + ")" if row.get("status") else "")}</option>'
        for row in rating_runs
    )
    rating_rows = _rating_rows(rating_snapshot)
    selected_rating_row = (
        _rating_row_by_checkpoint(rating_snapshot, selected_checkpoint_id)
        if selected_checkpoint_id
        else {}
    )
    rating_html = ""
    if rating_rows:
        provisional = bool(rating_snapshot.get("provisional"))
        ranking_title = "Live Leaderboard" if provisional else "Leaderboard"
        ranking_status = (
            "updating from finished games"
            if provisional
            else str(rating_snapshot.get("round_id", ""))
        )
        body = []
        for row in rating_rows:
            record = row if isinstance(row, Mapping) else {}
            checkpoint_id = str(record.get("checkpoint_id", ""))
            checkpoint_label = str(record.get("label") or checkpoint_id)
            href = _page_href(
                tournament_id=selected_tournament_id,
                rating_run_id=selected_rating_run_id,
                checkpoint_id=checkpoint_id,
            )
            selected_class = " selected-row" if checkpoint_id == selected_checkpoint_id else ""
            body.append(
                f"<tr class=\"{selected_class.strip()}\">"
                f"<td>{html.escape(str(record.get('rank', '')))}</td>"
                f"<td title=\"{html.escape(checkpoint_id)}\">"
                f"<a href=\"{html.escape(href)}\">{html.escape(checkpoint_label)}</a></td>"
                f"<td>{float(record.get('rating', 0.0)):.1f}</td>"
                f"<td>{html.escape(str(record.get('games', 0)))}</td>"
                f"<td>{html.escape(str(record.get('wins', 0)))}-"
                f"{html.escape(str(record.get('losses', 0)))}-"
                f"{html.escape(str(record.get('draws', 0)))}</td>"
                f"<td>{fmt_number(record.get('win_rate'), digits=3)}</td>"
                f"<td>{html.escape(str(record.get('distinct_opponents') or record.get('battles') or 0))}</td>"
                f"<td>{html.escape(str(record.get('failure_count', 0)))}</td>"
                "</tr>"
            )
        rating_html = f"""
        <section class="panel">
          <div class="panel-head">
            <h2>{html.escape(ranking_title)}</h2>
            <span>{html.escape(str(rating_snapshot.get("rating_run_id", "")))} / {html.escape(ranking_status)}</span>
          </div>
          <div class="scroll-panel rankings-scroll">
            <table>
              <thead><tr><th>Rank</th><th>Checkpoint</th><th>Rating</th><th>Games</th><th>W-L-D</th><th>Win rate</th><th>Opp.</th><th>Failures</th></tr></thead>
              <tbody>{"".join(body)}</tbody>
            </table>
          </div>
        </section>
        """
    elif rating_runs:
        status = str(rating_progress.get("status") or "")
        if status and status != "complete":
            friendly_status = _friendly_progress_label(rating_progress)
            rating_html = f"""
            <section class="panel">
              <div class="panel-head">
                <h2>Leaderboard</h2>
                <span>{html.escape(friendly_status)}</span>
              </div>
              <p class="in-panel">Leaderboard rows will appear after this rating round writes its first rating snapshot. This arena is selected and running; the leaderboard is pending, not missing.</p>
            </section>
            """
        else:
            rating_html = """
            <section class="panel">
              <div class="panel-head"><h2>Leaderboard</h2><span>empty</span></div>
              <p class="in-panel">This rating run exists, but no rating rows were found.</p>
            </section>
            """
    progress_html = ""
    if rating_progress:
        try:
            pct = 100.0 * float(rating_progress.get("estimated_completion_fraction") or rating_progress.get("completion_fraction") or 0.0)
        except (TypeError, ValueError):
            pct = 0.0
        progress_html = f"""
        <section class="panel" id="progress-panel" data-tournament-id="{html.escape(selected_tournament_id)}" data-rating-run-id="{html.escape(selected_rating_run_id)}" data-has-ratings="{html.escape('true' if rating_rows else 'false')}">
          <div class="panel-head">
            <h2>Progress</h2>
            <span data-progress-field="updated">{html.escape(str(rating_progress.get("updated_at") or ""))}</span>
          </div>
          <div class="progress-body">
            <div><strong data-progress-field="status">{html.escape(_friendly_progress_label(rating_progress))}</strong><span>state</span></div>
            <div><strong data-progress-field="phase">{html.escape(str(rating_progress.get("phase") or ""))}</strong><span>detail</span></div>
            <div><strong data-progress-field="pairs">{html.escape(str(rating_progress.get("started_pair_count") or 0))}/{html.escape(str(rating_progress.get("pair_count") or 0))}</strong><span>pairs started</span></div>
            <div><strong data-progress-field="games">{html.escape(str(rating_progress.get("estimated_seen_game_count") or rating_progress.get("completed_game_count") or 0))}/{html.escape(str(rating_progress.get("game_count") or 0))}</strong><span>games seen</span></div>
            <div><strong data-progress-field="percent">{pct:.1f}%</strong><span>estimated progress</span></div>
          </div>
        </section>
        """
    elif rating_runs:
        progress_html = f"""
        <section class="panel" id="progress-panel" data-tournament-id="{html.escape(selected_tournament_id)}" data-rating-run-id="{html.escape(selected_rating_run_id)}" data-has-ratings="{html.escape('true' if rating_rows else 'false')}">
          <div class="panel-head">
            <h2>Progress</h2>
            <span data-progress-field="updated"></span>
          </div>
          <p class="in-panel">Tournament state is loading. This page will check again automatically.</p>
        </section>
        """
    checkpoint_html = ""
    if selected_checkpoint_id:
        detail_name = str(
            selected_rating_row.get("label")
            or selected_rating_row.get("checkpoint_id")
            or selected_checkpoint_id
        )
        if selected_rating_row:
            checkpoint_html = f"""
            <section class="panel selected-checkpoint">
              <div class="panel-head">
                <h2>{html.escape(detail_name)}</h2>
                <a href="{html.escape(_page_href(tournament_id=selected_tournament_id, rating_run_id=selected_rating_run_id))}">Clear</a>
              </div>
              <div class="progress-body">
                <div><strong>{float(selected_rating_row.get("rating", 0.0)):.1f}</strong><span>rating</span></div>
                <div><strong>{html.escape(str(selected_rating_row.get("rank", "")))}</strong><span>rank</span></div>
                <div><strong>{html.escape(str(selected_rating_row.get("games", 0)))}</strong><span>games</span></div>
                <div><strong>{html.escape(str(selected_rating_row.get("wins", 0)))}-{html.escape(str(selected_rating_row.get("losses", 0)))}-{html.escape(str(selected_rating_row.get("draws", 0)))}</strong><span>W-L-D</span></div>
                <div><strong>{html.escape(str(battles.get("total", 0)))}</strong><span>battles shown below</span></div>
              </div>
            </section>
            """
        else:
            checkpoint_html = f"""
            <section class="panel selected-checkpoint">
              <div class="panel-head">
                <h2>{html.escape(selected_checkpoint_id)}</h2>
                <a href="{html.escape(_page_href(tournament_id=selected_tournament_id, rating_run_id=selected_rating_run_id))}">Clear</a>
              </div>
              <p class="in-panel">No rating row was found for this checkpoint in the selected rating run.</p>
            </section>
            """
    battle_html = '<p class="summary">Select a checkpoint row to inspect its battles.</p>'
    if selected_checkpoint_id:
        rank_by_checkpoint = _rating_rank_by_checkpoint(rating_snapshot)
        body = []
        battle_rows = battles.get("rows", [])
        sorted_battle_rows = _sort_checkpoint_battle_rows(
            battle_rows if isinstance(battle_rows, Sequence) else [],
            checkpoint_id=selected_checkpoint_id,
            rank_by_checkpoint=rank_by_checkpoint,
        )
        for sort_index, raw_row in enumerate(sorted_battle_rows):
            row = _review_battle_row(
                raw_row,
                selected_checkpoint_id,
                rank_by_checkpoint=rank_by_checkpoint,
            )
            opponent = row.get("opponent") if isinstance(row.get("opponent"), Mapping) else {}
            opponent_id = str(opponent.get("checkpoint_id") or "")
            opponent_label = str(opponent.get("label") or opponent_id or "unknown")
            summary_ref = html.escape(str(row.get("summary_ref") or ""))
            summary_link = f'<a href="/meta?ref={summary_ref}">JSON</a>' if summary_ref else ""
            gif_ref = html.escape(str(row.get("first_gif_ref") or ""))
            gif_link = f'<a href="/gif?ref={gif_ref}">GIF</a>' if gif_ref else "No GIF"
            battle_id = str(row.get("battle_id") or "")
            battle_href = _page_href(
                tournament_id=selected_tournament_id,
                rating_run_id=selected_rating_run_id,
                checkpoint_id=selected_checkpoint_id,
                battle_id=battle_id,
            )
            if battle_id:
                battle_href += "#battle-detail"
            selected_battle_class = (
                ' class="selected-row"' if battle_id and battle_id == selected_battle_id else ""
            )
            opponent_rank_sort = sort_number_attr(row.get("opponent_rank"))
            avg_steps_sort = sort_number_attr(row.get("average_physical_steps"))
            failure_count_sort = sort_number_attr(row.get("failure_count"))
            body.append(
                f"<tr{selected_battle_class} data-battle-row "
                f"data-sort-index=\"{sort_index}\" "
                f"data-sort-rank=\"{html.escape(opponent_rank_sort)}\" "
                f"data-sort-avg-steps=\"{html.escape(avg_steps_sort)}\" "
                f"data-sort-failures=\"{html.escape(failure_count_sort)}\">"
                f"<td>{html.escape(str(row.get('opponent_rank') or ''))}</td>"
                f"<td title=\"{html.escape(opponent_id)}\"><a href=\"{html.escape(battle_href)}\">{html.escape(opponent_label)}</a></td>"
                f"<td>{html.escape(str(row.get('checkpoint_wins', 0)))}-"
                f"{html.escape(str(row.get('opponent_wins', 0)))}-"
                f"{html.escape(str(row.get('draws', 0)))}</td>"
                f"<td>{html.escape(str(row.get('completed_count', 0)))}</td>"
                f"<td>{fmt_number(row.get('average_physical_steps'), digits=2)}</td>"
                f"<td>{html.escape(str(row.get('failure_count', 0)))}</td>"
                f"<td>{gif_link}</td>"
                f"<td>{summary_link}</td>"
                f"<td><a href=\"{html.escape(battle_href)}\">Games</a></td>"
                "</tr>"
            )
        if body:
            battle_html = f"""
            <section class="panel">
              <div class="panel-head">
                <h2>Battles</h2>
                <span>{html.escape(str(battles.get("total", 0)))} total</span>
              </div>
              <div class="scroll-panel battles-scroll">
                <table data-battle-table data-sort-key="rank" data-sort-direction="asc">
                  <thead><tr><th aria-sort="ascending"><button type="button" class="sort-button" data-battle-sort="rank">Opp. rank <span class="sort-indicator" data-sort-indicator="rank">asc</span></button></th><th>Opponent</th><th>W-L-D</th><th>Games</th><th><button type="button" class="sort-button" data-battle-sort="avgSteps">Avg steps <span class="sort-indicator" data-sort-indicator="avgSteps"></span></button></th><th><button type="button" class="sort-button" data-battle-sort="failures">Failures <span class="sort-indicator" data-sort-indicator="failures"></span></button></th><th>GIF</th><th>JSON</th><th>Battle</th></tr></thead>
                  <tbody>{"".join(body)}</tbody>
                </table>
              </div>
            </section>
            """
        else:
            battle_html = '<div class="empty">No battles found for this checkpoint.</div>'
    elif rating_progress and isinstance(rating_progress.get("recent_started_pairs"), Sequence):
        recent_rows = [
            row
            for row in rating_progress.get("recent_started_pairs", [])
            if isinstance(row, Mapping) and row.get("battle_id")
        ]
        if recent_rows:
            body = []
            for row in recent_rows[:50]:
                battle_id = str(row.get("battle_id") or "")
                battle_label = _short_battle_label(battle_id, row.get("pair_index"))
                battle_href = _page_href(
                    tournament_id=selected_tournament_id,
                    rating_run_id=selected_rating_run_id,
                    battle_id=battle_id,
                )
                body.append(
                    "<tr>"
                    f"<td>{html.escape(str(row.get('pair_index') if row.get('pair_index') is not None else ''))}</td>"
                    f"<td title=\"{html.escape(battle_id)}\"><a href=\"{html.escape(battle_href)}#battle-detail\">{html.escape(battle_label)}</a></td>"
                    f"<td>{html.escape(str(row.get('expected_game_count') or ''))}</td>"
                    f"<td>{html.escape('yes' if row.get('complete') else 'running')}</td>"
                    f"<td><a href=\"{html.escape(battle_href)}#battle-detail\">Games</a></td>"
                    "</tr>"
                )
            battle_html = f"""
            <section class="panel">
              <div class="panel-head">
                <h2>Recent Battles</h2>
                <span>live sample</span>
              </div>
              <div class="scroll-panel battles-scroll">
                <table>
                  <thead><tr><th>Pair</th><th>Battle</th><th>Games</th><th>State</th><th>Open</th></tr></thead>
                  <tbody>{"".join(body)}</tbody>
                </table>
              </div>
            </section>
            """
    battle_detail_html = (
        _render_battle_detail_section(
            payload=battle_detail or {"battle_id": selected_battle_id},
            rating_run_id=selected_rating_run_id,
            checkpoint_id=selected_checkpoint_id,
        )
        if selected_battle_id
        else ""
    )
    reload_html = (
        f"""
        <section class="panel">
          <div class="panel-head"><h2>Volume Refresh</h2><span>using last visible data</span></div>
          <p class="in-panel">{html.escape(volume_reload_error)}</p>
        </section>
        """
        if volume_reload_error
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CurvyTron Tournament</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f6f7f9; color: #202124; }}
    main {{ max-width: 1320px; margin: 0 auto; padding: 18px; }}
    header {{ display: flex; align-items: end; justify-content: space-between; gap: 12px; margin-bottom: 14px; }}
    h1 {{ margin: 0; font-size: 22px; }}
    form {{ display: flex; gap: 8px; align-items: end; }}
    select, button {{ height: 36px; border: 1px solid #dadce0; border-radius: 6px; padding: 0 8px; background: white; }}
    button {{ background: #1a73e8; border-color: #1a73e8; color: white; }}
    .summary {{ margin: 0 0 12px; color: #5f6368; font-size: 13px; }}
    .in-panel {{ margin: 0; padding: 12px; color: #5f6368; font-size: 13px; }}
    .panel {{ background: white; border: 1px solid #dadce0; border-radius: 8px; margin-bottom: 12px; overflow: hidden; }}
    .panel-head {{ display: flex; align-items: baseline; justify-content: space-between; gap: 12px; padding: 10px 12px; border-bottom: 1px solid #eef0f3; }}
    .panel h2 {{ margin: 0; font-size: 15px; }}
    .panel span {{ color: #5f6368; font-size: 12px; }}
    a {{ color: #1557b0; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .progress-body {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1px; background: #eef0f3; }}
    .progress-body div {{ display: grid; gap: 3px; padding: 12px; background: white; }}
    .progress-body strong {{ font-size: 18px; }}
    .progress-body span {{ color: #5f6368; font-size: 12px; }}
    .gif-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 10px; padding: 10px; }}
    .gif-card {{ display: grid; gap: 6px; color: #202124; }}
    .gif-card img {{ width: 100%; aspect-ratio: 1; object-fit: contain; background: #111827; }}
    .gif-card span {{ color: #5f6368; font-size: 12px; }}
    .scroll-panel {{ overflow: auto; }}
    .rankings-scroll {{ max-height: min(48vh, 520px); }}
    .battles-scroll {{ max-height: min(36vh, 380px); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 7px 10px; border-bottom: 1px solid #eef0f3; text-align: left; }}
    th {{ color: #5f6368; font-weight: 600; position: sticky; top: 0; z-index: 1; background: white; }}
    .sort-button {{ all: unset; display: inline-flex; align-items: center; gap: 4px; cursor: pointer; color: inherit; font: inherit; }}
    .sort-button:focus-visible {{ outline: 2px solid #1a73e8; outline-offset: 2px; border-radius: 4px; }}
    .panel .sort-indicator {{ min-width: 28px; color: #80868b; font-size: 11px; }}
    .pager {{ display: flex; justify-content: space-between; gap: 12px; padding: 10px 12px; border-top: 1px solid #eef0f3; font-size: 13px; }}
    .selected-row td {{ background: #eef4ff; }}
    td:nth-child(n+3) {{ white-space: nowrap; }}
    .empty {{ padding: 60px; text-align: center; background: white; border: 1px dashed #dadce0; border-radius: 8px; color: #80868b; }}
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>CurvyTron Tournament</h1>
      <p class="summary">Checkpoint battles. Score is who dies first.</p>
    </div>
    <form method="get" id="tournament-picker">
      <label>Tournament<br><select name="tournament_id" data-picker="tournament">{options}</select></label>
      <label>Rating<br><select name="rating_run_id" data-picker="rating">{rating_options}</select></label>
      <button type="submit">Open</button>
    </form>
  </header>
  {reload_html}
  {progress_html}
  {rating_html}
  {checkpoint_html}
  {battle_html}
  {battle_detail_html}
</main>
<script>
(() => {{
  const picker = document.getElementById("tournament-picker");
  const tournamentSelect = picker ? picker.querySelector("[name='tournament_id']") : null;
  const ratingSelect = picker ? picker.querySelector("[name='rating_run_id']") : null;
  const disablePicker = () => {{
    if (!picker) return;
    picker.querySelectorAll("select, button").forEach((node) => {{
      node.disabled = true;
    }});
  }};
  const navigatePicker = (changed) => {{
    if (!picker) return;
    const url = new URL(window.location.href);
    const tournamentId = tournamentSelect ? tournamentSelect.value : "";
    const ratingRunId = ratingSelect ? ratingSelect.value : "";
    if (tournamentId) {{
      url.searchParams.set("tournament_id", tournamentId);
    }} else {{
      url.searchParams.delete("tournament_id");
    }}
    if (changed === "tournament") {{
      url.searchParams.delete("rating_run_id");
    }} else if (ratingRunId) {{
      url.searchParams.set("rating_run_id", ratingRunId);
    }} else {{
      url.searchParams.delete("rating_run_id");
    }}
    url.searchParams.delete("checkpoint_id");
    url.searchParams.delete("battle_id");
    url.searchParams.delete("fresh");
    url.hash = "";
    disablePicker();
    window.location.assign(url.toString());
  }};
  if (tournamentSelect) {{
    tournamentSelect.addEventListener("change", () => navigatePicker("tournament"));
  }}
  if (ratingSelect) {{
    ratingSelect.addEventListener("change", () => navigatePicker("rating"));
  }}
  if (picker) {{
    picker.addEventListener("submit", (event) => {{
      event.preventDefault();
      navigatePicker("rating");
    }});
  }}
  const battleTable = document.querySelector("[data-battle-table]");
  if (battleTable) {{
    const tbody = battleTable.querySelector("tbody");
    const sortButtons = battleTable.querySelectorAll("[data-battle-sort]");
    const sortFields = {{
      rank: "sortRank",
      avgSteps: "sortAvgSteps",
      failures: "sortFailures",
    }};
    const sortValue = (row, key) => {{
      const field = sortFields[key];
      const raw = field ? row.dataset[field] : "";
      if (raw === undefined || raw === "") return null;
      const parsed = Number(raw);
      return Number.isFinite(parsed) ? parsed : null;
    }};
    const originalIndex = (row) => {{
      const parsed = Number(row.dataset.sortIndex || "0");
      return Number.isFinite(parsed) ? parsed : 0;
    }};
    const updateSortIndicators = (key, direction) => {{
      battleTable.dataset.sortKey = key;
      battleTable.dataset.sortDirection = direction;
      battleTable.querySelectorAll("th[aria-sort]").forEach((cell) => {{
        cell.removeAttribute("aria-sort");
      }});
      battleTable.querySelectorAll("[data-sort-indicator]").forEach((node) => {{
        node.textContent = node.dataset.sortIndicator === key ? direction : "";
      }});
      const activeButton = battleTable.querySelector(`[data-battle-sort="${{key}}"]`);
      if (activeButton && activeButton.closest("th")) {{
        activeButton.closest("th").setAttribute(
          "aria-sort",
          direction === "asc" ? "ascending" : "descending",
        );
      }}
    }};
    const applyBattleSort = (key, direction) => {{
      if (!tbody || !sortFields[key]) return;
      const multiplier = direction === "desc" ? -1 : 1;
      const rows = Array.from(tbody.querySelectorAll("[data-battle-row]"));
      rows.sort((a, b) => {{
        const left = sortValue(a, key);
        const right = sortValue(b, key);
        if (left === null && right === null) return originalIndex(a) - originalIndex(b);
        if (left === null) return 1;
        if (right === null) return -1;
        const diff = (left - right) * multiplier;
        if (diff !== 0) return diff;
        return originalIndex(a) - originalIndex(b);
      }});
      rows.forEach((row) => tbody.appendChild(row));
      updateSortIndicators(key, direction);
    }};
    sortButtons.forEach((button) => {{
      button.addEventListener("click", () => {{
        const key = button.dataset.battleSort || "rank";
        const currentKey = battleTable.dataset.sortKey || "rank";
        const currentDirection = battleTable.dataset.sortDirection || "asc";
        const nextDirection = key === currentKey && currentDirection === "asc" ? "desc" : "asc";
        applyBattleSort(key, nextDirection);
      }});
    }});
  }}
  const panel = document.getElementById("progress-panel");
  if (!panel) return;
  const fields = {{}};
  document.querySelectorAll("[data-progress-field]").forEach((node) => {{
    fields[node.dataset.progressField] = node;
  }});
  const text = (value) => value === null || value === undefined ? "" : String(value);
  const number = (value) => {{
    const parsed = Number(value || 0);
    return Number.isFinite(parsed) ? parsed : 0;
  }};
  const set = (name, value) => {{
    if (fields[name]) fields[name].textContent = value;
  }};
  const stateLabel = (progress) => {{
    if (progress.status === "complete") return "rankings ready";
    if (["game_map_started", "games_running", "all_games_seen"].includes(progress.phase)) return "running games";
    if (["reduced", "ratings_written"].includes(progress.phase)) return "finalizing rankings";
    if (progress.status === "pending") return "starting";
    return text(progress.status || progress.phase || "starting").replaceAll("_", " ");
  }};
  const reloadKey = `curvyzero:tournament-ratings-reloaded:${{panel.dataset.tournamentId}}:${{panel.dataset.ratingRunId}}`;
  let pollTimer = null;
  let inFlight = false;
  let failureCount = 0;
  const scheduleNext = (delayMs) => {{
    window.clearTimeout(pollTimer);
    pollTimer = window.setTimeout(() => refreshProgress().catch(() => {{}}), delayMs);
  }};
  async function refreshProgress() {{
    if (inFlight) {{
      scheduleNext(10000);
      return;
    }}
    inFlight = true;
    const params = new URLSearchParams({{
      tournament_id: panel.dataset.tournamentId || "",
      rating_run_id: panel.dataset.ratingRunId || "",
    }});
    const pollCount = Number(panel.dataset.pollCount || "0") + 1;
    panel.dataset.pollCount = String(pollCount);
    try {{
      const response = await fetch("/api/rating-progress?" + params.toString(), {{
        cache: "no-store",
      }});
      if (!response.ok) throw new Error(`progress ${{response.status}}`);
      const payload = await response.json();
      const progress = payload.progress || {{}};
      const seen = progress.estimated_seen_game_count ?? progress.completed_game_count ?? 0;
      const fraction = progress.estimated_completion_fraction ?? progress.completion_fraction ?? 0;
      set("status", stateLabel(progress));
      set("phase", text(progress.phase));
      set("pairs", `${{text(progress.started_pair_count ?? 0)}}/${{text(progress.pair_count ?? 0)}}`);
      set("games", `${{text(seen)}}/${{text(progress.game_count ?? 0)}}`);
      set("percent", `${{(100 * number(fraction)).toFixed(1)}}%`);
      set("updated", text(progress.updated_at));
      failureCount = 0;
      if (progress.status === "complete" && panel.dataset.hasRatings !== "true" && !sessionStorage.getItem(reloadKey)) {{
        sessionStorage.setItem(reloadKey, "1");
        const url = new URL(window.location.href);
        url.searchParams.set("fresh", "true");
        window.location.href = url.toString();
        return;
      }}
    }} catch (error) {{
      failureCount += 1;
    }} finally {{
      inFlight = false;
      const hiddenDelay = document.hidden ? 60000 : 10000;
      const retryDelay = Math.min(60000, hiddenDelay * Math.max(1, failureCount));
      scheduleNext(retryDelay);
    }}
  }}
  document.addEventListener("visibilitychange", () => {{
    if (!document.hidden) {{
      scheduleNext(250);
    }}
  }});
  scheduleNext(250);
}})();
</script>
</body>
</html>"""


def _render_battle_page(
    *,
    payload: Mapping[str, Any],
    rating_run_id: str = "latest",
    checkpoint_id: str = "",
) -> str:
    tournament_id = str(payload.get("selected_tournament_id") or "")
    battle_id = str(payload.get("battle_id") or "")
    back_href = _page_href(
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        checkpoint_id=checkpoint_id,
    )
    detail_html = _render_battle_detail_section(
        payload=payload,
        rating_run_id=rating_run_id,
        checkpoint_id=checkpoint_id,
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CurvyTron Battle</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f6f7f9; color: #202124; }}
    main {{ max-width: 1320px; margin: 0 auto; padding: 18px; }}
    header {{ display: grid; gap: 8px; margin-bottom: 14px; }}
    h1 {{ margin: 0; font-size: 20px; }}
    a {{ color: #1557b0; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .summary {{ margin: 0 0 12px; color: #5f6368; font-size: 13px; }}
    .panel {{ background: white; border: 1px solid #dadce0; border-radius: 8px; margin-bottom: 12px; overflow: hidden; }}
    .panel-head {{ display: flex; align-items: baseline; justify-content: space-between; gap: 12px; padding: 10px 12px; border-bottom: 1px solid #eef0f3; }}
    .panel h2 {{ margin: 0; font-size: 15px; }}
    .panel span {{ color: #5f6368; font-size: 12px; }}
    .pager {{ display: flex; justify-content: space-between; gap: 12px; padding: 10px 12px; border-top: 1px solid #eef0f3; font-size: 13px; }}
    .progress-body {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1px; background: #eef0f3; }}
    .progress-body div {{ display: grid; gap: 3px; padding: 12px; background: white; }}
    .progress-body strong {{ font-size: 18px; }}
    .progress-body span {{ color: #5f6368; font-size: 12px; }}
    .gif-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 10px; padding: 10px; }}
    .gif-card {{ display: grid; gap: 6px; color: #202124; }}
    .gif-card img {{ width: 100%; aspect-ratio: 1; object-fit: contain; background: #111827; }}
    .gif-card span {{ font-size: 12px; color: #5f6368; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 7px 10px; border-bottom: 1px solid #eef0f3; text-align: left; }}
    th {{ color: #5f6368; font-weight: 600; }}
    td:nth-child(n+3) {{ white-space: nowrap; }}
    .empty {{ padding: 60px; text-align: center; background: white; border: 1px dashed #dadce0; border-radius: 8px; color: #80868b; }}
  </style>
</head>
<body>
<main>
  <header>
    <a href="{html.escape(back_href)}">Back to checkpoint</a>
    <h1>{html.escape(battle_id)}</h1>
  </header>
  {detail_html}
</main>
</body>
</html>"""
