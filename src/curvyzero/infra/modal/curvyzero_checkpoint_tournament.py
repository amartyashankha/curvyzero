"""Modal app for CurvyTron checkpoint tournaments.

One app owns the whole tournament lane. The lowest-level function runs one game.
Higher functions fan out over games and checkpoint pairs.

Checkpoint files are read from the training Volume. Tournament summaries and
GIFs are written to a separate v2 Volume.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import modal

from curvyzero.infra.modal import run_management as runs
from curvyzero.tournament import curvytron_checkpoint_tournament as arena


APP_NAME = "curvyzero-checkpoint-tournament"
CHECKPOINT_VOLUME_NAME = "curvyzero-runs"
TOURNAMENT_VOLUME_NAME = "curvyzero-curvytron-tournaments"
LIGHTZERO_VERSION = "0.2.0"
REMOTE_ROOT = Path("/repo")
RUNS_MOUNT = Path("/runs")
TOURNAMENT_MOUNT = Path("/tournament-runs")
CURVYTRON_BONUS_SPRITE_SHEET_RELATIVE_PATH = (
    "third_party/curvytron-reference/web/images/bonus.png"
)
DEFAULT_LIMIT = 50
MAX_LIMIT = 500
GIF_CACHE_MAX_AGE_SECONDS = 86_400
DYNAMIC_HEADERS = {"Cache-Control": "no-store"}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        f"LightZero=={LIGHTZERO_VERSION}",
        "numpy>=1.26",
        "cloudpickle>=3",
        "pillow>=10",
        "fastapi>=0.110",
    )
    .env({"PYTHONPATH": str(REMOTE_ROOT / "src")})
    .add_local_dir(Path.cwd() / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
    .add_local_file(
        Path.cwd() / CURVYTRON_BONUS_SPRITE_SHEET_RELATIVE_PATH,
        remote_path=str(REMOTE_ROOT / CURVYTRON_BONUS_SPRITE_SHEET_RELATIVE_PATH),
        copy=True,
    )
)
checkpoint_volume = modal.Volume.from_name(
    CHECKPOINT_VOLUME_NAME,
    create_if_missing=True,
).read_only()
tournament_volume = modal.Volume.from_name(
    TOURNAMENT_VOLUME_NAME,
    create_if_missing=True,
    version=2,
)
app = modal.App(APP_NAME)


def _checkpoint_volumes() -> dict[str, Any]:
    return {RUNS_MOUNT.as_posix(): checkpoint_volume}


def _tournament_volumes() -> dict[str, Any]:
    return {TOURNAMENT_MOUNT.as_posix(): tournament_volume}


def _game_volumes() -> dict[str, Any]:
    return {**_checkpoint_volumes(), **_tournament_volumes()}


def _commit_volume(volume: Any = tournament_volume) -> str | None:
    if not hasattr(volume, "commit"):
        return None
    try:
        volume.commit()
    except Exception as exc:  # pragma: no cover - remote Volume resilience.
        return f"{type(exc).__name__}: {exc}"
    return None


def _reload_volume(volume: Any, *, force: bool = False) -> str | None:
    if volume is None or not hasattr(volume, "reload"):
        return None
    try:
        volume.reload()
    except Exception as exc:  # pragma: no cover - remote Volume resilience.
        return f"{type(exc).__name__}: {exc}"
    return None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _path_for_ref(ref: str | Path) -> Path:
    return runs.volume_path(TOURNAMENT_MOUNT, arena.validate_tournament_artifact_ref(ref))


def _write_tournament_marker(tournament_id: str) -> dict[str, Any]:
    ref = arena.tournament_marker_ref(tournament_id)
    payload = {
        "schema_id": "curvyzero_curvytron_tournament_browser_marker/v0",
        "artifact_volume_name": TOURNAMENT_VOLUME_NAME,
        "tournament_id": runs.clean_id(tournament_id, label="tournament_id"),
        "created_at": runs.utc_timestamp(),
    }
    return arena.write_json_artifact(TOURNAMENT_MOUNT, ref, payload)


def _write_tournament_manifest(spec: Mapping[str, Any], *, status: str) -> dict[str, Any]:
    tournament_id = runs.clean_id(str(spec["tournament_id"]), label="tournament_id")
    ref = arena.tournament_manifest_ref(tournament_id)
    payload = {
        "schema_id": arena.TOURNAMENT_SCHEMA_ID,
        "app_name": APP_NAME,
        "artifact_volume_name": TOURNAMENT_VOLUME_NAME,
        "checkpoint_volume_name": CHECKPOINT_VOLUME_NAME,
        "tournament_id": tournament_id,
        "status": status,
        "updated_at": runs.utc_timestamp(),
        "spec": arena._to_plain(dict(spec)),
    }
    return arena.write_json_artifact(TOURNAMENT_MOUNT, ref, payload)


@app.function(
    image=image,
    volumes=_game_volumes(),
    cpu=1.0,
    memory=4096,
    timeout=30 * 60,
    max_containers=500,
)
def curvytron_tournament_game(game_spec: dict[str, Any]) -> dict[str, Any]:
    checkpoint_reload_error = _reload_volume(checkpoint_volume)
    try:
        result = arena.run_checkpoint_game(
            game_spec,
            checkpoint_mount=RUNS_MOUNT,
            artifact_mount=TOURNAMENT_MOUNT,
            remote_root=REMOTE_ROOT,
        )
    except Exception as exc:  # pragma: no cover - remote dependency diagnosis.
        result = arena.failure_game_summary(
            game_spec,
            exc,
            artifact_mount=TOURNAMENT_MOUNT,
        )
    commit_error = _commit_volume(tournament_volume)
    if checkpoint_reload_error:
        result["checkpoint_reload_error"] = checkpoint_reload_error
    if commit_error:
        result["commit_error"] = commit_error
    print(json.dumps(arena._to_plain(arena._compact_game_result(result)), sort_keys=True))
    return arena._to_plain(result)


@app.function(
    image=image,
    volumes=_tournament_volumes(),
    cpu=0.5,
    memory=1024,
    timeout=4 * 60 * 60,
    max_containers=100,
)
def curvytron_tournament_pair(pair_spec: dict[str, Any]) -> dict[str, Any]:
    _reload_volume(tournament_volume)
    pair = arena.normalize_pair_spec(pair_spec)
    started_at = runs.utc_timestamp()
    spec_ref = arena.battle_root_ref(pair["tournament_id"], pair["battle_id"]) / "pair_spec.json"
    arena.write_json_artifact(TOURNAMENT_MOUNT, spec_ref, pair)
    game_specs = arena.build_game_specs_for_pair(pair)
    game_results = list(
        curvytron_tournament_game.map(
            game_specs,
            order_outputs=False,
        )
    )
    summary = arena.summarize_pair_results(pair, game_results)
    summary["started_at"] = started_at
    summary["ended_at"] = runs.utc_timestamp()
    summary["summary_ref"] = arena.battle_summary_ref(
        pair["tournament_id"],
        pair["battle_id"],
    ).as_posix()
    arena.write_json_artifact(
        TOURNAMENT_MOUNT,
        arena.battle_summary_ref(pair["tournament_id"], pair["battle_id"]),
        summary,
    )
    commit_error = _commit_volume(tournament_volume)
    if commit_error:
        summary["commit_error"] = commit_error
    print(json.dumps(arena._to_plain({"battle_id": pair["battle_id"], "tally": summary["tally"]}), sort_keys=True))
    return arena._to_plain(summary)


@app.function(
    image=image,
    volumes=_tournament_volumes(),
    cpu=1.0,
    memory=2048,
    timeout=12 * 60 * 60,
    max_containers=10,
)
def curvytron_tournament_run(tournament_spec: dict[str, Any]) -> dict[str, Any]:
    _reload_volume(tournament_volume)
    spec = dict(tournament_spec)
    tournament_id = runs.clean_id(str(spec.get("tournament_id") or runs.new_run_id("arena")), label="tournament_id")
    spec["tournament_id"] = tournament_id
    started_at = runs.utc_timestamp()
    _write_tournament_marker(tournament_id)
    _write_tournament_manifest(spec, status="running")

    checkpoints = spec.get("checkpoints") or spec.get("checkpoint_refs") or []
    if isinstance(checkpoints, str):
        checkpoints = arena.parse_checkpoint_refs(checkpoints)
    if not isinstance(checkpoints, list):
        raise ValueError("tournament spec needs a checkpoints list or comma-separated checkpoint_refs")
    pair_specs = arena.build_pair_specs(
        tournament_id=tournament_id,
        checkpoints=checkpoints,
        games_per_pair=int(spec.get("games_per_pair", arena.DEFAULT_GAMES_PER_PAIR)),
        ordered_pairs=bool(spec.get("ordered_pairs", arena.DEFAULT_ORDERED_PAIRS)),
        include_self_pairs=bool(
            spec.get("include_self_pairs", arena.DEFAULT_INCLUDE_SELF_PAIRS)
        ),
        seed=int(spec.get("seed", 0)),
        max_steps=int(spec.get("max_steps", arena.DEFAULT_MAX_STEPS)),
        decision_ms=float(spec.get("decision_ms", arena.DEFAULT_DECISION_MS)),
        num_simulations=int(spec.get("num_simulations", arena.DEFAULT_NUM_SIMULATIONS)),
        policy_batch_size=int(
            spec.get("policy_batch_size", arena.DEFAULT_POLICY_BATCH_SIZE)
        ),
        policy_mode=str(spec.get("policy_mode", arena.POLICY_MODE_EVAL)),
        collect_temperature=float(
            spec.get("collect_temperature", arena.DEFAULT_COLLECT_TEMPERATURE)
        ),
        collect_epsilon=float(spec.get("collect_epsilon", arena.DEFAULT_COLLECT_EPSILON)),
        natural_bonus_spawn=bool(spec.get("natural_bonus_spawn", True)),
        trail_render_mode=spec.get("trail_render_mode"),
        frame_stride=int(spec.get("frame_stride", arena.DEFAULT_FRAME_STRIDE)),
        frame_size=int(spec.get("frame_size", arena.DEFAULT_FRAME_SIZE)),
        gif_fps=float(spec.get("gif_fps", arena.DEFAULT_GIF_FPS)),
        save_gif=bool(spec.get("save_gif", arena.DEFAULT_SAVE_GIF)),
        save_frames_npz=bool(
            spec.get("save_frames_npz", arena.DEFAULT_SAVE_FRAMES_NPZ)
        ),
        action_trace_limit=int(spec.get("action_trace_limit", 128)),
    )
    pair_results = list(
        curvytron_tournament_pair.map(
            pair_specs,
            order_outputs=False,
        )
    )
    standings = arena.standings_from_pair_results(pair_results)
    standings["tournament_id"] = tournament_id
    standings["created_at"] = runs.utc_timestamp()
    arena.write_json_artifact(
        TOURNAMENT_MOUNT,
        arena.tournament_standings_ref(tournament_id),
        standings,
    )
    complete = {
        "schema_id": arena.TOURNAMENT_SCHEMA_ID,
        "ok": all(bool(pair.get("ok")) for pair in pair_results),
        "app_name": APP_NAME,
        "artifact_volume_name": TOURNAMENT_VOLUME_NAME,
        "checkpoint_volume_name": CHECKPOINT_VOLUME_NAME,
        "tournament_id": tournament_id,
        "started_at": started_at,
        "ended_at": runs.utc_timestamp(),
        "checkpoint_count": len(checkpoints),
        "pair_count": len(pair_results),
        "games_per_pair": int(spec.get("games_per_pair", arena.DEFAULT_GAMES_PER_PAIR)),
        "pair_summary_refs": [
            pair.get("summary_ref") for pair in pair_results if pair.get("summary_ref")
        ],
        "standings_ref": arena.tournament_standings_ref(tournament_id).as_posix(),
    }
    arena.write_json_artifact(
        TOURNAMENT_MOUNT,
        arena.tournament_complete_ref(tournament_id),
        complete,
    )
    _write_tournament_manifest({**spec, "pair_count": len(pair_specs)}, status="completed")
    commit_error = _commit_volume(tournament_volume)
    if commit_error:
        complete["commit_error"] = commit_error
    print(json.dumps(arena._to_plain(complete), sort_keys=True))
    return arena._to_plain({"complete": complete, "standings": standings})


def _list_tournaments(mount: Path) -> list[dict[str, Any]]:
    base = runs.volume_path(mount, arena.TOURNAMENT_BASE_REF)
    rows = []
    if not base.exists():
        return []
    for marker in base.glob(f"*/{arena.TOURNAMENT_RUN_MARKER_FILENAME}"):
        root = marker.parent
        tournament_id = root.name
        manifest = _read_json(root / "tournament.json")
        complete = _read_json(root / "complete.json")
        updated_path = root / "complete.json" if (root / "complete.json").exists() else marker
        rows.append(
            {
                "tournament_id": tournament_id,
                "status": complete.get("status") or manifest.get("status"),
                "updated_ts": updated_path.stat().st_mtime,
                "updated_at": complete.get("ended_at") or manifest.get("updated_at"),
                "pair_count": complete.get("pair_count"),
                "checkpoint_count": complete.get("checkpoint_count"),
            }
        )
    rows.sort(key=lambda row: (-float(row.get("updated_ts") or 0), row["tournament_id"]))
    return rows


def _list_battles(mount: Path, *, tournament_id: str, limit: int, offset: int) -> dict[str, Any]:
    clean_id = runs.clean_id(tournament_id, label="tournament_id")
    root = runs.volume_path(mount, arena.tournament_root_ref(clean_id)) / "battles"
    summaries = sorted(
        root.glob("*/battle.json") if root.exists() else [],
        key=lambda path: (-path.stat().st_mtime, path.as_posix()),
    )
    total = len(summaries)
    rows = []
    for path in summaries[offset : offset + limit]:
        row = _read_json(path)
        if not row:
            continue
        first_gif_ref = row.get("first_gif_ref")
        rows.append(
            {
                "tournament_id": row.get("tournament_id"),
                "battle_id": row.get("battle_id"),
                "players": row.get("players"),
                "tally": row.get("tally"),
                "ok": row.get("ok"),
                "summary_ref": runs.file_ref(path, mount=mount),
                "first_gif_ref": first_gif_ref,
                "updated_ts": path.stat().st_mtime,
            }
        )
    return {
        "rows": rows,
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_older": offset + limit < total,
        "has_newer": offset > 0,
    }


def _default_tournament_id(rows: list[dict[str, Any]], requested: str) -> str:
    if requested:
        clean = runs.clean_id(requested, label="tournament_id")
        if any(row["tournament_id"] == clean for row in rows):
            return clean
    return str(rows[0]["tournament_id"]) if rows else ""


def _render_page(
    *,
    tournaments: list[dict[str, Any]],
    selected_tournament_id: str,
    battles: dict[str, Any],
) -> str:
    import html

    options = "\n".join(
        f'<option value="{html.escape(row["tournament_id"])}" '
        f'{"selected" if row["tournament_id"] == selected_tournament_id else ""}>'
        f'{html.escape(row["tournament_id"])}</option>'
        for row in tournaments
    )
    cards = []
    for row in battles["rows"]:
        players = row.get("players") or []
        labels = " vs ".join(html.escape(str(player.get("label") or player.get("checkpoint_id"))) for player in players)
        tally = row.get("tally") or {}
        outcomes = tally.get("outcomes") or {}
        wins = tally.get("wins_by_seat") or {}
        gif_html = ""
        if row.get("first_gif_ref"):
            gif_ref = html.escape(str(row["first_gif_ref"]))
            gif_html = f'<a href="/gif?ref={gif_ref}"><img src="/gif?ref={gif_ref}" alt=""></a>'
        cards.append(
            f"""
            <article class="card">
              <div class="gif">{gif_html or '<span>No GIF</span>'}</div>
              <div class="body">
                <h2>{html.escape(str(row.get("battle_id")))}</h2>
                <p>{labels}</p>
                <p>{html.escape(str(tally.get("completed_count", 0)))} games, {html.escape(str(tally.get("failure_count", 0)))} failures</p>
                <p>seat 0 wins {html.escape(str(wins.get("seat_0", 0)))}, seat 1 wins {html.escape(str(wins.get("seat_1", 0)))}, draws {html.escape(str(outcomes.get("draw", 0)))}</p>
                <a href="/meta?ref={html.escape(str(row.get("summary_ref")))}">JSON</a>
              </div>
            </article>
            """
        )
    gallery = "\n".join(cards) or '<div class="empty">No battles yet.</div>'
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
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(min(420px, 100%), 1fr)); gap: 12px; }}
    .card {{ display: grid; grid-template-columns: 132px minmax(0, 1fr); background: white; border: 1px solid #dadce0; border-radius: 8px; overflow: hidden; }}
    .gif {{ display: grid; place-items: center; aspect-ratio: 1; background: #111827; color: #f8fafd; font-size: 13px; }}
    .gif img {{ width: 100%; height: 100%; object-fit: contain; }}
    .body {{ padding: 10px; min-width: 0; }}
    .body h2 {{ margin: 0 0 6px; font-size: 14px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .body p {{ margin: 4px 0; color: #5f6368; font-size: 13px; }}
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
    <form method="get">
      <label>Tournament<br><select name="tournament_id">{options}</select></label>
      <button type="submit">Open</button>
    </form>
  </header>
  <p class="summary">{html.escape(str(len(battles["rows"])))} shown / {html.escape(str(battles["total"]))} battles</p>
  <section class="grid">{gallery}</section>
</main>
</body>
</html>"""


def _build_fastapi_app(volume: Any):
    from fastapi import FastAPI, Header, Query
    from fastapi.responses import HTMLResponse, JSONResponse, Response

    web_app = FastAPI(title="CurvyTron checkpoint tournament")

    @web_app.get("/")
    def index(
        tournament_id: str = "",
        limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
        offset: int = Query(0, ge=0),
        fresh: bool = False,
    ) -> HTMLResponse:
        if fresh:
            _reload_volume(volume, force=True)
        tournaments = _list_tournaments(TOURNAMENT_MOUNT)
        selected = _default_tournament_id(tournaments, tournament_id)
        battles = (
            _list_battles(TOURNAMENT_MOUNT, tournament_id=selected, limit=limit, offset=offset)
            if selected
            else {"rows": [], "total": 0, "limit": limit, "offset": offset}
        )
        return HTMLResponse(
            _render_page(
                tournaments=tournaments,
                selected_tournament_id=selected,
                battles=battles,
            ),
            headers=DYNAMIC_HEADERS,
        )

    @web_app.get("/api/tournaments")
    def tournaments(fresh: bool = False) -> JSONResponse:
        if fresh:
            _reload_volume(volume, force=True)
        return JSONResponse({"tournaments": _list_tournaments(TOURNAMENT_MOUNT)}, headers=DYNAMIC_HEADERS)

    @web_app.get("/api/battles")
    def battles(
        tournament_id: str = "",
        limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
        offset: int = Query(0, ge=0),
        fresh: bool = False,
    ) -> JSONResponse:
        if fresh:
            _reload_volume(volume, force=True)
        tournaments = _list_tournaments(TOURNAMENT_MOUNT)
        selected = _default_tournament_id(tournaments, tournament_id)
        page = (
            _list_battles(TOURNAMENT_MOUNT, tournament_id=selected, limit=limit, offset=offset)
            if selected
            else {"rows": [], "total": 0, "limit": limit, "offset": offset}
        )
        return JSONResponse({"selected_tournament_id": selected, **page}, headers=DYNAMIC_HEADERS)

    @web_app.get("/gif")
    def gif(ref: str, if_none_match: str = Header(default="")) -> Response:
        try:
            safe_ref = arena.validate_tournament_artifact_ref(ref)
        except ValueError as exc:
            return Response(str(exc), status_code=400)
        if safe_ref.name != "game.gif":
            return Response("not a tournament GIF ref", status_code=400)
        path = runs.volume_path(TOURNAMENT_MOUNT, safe_ref)
        if not path.is_file():
            return Response("GIF not found", status_code=404)
        stat = path.stat()
        etag = f'W/"{stat.st_mtime_ns}-{stat.st_size}"'
        headers = {
            "Cache-Control": f"public, max-age={GIF_CACHE_MAX_AGE_SECONDS}, immutable",
            "ETag": etag,
            "Content-Length": str(stat.st_size),
        }
        if if_none_match == etag:
            return Response(status_code=304, headers=headers)
        return Response(path.read_bytes(), media_type="image/gif", headers=headers)

    @web_app.get("/meta")
    def meta(ref: str) -> Response:
        try:
            safe_ref = arena.validate_tournament_artifact_ref(ref)
        except ValueError as exc:
            return Response(str(exc), status_code=400)
        if safe_ref.suffix != ".json":
            return Response("not a JSON ref", status_code=400)
        path = runs.volume_path(TOURNAMENT_MOUNT, safe_ref)
        if not path.is_file():
            return Response("JSON not found", status_code=404)
        return Response(
            path.read_bytes(),
            media_type="application/json",
            headers={"Cache-Control": "no-cache"},
        )

    return web_app


@app.function(
    image=image,
    volumes=_tournament_volumes(),
    timeout=300,
    cpu=4,
    memory=4096,
    max_containers=2,
)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def curvytron_tournament_browser():
    return _build_fastapi_app(tournament_volume)


@app.local_entrypoint()
def main(
    mode: str = "pair",
    tournament_id: str = "",
    checkpoint_refs: str = "",
    games_per_pair: int = 2,
    seed: int = 0,
    max_steps: int = arena.DEFAULT_MAX_STEPS,
    policy_mode: str = arena.POLICY_MODE_EVAL,
    collect_temperature: float = arena.DEFAULT_COLLECT_TEMPERATURE,
    collect_epsilon: float = arena.DEFAULT_COLLECT_EPSILON,
    num_simulations: int = arena.DEFAULT_NUM_SIMULATIONS,
    save_gif: bool = True,
    wait: bool = False,
) -> None:
    refs = arena.parse_checkpoint_refs(checkpoint_refs)
    resolved_tournament_id = tournament_id or runs.new_run_id("arena")
    if mode not in {"game", "pair", "tournament"}:
        raise ValueError("mode must be one of: game, pair, tournament")
    if mode in {"game", "pair"} and len(refs) != 2:
        raise ValueError("game/pair mode needs exactly two checkpoint refs")
    if mode == "tournament" and len(refs) < 2:
        raise ValueError("tournament mode needs at least two checkpoint refs")

    common = {
        "games_per_pair": int(games_per_pair),
        "seed": int(seed),
        "max_steps": int(max_steps),
        "policy_mode": policy_mode,
        "collect_temperature": float(collect_temperature),
        "collect_epsilon": float(collect_epsilon),
        "num_simulations": int(num_simulations),
        "save_gif": bool(save_gif),
    }
    if mode == "tournament":
        spec = {"tournament_id": resolved_tournament_id, "checkpoints": refs, **common}
        call = curvytron_tournament_run.spawn(spec)
    else:
        pair = arena.build_pair_specs(
            checkpoints=refs,
            tournament_id=resolved_tournament_id,
            **common,
        )[0]
        if mode == "pair":
            call = curvytron_tournament_pair.spawn(pair)
        else:
            game = arena.build_game_specs_for_pair(pair)[0]
            call = curvytron_tournament_game.spawn(game)
    call_id = getattr(call, "object_id", None) or getattr(call, "id", None)
    payload: dict[str, Any] = {
        "status": "spawned",
        "app_name": APP_NAME,
        "mode": mode,
        "tournament_id": resolved_tournament_id,
        "function_call_id": call_id,
        "browser_url_hint": (
            "deploy this module, then open the curvytron_tournament_browser web endpoint"
        ),
    }
    if wait:
        payload["result"] = call.get()
    print(json.dumps(arena._to_plain(payload), indent=2, sort_keys=True))
