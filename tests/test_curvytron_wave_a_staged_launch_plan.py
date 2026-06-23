from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "plan_curvytron_wave_a_staged_launch.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("plan_curvytron_wave_a_staged_launch", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rnd_rows() -> list[dict]:
    weights = [0.003, 0.01, 0.03, 0.1, 0.3, 0.6, 1.0]
    rows = []
    for copy in range(5):
        seed = 20260519 + copy
        base = copy * 9
        specs = [("none", 0.0), ("rnd_meter_v0", 0.0)] + [
            ("rnd_replay_target_v0", weight) for weight in weights
        ]
        for offset, (mode, weight) in enumerate(specs, start=1):
            row_id = f"r{base + offset:03d}"
            rows.append(
                {
                    "row_id": row_id,
                    "run_id": f"rnd-{row_id}",
                    "training_seed": seed,
                    "label": f"{mode}-{weight}-copy{copy:02d}",
                    "exploration_bonus": {"mode": mode, "weight": weight},
                    "train_kwargs": {"exploration_bonus_mode": mode},
                }
            )
    return rows


def _non_rnd_rows(prefix: str) -> list[dict]:
    variants = [
        "sparse_outcome",
        "survival_plus_bonus_no_outcome",
        "survival_plus_bonus_plus_outcome",
    ]
    rows = []
    for index in range(1, 19):
        rows.append(
            {
                "row_id": f"r{index:03d}",
                "run_id": f"{prefix}-r{index:03d}",
                "reward_variant": variants[(index - 1) % len(variants)],
            }
        )
    return rows


def _write_manifest(path: Path, rows: list[dict]) -> None:
    _write_json(path, {"schema_id": "test", "rows": rows})


def _write_default_manifests(repo_root: Path, planner) -> None:
    for lane in planner._default_lanes().values():
        rows = _rnd_rows() if lane.lane_group == "rnd" else _non_rnd_rows(lane.lane_id)
        _write_manifest(repo_root / lane.manifest_relpath, rows)


def test_mid36_profile_preserves_rnd_and_static_controls(tmp_path):
    planner = _load_script()
    _write_default_manifests(tmp_path, planner)

    report = planner.build_report(
        planner.parse_args(["--repo-root", str(tmp_path), "--profile", "mid36"])
    )

    assert report["ok"] is True
    assert report["total_selected_rows"] == 36
    assert report["rnd_selected_rows"] == 18
    assert report["non_rnd_selected_rows"] == 18
    assert report["max_active_h100_rows"] == 40
    assert report["rnd_counts"]["none:0.0"] == 2
    assert report["rnd_counts"]["rnd_meter_v0:0.0"] == 2
    assert report["rnd_counts"]["rnd_replay_target_v0:1.0"] == 2
    assert any(lane["lane_id"] == "rnd-blank-sweep" and lane["partial_launch"] for lane in report["lanes"])


def test_long19_profile_focuses_low_weight_rnd_with_non_rnd_triad(tmp_path):
    planner = _load_script()
    _write_default_manifests(tmp_path, planner)

    report = planner.build_report(
        planner.parse_args(
            ["--repo-root", str(tmp_path), "--profile", "long19_low_weight_replicated"]
        )
    )

    assert report["ok"] is True
    assert report["total_selected_rows"] == 19
    assert report["rnd_selected_rows"] == 10
    assert report["non_rnd_selected_rows"] == 9
    assert report["rnd_counts"]["rnd_replay_target_v0:0.003"] == 2
    assert report["rnd_counts"]["rnd_replay_target_v0:0.03"] == 2
    assert "rnd_replay_target_v0:1.0" not in report["rnd_counts"]
    assert all("--allow-launch" in command for command in report["commands"])
    assert any("--allow-partial-launch" in command for command in report["commands"])


def test_profile_reports_missing_manifest(tmp_path):
    planner = _load_script()
    _write_manifest(
        tmp_path / planner._default_lanes()["rnd-blank-sweep"].manifest_relpath,
        _rnd_rows(),
    )

    report = planner.build_report(
        planner.parse_args(["--repo-root", str(tmp_path), "--profile", "mid36"])
    )

    assert report["ok"] is False
    assert any(error["message"] == "manifest missing" for error in report["errors"])
