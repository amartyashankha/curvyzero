from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_curvytron_opponent_mixture_manifest.py"
SUBMIT_SCRIPT = ROOT / "scripts" / "submit_curvytron_survivaldiag_manifest.py"


MAIN_RECIPES = (
    "r50-blank50",
    "r50-mid50",
    "r50-old50",
    "r50-scr50",
    "r50-pass50",
    "r50-blank25-scr25",
    "r50-mid25-old25",
    "r50-blank20-mid15-scr15",
)
CONTROL_RECIPES = (
    "recent100",
    "mid100",
    "old100",
    "blank100",
    "scr100",
    "pass100",
)
NEXT_WAVE_MAIN_RECIPES = (
    "r25-blank75",
    "r50-blank50",
    "r75-blank25",
    "r50-scr50",
    "r50-mid25-old25",
    "r40-blank20-mid20-scr20",
)
NEXT_WAVE_CONTROL_RECIPES = (
    "recent100",
    "blank100",
    "scr100",
    "mid100",
    "old100",
)
NEXT_WAVE_COMPUTE_RECIPES = (
    "r50-blank50",
    "r75-blank25",
    "r50-scr50",
    "r50-mid25-old25",
    "r40-blank20-mid20-scr20",
)
CORE_BASES = (
    "rf-s8-c32-l32-rep0-k10",
    "rf-s8-c32-l32-repM-k10",
    "rf-s8-c32-l32-repH-k10",
    "rb-s8-c32-l32-rep0-k10",
    "rb-s8-c32-l32-repM-k10",
    "rb-s8-c32-l32-repH-k10",
)
SENTINEL_BASES = (
    "rf-s16-c32-l32-repM-k10",
    "rf-s8-c64-l32-repM-k10",
    "rf-s8-c32-l64-repM-k10",
    "rb-s16-c32-l32-repM-k10",
    "rb-s8-c64-l32-repM-k10",
    "rb-s8-c32-l64-repM-k10",
)


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _manifest(extra_args: list[str] | None = None) -> dict:
    module = _load_script(SCRIPT, "build_curvytron_opponent_mixture_manifest_for_test")
    args = module.parse_args(["--stdout-only", *(extra_args or [])])
    return module.build_manifest(args)


def test_opponent_mixture_batch_shape_and_readable_names():
    module = _load_script(SCRIPT, "build_curvytron_opponent_mixture_manifest_for_test")
    manifest = _manifest()

    assert manifest["matrix_name"] == "curvy-mix2-20260513a"
    assert manifest["profile"] == "batch"
    assert manifest["batch_scope"] == "full"
    assert manifest["row_count"] == 228
    assert manifest["recipe_count"] == 14
    assert manifest["main_recipe_count"] == 8
    assert manifest["control_recipe_count"] == 6
    assert manifest["core_profile_count"] == 6
    assert manifest["sentinel_profile_count"] == 6
    assert manifest["guards"]["deployed_app_name"] == (
        "curvyzero-lightzero-curvytron-visual-survival-train"
    )

    run_ids = [row["run_id"] for row in manifest["rows"]]
    attempt_ids = [row["attempt_id"] for row in manifest["rows"]]
    run_pattern = re.compile(
        r"^curvy-mix2-[a-z0-9-]+-(?:rf|rb)-s(?:8|16)-c(?:32|64)-"
        r"l(?:32|64)-rep(?:0|M|H)-k10-c[1-3]-s\d+$"
    )
    attempt_pattern = re.compile(
        r"^try-mix2-[a-z0-9-]+-(?:rf|rb)-s(?:8|16)-c(?:32|64)-"
        r"l(?:32|64)-rep(?:0|M|H)-k10-c[1-3]-s\d+$"
    )
    assert len(run_ids) == len(set(run_ids))
    assert len(attempt_ids) == len(set(attempt_ids))
    assert all(run_pattern.fullmatch(run_id) for run_id in run_ids)
    assert all(attempt_pattern.fullmatch(attempt_id) for attempt_id in attempt_ids)
    assert all(len(run_id) <= 96 for run_id in run_ids)
    assert all(len(attempt_id) <= 96 for attempt_id in attempt_ids)
    assert not any("scripted" in run_id or "passive" in run_id for run_id in run_ids)
    assert not any(re.search(r"-r\d{3}(?:-|$)", run_id) for run_id in run_ids)

    for row in manifest["rows"]:
        assert row["mode"] == "train"
        assert row["calls_stock_train_muzero"] is True
        assert row["env_variant"] == "source_state_fixed_opponent"
        assert row["reward_variant"] == "survival_plus_bonus_no_outcome"
        assert row["source_max_steps"] == 65536
        assert row["train_kwargs"]["decision_ms"] == module.DECISION_MS
        assert row["train_kwargs"]["decision_ms"] < 20.0
        assert row["train_kwargs"]["background_eval_enabled"] is True
        assert row["train_kwargs"]["background_gif_enabled"] is True
        assert row["train_kwargs"]["lightzero_eval_freq"] == 0
        assert row["train_kwargs"]["opponent_checkpoint_ref"] is None
        assert row["poller_kwargs"]["opponent_checkpoint_ref"] is None
        assert (
            row["poller_kwargs"]["opponent_mixture_spec"]
            == (row["train_kwargs"]["opponent_mixture_spec"])
        )
        assert row["deployed_app_submission"]["train_function"] == (
            "lightzero_curvytron_visual_survival_gpu_cpu40"
        )
        assert row["deployed_app_submission"]["poller_function"] == (
            "lightzero_curvytron_visual_survival_checkpoint_eval_poller"
        )


def test_opponent_mixture_batch_recipes_and_base_counts():
    manifest = _manifest()
    recipe_ids = [recipe["recipe_id"] for recipe in manifest["recipes"]]

    assert recipe_ids == [*MAIN_RECIPES, *CONTROL_RECIPES]
    recipes = {recipe["recipe_id"]: recipe["weights"] for recipe in manifest["recipes"]}
    assert recipes["r50-blank50"] == {"recent": 50, "blank": 50}
    assert recipes["r50-scr50"] == {"recent": 50, "scripted": 50}
    assert recipes["r50-pass50"] == {"recent": 50, "passive": 50}
    assert recipes["r50-blank25-scr25"] == {
        "recent": 50,
        "blank": 25,
        "scripted": 25,
    }
    assert recipes["r50-blank20-mid15-scr15"] == {
        "recent": 50,
        "blank": 20,
        "mid": 15,
        "scripted": 15,
    }
    assert recipes["recent100"] == {"recent": 100}
    assert recipes["mid100"] == {"mid": 100}
    assert recipes["old100"] == {"old": 100}
    assert recipes["blank100"] == {"blank": 100}
    assert recipes["scr100"] == {"scripted": 100}
    assert recipes["pass100"] == {"passive": 100}

    by_recipe = Counter(row["recipe_id"] for row in manifest["rows"])
    assert {recipe_id: by_recipe[recipe_id] for recipe_id in MAIN_RECIPES} == {
        recipe_id: 24 for recipe_id in MAIN_RECIPES
    }
    assert {recipe_id: by_recipe[recipe_id] for recipe_id in CONTROL_RECIPES} == {
        recipe_id: 6 for recipe_id in CONTROL_RECIPES
    }

    copies_by_recipe_base = Counter(
        (row["recipe_id"], row["base_token"]) for row in manifest["rows"]
    )
    for recipe_id in MAIN_RECIPES:
        for base in CORE_BASES:
            assert copies_by_recipe_base[(recipe_id, base)] == 3
        for base in SENTINEL_BASES:
            assert copies_by_recipe_base[(recipe_id, base)] == 1
    for recipe_id in CONTROL_RECIPES:
        for base in CORE_BASES:
            assert copies_by_recipe_base[(recipe_id, base)] == 1
        for base in SENTINEL_BASES:
            assert copies_by_recipe_base[(recipe_id, base)] == 0


def test_opponent_mixture_base_tokens_match_trainer_settings_and_cadence():
    manifest = _manifest()
    base_profiles = {
        profile["token"]: profile
        for group in manifest["base_profiles"].values()
        for profile in group
    }
    assert set(base_profiles) == {*CORE_BASES, *SENTINEL_BASES}

    for row in manifest["rows"]:
        base = base_profiles[row["base_token"]]
        assert row["base_settings"]["save_ckpt_after_iter"] == 10000
        assert row["train_kwargs"]["save_ckpt_after_iter"] == 10000
        assert row["base_settings"]["source_state_trail_render_mode"] == (
            "body_circles_fast" if base["render_token"] == "rf" else "browser_lines"
        )
        assert (
            row["source_state_trail_render_mode"]
            == (row["base_settings"]["source_state_trail_render_mode"])
        )
        assert (
            row["train_kwargs"]["source_state_trail_render_mode"]
            == (row["base_settings"]["source_state_trail_render_mode"])
        )
        for key in ("num_simulations", "collector_env_num", "n_episode", "batch_size"):
            assert row["base_settings"][key] == base[key]
            assert row["train_kwargs"][key] == base[key]
        assert row["base_settings"]["collector_env_num"] == row["base_settings"]["n_episode"]

        repeat = base["repeat_token"]
        expected_repeat = {
            "rep0": (1, 1, 0.0, "none"),
            "repM": (1, 3, 0.20, "policy_action_repeat_medium"),
            "repH": (1, 3, 0.35, "policy_action_repeat_high"),
        }[repeat]
        assert row["train_kwargs"]["policy_action_repeat_min"] == expected_repeat[0]
        assert row["train_kwargs"]["policy_action_repeat_max"] == expected_repeat[1]
        assert row["train_kwargs"]["policy_action_repeat_extra_probability"] == (expected_repeat[2])
        assert row["train_kwargs"]["control_noise_profile_id"] == expected_repeat[3]


def test_opponent_mixture_batch_recipes_keep_immutable_checkpoint_refs():
    manifest = _manifest()

    for row in manifest["rows"]:
        mixture = row["opponent_mixture_spec"]
        assert mixture["schema_id"] == "curvyzero_episode_opponent_mixture/v0"
        assert mixture["selection_unit"] == "episode_reset"
        assert mixture["total_weight"] == 100.0
        for entry in mixture["entries"]:
            if entry["opponent_policy_kind"] == "frozen_lightzero_checkpoint":
                assert entry["opponent_death_mode"] == "immortal"
                assert entry["opponent_checkpoint_ref"].endswith(".pth.tar")
                assert "latest" not in entry["opponent_checkpoint_ref"]
                assert "ckpt_best" not in entry["opponent_checkpoint_ref"]
                assert re.search(r"/iteration_\d+\.pth\.tar$", entry["opponent_checkpoint_ref"])
            elif entry["name"] == "blank":
                assert entry["opponent_runtime_mode"] == "blank_canvas_noop"
            elif entry["name"] in {"scripted", "passive"}:
                assert entry["opponent_death_mode"] == "immortal"


def test_opponent_mixture_canary_shape_is_six_rows_and_launchable_by_grouped_submitter(
    tmp_path,
):
    manifest = _manifest(
        [
            "--profile",
            "canary",
            "--matrix-name",
            "curvy-mix2-canary-20260513a",
            "--run-prefix",
            "curvy-mix2-canary",
            "--attempt-prefix",
            "try-mix2-canary",
        ]
    )
    submit = _load_script(SUBMIT_SCRIPT, "submit_curvytron_survivaldiag_manifest_mix_test")
    manifest_path = tmp_path / "mix_canary.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    args = submit.parse_args([str(manifest_path)])
    rows = submit._selected_rows(submit._load_manifest(manifest_path), args)
    records = [
        submit._launch_row(
            row,
            app_name=manifest["guards"]["deployed_app_name"],
            dry_run=True,
        )
        for row in rows
    ]

    assert manifest["row_count"] == 6
    assert [(row["recipe_id"], row["base_token"]) for row in manifest["rows"]] == [
        ("r50-blank25-scr25", "rf-s8-c32-l32-repM-k10"),
        ("r50-blank25-scr25", "rb-s8-c32-l32-repM-k10"),
        ("r50-mid25-old25", "rf-s8-c32-l32-repM-k10"),
        ("r50-mid25-old25", "rb-s8-c32-l32-repM-k10"),
        ("r50-pass50", "rf-s8-c32-l32-repH-k10"),
        ("r50-scr50", "rb-s8-c32-l32-repH-k10"),
    ]
    assert all(row["copy_index"] == 1 for row in manifest["rows"])
    assert all(record["status"] == "dry_run" for record in records)
    assert all(record["app_name"] == manifest["guards"]["deployed_app_name"] for record in records)


def test_opponent_mixture_core_batch_omits_heavy_sentinels():
    manifest = _manifest(
        [
            "--batch-scope",
            "core",
            "--matrix-name",
            "curvy-mix2-core-20260513a",
            "--run-prefix",
            "curvy-mix2core",
            "--attempt-prefix",
            "try-mix2core",
        ]
    )

    assert manifest["matrix_name"] == "curvy-mix2-core-20260513a"
    assert manifest["batch_scope"] == "core"
    assert manifest["row_count"] == 180
    assert manifest["sentinel_profile_count"] == 0
    assert manifest["main_sentinel_copies"] == 0
    assert manifest["base_profiles"]["sentinel"] == []
    assert manifest["base_settings"]["num_simulations_values"] == [8]
    assert manifest["base_settings"]["collector_env_num_values"] == [32]
    assert manifest["base_settings"]["batch_size_values"] == [32]

    by_recipe = Counter(row["recipe_id"] for row in manifest["rows"])
    assert {recipe_id: by_recipe[recipe_id] for recipe_id in MAIN_RECIPES} == {
        recipe_id: 18 for recipe_id in MAIN_RECIPES
    }
    assert {recipe_id: by_recipe[recipe_id] for recipe_id in CONTROL_RECIPES} == {
        recipe_id: 6 for recipe_id in CONTROL_RECIPES
    }
    assert not any(row["base_profile_kind"] == "sentinel" for row in manifest["rows"])
    assert set(row["base_token"] for row in manifest["rows"]) == set(CORE_BASES)


def test_opponent_mixture_recipe_filter_builds_pruned_launch_set():
    selected_recipes = [
        "r50-blank50",
        "r50-mid50",
        "r50-old50",
        "r50-scr50",
        "r50-blank25-scr25",
        "r50-mid25-old25",
        "r50-blank20-mid15-scr15",
        "recent100",
        "mid100",
        "old100",
        "blank100",
        "scr100",
    ]
    manifest = _manifest(
        [
            "--batch-scope",
            "core",
            *[arg for recipe_id in selected_recipes for arg in ("--recipe-id", recipe_id)],
        ]
    )

    assert manifest["row_count"] == 156
    assert manifest["recipe_count"] == 12
    assert manifest["main_recipe_count"] == 7
    assert manifest["control_recipe_count"] == 5
    assert manifest["excluded_recipe_ids"] == ["r50-pass50", "pass100"]
    assert not any("passive" in row["opponent_components"] for row in manifest["rows"])

    by_recipe = Counter(row["recipe_id"] for row in manifest["rows"])
    assert by_recipe["r50-pass50"] == 0
    assert by_recipe["pass100"] == 0
    assert {recipe_id: by_recipe[recipe_id] for recipe_id in selected_recipes[:7]} == {
        recipe_id: 18 for recipe_id in selected_recipes[:7]
    }
    assert {recipe_id: by_recipe[recipe_id] for recipe_id in selected_recipes[7:]} == {
        recipe_id: 6 for recipe_id in selected_recipes[7:]
    }


def test_opponent_mixture_builder_rejects_mutable_checkpoint_refs():
    module = _load_script(
        SCRIPT,
        "build_curvytron_opponent_mixture_manifest_mutable_ref_test",
    )
    args = module.parse_args(
        [
            "--stdout-only",
            "--recent-opponent-checkpoint-ref",
            "training/run/checkpoints/lightzero/latest.pth.tar",
        ]
    )

    try:
        module.build_manifest(args)
    except ValueError as exc:
        assert "mutable" in str(exc)
    else:
        raise AssertionError("mutable checkpoint ref was accepted")


def test_opponent_mixture_next_wave_shape_recipe_counts_and_no_passive_rows():
    manifest = _manifest(["--profile", "next-wave"])

    assert manifest["matrix_name"] == "curvy-mix3-nextwave-20260513a"
    assert manifest["profile"] == "next-wave"
    assert manifest["dry_run_only"] is True
    assert manifest["launches_modal"] is False
    assert manifest["row_count"] == 300
    assert manifest["recipe_count"] == 11
    assert manifest["main_recipe_count"] == 6
    assert manifest["control_recipe_count"] == 5
    assert manifest["compute_probe_recipe_count"] == 5
    assert manifest["main_core_copies"] == 5
    assert manifest["control_copies"] == 2
    assert manifest["compute_probe_copies"] == 1
    assert manifest["sentinel_profile_count"] == 0
    assert manifest["compute_probe_profile_count"] == 12

    recipe_ids = [recipe["recipe_id"] for recipe in manifest["recipes"]]
    assert recipe_ids == [*NEXT_WAVE_MAIN_RECIPES, *NEXT_WAVE_CONTROL_RECIPES]
    assert not any("pass" in recipe_id for recipe_id in recipe_ids)
    assert not any("passive" in row["opponent_components"] for row in manifest["rows"])
    assert all(
        row["row_kind"] == "opponent_mixture_next_wave_candidate" for row in manifest["rows"]
    )

    by_block = Counter(row["next_wave_block"] for row in manifest["rows"])
    assert by_block == {"main": 180, "control": 60, "compute_probe": 60}

    by_recipe = Counter(row["recipe_id"] for row in manifest["rows"])
    assert by_recipe["r25-blank75"] == 30
    for recipe_id in NEXT_WAVE_COMPUTE_RECIPES:
        assert by_recipe[recipe_id] == 42
    for recipe_id in NEXT_WAVE_CONTROL_RECIPES:
        assert by_recipe[recipe_id] == 12


def test_opponent_mixture_next_wave_render_pairing_is_complete():
    manifest = _manifest(["--profile", "next-wave"])

    pairs: dict[str, list[dict]] = {}
    for row in manifest["rows"]:
        pairs.setdefault(row["launch_pair_id"], []).append(row)

    assert len(pairs) == 150
    assert all(len(rows) == 2 for rows in pairs.values())
    for rows in pairs.values():
        assert {row["render_role"] for row in rows} == {"fast", "browser"}
        assert len({row["launch_pair_key"] for row in rows}) == 1
        assert len({row["recipe_id"] for row in rows}) == 1
        assert len({row["copy_index"] for row in rows}) == 1
        assert len({row["base_settings"]["num_simulations"] for row in rows}) == 1
        assert len({row["base_settings"]["collector_env_num"] for row in rows}) == 1
        assert len({row["base_settings"]["batch_size"] for row in rows}) == 1
        assert len({row["base_settings"]["control_noise_profile_id"] for row in rows}) == 1

    main_pairs = {
        row["launch_pair_key"] for row in manifest["rows"] if row["next_wave_block"] == "main"
    }
    assert len(main_pairs) == 90
    control_pairs = {
        row["launch_pair_key"] for row in manifest["rows"] if row["next_wave_block"] == "control"
    }
    assert len(control_pairs) == 30
    compute_pairs = {
        row["launch_pair_key"]
        for row in manifest["rows"]
        if row["next_wave_block"] == "compute_probe"
    }
    assert len(compute_pairs) == 30


def test_opponent_mixture_next_wave_balances_launch_order():
    manifest = _manifest(["--profile", "next-wave"])

    assert manifest["launch_order_strategy"] == "matched_render_pairs_alternating_lead"
    assert (
        "does not make browser a checkpoint-speed blocker" in manifest["render_pairing_rationale"]
    )

    running = Counter()
    for row in manifest["rows"]:
        running[row["render_role"]] += 1
        assert abs(running["fast"] - running["browser"]) <= 1

    pair_leads = [
        rows[0]["render_role"]
        for _pair_id, rows in sorted(
            (
                (pair_id, sorted(pair_rows, key=lambda row: row["launch_order_index"]))
                for pair_id, pair_rows in {
                    row["launch_pair_id"]: [
                        candidate
                        for candidate in manifest["rows"]
                        if candidate["launch_pair_id"] == row["launch_pair_id"]
                    ]
                    for row in manifest["rows"]
                }.items()
            ),
            key=lambda item: item[1][0]["launch_order_index"],
        )
    ]
    assert pair_leads[:6] == ["fast", "browser", "fast", "browser", "fast", "browser"]
    assert Counter(pair_leads) == {"fast": 75, "browser": 75}
