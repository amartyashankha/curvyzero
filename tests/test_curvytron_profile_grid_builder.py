import argparse
import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / (
    "build_curvytron_profile_grid.py"
)
SPEC = importlib.util.spec_from_file_location("curvytron_profile_grid_builder", SCRIPT_PATH)
assert SPEC is not None
builder = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(builder)


def _args(**overrides):
    defaults = {
        "experiment_id": "unit-profile-grid",
        "family": "unit",
        "run_prefix": "unit-profile-grid",
        "attempt_prefix": "profile",
        "seed": 304,
        "seeds": None,
        "computes": ["gpu-h100-cpu40"],
        "env_manager_types": ["subprocess"],
        "collectors": [512],
        "batch_sizes": [64],
        "num_simulations": [4],
        "exploration_bonus_modes": ["none"],
        "exploration_bonus_weight": 0.0,
        "exploration_bonus_rnd_batch_size": 64,
        "exploration_bonus_rnd_update_per_collect": 100,
        "source_max_steps": 512,
        "max_train_iter": 96,
        "max_env_step": 200_000,
        "save_ckpt_after_iter": 999_999,
        "stop_after_learner_train_calls": 12,
        "env_telemetry_stride": 256,
        "env_variant": "source_state_fixed_opponent",
        "reward_variant": "sparse_outcome",
        "opponent_policy_kind": "fixed_straight",
        "trail_render_mode": "browser_lines",
        "bonus_render_mode": "simple_symbols",
        "policy_observation_backend": "cpu_oracle",
        "collect_search_backends": ["stock"],
        "collect_search_ctree_backends": ["lightzero"],
        "disable_death_for_profile": True,
        "detached": True,
        "stdout_only": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_profile_grid_builder_can_emit_seed_axis_for_repeated_baseline():
    manifest = builder.build_manifest(_args(seeds=[304, 305, 306]))

    assert manifest["row_count"] == 3
    assert manifest["guardrails"]["seeds"] == [304, 305, 306]
    assert [row["seed"] for row in manifest["rows"]] == [304, 305, 306]
    assert [row["row_id"] for row in manifest["rows"]] == ["001", "002", "003"]
    assert all("--profile-spawn" in row["command"] for row in manifest["rows"])
    assert all("--detach" in row["command"] for row in manifest["rows"])
    assert all("-s" in row["label"] for row in manifest["rows"])


def test_profile_grid_builder_emits_collect_search_backend_flags():
    manifest = builder.build_manifest(
        _args(
            collect_search_backends=["direct_ctree_gpu_latent"],
            collect_search_ctree_backends=["flat_a3"],
        )
    )

    row = manifest["rows"][0]
    assert row["collect_search_backend"] == "direct_ctree_gpu_latent"
    assert row["collect_search_ctree_backend"] == "flat_a3"
    assert "--collect-search-backend" in row["command"]
    assert "direct_ctree_gpu_latent" in row["command"]
    assert "--collect-search-ctree-backend" in row["command"]
    assert "flat_a3" in row["command"]
    assert "directsearch-flat_a3" in row["label"]


def test_profile_grid_builder_rejects_flat_ctree_without_direct_search():
    try:
        builder.build_manifest(
            _args(
                collect_search_backends=["stock"],
                collect_search_ctree_backends=["flat_a3"],
            )
        )
    except ValueError as exc:
        assert "requires collect_search_backend='direct_ctree_gpu_latent'" in str(exc)
    else:
        raise AssertionError("expected flat_a3 stock-search manifest rejection")


def test_profile_grid_builder_matched_denominator_stock_preset_labels_currency():
    args = _args(
        computes=["gpu-l4-t4-cpu40"],
        collectors=[1],
        batch_sizes=[1],
        num_simulations=[1],
        disable_death_for_profile=False,
        detached=False,
    )

    builder.apply_next_matched_denominator_stock_preset(args)
    manifest = builder.build_manifest(args)

    row = manifest["rows"][0]
    assert manifest["matched_denominator"]["id"] == builder.MATCHED_DENOMINATOR_ID
    assert row["matched_pair_role"] == "stock_reference"
    assert row["speed_currency"] == builder.MATCHED_STOCK_SPEED_CURRENCY
    assert row["row_purpose"] == builder.MATCHED_DENOMINATOR_ROW_PURPOSE
    assert row["promotion_claim"] is False
    assert row["compute"] == "gpu-h100-cpu40"
    assert row["collector_env_num"] == 512
    assert row["batch_size"] == 64
    assert row["num_simulations"] == 8
    assert row["collect_search_backend"] == "stock"
    assert row["collect_search_ctree_backend"] == "lightzero"
    assert row["exploration_bonus_mode"] == "none"
    assert row["fixed_denominator"]["speed_currency"] == row["speed_currency"]
    assert "--detach" in row["command"]
    assert "--profile-spawn" in row["command"]
    assert "--disable-death-for-profile" in row["command"]
