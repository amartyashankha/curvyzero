import json

from curvyzero.training.curvytron_inspector import build_inspector_report


def _row(
    checkpoint: str,
    seed: int,
    steps: int,
    *,
    cap: int = 100,
    ok: object = True,
    strict_load: object = True,
    actions: dict[str, int] | None = None,
    terminal_reason: str = "round_survivor_win",
    death_cause: str | None = "wall",
) -> dict[str, object]:
    row: dict[str, object] = {
        "checkpoint_label": checkpoint,
        "seed": seed,
        "steps_survived": steps,
        "cap": cap,
        "ok": ok,
        "strict_load": strict_load,
        "terminal_reason": terminal_reason,
        "action_histogram": actions or {"left": 3, "straight": 4, "right": 3},
    }
    if death_cause is not None:
        row["death_cause"] = death_cause
    return row


def _report(
    rows: list[dict[str, object]],
    *,
    opponent_policy_kind: str = "self_play_current",
    opponent_training_relation: str = "current_policy_self_play",
    include_train_summary: bool = True,
    train_summary_overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    train_summary: dict[str, object] | None = None
    if include_train_summary:
        train_summary = {
            "run_id": "run-fixture",
            "current_policy_self_play": True,
            "opponent_training_relation": opponent_training_relation,
            "learning_proof": None,
            "debug_fidelity_only": False,
        }
        if train_summary_overrides:
            train_summary.update(train_summary_overrides)

    return build_inspector_report(
        {
            "ok": True,
            "eval_id": "eval-fixture",
            "config": {
                "run_id": "run-fixture",
                "opponent_policy_kind": opponent_policy_kind,
            },
            "table": rows,
        },
        train_summary=train_summary,
        created_at="2026-05-11T00:00:00+00:00",
    )


def _warning_text(report: dict[str, object]) -> str:
    return " ".join(str(item) for item in report["warnings"])


def test_seed_counter_mismatch_blocks_improved_learning_claim():
    report = _report(
        [
            _row("iteration_0", 1, 10),
            _row("iteration_0", 1, 10),
            _row("iteration_0", 2, 10),
            _row("iteration_10", 1, 30),
            _row("iteration_10", 2, 30),
            _row("iteration_10", 2, 30),
        ]
    )

    assert report["eval"]["row_count"] == 6
    assert report["eval"]["unique_seed_count"] == 2
    assert report["eval"]["seed_sets_match"] is False
    assert "seed_counter_mismatch" in report["eval"]["comparability_blockers"]
    assert report["verdict"]["raw_survival_read"] == "improved"
    assert report["verdict"]["survival_read"] == "unknown"
    assert report["verdict"]["learning_claim"] == "blocked_by_non_comparable_eval"
    assert "seed_counter_mismatch" in report["verdict"]["claim_blockers"]
    assert report["verdict"]["coach_next_move"].startswith("Rerun a comparable panel first")


def test_mixed_caps_block_improved_learning_claim():
    report = _report(
        [
            _row("iteration_0", 1, 10, cap=50),
            _row("iteration_0", 2, 12, cap=50),
            _row("iteration_10", 1, 35, cap=200),
            _row("iteration_10", 2, 37, cap=200),
        ]
    )

    assert "mixed_caps" in report["eval"]["comparability_blockers"]
    assert report["verdict"]["raw_survival_read"] == "improved"
    assert report["verdict"]["survival_read"] == "unknown"
    assert "mixed caps" in _warning_text(report)


def test_failed_and_non_strict_rows_do_not_drive_primary_survival():
    report = _report(
        [
            _row("iteration_0", 1, 10),
            _row("iteration_0", 2, 10),
            _row("iteration_10", 1, 10),
            _row("iteration_10", 2, 100, ok="false", strict_load="false"),
        ]
    )

    latest = report["eval"]["checkpoints"][1]
    assert latest["row_count"] == 2
    assert latest["trusted_row_count"] == 1
    assert latest["mean_steps"] == 10
    assert "failed_eval_rows" in report["eval"]["comparability_blockers"]
    assert "non_strict_or_missing_checkpoint_load" in report["eval"]["comparability_blockers"]
    assert report["verdict"]["latest_mean_steps"] == 10
    assert report["verdict"]["survival_read"] == "unknown"


def test_matching_detailed_seeds_use_paired_first_latest_deltas():
    report = _report(
        [
            _row("iteration_0", 1, 10),
            _row("iteration_0", 2, 20),
            _row("iteration_10", 1, 20),
            _row("iteration_10", 2, 32),
        ]
    )

    paired = report["eval"]["paired_first_latest_survival_delta"]
    assert paired["same_seed_set"] is True
    assert paired["mean_delta_steps"] == 11
    assert paired["seed_deltas"] == [
        {"seed": 1, "first_steps": 10.0, "latest_steps": 20.0, "delta_steps": 10.0},
        {"seed": 2, "first_steps": 20.0, "latest_steps": 32.0, "delta_steps": 12.0},
    ]
    assert report["verdict"]["delta_basis"] == "paired same-seed"
    assert report["verdict"]["latest_minus_first_steps"] == 11
    assert report["verdict"]["survival_read"] == "improved"
    assert report["verdict"]["learning_claim"] == "incomplete_due_to_claim_blockers"
    assert report["verdict"]["claim_blockers"] == ["missing_baseline_panel"]


def test_per_row_action_collapse_is_not_hidden_by_aggregate_mix():
    report = _report(
        [
            _row("iteration_0", 1, 10, actions={"left": 100}),
            _row("iteration_0", 2, 10, actions={"right": 100}),
            _row("iteration_10", 1, 10),
            _row("iteration_10", 2, 10),
        ]
    )

    first = report["eval"]["checkpoints"][0]
    assert first["top_action_fraction"] == 0.5
    assert first["max_row_top_action_fraction"] == 1.0
    assert first["row_action_collapsed_count"] == 2
    assert report["verdict"]["action_collapse_seen"] is True
    assert "action_collapse" in report["verdict"]["claim_blockers"]
    assert "Per-row action collapse" in _warning_text(report)


def test_missing_death_cause_stays_visible():
    report = _report(
        [
            _row("iteration_0", 1, 10, death_cause=None),
            _row("iteration_0", 2, 10, death_cause=None),
            _row("iteration_10", 1, 10, death_cause=None),
            _row("iteration_10", 2, 10, death_cause=None),
        ]
    )

    warnings = _warning_text(report)
    assert "real death cause is missing" in warnings
    assert "No death-cause fields" in warnings
    assert "missing_death_cause" in report["verdict"]["claim_blockers"]
    assert "death cause is missing" in report["verdict"]["plain_read"]


def test_named_death_cause_clears_missing_death_cause_blocker():
    row = _row("iteration_0", 1, 10, death_cause=1)
    row["death_cause_name"] = "wall"
    row["death_hit_owner"] = -1
    latest = _row("iteration_10", 1, 12, death_cause=3)
    latest["death_cause_name"] = "opponent_trail"
    latest["death_hit_owner"] = 0
    report = _report([row, latest])

    checkpoints = report["eval"]["checkpoints"]
    assert checkpoints[0]["death_cause_counts"] == {"wall": 1}
    assert checkpoints[1]["death_cause_counts"] == {"opponent_trail:hit_owner=0": 1}
    assert "missing_death_cause" not in report["verdict"]["claim_blockers"]


def test_local_replay_recovers_missing_manifest_death_cause(tmp_path):
    manifest_path = tmp_path / "s102_fixed64_manifest.json"
    artifact_path = (
        tmp_path
        / "s102_fixed64"
        / "iteration_0_steps1024_seed1297473639"
        / "curvytron_visual_survival_eval_iteration_0_steps1024_seed1297473639.json"
    )
    sidecar_path = artifact_path.with_suffix(".env_steps.jsonl")
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text(
        json.dumps(
            {
                "config": {
                    "seed": 1297473639,
                    "source_max_steps": 1024,
                    "opponent_policy_kind": "fixed_straight",
                },
                "episode": {"seed": 1297473639, "actions": [0] * 33},
            }
        ),
        encoding="utf-8",
    )
    sidecar_path.write_text(
        json.dumps({"step_index": 32, "trace_hash": "30c77a4dedae3f35"}) + "\n",
        encoding="utf-8",
    )
    row = _row("iteration_0", 1297473639, 32, death_cause=None)
    row["artifact_ref"] = (
        "training/lightzero-curvytron-visual-survival/run/attempts/attempt/"
        "eval/s102_fixed64/iteration_0_steps1024_seed1297473639/"
        "curvytron_visual_survival_eval_iteration_0_steps1024_seed1297473639.json"
    )

    report = build_inspector_report(
        {
            "ok": True,
            "eval_id": "s102_fixed64",
            "config": {"run_id": "run-fixture", "opponent_policy_kind": "fixed_straight"},
            "table": [row],
        },
        eval_manifest_path=manifest_path,
        train_summary={
            "run_id": "run-fixture",
            "current_policy_self_play": False,
            "opponent_training_relation": "learner_vs_fixed_straight",
            "learning_proof": False,
            "debug_fidelity_only": True,
        },
        created_at="2026-05-11T00:00:00+00:00",
    )

    replay = report["replay_inspection"]
    assert replay["status"] == "ok"
    assert replay["death_cause_counts"] == {"wall": 1}
    assert replay["trace_hash_match_count"] == 1
    assert "missing_death_cause" not in report["verdict"]["claim_blockers"]
    assert "Local replay found death causes: wall:1" in report["verdict"]["plain_read"]


def test_fixed_and_frozen_opponents_are_labeled():
    fixed = _report(
        [
            _row("iteration_0", 1, 10),
            _row("iteration_10", 1, 10),
        ],
        opponent_policy_kind="fixed_random",
    )
    frozen = _report(
        [
            _row("iteration_0", 1, 10),
            _row("iteration_10", 1, 10),
        ],
        opponent_policy_kind="frozen_lightzero_checkpoint",
    )

    assert fixed["run"]["opponent_comparison"] == "fixed_opponent"
    assert "fixed-opponent eval" in _warning_text(fixed)
    assert "fixed_opponent" in fixed["verdict"]["claim_blockers"]
    assert frozen["run"]["opponent_comparison"] == "frozen_opponent"
    assert "frozen-opponent eval" in _warning_text(frozen)
    assert "frozen_opponent" in frozen["verdict"]["claim_blockers"]


def test_heavy_capped_rows_warn_and_block_learning_claim():
    report = _report(
        [
            _row("iteration_0", 1, 10),
            _row("iteration_0", 2, 10),
            _row("iteration_10", 1, 100, terminal_reason="cap"),
            _row("iteration_10", 2, 100, terminal_reason="cap"),
        ]
    )

    latest = report["eval"]["checkpoints"][1]
    assert latest["capped_fraction"] == 1.0
    assert "heavy_capped_rows" in report["eval"]["comparability_blockers"]
    assert report["verdict"]["raw_survival_read"] == "improved"
    assert report["verdict"]["survival_read"] == "unknown"
    assert "heavy_capped_rows" in report["verdict"]["claim_blockers"]
    assert "capped/right-censored" in _warning_text(report)


def test_improved_fixed_opponent_without_train_summary_is_not_learning_claim():
    report = _report(
        [
            _row("iteration_0", 1, 10),
            _row("iteration_0", 2, 12),
            _row("iteration_10", 1, 35),
            _row("iteration_10", 2, 37),
        ],
        opponent_policy_kind="fixed_random",
        include_train_summary=False,
    )

    assert report["verdict"]["raw_survival_read"] == "improved"
    assert report["verdict"]["survival_read"] == "improved"
    assert report["verdict"]["learning_claim"] == "incomplete_due_to_claim_blockers"
    assert "missing_train_summary" in report["verdict"]["claim_blockers"]
    assert "fixed_opponent" in report["verdict"]["claim_blockers"]
    assert "missing_baseline_panel" in report["verdict"]["claim_blockers"]
    assert "narrow survival evidence" in report["verdict"]["coach_next_move"]


def test_improved_action_collapse_is_not_positive_learning_claim():
    report = _report(
        [
            _row("iteration_0", 1, 10),
            _row("iteration_0", 2, 12),
            _row("iteration_10", 1, 35, actions={"left": 100}),
            _row("iteration_10", 2, 37, actions={"left": 100}),
        ]
    )

    assert report["verdict"]["raw_survival_read"] == "improved"
    assert report["verdict"]["survival_read"] == "improved"
    assert report["verdict"]["learning_claim"] == "incomplete_due_to_claim_blockers"
    assert "action_collapse" in report["verdict"]["claim_blockers"]
    assert "narrow survival evidence" in report["verdict"]["coach_next_move"]


def test_worse_survival_coach_says_stop_and_inspect_deaths():
    report = _report(
        [
            _row("iteration_0", 1, 30),
            _row("iteration_0", 2, 40),
            _row("iteration_10", 1, 12),
            _row("iteration_10", 2, 18),
        ]
    )

    assert report["verdict"]["survival_read"] == "worse"
    assert report["verdict"]["learning_claim"] == "survival_got_worse_in_this_eval"
    coach_next_move = report["verdict"]["coach_next_move"]
    assert "Stop or avoid scaling" in coach_next_move
    assert "before more training" in coach_next_move


def test_flat_survival_coach_says_do_not_scale_and_compare_baselines():
    report = _report(
        [
            _row("iteration_0", 1, 20),
            _row("iteration_0", 2, 24),
            _row("iteration_10", 1, 21),
            _row("iteration_10", 2, 23),
        ]
    )

    assert report["verdict"]["survival_read"] == "flat"
    assert report["verdict"]["learning_claim"] == "no_clear_survival_lift"
    coach_next_move = report["verdict"]["coach_next_move"]
    assert "Do not scale from this run" in coach_next_move
    assert "inspect the shortest deaths" in coach_next_move
    assert "compare baselines" in coach_next_move


def test_train_summary_learning_flags_block_positive_learning_claim():
    report = _report(
        [
            _row("iteration_0", 1, 10),
            _row("iteration_0", 2, 12),
            _row("iteration_10", 1, 35),
            _row("iteration_10", 2, 37),
        ],
        train_summary_overrides={
            "current_policy_self_play": False,
            "debug_fidelity_only": True,
            "learning_proof": False,
        },
    )

    assert report["verdict"]["survival_read"] == "improved"
    assert report["verdict"]["learning_claim"] == "incomplete_due_to_claim_blockers"
    assert "train_summary_current_policy_self_play_false" in report["verdict"]["claim_blockers"]
    assert "debug_fidelity_only_true" in report["verdict"]["claim_blockers"]
    assert "learning_proof_false" in report["verdict"]["claim_blockers"]
