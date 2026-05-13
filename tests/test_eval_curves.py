import json

from curvyzero.analysis.eval_curves import (
    CurvePoint,
    EvalCurve,
    METRIC_SCHEMA_BY_NAME,
    build_curves,
    load_curves,
    score_curve,
    score_curves,
    score_curves_multi,
    summarize_curve_metrics,
)


def _curve(run_id, values, *, collapsed=False):
    return EvalCurve(
        run_id=run_id,
        points=tuple(
            CurvePoint(
                iteration=i * 100,
                metrics={
                    "mean_survival": value,
                    "top_action_fraction": 0.99 if collapsed and i == len(values) - 1 else 0.5,
                },
            )
            for i, value in enumerate(values)
        ),
    )


def test_scores_increasing_curve():
    score = score_curve(_curve("increasing", [8, 10, 13, 18]))

    assert score["first"] == 8
    assert score["latest"] == 18
    assert score["best"] == 18
    assert score["delta"] == 10
    assert score["best_delta"] == 10
    assert score["early_slope"] > 0
    assert score["late_slope"] > 0
    assert score["peak_signal"] is True
    assert score["late_bloom"] is True
    assert score["flat"] is False
    assert score["peak_then_crash"] is False


def test_scores_flat_curve():
    score = score_curve(_curve("flat", [8.0, 8.2, 7.9, 8.1]), flat_epsilon=0.5)

    assert score["flat"] is True
    assert score["best_delta"] == 0.1999999999999993
    assert score["peak_then_crash"] is False


def test_scores_peak_then_crash_curve():
    score = score_curve(_curve("crash", [8, 14, 22, 11]), crash_drop=5)

    assert score["best"] == 22
    assert score["latest"] == 11
    assert score["peak_signal"] is True
    assert score["late_bloom"] is True
    assert score["peak_then_crash"] is True
    assert score["late_slope"] < 0


def test_scores_late_blooming_curve():
    score = score_curve(_curve("late", [8, 8, 9, 18]))

    assert score["early_slope"] == 0
    assert score["late_slope"] > 0
    assert score["best_delta"] == 10
    assert score["late_bloom"] is True
    assert score["flat"] is False


def test_collapse_flag_from_latest_top_action_fraction():
    score = score_curve(_curve("collapsed", [8, 9, 10], collapsed=True))

    assert score["collapsed"] is True


def test_builds_curves_from_eval_summary_rows_with_manifest_axes(tmp_path):
    manifest = {
        "rows": [
            {
                "run_id": "run-a",
                "attempt_id": "attempt-a",
                "label": "fixed-sparse",
                "command": [
                    "--reward-variant",
                    "sparse_outcome",
                    "--source-state-trail-render-mode",
                    "body_circles_fast",
                    "--collector-env-num",
                    "32",
                ],
            }
        ]
    }
    rows = [
        {
            "short_name": "run-a",
            "first_iter": 0,
            "first_mean": 8,
            "best_iter": 1000,
            "best_mean": 18,
            "latest_iter": 2000,
            "latest_mean": 14,
            "latest_top_fraction": 0.5,
            "any_collapsed": False,
        }
    ]
    manifest_path = tmp_path / "manifest.json"
    rows_path = tmp_path / "rows.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    rows_path.write_text(json.dumps(rows), encoding="utf-8")

    curves = load_curves(rows_path, manifest_path=manifest_path)
    scores = score_curves(curves)

    assert len(curves) == 1
    assert curves[0].run_id == "run-a"
    assert curves[0].axes["label"] == "fixed-sparse"
    assert curves[0].axes["reward_variant"] == "sparse_outcome"
    assert curves[0].axes["collector_env_num"] == 32
    assert scores[0]["best"] == 18
    assert scores[0]["peak_then_crash"] is True


def test_builds_canonical_curve_payload():
    curves = build_curves(
        {
            "curves": [
                {
                    "run_id": "run-b",
                    "axes": {"opponent": "fixed"},
                    "points": [
                        {"iteration": 0, "metrics": {"mean_survival": 5}},
                        {"checkpoint": "iteration_10", "metrics": {"mean_survival": 7}},
                    ],
                }
            ]
        }
    )

    assert curves[0].axes == {"opponent": "fixed"}
    assert score_curve(curves[0])["latest"] == 7


def test_scores_multiple_metrics_without_single_truth_metric():
    curve = EvalCurve(
        run_id="multi",
        points=(
            CurvePoint(iteration=0, metrics={"win_rate": 0.1, "mean_survival": 8, "mean_reward": 0.0}),
            CurvePoint(iteration=100, metrics={"win_rate": 0.4, "mean_survival": 9, "mean_reward": 0.2}),
            CurvePoint(iteration=200, metrics={"win_rate": 0.2, "mean_survival": 16, "mean_reward": 0.5}),
        ),
    )

    scores = score_curves_multi(
        [curve],
        metrics=("win_rate", "mean_survival", "mean_reward"),
        signal_delta=0.1,
    )
    by_metric = {score["metric"]: score for score in scores}

    assert set(by_metric) == {"win_rate", "mean_survival", "mean_reward"}
    assert by_metric["win_rate"]["best_delta"] == 0.30000000000000004
    assert by_metric["win_rate"]["peak_signal"] is True
    assert by_metric["mean_survival"]["late_bloom"] is True
    assert by_metric["mean_reward"]["latest"] == 0.5


def test_eval_summary_rows_can_carry_win_rate_survival_and_reward(tmp_path):
    rows = [
        {
            "short_name": "run-metrics",
            "first_iter": 0,
            "first_win_rate": 0.125,
            "first_mean_survival": 8,
            "first_mean_reward": -0.1,
            "best_iter": 1000,
            "best_win_rate": 0.5,
            "best_mean_survival": 18,
            "best_mean_reward": 0.25,
            "latest_iter": 2000,
            "latest_win_rate": 0.25,
            "latest_mean_survival": 14,
            "latest_mean_reward": 0.1,
        }
    ]
    rows_path = tmp_path / "rows.json"
    rows_path.write_text(json.dumps(rows), encoding="utf-8")

    scores = score_curves_multi(
        load_curves(rows_path),
        metrics=("win_rate", "mean_survival", "mean_reward"),
        signal_delta=0.1,
    )
    by_metric = {score["metric"]: score for score in scores}

    assert by_metric["win_rate"]["best"] == 0.5
    assert by_metric["win_rate"]["peak_then_crash"] is True
    assert by_metric["mean_survival"]["peak_then_crash"] is True
    assert by_metric["mean_reward"]["best_delta"] == 0.35


def test_builds_curve_from_status_eval_checkpoints_rows():
    curves = build_curves(
        {
            "rows": [
                {
                    "short_name": "run-eval",
                    "attempt_id": "attempt-eval",
                    "eval_checkpoints": [
                        {
                            "checkpoint": "iteration_0",
                            "seeds": 8,
                            "mean_steps": 8.0,
                            "median_steps": 8.0,
                            "max_steps": 11,
                            "failure_count": 0,
                            "outcome_histogram": {"loss": 7, "win": 1},
                            "action_summary": {
                                "top_action": "0",
                                "top_action_fraction": 0.4,
                                "collapsed": False,
                            },
                        },
                        {
                            "checkpoint": "iteration_1000",
                            "seeds": 8,
                            "mean_steps": 16.0,
                            "median_steps": 15.0,
                            "max_steps": 30,
                            "failure_count": 0,
                            "outcome_histogram": {"win": 8},
                            "action_summary": {
                                "top_action": "1",
                                "top_action_fraction": 0.99,
                                "collapsed": True,
                            },
                        },
                    ],
                }
            ]
        }
    )

    scores = score_curves_multi(curves, metrics=("win_rate", "mean_survival"))
    by_metric = {score["metric"]: score for score in scores}

    assert curves[0].run_id == "run-eval"
    assert curves[0].attempt_id == "attempt-eval"
    assert by_metric["win_rate"]["first"] == 0.125
    assert by_metric["win_rate"]["latest"] == 1.0
    assert by_metric["mean_survival"]["best_delta"] == 8.0
    assert by_metric["mean_survival"]["collapsed"] is True


def test_metric_schema_and_summary_cover_survivaldiag_fields():
    curves = build_curves(
        {
            "rows": [
                {
                    "short_name": "run-survivaldiag",
                    "attempt_id": "attempt-survivaldiag",
                    "eval_checkpoints": [
                        {
                            "checkpoint": "iteration_0",
                            "seeds": 4,
                            "mean_steps": 10.0,
                            "median_steps": 9.0,
                            "max_steps": 20,
                            "mean_training_reward": 0.25,
                            "mean_bonus_pickup_count": 0.5,
                            "bonus_reward": 0.05,
                            "ok_count": 4,
                            "failure_count": 0,
                            "outcome_histogram": {"win": 1, "loss": 3},
                            "terminal_reason_histogram": {"normal_wall": 3, "cap": 1},
                            "action_histogram": {"0": 1, "1": 2, "2": 1},
                            "action_summary": {
                                "top_action": "1",
                                "top_action_fraction": 0.5,
                                "collapsed": False,
                            },
                        },
                        {
                            "checkpoint": "iteration_100",
                            "seeds": 4,
                            "mean_steps": 18.0,
                            "median_steps": 17.0,
                            "max_steps": 40,
                            "mean_training_reward": 0.75,
                            "mean_bonus_pickup_count": 1.25,
                            "bonus_reward": 0.15,
                            "ok_count": 3,
                            "failure_count": 1,
                            "outcome_histogram": {"win": 2, "loss": 1, "error": 1},
                            "terminal_reason_histogram": {"own_trail": 1, "normal_wall": 2, "cap": 1},
                            "action_histogram": {"0": 0, "1": 4, "2": 0},
                            "action_summary": {
                                "top_action": "1",
                                "top_action_fraction": 1.0,
                                "collapsed": True,
                            },
                        },
                    ],
                }
            ]
        }
    )

    curve = curves[0]
    scores = score_curves_multi(
        [curve],
        metrics=(
            "mean_survival",
            "mean_training_reward",
            "bonus_pickup_count",
            "wall_rate",
            "timeout_rate",
            "straight_rate",
            "action_entropy",
            "failure_rate",
        ),
    )
    by_metric = {score["metric"]: score for score in scores}
    summary = summarize_curve_metrics(curve)

    assert METRIC_SCHEMA_BY_NAME["bonus_pickup_count"].family == "bonus"
    assert by_metric["mean_survival"]["latest"] == 18.0
    assert by_metric["mean_training_reward"]["delta"] == 0.5
    assert by_metric["bonus_pickup_count"]["latest"] == 1.25
    assert by_metric["wall_rate"]["latest"] == 0.5
    assert by_metric["timeout_rate"]["latest"] == 0.25
    assert by_metric["straight_rate"]["latest"] == 1.0
    assert by_metric["action_entropy"]["latest"] == 0.0
    assert by_metric["failure_rate"]["latest"] == 0.25
    assert summary["families"]["survival"][0]["metric"] == "mean_survival"
    assert summary["latest_terminal_cause"] == "normal_wall"
    assert summary["latest_top_action"] == "1"
    assert summary["collapsed"] is True
    assert summary["eval_health"] == "has_failures"
