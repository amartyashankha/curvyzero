from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = ROOT / "scripts" / "build_curvytron_survivaldiag_manifest.py"
SUBMIT_SCRIPT = ROOT / "scripts" / "submit_curvytron_survivaldiag_manifest.py"


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _manifest_payload() -> dict:
    module = _load_script(BUILD_SCRIPT, "build_curvytron_survivaldiag_manifest_for_submit")
    args = module.parse_args(["--stdout-only"])
    return module.build_manifest(args)


def test_grouped_submitter_dry_run_selects_rows_and_preserves_two_call_shape(tmp_path):
    submit = _load_script(SUBMIT_SCRIPT, "submit_curvytron_survivaldiag_manifest_for_test")
    manifest = _manifest_payload()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    args = submit.parse_args([str(manifest_path), "--limit", "3"])
    loaded = submit._load_manifest(manifest_path)
    rows = submit._selected_rows(loaded, args)
    records = [
        submit._launch_row(
            row,
            app_name=loaded["guards"]["deployed_app_name"],
            modal_env=None,
            dry_run=True,
        )
        for row in rows
    ]

    assert len(records) == 3
    assert all(record["status"] == "dry_run" for record in records)
    assert all(
        record["app_name"] == "curvyzero-lightzero-curvytron-visual-survival-train-v2"
        for record in records
    )
    assert all(
        record["poller_function"]
        == "lightzero_curvytron_visual_survival_checkpoint_eval_poller"
        for record in records
    )
    assert all(
        record["train_function"] == "lightzero_curvytron_visual_survival_gpu_cpu40"
        for record in records
    )


def test_grouped_submitter_rejects_incomplete_train_kwargs():
    submit = _load_script(SUBMIT_SCRIPT, "submit_curvytron_survivaldiag_submit_missing")
    manifest = _manifest_payload()
    row = dict(manifest["rows"][0])
    row["train_kwargs"] = dict(row["train_kwargs"])
    row["train_kwargs"].pop("decision_ms")

    try:
        submit._launch_row(
            row,
            app_name=manifest["guards"]["deployed_app_name"],
            modal_env=None,
            dry_run=True,
        )
    except ValueError as exc:
        assert "train_kwargs missing required keys" in str(exc)
        assert "decision_ms" in str(exc)
    else:
        raise AssertionError("incomplete train kwargs were accepted")
