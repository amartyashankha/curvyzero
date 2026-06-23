from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "rematerialize_curvytron_checkpoint_refs.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("rematerialize_curvytron_checkpoint_refs", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _checkpoint_ref(iteration: int = 10000) -> str:
    return (
        "training/lightzero-curvytron-visual-survival/run-a/attempts/try-a/"
        f"train/lightzero_exp/ckpt/iteration_{iteration}.pth.tar"
    )


def test_rematerialize_refs_file_dry_run(tmp_path):
    script = _load_script()
    refs_file = tmp_path / "refs.txt"
    refs_file.write_text(_checkpoint_ref() + "\n", encoding="utf-8")

    args = script.parse_args(
        [
            "--refs-file",
            str(refs_file),
            "--allow-non-v2-source",
            "--dry-run",
        ]
    )
    report = script.rematerialize(args)

    assert report["dry_run"] is True
    assert report["ref_count"] == 1
    assert report["records"][0]["status"] == "dry_run"
    assert report["records"][0]["remote_path"] == "/" + _checkpoint_ref()


def test_rematerialize_rejects_non_v2_target(tmp_path):
    script = _load_script()
    refs_file = tmp_path / "refs.txt"
    refs_file.write_text(_checkpoint_ref() + "\n", encoding="utf-8")

    args = script.parse_args(
        ["--refs-file", str(refs_file), "--target-volume", "curvyzero-runs"]
    )
    try:
        script.rematerialize(args)
    except ValueError as exc:
        assert "all-v2" in str(exc)
    else:
        raise AssertionError("non-v2 target volume was accepted")


def test_rematerialize_rejects_mutable_ref(tmp_path):
    script = _load_script()
    refs_file = tmp_path / "refs.txt"
    refs_file.write_text(_checkpoint_ref(10000).replace("iteration_10000", "ckpt_best") + "\n")

    args = script.parse_args(
        ["--refs-file", str(refs_file), "--allow-non-v2-source", "--dry-run"]
    )
    try:
        script.rematerialize(args)
    except ValueError as exc:
        assert "mutable" in str(exc)
    else:
        raise AssertionError("mutable checkpoint ref was accepted")
