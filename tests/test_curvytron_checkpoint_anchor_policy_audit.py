from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_curvytron_checkpoint_anchor_policy.py"


HISTORICAL_BEST = (
    "training/lightzero-curvytron-visual-survival/"
    "curvy-r18fresh-survbonusout-blank20-wall5-rank1_70-rank1imm5-so10rep10-s134842423/"
    "attempts/try-r18fresh-survbonusout-blank20-wall5-rank1_70-rank1imm5-so10rep10-s134842423/"
    "train/lightzero_exp/ckpt/iteration_180000.pth.tar"
)
TOP4_RANK1 = (
    "training/lightzero-curvytron-visual-survival/"
    "curvy-r18fresh-sparse-blank10-wall10-rank4_10-rank3_15-rank2_20-rank1_30-rank1imm5-cl-da1a498fd8/"
    "attempts/try-r18fresh-sparse-blank10-wall10-rank4_10-rank3_15-rank2_20-rank1_30-rank1imm5-clea-6132ae9835/"
    "train/lightzero_exp/ckpt/iteration_40000.pth.tar"
)


def _load_script():
    spec = importlib.util.spec_from_file_location("audit_curvytron_checkpoint_anchor_policy", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_base_inputs(repo_root: Path, audit) -> None:
    top10 = repo_root / audit.TOP10_REFS_FILE
    top10.parent.mkdir(parents=True, exist_ok=True)
    top10.write_text(
        "\n".join(
            [
                "# Raw active top-10 checkpoint refs",
                "rank=1 rating=1632.6800339881258 checkpoint_id=ckpt-432",
                HISTORICAL_BEST,
                "",
                "rank=6 rating=1588.1493412491225 checkpoint_id=ckpt-555",
                HISTORICAL_BEST.replace("iteration_180000", "iteration_260000"),
                "",
            ]
        ),
        encoding="utf-8",
    )
    refs_file = repo_root / audit.STATIC_TOP4NZ_REFS_FILE
    refs_file.parent.mkdir(parents=True, exist_ok=True)
    refs_file.write_text(
        "\n".join(
            [
                TOP4_RANK1,
                TOP4_RANK1.replace("iteration_40000", "iteration_160000"),
                TOP4_RANK1.replace("iteration_40000", "iteration_120000"),
                TOP4_RANK1.replace("iteration_40000", "iteration_140000"),
                "",
            ]
        ),
        encoding="utf-8",
    )
    _write_json(
        repo_root / audit.STATIC_TOP4NZ_MODAL_AUDIT,
        {
            "ok": True,
            "ref_count": 4,
            "missing_ref_count": 0,
            "modal_parent_error_count": 0,
        },
    )


def _write_manifests(repo_root: Path, audit, seed_ref: str) -> None:
    for spec in audit._default_manifest_specs():
        rows = [
            {
                "row_id": f"r{index:03d}",
                "initial_policy_checkpoint_ref": seed_ref,
            }
            for index in range(1, 19)
        ]
        _write_json(repo_root / spec.manifest_relpath, {"schema_id": "test", "rows": rows})


def test_anchor_audit_warns_when_repaired_manifests_use_top4nz_seed(tmp_path):
    audit = _load_script()
    _write_base_inputs(tmp_path, audit)
    _write_manifests(tmp_path, audit, TOP4_RANK1)

    report = audit.build_report(audit.parse_args(["--repo-root", str(tmp_path)]))

    assert report["ok"] is True
    assert report["historical_best_seed"]["checkpoint_ref"] == HISTORICAL_BEST
    assert report["static_top4nz_modal_audit"]["ok"] is True
    assert report["manifest_seed_summary"]["top4nz_seed_manifest_count"] == 10
    assert report["manifest_seed_summary"]["historical_best_seed_manifest_count"] == 0
    assert any("do not use the historical" in warning["message"] for warning in report["warnings"])


def test_anchor_audit_can_require_historical_best_seed(tmp_path):
    audit = _load_script()
    _write_base_inputs(tmp_path, audit)
    _write_manifests(tmp_path, audit, TOP4_RANK1)

    report = audit.build_report(
        audit.parse_args(["--repo-root", str(tmp_path), "--require-best-known-seed"])
    )

    assert report["ok"] is False
    assert any("violate --require-best-known-seed" in error["message"] for error in report["errors"])


def test_anchor_audit_passes_when_manifests_use_historical_best_seed(tmp_path):
    audit = _load_script()
    _write_base_inputs(tmp_path, audit)
    _write_manifests(tmp_path, audit, HISTORICAL_BEST)

    report = audit.build_report(
        audit.parse_args(["--repo-root", str(tmp_path), "--require-best-known-seed"])
    )

    assert report["ok"] is True
    assert report["manifest_seed_summary"]["historical_best_seed_manifest_count"] == 10
    assert report["manifest_seed_summary"]["top4nz_seed_manifest_count"] == 0
