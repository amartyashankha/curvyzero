import importlib.util
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "compare_2p_raw_visual_observation",
    REPO_ROOT / "scripts" / "compare_2p_raw_visual_observation.py",
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("could not load compare_2p_raw_visual_observation script")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
run_comparison = _MODULE.run_comparison


def test_compare_2p_raw_visual_observation_matches_short_rollout():
    report = run_comparison(max_steps=4, include_original_js_reset=False)

    assert report["source_arena_size"] == 88
    assert report["raw_observation_shape"] == [1, 64, 64]
    assert report["browser_canvas_pixel_fidelity"] is False
    assert report["match"] is True


def test_compare_2p_raw_visual_observation_matches_original_js_reset_when_available():
    if shutil.which("node") is None:
        return

    report = run_comparison(max_steps=0, include_original_js_reset=True)
    js_check = report["original_js_reset_check"]

    assert js_check["available"] is True
    assert js_check["loaded_original_source"] is True
    assert js_check["source_arena_size"] == 88
    assert js_check["match"] is True
