import argparse
import importlib.util
from pathlib import Path
import sys


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "benchmark_selfplay_parallel_bridge.py"
)
_SPEC = importlib.util.spec_from_file_location("benchmark_selfplay_parallel_bridge", _SCRIPT_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
parallel_benchmark = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = parallel_benchmark
_SPEC.loader.exec_module(parallel_benchmark)


def test_parallel_bridge_reports_ranked_bottleneck_summary():
    args = argparse.Namespace(
        batch=2,
        steps=2,
        warmup=0,
        workers=1,
        seed=0,
        hidden_dim=4,
        action_repeat=1,
        max_ticks=2000,
        modes=("serial",),
        format="json",
    )

    summary = parallel_benchmark.run(args)

    result = summary["results"][0]
    bottleneck = result["bottleneck_summary"]
    ranked = bottleneck["ranked_buckets"]

    assert bottleneck["schema"] == parallel_benchmark.BOTTLENECK_SUMMARY_SCHEMA
    assert ranked
    assert bottleneck["top_bucket"] == ranked[0]["bucket"]
    assert bottleneck["top_bucket"] in parallel_benchmark.TIMER_DEFINITIONS
    assert bottleneck["top_bucket_pct_serialized"] == ranked[0][
        "pct_serialized_shard_loop"
    ]
    assert bottleneck["env_step_pct_serialized"] == result[
        "bucket_pct_serialized_shard_loop"
    ]["env_step"]
    assert bottleneck["policy_batch_pct_serialized"] == result[
        "bucket_pct_serialized_shard_loop"
    ]["policy_batch"]
    assert bottleneck["estimated_sync_actor_step_p95_ms"] >= 0
    assert bottleneck["estimated_sync_actor_step_p99_ms"] >= (
        bottleneck["estimated_sync_actor_step_p95_ms"]
    )
    assert ranked == sorted(
        ranked,
        key=lambda item: (-item["pct_serialized_shard_loop"], item["bucket"]),
    )
