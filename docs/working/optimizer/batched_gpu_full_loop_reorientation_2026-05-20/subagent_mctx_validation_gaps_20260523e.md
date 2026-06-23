# MCTX Validation Gaps 2026-05-23e

Scope: validation sidecar only. No runtime code changes.

Small tests added now:

- `tests/test_mctx_compact_search_service.py`
  - zero-active roots keep profile-only labels and return empty action/policy arrays.
  - active roots with no legal actions fail before importing JAX/MCTX.
  - inactive roots fail in the current fixed-shape mode before importing JAX/MCTX.
- `tests/test_compact_search_replay_contract.py`
  - raw visit counts cannot put mass on illegal actions.
  - compact slab promotes MCTX profile telemetry into the shared summary fields.

Remaining gap:

- A true non-prefix active-root MCTX run still needs a real JAX/MCTX runtime because the service only skips imports for zero-active roots. The compact contract already checks non-prefix replay identity with fake search arrays. A service-level non-prefix MCTX test should wait until we either install MCTX in the local test image or add a tiny injectable backend seam. Do not add that seam casually; it is runtime surface area.

Validation intent:

- Keep the profile-only labels loud.
- Catch action/replay identity mistakes at the compact contract boundary.
- Catch bad masks before a GPU profile hides them.
- Catch summary regression so MCTX speed numbers stay visible in profile results.
