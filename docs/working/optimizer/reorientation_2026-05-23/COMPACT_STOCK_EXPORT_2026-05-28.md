# Compact Stock Export Gate

Date: 2026-05-28

Status: local loader-facing contract, strict-load verifier harness,
stock-model verifier smoke, tournament policy-loader smoke, evidence bundle
attachment policy, and one-game/GIF/eval smoke implemented.
Role: derived stock-shaped export artifact for compact-owned trainer checkpoints.
Authority: current local truth for the stock-export slice; promotion still
requires explicit compact Coach speed-row evidence after the later lifecycle
closeout.

## What Landed

Added `src/curvyzero/training/compact_stock_checkpoint_export.py`.

The compact-native checkpoint remains the resume artifact:

```text
CompactTrainerCheckpointV1
-> model state
-> optimizer state
-> compact replay-store state
-> loop runtime state
-> counters/config/metrics/lineage
```

The new export is a separate derived eval-only payload:

```python
{
    "model": checkpoint.model_state_dict,
    "metadata": {...},
    "compact_trainer_checkpoint_metadata": {...},
}
```

It writes the stock sidecar path expected by the existing frozen-opponent and
tournament tooling:

```text
<checkpoint filename>.metadata.json
```

The sidecar schema is:

```text
curvyzero_checkpoint_policy_metadata/v0
```

## Guardrails

The export validates and preserves:

- stock state-dict discovery under `model`;
- current policy-observation metadata:
  `browser_lines`, `simple_symbols`, `cpu_oracle`,
  `curvyzero_policy_observation_surface/v1`,
  controlled-player perspective schema;
- full nested `observation_contract`, including stack shape `[4, 64, 64]` and
  single-frame shape `[1, 64, 64]`;
- model/runtime fields needed by tournament loader reconstruction:
  `model_env_variant=source_state_fixed_opponent`,
  `model_reward_variant=survival_plus_bonus_no_outcome`,
  `env_variant`, `reward_variant`, `decision_source_frames`,
  `source_physics_step_ms`, `decision_ms`, `source_max_steps`,
  `source_max_steps_semantics`, and `learner_seat_mode`;
- compact source checkpoint id, trainer id, policy/model refs, policy source,
  and non-claims.

It deliberately does not expose optimizer/replay state in the stock-shaped
payload.

The same module now also exposes
`verify_compact_stock_export_model_contract_v1(...)`. It loads the export,
requires the adjacent sidecar by default, validates policy-observation
metadata, selects the state dict under `model`, infers support metadata, and
then calls the existing stock LightZero strict-loader path. The report schema is:

```text
curvyzero_compact_stock_model_contract_verification/v1
```

The verifier can report `stock_model_contract_verified=true` only if the stock
loader actually strict-loads the model. Missing sidecars and strict-load
exceptions return `ok=false` reports unless the caller explicitly requests
raising.

## Local Strict-Load Smoke

Durable local smoke:

```text
artifacts/local/curvytron_compact_stock_export_verifier_results/optimizer-compact-stock-export-verifier-smoke-20260528/iteration_0.pth.tar
artifacts/local/curvytron_compact_stock_export_verifier_results/optimizer-compact-stock-export-verifier-smoke-20260528/iteration_0.pth.tar.metadata.json
artifacts/local/curvytron_compact_stock_export_verifier_results/optimizer-compact-stock-export-verifier-smoke-20260528/verification_report.json
```

Shape:

- compact checkpoint wrapper around a stock LightZero `MuZeroPolicy._model`;
- exported under `model`;
- current policy-observation sidecar;
- verifier used stock loader with `num_simulations=1`, `batch_size=4`,
  CPU-only.

Result:

```text
ok=true
state_key=model
strict_load=true
load_summary.candidate=as_is
stock_model_contract_verified=true
strict_stock_model_load_verified=true
stock_eval_tournament_loadable=false
checkpoint_inferred_model_support_config={support_scale: 300, reward_support_size: 601, value_support_size: 601}
```

Read: the export and verifier path can produce a true strict-load report when
the compact checkpoint contains a stock-shaped LightZero model. This is not yet
an eval/GIF/tournament smoke and not a proof that every compact learner model is
stock-compatible.

## Local Tournament Loader Smoke

Durable local loader report:

```text
artifacts/local/curvytron_compact_stock_export_verifier_results/optimizer-compact-stock-export-verifier-smoke-20260528/tournament_loader_report.json
```

Shape:

- same durable export as the strict-load smoke;
- relative artifact checkpoint ref, no Modal volume mutation;
- tournament `_load_policy_from_checkpoint(...)`;
- explicit `checkpoint_state_key=model`;
- real stock LightZero policy construction through `_make_policy_and_env`;
- no game rollout and no GIF render.

Result:

```text
ok=true
schema_id=curvyzero_compact_stock_export_tournament_loader_smoke/v1
smoke_scope=local_tournament_policy_loader_real_stock_model_construction_no_game_or_gif
checkpoint_state_key=model
model_env_variant=source_state_fixed_opponent
model_reward_variant=survival_plus_bonus_no_outcome
surface.load_state_dict.ok=true
surface.load_state_dict.strict=true
surface.load_state_dict.candidate=as_is
surface.load_state_dict.missing_keys=[]
surface.load_state_dict.unexpected_keys=[]
```

Read: the tournament policy-loader path can consume the stock-shaped export,
select `model`, accept policy-observation metadata, propagate checkpoint runtime
settings into policy construction, and strict-load the stock-shaped model. This
closes the loader-construction slice, not full gameplay/GIF/eval.

## Evidence Bundle Attachment

Decision: verified evidence stays in a sibling bundle, not by mutating the base
export.

Bundle schema:

```text
curvyzero_compact_stock_export_evidence_bundle/v1
```

Durable local bundle smoke:

```text
artifacts/local/curvytron_compact_stock_export_evidence_bundle_results/optimizer-compact-stock-export-evidence-bundle-smoke-20260528/iteration_0.pth.tar
artifacts/local/curvytron_compact_stock_export_evidence_bundle_results/optimizer-compact-stock-export-evidence-bundle-smoke-20260528/iteration_0.pth.tar.metadata.json
artifacts/local/curvytron_compact_stock_export_evidence_bundle_results/optimizer-compact-stock-export-evidence-bundle-smoke-20260528/verification_report.json
artifacts/local/curvytron_compact_stock_export_evidence_bundle_results/optimizer-compact-stock-export-evidence-bundle-smoke-20260528/tournament_loader_report.json
artifacts/local/curvytron_compact_stock_export_evidence_bundle_results/optimizer-compact-stock-export-evidence-bundle-smoke-20260528/iteration_0.pth.tar.evidence.json
```

Result:

```text
ok=true
base_export_claims_mutated=false
stock_model_state_key=model
strict_stock_model_load_verified=true
tournament_loader_constructed=true
stock_eval_tournament_loadable_by_evidence=false
eval_gif_tournament_loadable_by_evidence=false
promotion_claim=false
training_speed_claim=false
optimizer_resume_claim=false
```

Read: the evidence bundle can attach strict-load and tournament-loader proof to
one exact export/report set by file hash. It now deliberately does not claim
full eval/GIF/tournament loadability without the separate current-chain
gameplay/GIF/eval contract. The base export still says
`stock_model_contract_verified=false`,
`stock_eval_tournament_loadable=false`, and `promotion_claim=false`.

## One-Game/GIF/Eval Smoke

Durable local smoke:

```text
artifacts/local/curvytron_compact_stock_export_gameplay_smoke_results/optimizer-compact-stock-export-one-game-gif-smoke-20260528/one_game_gif_smoke_report.json
```

Result:

```text
schema_id=curvyzero_compact_stock_export_one_game_gif_smoke/v1
ok=true
checkpoint_state_key=model
physical_steps=4
first joint action=[1,1]
gif_ref=tournaments/curvytron/optimizer-compact-stock-export-one-game-gif-smoke-20260528/battles/compact-stock-export-self-play-smoke/games/game-000000/game.gif
frames_ref=tournaments/curvytron/optimizer-compact-stock-export-one-game-gif-smoke-20260528/battles/compact-stock-export-self-play-smoke/games/game-000000/frames.npz
promotion_claim=false
training_speed_claim=false
calls_train_muzero=false
touches_live_runs=false
```

Read: the bundled stock-shaped compact export can drive the existing tournament
eval policy path and GIF renderer for a tiny local self-play smoke. The same
export was loaded under `model` for both seats, the source-state game loop ran 4
physical steps, the first joint action was `[1,1]`, and a 704x704 GIF with 5
frames was written. The report now also embeds a standalone visual-survival eval
core result: strict load ok, env reset ok, policy acted in the real eval env,
and `steps_survived=8`. This closes the local gameplay-side loader/eval smoke
for this exact bundle, not promotion or training speed.

## Non-Claims

The export currently records:

```text
eval_only_export=true
stock_payload_mapping=true
stock_state_dict_discovery_verified=true
stock_model_contract_verified=false
stock_model_contract_verification_required=true
stock_eval_tournament_loadable=false
stock_eval_tournament_load_status=strict_stock_model_load_not_run
stock_resume_claim=false
optimizer_resume_supported=false
calls_train_muzero=false
touches_live_runs=false
promotion_claim=false
training_speed_claim=false
```

That is the important honesty line: mapping discovery and sidecar metadata are
now local. Strict load and loader construction are attached evidence, not base
export mutation. A caller cannot flip the export into loadable status by
passing a boolean.

## Tests

Focused validation:

```text
uv run ruff check src/curvyzero/training/compact_coach_compatibility.py src/curvyzero/training/compact_owned_loop.py src/curvyzero/training/compact_owned_trainer.py src/curvyzero/training/compact_trainer_checkpoint.py src/curvyzero/training/compact_stock_checkpoint_export.py tests/test_compact_coach_compatibility.py tests/test_compact_owned_loop.py tests/test_compact_owned_trainer.py tests/test_compact_trainer_checkpoint.py tests/test_compact_stock_checkpoint_export.py

uv run pytest tests/test_compact_coach_compatibility.py tests/test_compact_owned_loop.py tests/test_compact_owned_trainer.py tests/test_compact_trainer_checkpoint.py tests/test_compact_stock_checkpoint_export.py -q
```

Result:

```text
ruff passed
33 passed, 2 warnings
```

Coverage added:

- compact export payload exposes state under the stock `model` key;
- saved `.pth.tar` can be reloaded and discovered by the existing eval helper;
- adjacent sidecar is accepted by
  `lightzero_checkpoint_opponent_provider.require_checkpoint_policy_observation_metadata`;
- tournament payload helpers recover model contract and runtime settings from
  export metadata;
- missing policy-observation metadata or runtime/model metadata fails closed;
- export remains eval-only and does not claim stock resume or stock
  eval/tournament loadability;
- `CompactOwnedTrainerV1.save_stock_eval_export(...)` writes the derived export
  without changing the compact-native checkpoint contract.
- verifier reports strict-load success when the stock loader succeeds;
- verifier returns fail-closed reports for strict-load errors and missing
  sidecars.
- tournament `_load_policy_from_checkpoint(...)` consumes the export under
  `model`, propagates runtime/model metadata into policy construction, and
  records the loaded policy metadata.
- evidence bundle builder/validator records file hashes and rejects stale
  hashes, failed verifier reports, loader runtime drift, and promotion claims.

## Subagent Inputs

Ada confirmed the stock loader contract:

- use a normal `torch.save` mapping, not the compact dataclass;
- prefer `model` as the state key;
- write `<checkpoint>.metadata.json`;
- include full policy-observation contract and runtime/model variant fields.

Tarski confirmed the route decision:

- choose stock-compatible export first, not compact eval adapter;
- keep `CompactTrainerCheckpointV1` compact-native;
- promotion still needs strict stock model-load evidence because compact learner
  models are not guaranteed to be stock LightZero models.

Carson confirmed the tournament loader path:

- `_load_policy_from_checkpoint(...)` is the smallest loader smoke target;
- use a relative ref or mounted volume ref, not an absolute path;
- `model` is the exact compact stock-export state key;
- sidecar lookup is strict about policy-observation metadata.

Huygens confirmed the attachment policy:

- use a separate evidence bundle, not checkpoint mutation;
- bundle export, sidecar, verifier report, and loader report by hash;
- allow evidence-scoped loadability while keeping base export non-loadable.

## Next Gate

The compact Coach-compatibility report has been refreshed with the now-available
stock export evidence:

```text
checkpoint/export ref
-> policy sidecar ref
-> strict verifier report ref
-> tournament-loader report ref
-> evidence bundle ref
-> one-game/GIF smoke report ref
-> reward/RND contract ref
-> historical missing lifecycle gates remain explicit
```

Durable refresh:

```text
artifacts/local/curvytron_compact_coach_compatibility_results/optimizer-compact-coach-compatibility-after-reward-rnd-contract-20260528/compatibility_report.json
```

The selected next gate is `death_terminal_contract`.

The local verifier report can say `stock_model_contract_verified=true` for the
exact export it checked. The export payload itself still defaults to
`stock_model_contract_verified=false`; the evidence bundle is the attachment
contract. Even with strict load, loader construction, and a tiny gameplay/GIF
smoke, this does not prove Coach training speed, stock resume, policy-refresh
integration, full tournament/rating behavior, or live-run safety.
