# Non-Search Materialization Audit - 2026-05-23

Read-only critique pass. I did not edit code or touch live runs.

Goal: find the places that will dominate after search gets faster because they still turn compact CurvyTron data into Python objects, NumPy copies, full observation rows, or CPU/GPU sync points.

Notation:

- `B` = physical env rows.
- `P` = players, currently usually `2`.
- `R = B * P` policy roots.
- One float32 stack `[4,64,64]` = `65,536 bytes` = `64 KiB/root`.
- One uint8 stack `[4,64,64]` = `16 KiB/root`.
- One float32 latest frame `[1,64,64]` = `16 KiB/root`.
- At `B=512, P=2, R=1024`: float32 roots are about `64 MiB`; uint8 roots are about `16 MiB`; target rows with `observation + next_observation` are about `128 MiB`.

## Ranked Boundaries

| Rank | Boundary | Current materialization | Scale | Can it stay compact/device-resident? | Lowest-risk canary |
| ---: | --- | --- | --- | --- | --- |
| 1 | Scalar LightZero timestep split and ready-observation dicts | `materialize_*_timestep` flattens stacks, builds `MockBaseEnvTimestep`, then `_ready_obs_by_env_id` and `_split_timestep_by_env_id` copy one `[4,64,64]` observation per scalar env into Python dicts and per-env timestep objects. See `source_state_batched_observation_mock_collector.py`. | `64 KiB/root` float32 copied at least once; more if both ready obs and split timesteps are built. At `R=1024`, this is about `64 MiB` per full scalarization pass plus Python object fanout. | Yes for profile/optimizer lane: keep one batched array plus env ids. For stock LightZero this is compatibility glue, so treat it as a boundary to bypass, not mutate silently. | Add a same-denominator profile row with scalar timestep materialization off versus on, requiring identical selected actions and compact replay proof. Fail if `ready_obs_count > 0` in compact hot path. |
| 2 | Target-row materialization | `build_source_state_multiplayer_target_rows_v0`, `build_compact_target_rows_from_search_arrays_v0`, and `materialize_compact_target_rows_from_index_rows_v1` build Python row dicts, copy `observation`, copy `next_observation`, then stack them. See `multiplayer_source_state_target_rows.py` and `compact_policy_row_bridge.py`. | `128 KiB/target row` for float32 `observation + next_observation`; about `128 MiB` for `1024` rows, before metadata/list overhead. | Mostly yes. Collection should store `CompactReplayIndexRowsV1` plus references into a replay array/ring. Materialize only at sampler or validation edge. | Two-record compact replay canary: search result action must drive the next env step; index rows must rebuild the same target rows only when explicitly requested; metadata must say `observation_materialized=False` during collection. |
| 3 | Replay recorder/chunk build | `SourceStateMultiplayerTrainerReplayRecorder.record` copies every trainer step array; `build_chunk` stacks over time, including full `observation` and full `final_observation`. See `multiplayer_source_state_trainer_replay.py`. | Per recorded step stores full `observation` plus full `final_observation`: about `128 MiB` at `R=1024`, even when terminal rows are sparse. Over `T` steps this grows as `T * B * P * stack`. | Yes in a new compact replay owner. Store frame/stack ring plus row metadata and terminal final refs. Full chunk build should be validation/export only. | Replay ring canary with normal, terminal, autoreset, and no-death rows. Sampled target rows must match existing chunk path exactly for selected rows. Fail if final observation comes from reset frame. |
| 4 | RND buffer and estimate path | `collect_data` converts `obs_segment` to NumPy float32, slices latest frames, then appends one cloned Torch tensor per frame into `train_obs`. `estimate` extracts latest frames to NumPy, moves to Torch, reads MSE back to CPU, and optionally writes CPU metrics. See `exploration_bonus.py`. | Latest frame is `16 KiB/root` float32. A `100,000` frame buffer is about `1.6 GiB` raw tensor data plus Python list/tensor-object overhead. Per-estimate D2H is small by bytes but synchronizes. | Yes. Use a preallocated latest-frame ring tensor or compact NumPy slab. Keep metrics sampled or amortized. RND can be independent from search if it consumes the same compact latest-frame contract. | RND meter canary: latest frame checksum matches policy stack, `train_cnt_rnd > 0`, target hash unchanged, predictor hash changes, `rnd_meter_v0` does not change target rewards. Add a buffer-shape check that no per-frame Python tensor list is used in the hot path. |
| 5 | Root-batch construction with copied observations | `build_compact_root_batch_v1(copy_observation=True)` reshapes `[B,P,4,64,64]` into `[R,4,64,64]` and copies root observations and final observations by default. See `compact_policy_row_bridge.py`. | `64 KiB/root` float32 or `16 KiB/root` uint8. At `R=1024`, `64 MiB` float32 or `16 MiB` uint8 per root-batch copy. Final observations can double this. | Yes. For resident/device lanes, root batch should carry a view or a handle plus ids/masks. Copying is fine for validation rows only. | Same root batch canary with `copy_observation=False`: active root order, legal masks, env_row/player/policy_env_id, selected actions, and replay proof must match copied mode. |
| 6 | Per-root policy record objects | `build_policy_row_records_from_compact_search_v0` allocates one `PolicyRowRecordV0` per active root and copies action masks/policy targets. | Small by bytes but high Python allocation/control overhead at large `R` and long games. | Yes. Prefer compact arrays or `CompactReplayIndexRowsV1`; object records should be legacy/parity edges. | Policy-record-free canary: compact target rows equal object target rows for mixed active roots, non-prefix roots, terminal rows, and swapped-player poison tests. |
| 7 | Stock env-manager conversion | `BatchedLightZeroStockEnvManagerAdapter.step` loops over scalar timesteps, extracts scalar reward/done via NumPy, then creates stock `BaseEnvTimestep`-like objects. | Small bytes, high Python loop and object allocation cost. | For stock LightZero compatibility, no. For optimizer/trainer rewrite, yes: skip this adapter and consume batched compact arrays directly. | Profile stock adapter conversion time with `R=256,512,1024`; fail any compact candidate row that still calls stock conversion. |
| 8 | Telemetry and JSON/plain conversion | `_plain_telemetry_value`, `_plain`, `.tolist()`, metrics snapshots, and policy/env id lists convert arrays to Python lists for summaries. | Usually small if summary-only, but dangerous if run per hot step or over full arrays. | Yes. Keep checksums/counts/hashes in hot telemetry; dump arrays only for debug samples. | Telemetry canary: production/profile hot row exposes counts/bytes/checksums only. Fail if a hot-loop payload includes full observation lists or per-root JSON rows. |
| 9 | RND hashes and metric reads | `_state_hash` copies model state to CPU NumPy; `estimate` calls `.cpu().numpy()` and `.item()` for metrics. | Model state is small relative to observation stacks, but these force device syncs. | Mostly yes. Hash at init/checkpoint/debug cadence, not every collect/estimate. Keep GPU-side metrics until summary cadence. | RND cadence canary with metrics frequency set low: predictor still trains, target unchanged, wall time improves or sync count drops. |
| 10 | Actor render-state/state handoff | Actor payload and render-state helpers copy arrays when parent owns render state, including compact payload copies per step. | State bytes are smaller than full observations but can grow with trail capacity and actor count; copy cost becomes visible once search is fixed. | Partly. Actor-owned render/env state or a shared/native slab is the clean direction. | Actor-owned state canary: same actions and seeds produce same compact observations, dones, rewards, bonus state, and replay ids with parent-copy mode disabled. |

## Simple Read

The search boundary is no longer the only story. If search becomes fast, the next wall is full observation materialization at the scalar timestep, replay, target-row, sample, and RND edges.

The clean target shape is:

1. env step owns compact row/player state;
2. observation stack is a batched slab, preferably uint8 or resident;
3. search consumes a compact root batch and returns compact arrays;
4. collection writes compact replay index rows, not full target rows;
5. RND consumes a latest-frame ring, not a Python list of cloned per-frame tensors;
6. target rows and learner batches materialize only at the sampler boundary, with explicit bytes and checksums.

## Next Canaries

1. **Scalarization off canary**: compact hot path must not build ready-observation dicts, split timesteps, or stock env timesteps.
2. **Index replay canary**: collection must store `CompactReplayIndexRowsV1` with `observation_materialized=False`, then rebuild exact target rows only on request.
3. **RND latest-frame canary**: RND latest frame must match the policy stack and must not mutate target rewards in meter mode.
4. **Terminal final-observation canary**: terminal rows must sample final observation before autoreset.
5. **Materialization ledger canary**: every profile row must report root-copy bytes, Python rows materialized, RND materialized rows, and whether scalar timesteps were built.

## Recommendation

Do not spend the next pass on another small search wrapper unless these boundaries are measured in the same denominator. The highest-leverage non-search move is a compact replay/RND/sample owner: keep index rows and frame slabs hot, then materialize full learner rows only at the final edge that actually requires them.
