# CurvyZero

CurvyZero is an investigation and implementation workspace for training agents on a CurvyTron-like environment.

Current direction:

1. Build one fast, source-faithful CurvyTron environment.
2. Use the JS oracle and `CurvyTronSourceEnv` as temporary proof tools, not as the final runtime.
3. Treat the strict 1v1/no-bonus path as a boundary proof for reset, observation, reward, replay, and speed.
4. Keep LightZero, Modal, JAX/Mctx, and vectorized simulation as contract or runtime evidence until they run the real environment.

Start with [docs/README.md](docs/README.md) for the project map.
