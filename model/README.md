# Model

This folder contains the neural-network side of the DUM-E playground.

## Boundaries

```txt
model/common/           Feature contracts, scalers, and DUM-E kinematics.
model/models_design/    Neural architectures only.
model/training/         Training orchestration for the World Model and controller.
model/artifacts/        Local trained weights, ignored by Git.
```

Dataset generation now lives outside this folder in `dataset_generation/`.
Serving now lives outside this folder in `server/`.

## World Model

The World Model consumes recent ball trajectory state plus throw context and predicts:

- future ball positions
- a compact catch plan
- catch probability

```bash
.venv/bin/python -m model.training.train_world_model --shots 800 --epochs 14 --output-dir model/artifacts
```

## Controller

The controller uses the frozen World Model and predicts DUM-E joint motion for:

```txt
base, coude, poignet
```

```bash
.venv/bin/python -m model.training.train_catch_controller --world-model-path model/artifacts/dum_e_world_model.pt --shots 500 --epochs 8 --output-dir model/artifacts
```

## Server

Run the server from the top-level `server/` package:

```bash
.venv/bin/python -m server.server --model-path model/artifacts/dum_e_world_model.pt --controller-path model/artifacts/dum_e_catch_controller.pt --port 8765
```
