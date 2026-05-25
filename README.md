# DUM-E Catch Playground

A local playground for generating synthetic DUM-E catch data, training a World Model, training a robot-arm controller on top of it, and visualizing predictions in the browser.

## Architecture

```txt
interface/              Browser UI and canvas simulation.
dataset_generation/     Synthetic throw physics, wind, DUM-E catch labels, dataset tools.
model/                  Feature contracts, kinematics, architectures, and training scripts.
server/                 Local HTTP inference server used by the WM button.
model/artifacts/        Local trained weights, gitignored.
```

The separation rule is simple:

```txt
interface -> visualizes and sends /predict requests
dataset_generation -> creates trajectories and training windows
model -> defines and trains neural architectures
server -> loads artifacts and answers browser predictions
```

## Setup

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Interface

Open the root `index.html`, which redirects to `interface/index.html`, or open the interface directly:

```txt
interface/index.html
```

The UI works without the server. When the server is running, the `WM` button overlays predicted ball states and DUM-E ghost arms.

## Dataset Generation

The current training dataset is generated programmatically by:

```txt
dataset_generation/synthetic.py
```

It contains the synthetic throw simulator, wind regimes, DUM-E catch labels, and window builders used by training.

Render AI camera frames from a generated manifest if needed:

```bash
node dataset_generation/render_frames.js --dataset-dir generated_dataset
```

## Train World Model

```bash
.venv/bin/python -m model.training.train_world_model --shots 800 --epochs 14 --output-dir model/artifacts
```

Fast smoke run:

```bash
.venv/bin/python -m model.training.train_world_model --shots 12 --epochs 1 --output-dir /tmp/dum_e_model_smoke
```

## Train Controller

The controller uses the frozen World Model and learns DUM-E joint motion over `base`, `coude`, and `poignet`.

```bash
.venv/bin/python -m model.training.train_catch_controller --world-model-path model/artifacts/dum_e_world_model.pt --shots 500 --epochs 8 --output-dir model/artifacts
```

Fast smoke run:

```bash
.venv/bin/python -m model.training.train_catch_controller --shots 12 --epochs 1 --max-windows-per-shot 4 --output-dir /tmp/dum_e_controller_smoke
```

## Serve Predictions

One-command launcher:

```bash
./start_server.sh
```

Stop it with `Ctrl+C`.

Equivalent explicit command:

```bash
.venv/bin/python -m server.server --model-path model/artifacts/dum_e_world_model.pt --controller-path model/artifacts/dum_e_catch_controller.pt --port 8765
```
sinon:
```
cd Desktop/dum-e/dum-E
./start_server.sh
```

Endpoint used by the browser:

```txt
POST http://127.0.0.1:8765/predict
```

If artifacts are missing, the server still responds with deterministic fallback predictions so the interface can keep testing the contract.

## Git Notes

Heavy local outputs are ignored:

```txt
.venv/
model/artifacts/
generated_dataset*/
node_modules/
__pycache__/
```
