# Basketball Shot Simulator

A local 3D basketball-shot simulator with deterministic physics, wind regimes, dataset export, a Temporal Transformer world model, and an XGBoost momentum classifier trained on world-model latents.

The browser app runs from static files. The ML stack is organized as a Python package so dataset generation, model training, latent extraction, and serving stay separated.

## Project Layout

```txt
index.html                     Browser simulator entry point
app.js                         Main simulator and UI logic
styles.css                     Shared UI styles
ai-cameras.html                Headless rendering target for dataset frames
ai-cameras.js                  Camera/rendering logic for AI views
dataset_builder/render_frames.js
                               PNG renderer for frames_manifest.csv

basketball_sim/
  dataset/
    config.py                  Default dataset configuration
    config_test.py             Tiny smoke-test configuration
    grid.py                    Parameter-grid enumeration and summaries
    generate.py                CSV and frame-manifest generation
  models/
    features.py                Shared feature/scaler contracts
    world_model.py             Temporal Transformer architecture
    train_world_model.py       World-model training and validation
    build_momentum_latents.py  Latent extraction for classification
    train_momentum_xgb.py      XGBoost momentum training
  serving/
    world_model_server.py      Local HTTP inference server

dataset_builder/*.py           Backward-compatible wrappers
live_world_model_server.py     Backward-compatible server wrapper
models/                        Trained artifacts, gitignored
generated_dataset*/            Generated datasets, gitignored
```

## Setup

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Run The App

Open `index.html` directly in a browser or serve the folder with any static server.

To enable live world-model predictions and momentum probabilities:

```bash
.venv/bin/python -m basketball_sim.serving.world_model_server --model-dir models/world_model --momentum-model models/momentum_xgb/momentum_xgb.json --port 8765
```

The `WM` button in the browser sends the current trajectory history to the local server, draws future predicted positions in blue, and shows the XGBoost momentum probability when the classifier artifact is available.

## Physics Model

The shot is integrated with fixed-step RK4 until ground impact or `MAX_STEPS`.

```txt
r(t) = [x, y, z]
v(t) = [vx, vy, vz]
W(x, y, z, t) = [wind_x, wind_y, wind_z]

dr/dt = v
dv/dt = [0, 0, -g] - (drag_coeff / ball_mass) * ||v - W|| * (v - W)
```

Drag uses the relative ball-air velocity, not absolute ball velocity. Wind direction uses horizontal azimuth plus vertical elevation.

## Dataset Generation

Count the dataset grid:

```bash
.venv/bin/python -m basketball_sim.dataset.grid --config dataset_config
```

Generate CSV files and a frame manifest:

```bash
.venv/bin/python -m basketball_sim.dataset.generate --config dataset_config --output-dir generated_dataset --max-shots 10
```

Render PNG frames from the exact `ai-cameras.html` view:

```bash
node dataset_builder/render_frames.js --dataset-dir generated_dataset
```

Tiny smoke dataset:

```bash
.venv/bin/python -m basketball_sim.dataset.config_test
.venv/bin/python -m basketball_sim.dataset.grid --config dataset_config_test
.venv/bin/python -m basketball_sim.dataset.generate --config dataset_config_test --output-dir generated_dataset_config_test --max-shots 0 --frame-stride 250
```

The generated dataset contains:

```txt
shots.csv
trajectory.csv
wind_field.csv
labels.csv
frames_manifest.csv
```

## World Model

The world model is an encoder-only Temporal Transformer:

```txt
history[t-k:t] + physical context -> latent state -> positions[t+1:t+H]
```

It predicts future positions, not the make/miss label. Validation can split by full unseen shots and report RMSE by horizon.

Smoke training:

```bash
.venv/bin/python -m basketball_sim.models.train_world_model --config dataset_config_test --max-shots 2 --epochs 1 --history-steps 6 --horizon-steps 8 --d-model 48 --num-heads 12 --num-layers 1 --output-dir models/world_model_smoke
```

Larger training run:

```bash
.venv/bin/python -m basketball_sim.models.train_world_model --config dataset_config --max-shots 3000 --epochs 8 --history-steps 12 --horizon-steps 30 --d-model 192 --num-heads 12 --num-layers 6 --batch-size 256 --augmentation mixed --augmented-fraction 0.85 --min-label1-fraction 0.20 --candidate-limit 30000 --split-by shot --metric-horizons 1,5,10,30 --output-dir models/world_model
```

Important training options:

```txt
--augmentation mixed          Mix grid shots with continuous synthetic shots.
--min-label1-fraction 0.20    Keep at least 20% made-shot labels when possible.
--split-by shot               Validate on full shots never seen during training.
--metric-horizons 1,5,10,30   Report RMSE at selected forecast horizons.
```

Current reference run:

```txt
shots: 3000
label 1 shots: 600 (20.0%)
validation split: full unseen shots
t+1 RMSE: 0.0696 m
t+5 RMSE: 0.0648 m
t+10 RMSE: 0.0633 m
t+30 RMSE: 0.1255 m
```

## Momentum Classifier

Definition used in this project:

```txt
momentum = 1 when the shot is made
momentum = 0 otherwise
```

The XGBoost classifier does not consume raw simulator features directly. It receives the latent vector produced by the Temporal Transformer:

```txt
history + context -> world_model.encode(...) -> latent -> XGBoost -> P(momentum)
```

Extract latents:

```bash
.venv/bin/python -m basketball_sim.models.build_momentum_latents --config dataset_config --world-model-dir models/world_model --output-dir models/momentum_xgb --max-shots 3000 --augmentation mixed --augmented-fraction 0.85 --min-label1-fraction 0.20 --candidate-limit 30000
```

Train XGBoost:

```bash
.venv/bin/python -m basketball_sim.models.train_momentum_xgb --latents-path models/momentum_xgb/momentum_latents.npz --metadata-path models/momentum_xgb/momentum_latents_config.json --output-dir models/momentum_xgb --xgb-rounds 450 --early-stopping-rounds 35 --threshold best-f1
```

Current reference classifier:

```txt
latent_dim: 192
validation AUC: 0.8869
threshold: 0.545
validation accuracy: 0.8044
validation precision: 0.4916
validation recall: 0.8408
validation F1: 0.6205
```

## Backward Compatibility

Old commands still work through thin wrappers:

```bash
.venv/bin/python dataset_builder/train_world_model.py --config dataset_config_test
.venv/bin/python live_world_model_server.py --model-dir models/world_model
```

Prefer the package commands for new work:

```bash
.venv/bin/python -m basketball_sim.models.train_world_model
.venv/bin/python -m basketball_sim.serving.world_model_server
```

## Git Notes

The repository is prepared to keep source code in Git while excluding heavy local artifacts:

```txt
.venv/
models/
generated_dataset*/
node_modules/
__pycache__/
```

Before pushing, regenerate model artifacts locally if needed, but do not commit trained weights or generated datasets unless you intentionally move them to a release or external storage.
