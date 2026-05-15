"""Local HTTP inference server for the basketball Temporal Transformer world model.

Start it from the project root after training a model:

    .venv/bin/python -m basketball_sim.serving.world_model_server --model-dir models/world_model --port 8765
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from math import exp, isfinite, isnan, log
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch


from basketball_sim.models.features import (
    apply_scaler,
    context_vector,
    invert_scaler,
    sequence_vector,
)
from basketball_sim.models.world_model import TemporalTransformerWorldModel, WorldModelConfig


class JsonXGBoostBinaryClassifier:
    """Tiny JSON predictor for XGBoost binary:logistic tree ensembles.

    This avoids loading the native XGBoost runtime in the live PyTorch server,
    which can conflict with PyTorch's OpenMP runtime on macOS.
    """

    def __init__(self, model: dict):
        learner = model["learner"]
        self.trees = learner["gradient_booster"]["model"]["trees"]
        attributes = learner.get("attributes", {})
        self.best_iteration = int(attributes.get("best_iteration", len(self.trees) - 1))
        self.base_margin = self._base_margin(learner["learner_model_param"].get("base_score", "0.5"))

    @staticmethod
    def _base_margin(raw_value: str) -> float:
        value = float(str(raw_value).strip("[]"))
        value = min(1.0 - 1e-12, max(1e-12, value))
        return log(value / (1.0 - value))

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            z = exp(-value)
            return 1.0 / (1.0 + z)
        z = exp(value)
        return z / (1.0 + z)

    def predict_one(self, features: np.ndarray) -> float:
        score = self.base_margin
        for tree in self.trees[: self.best_iteration + 1]:
            node = 0
            left = tree["left_children"]
            right = tree["right_children"]
            splits = tree["split_indices"]
            conditions = tree["split_conditions"]
            default_left = tree["default_left"]
            while left[node] != -1:
                value = float(features[splits[node]])
                if not isfinite(value) or isnan(value):
                    node = left[node] if default_left[node] else right[node]
                elif value < conditions[node]:
                    node = left[node]
                else:
                    node = right[node]
            score += conditions[node]
        return self._sigmoid(score)


@dataclass
class InferenceEngine:
    model: TemporalTransformerWorldModel
    scalers: dict
    device: torch.device
    model_dir: Path
    momentum_model: Any | None = None
    momentum_metadata: dict | None = None

    @property
    def config(self) -> WorldModelConfig:
        return self.model.config

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        params = payload.get("params") or {}
        history = payload.get("history") or []
        if not history:
            raise ValueError("payload.history must contain at least one trajectory row.")

        history_steps = self.config.history_steps
        history_rows = list(history[-history_steps:])
        if len(history_rows) < history_steps:
            history_rows = [history_rows[0]] * (history_steps - len(history_rows)) + history_rows

        sequence = np.asarray([sequence_vector(row) for row in history_rows], dtype=np.float32)[None, :, :]
        context = context_vector(params, gravity=float(payload.get("gravity", params.get("gravity", 9.81))))[None, :]

        sequence = apply_scaler(sequence, self.scalers["sequence"]).astype(np.float32)
        context = apply_scaler(context, self.scalers["context"]).astype(np.float32)

        with torch.no_grad():
            sequence_tensor = torch.from_numpy(sequence).to(self.device)
            context_tensor = torch.from_numpy(context).to(self.device)
            latent = self.model.encode(sequence_tensor, context_tensor)
            prediction = self.model.decode(latent).cpu().numpy()[0]
            latent_np = latent.cpu().numpy().astype(np.float32)

        prediction = invert_scaler(prediction, self.scalers["target"])
        horizon = max(1, min(int(payload.get("horizon", self.config.horizon_steps)), self.config.horizon_steps))
        prediction = prediction[:horizon]

        last_row = history_rows[-1]
        last_timestep = int(last_row.get("timestep", 0))
        last_time = float(last_row.get("time", 0.0))
        dt = float(payload.get("dt", 0.02))

        predictions = []
        for index, point in enumerate(prediction, start=1):
            x, y, z = (float(point[0]), float(point[1]), float(point[2]))
            predictions.append(
                {
                    "timestep": last_timestep + index,
                    "time": last_time + dt * index,
                    "ball_x": x,
                    "ball_y": y,
                    "ball_z": z,
                    "x": x,
                    "y": y,
                    "z": z,
                }
            )

        momentum = None
        if self.momentum_model is not None:
            probability = float(self.momentum_model.predict_one(latent_np[0]))
            threshold = float((self.momentum_metadata or {}).get("threshold", 0.5))
            momentum = {
                "probability": probability,
                "label": int(probability >= threshold),
                "threshold": threshold,
                "latent_dim": int(latent_np.shape[1]),
            }

        response = {
            "ok": True,
            "history_steps": history_steps,
            "horizon_steps": horizon,
            "predictions": predictions,
        }
        if momentum is not None:
            response["momentum"] = momentum
        return response


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_momentum_model(path: Path):
    if not path.exists():
        return None, None
    with path.open("r", encoding="utf-8") as file:
        booster = JsonXGBoostBinaryClassifier(json.load(file))
    metadata_path = path.with_name("momentum_xgb_config.json")
    metadata = None
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
    return booster, metadata


def load_engine(model_dir: Path, device_name: str, momentum_model_path: Path | None) -> InferenceEngine:
    checkpoint_path = model_dir / "world_model.pt"
    scaler_path = model_dir / "world_model_scaler.json"
    if not checkpoint_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            f"Missing model artifacts in {model_dir}. "
            "Train first with python -m basketball_sim.models.train_world_model."
        )

    device = pick_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    with scaler_path.open("r", encoding="utf-8") as file:
        scalers = json.load(file)

    config = WorldModelConfig.from_dict(checkpoint["config"])
    model = TemporalTransformerWorldModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    momentum_model, momentum_metadata = (None, None)
    if momentum_model_path is not None:
        momentum_model, momentum_metadata = load_momentum_model(momentum_model_path)
    return InferenceEngine(
        model=model,
        scalers=scalers,
        device=device,
        model_dir=model_dir,
        momentum_model=momentum_model,
        momentum_metadata=momentum_metadata,
    )


def make_handler(engine: InferenceEngine | None, load_error: str | None):
    class Handler(BaseHTTPRequestHandler):
        server_version = "BasketballWorldModel/1.0"

        def end_headers(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            super().end_headers()

        def log_message(self, format: str, *args: Any) -> None:
            sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))

        def send_json(self, status: int, data: dict[str, Any]) -> None:
            body = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self.end_headers()

        def do_GET(self) -> None:
            if self.path not in {"/", "/health"}:
                self.send_json(404, {"ok": False, "error": "not found"})
                return
            if engine is None:
                self.send_json(200, {"ok": False, "error": load_error})
                return
            self.send_json(
                200,
                {
                    "ok": True,
                    "model_dir": str(engine.model_dir),
                    "device": str(engine.device),
                    "history_steps": engine.config.history_steps,
                    "horizon_steps": engine.config.horizon_steps,
                    "d_model": engine.config.d_model,
                    "num_heads": engine.config.num_heads,
                    "num_layers": engine.config.num_layers,
                    "momentum_model": engine.momentum_model is not None,
                    "momentum_threshold": None
                    if engine.momentum_metadata is None
                    else engine.momentum_metadata.get("threshold"),
                },
            )

        def do_POST(self) -> None:
            if self.path != "/predict":
                self.send_json(404, {"ok": False, "error": "not found"})
                return
            if engine is None:
                self.send_json(503, {"ok": False, "error": load_error})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                self.send_json(200, engine.predict(payload))
            except Exception as exc:  # pragma: no cover - keeps the browser client debuggable.
                self.send_json(400, {"ok": False, "error": str(exc)})

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/world_model")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--momentum-model", default="models/momentum_xgb/momentum_xgb.json")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    momentum_model_path = Path(args.momentum_model).resolve() if args.momentum_model else None
    engine = None
    load_error = None
    try:
        engine = load_engine(model_dir, args.device, momentum_model_path)
        print(
            f"Loaded world model from {model_dir} "
            f"({engine.config.num_layers} layers, {engine.config.num_heads} heads) on {engine.device}."
        )
        if engine.momentum_model is not None:
            print(f"Loaded momentum XGBoost model from {momentum_model_path}.")
        else:
            print("Momentum XGBoost model unavailable; /predict will return trajectory only.")
    except Exception as exc:
        load_error = str(exc)
        print(f"World model unavailable: {load_error}")

    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine, load_error))
    print(f"World model server: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
