"""Local inference server for the DUM-E catch world model."""

from __future__ import annotations

import argparse
import json
import math
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import torch

warnings.filterwarnings("ignore", message="enable_nested_tensor is True.*", category=UserWarning)

from model.common.features import (
    ACTION_FEATURES,
    apply_scaler,
    context_vector,
    invert_scaler,
    plan_from_future,
    sequence_vector,
)
from model.common.kinematics import REST_AXIS_BASE, REST_AXIS_ELBOW, REST_AXIS_WRIST
from model.models_design.catch_controller import DUMECatchController, DUMECatchControllerConfig
from model.models_design.world_model import DUMEWorldModel, DUMEWorldModelConfig


class Predictor:
    def __init__(self, model_path: Path | None, controller_path: Path | None = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = None
        self.artifact = None
        self.controller = None
        self.controller_artifact = None
        if model_path and model_path.exists():
            self.artifact = torch.load(model_path, map_location=self.device)
            config = DUMEWorldModelConfig.from_dict(self.artifact["config"])
            self.model = DUMEWorldModel(config).to(self.device)
            self.model.load_state_dict(self.artifact["model_state"])
            self.model.eval()
            if controller_path and controller_path.exists():
                self.controller_artifact = torch.load(controller_path, map_location=self.device)
                controller_config = DUMECatchControllerConfig.from_dict(self.controller_artifact["config"])
                self.controller = DUMECatchController(controller_config, self.model).to(self.device)
                self.controller.load_state_dict(self.controller_artifact["model_state"])
                self.controller.eval()

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.artifact is not None

    @property
    def controller_loaded(self) -> bool:
        return self.controller is not None and self.controller_artifact is not None

    def _pad_history(self, history: list[dict], history_steps: int) -> list[dict]:
        if not history:
            raise ValueError("history must contain at least one row.")
        if len(history) >= history_steps:
            return history[-history_steps:]
        return [history[0]] * (history_steps - len(history)) + history

    def _fallback(self, payload: dict[str, Any]) -> dict[str, Any]:
        history = payload.get("history") or []
        if not history:
            raise ValueError("history must contain at least one row.")
        params = payload.get("params") or {}
        horizon = int(payload.get("horizon") or 30)
        dt = float(payload.get("dt") or 0.02)
        last = history[-1]
        position = np.asarray([last["ball_x"], last["ball_y"], max(last["ball_z"], 0.0)], dtype=np.float32)
        velocity = np.asarray([last["ball_vx"], last["ball_vy"], last["ball_vz"]], dtype=np.float32)
        acceleration = np.asarray(
            [last.get("ball_ax", 0.0), last.get("ball_ay", 0.0), last.get("ball_az", -9.81)],
            dtype=np.float32,
        )
        future_rows = []
        current_time = float(last.get("time", 0.0))
        for step in range(1, horizon + 1):
            t = step * dt
            next_position = position + velocity * t + 0.5 * acceleration * t * t
            future_rows.append(
                {
                    "time": current_time + t,
                    "ball_x": float(next_position[0]),
                    "ball_y": float(next_position[1]),
                    "ball_z": float(max(next_position[2], 0.0)),
                }
            )
        action = plan_from_future(future_rows, params, current_time)
        plan = self._plan_dict(action)
        robot_motion = self._fallback_motion(plan, current_time, dt, horizon)
        return {
            "ok": True,
            "model_loaded": False,
            "controller_loaded": False,
            "predictions": future_rows,
            "catch_plan": plan,
            "robot_motion": robot_motion,
            "momentum": {"probability": plan["catch_probability"]},
        }

    def _plan_dict(self, action: np.ndarray) -> dict[str, float]:
        plan = {key: float(action[index]) for index, key in enumerate(ACTION_FEATURES)}
        plan["catch_probability"] = max(0.0, min(1.0, plan["catch_probability"]))
        return plan

    def _fallback_motion(
        self,
        plan: dict[str, float],
        last_time: float,
        dt: float,
        steps: int,
    ) -> list[dict[str, float]]:
        rest = np.asarray([REST_AXIS_BASE, REST_AXIS_ELBOW, REST_AXIS_WRIST], dtype=np.float32)
        target = np.asarray(
            [
                float(plan.get("axis_base", REST_AXIS_BASE)),
                float(plan.get("axis_elbow", REST_AXIS_ELBOW)),
                float(plan.get("axis_wrist", REST_AXIS_WRIST)),
            ],
            dtype=np.float32,
        )
        progress = np.linspace(1.0 / max(1, steps), 1.0, max(1, steps), dtype=np.float32)
        progress = progress * progress * (3.0 - 2.0 * progress)
        motion = rest[None, :] + progress[:, None] * (target[None, :] - rest[None, :])
        return self._motion_dict(motion, last_time, dt, probability=plan.get("catch_probability"))

    def _motion_dict(
        self,
        motion: np.ndarray,
        last_time: float,
        dt: float,
        probability: float | None = None,
    ) -> list[dict[str, float]]:
        rows = []
        for index, axes in enumerate(motion):
            row = {
                "step": index + 1,
                "time": last_time + (index + 1) * dt,
                "axis_base": float(axes[0]),
                "axis_elbow": float(axes[1]),
                "axis_wrist": float(axes[2]),
            }
            if probability is not None:
                row["catch_probability"] = float(probability)
            rows.append(row)
        return rows

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.loaded:
            return self._fallback(payload)

        assert self.artifact is not None and self.model is not None
        params = payload.get("params") or {}
        history_steps = int(self.artifact["config"]["history_steps"])
        history = self._pad_history(payload.get("history") or [], history_steps)

        sequence = np.asarray([sequence_vector(row, params) for row in history], dtype=np.float32)[None, :, :]
        context = context_vector(params, gravity=float(params.get("gravity", 9.81)))[None, :]
        sequence = apply_scaler(sequence, self.artifact["scalers"]["sequence"]).astype(np.float32)
        context = apply_scaler(context, self.artifact["scalers"]["context"]).astype(np.float32)

        sequence_tensor = torch.from_numpy(sequence).to(self.device)
        context_tensor = torch.from_numpy(context).to(self.device)
        with torch.no_grad():
            if self.controller_loaded:
                assert self.controller is not None
                outputs = self.controller(sequence_tensor, context_tensor)
                future_output = outputs["world_future"]
                action_output = outputs["world_action"]
                joint_output = outputs["joints"]
                controller_probability = 1.0 / (1.0 + math.exp(-float(outputs["catch_logit"].detach().cpu().numpy()[0, 0])))
            else:
                outputs = self.model(sequence_tensor, context_tensor)
                future_output = outputs["future"]
                action_output = outputs["action"]
                joint_output = None
                controller_probability = None
        future = future_output.detach().cpu().numpy()
        future = invert_scaler(future, self.artifact["scalers"]["target"])[0]
        action = action_output.detach().cpu().numpy()[0]
        action[:-1] = invert_scaler(action[:-1], self.artifact["scalers"]["action"])
        action[-1] = 1.0 / (1.0 + math.exp(-float(action[-1])))

        last_time = float(history[-1].get("time", 0.0))
        dt = float(payload.get("dt") or 0.02)
        predictions = [
            {
                "time": last_time + (index + 1) * dt,
                "ball_x": float(point[0]),
                "ball_y": float(point[1]),
                "ball_z": float(max(point[2], 0.0)),
            }
            for index, point in enumerate(future)
        ]
        plan = self._plan_dict(action)
        if joint_output is not None and self.controller_artifact is not None:
            motion = joint_output.detach().cpu().numpy()[0]
            motion = invert_scaler(motion, self.controller_artifact["scalers"]["joint"])
            robot_motion = self._motion_dict(motion, last_time, dt, probability=controller_probability)
        else:
            robot_motion = self._fallback_motion(plan, last_time, dt, len(predictions))
        return {
            "ok": True,
            "model_loaded": True,
            "controller_loaded": self.controller_loaded,
            "predictions": predictions,
            "catch_plan": plan,
            "robot_motion": robot_motion,
            "momentum": {"probability": plan["catch_probability"]},
        }


def make_handler(predictor: Predictor):
    class Handler(BaseHTTPRequestHandler):
        server_version = "DUMEWorldModel/0.1"

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "content-type")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self) -> None:  # noqa: N802
            self._send_json(200, {"ok": True})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/predict":
                self._send_json(404, {"ok": False, "error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                self._send_json(200, predictor.predict(payload))
            except Exception as exc:  # pragma: no cover - defensive for browser use.
                self._send_json(500, {"ok": False, "error": str(exc)})

        def log_message(self, format: str, *args) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="model/artifacts/dum_e_world_model.pt")
    parser.add_argument("--controller-path", default="model/artifacts/dum_e_catch_controller.pt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    controller_path = Path(args.controller_path) if args.controller_path else None
    predictor = Predictor(model_path=model_path, controller_path=controller_path, device=args.device)
    state = "loaded" if predictor.loaded else "fallback heuristic"
    controller_state = "controller loaded" if predictor.controller_loaded else "controller fallback"
    print(
        f"DUM-E world model server ({state}, {controller_state}) "
        f"on http://{args.host}:{args.port}/predict",
        flush=True,
    )
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(predictor))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nDUM-E world model server stopped.", flush=True)
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
