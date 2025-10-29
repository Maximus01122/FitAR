"""Standalone MoveNet inference microservice.

Run this inside the TensorFlow environment:

    python backend/services/movenet_service.py --model /path/to/movenet.tflite

The service exposes a single POST /infer endpoint that accepts a JSON body:
    {"frame": "<base64 JPEG>"}
and responds with:
    {"keypoints": [[y, x, score], ...], "score": float}
"""

from __future__ import annotations

import argparse
import base64
import logging
import os
import time
from typing import Tuple

import cv2
import numpy as np
from flask import Flask, jsonify, request


try:
    import tensorflow as tf  # type: ignore
except ImportError as exc:
    raise SystemExit(
        "tensorflow is required to run the MoveNet service. "
        "Install tensorflow-macos/tensorflow-metal inside this environment."
    ) from exc


def build_interpreter(model_path: str, fallback_size: int):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    input_details = interpreter.get_input_details()[0]
    resize_shape = list(input_details["shape"])
    if len(resize_shape) == 4:
        if resize_shape[1] <= 1:
            resize_shape[1] = fallback_size
        if resize_shape[2] <= 1:
            resize_shape[2] = fallback_size
        if resize_shape != list(input_details["shape"]):
            interpreter.resize_tensor_input(input_details["index"], resize_shape)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    return interpreter, input_details, output_details


def letterbox(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h))
    canvas = np.zeros((target_h, target_w, 3), dtype=image.dtype)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def preprocess(frame_bgr: np.ndarray, input_details, fallback_size: int) -> np.ndarray:
    height = int(input_details["shape"][1])
    width = int(input_details["shape"][2])
    if height <= 1 or width <= 1:
        height = width = fallback_size
    padded = letterbox(frame_bgr, height, width)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

    dtype = input_details["dtype"]
    quant_params = input_details.get("quantization", (0.0, 0))
    scale, zero_point = quant_params if quant_params else (0.0, 0)

    if dtype == np.uint8:
        if scale and scale > 0:
            normalized = rgb.astype(np.float32) / 255.0
            quantized = normalized / scale + zero_point
            quantized = np.clip(np.round(quantized), 0, 255).astype(np.uint8)
            tensor = quantized
        else:
            tensor = rgb.astype(np.uint8)
    elif dtype == np.int32:
        tensor = rgb.astype(np.int32)
    else:
        tensor = (rgb.astype(np.float32) - 127.5) / 127.5

    return np.expand_dims(tensor, axis=0).astype(dtype)


def run_inference(
    interpreter: tf.lite.Interpreter,
    input_details,
    output_details,
    frame_bgr: np.ndarray,
    fallback_size: int,
) -> Tuple[np.ndarray, float]:
    input_tensor = preprocess(frame_bgr, input_details, fallback_size)
    interpreter.set_tensor(input_details["index"], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])  # [1, 6, 56]
    detections = output[0]
    best_idx = np.argmax(detections[:, -1])
    best_score = detections[best_idx, -1]
    keypoints = detections[best_idx, : 17 * 3].reshape(17, 3)
    return keypoints, float(best_score)


def create_app(model_path: str, fallback_size: int) -> Flask:
    app = Flask(__name__)
    interpreter, input_details, output_details = build_interpreter(model_path, fallback_size)

    @app.route("/infer", methods=["POST"])
    def infer():
        start_time = time.time()
        payload = request.get_json(silent=True) or {}
        frame_b64 = payload.get("frame")
        if not frame_b64:
            return jsonify({"error": "frame field missing"}), 400

        if isinstance(frame_b64, str) and frame_b64.startswith("data:image"):
            frame_b64 = frame_b64.split(",", 1)[1]

        try:
            frame_bytes = base64.b64decode(frame_b64)
        except Exception:
            return jsonify({"error": "invalid base64 frame"}), 400

        np_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "invalid image data"}), 400

        keypoints, score = run_inference(
            interpreter, input_details, output_details, frame, fallback_size
        )
        app.logger.debug("Top keypoint sample: %s", keypoints[0])
        latency_ms = (time.time() - start_time) * 1000.0
        app.logger.debug("MoveNet inference latency %.2f ms, score %.3f", latency_ms, score)
        return jsonify(
            {
                "keypoints": keypoints.tolist(),
                "score": score,
                "latency_ms": latency_ms,
            }
        )

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="MoveNet inference service")
    parser.add_argument("--model", required=True, help="Path to MoveNet TFLite model file")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8502)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--input-size",
        type=int,
        default=int(os.getenv("MOVENET_INPUT_SIZE", "256")),
        help="Fallback square input size when the model reports dynamic dims",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    app = create_app(args.model, args.input_size)
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
