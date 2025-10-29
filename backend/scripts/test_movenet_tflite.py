"""
Utility script to sanity-check a MoveNet TFLite model.

Example:
    python backend/scripts/test_movenet_tflite.py \
        --model /path/to/multi_pose_lightning.tflite \
        --image /path/to/frame.jpg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

try:
    import tensorflow as tf  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "tensorflow is required to run this script. "
        "Install tensorflow-macos / tensorflow-metal (on Apple silicon) or the "
        "appropriate TensorFlow package for your platform."
    ) from exc


def load_interpreter(
    model_path: Path, fallback_size: int
) -> Tuple[tf.lite.Interpreter, dict, dict]:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
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
    """Resize preserving aspect ratio with zero padding."""
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h))
    canvas = np.zeros((target_h, target_w, 3), dtype=image.dtype)
    top = (target_h - resized.shape[0]) // 2
    left = (target_w - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return canvas


def prepare_input(
    image_bgr: np.ndarray,
    input_details: dict,
    fallback_size: int,
) -> np.ndarray:
    """Convert an OpenCV BGR image into the tensor format the model expects."""
    target_h = int(input_details["shape"][1])
    target_w = int(input_details["shape"][2])
    if target_h <= 1 or target_w <= 1:
        target_h = target_w = fallback_size
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgb = letterbox(rgb, target_h, target_w)

    dtype = input_details["dtype"]
    scale, zero_point = input_details.get("quantization", (0.0, 0))

    if dtype == np.uint8:
        if scale and scale > 0:
            normalized = rgb.astype(np.float32) / 255.0
            quantized = normalized / scale + zero_point
            tensor = np.clip(np.round(quantized), 0, 255).astype(np.uint8)
        else:
            tensor = rgb.astype(np.uint8)
    elif dtype == np.int32:
        tensor = rgb.astype(np.int32)
    else:  # assume float32
        tensor = rgb.astype(np.float32)
        if scale and scale > 0:
            tensor = tensor * scale + zero_point

    return np.expand_dims(tensor, axis=0)


def decode_output(output: np.ndarray) -> list:
    """Return detections sorted by score."""
    detections = output[0]  # [6, 56]
    result = []
    for det in detections:
        keypoints = det[: 17 * 3].reshape(17, 3)
        ymin, xmin, ymax, xmax, score = det[51:]
        result.append(
            {
                "score": float(score),
                "bbox": [float(ymin), float(xmin), float(ymax), float(xmax)],
                "keypoints": keypoints.tolist(),
            }
        )
    result.sort(key=lambda d: d["score"], reverse=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Test a MoveNet TFLite model.")
    parser.add_argument("--model", required=True, type=Path, help="Path to *.tflite file")
    parser.add_argument("--image", required=True, type=Path, help="Path to input image")
    parser.add_argument("--topk", type=int, default=3, help="Number of detections to display")
    parser.add_argument(
        "--input-size",
        type=int,
        default=256,
        help="Fallback square size when model input shape is dynamic",
    )
    parser.add_argument("--save-json", type=Path, help="Optional path to dump raw detections")
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Model file not found: {args.model}")
    if not args.image.exists():
        raise SystemExit(f"Image file not found: {args.image}")

    interpreter, input_details, output_details = load_interpreter(args.model, args.input_size)
    print(f"Input:  shape={input_details['shape']}, dtype={input_details['dtype']}")
    print(f"Output: shape={output_details['shape']}, dtype={output_details['dtype']}")

    image_bgr = cv2.imread(str(args.image))
    if image_bgr is None:
        raise SystemExit(f"Failed to read image: {args.image}")

    input_tensor = prepare_input(image_bgr, input_details, args.input_size)
    interpreter.set_tensor(input_details["index"], input_tensor)
    interpreter.invoke()
    raw_output = interpreter.get_tensor(output_details["index"])

    detections = decode_output(raw_output)
    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(detections, indent=2))
        print(f"Saved detections to {args.save_json}")

    print("\nTop detections:")
    for det in detections[: args.topk]:
        print(
            f"  score={det['score']:.3f} "
            f"bbox={[round(v, 3) for v in det['bbox']]}"
        )
        first_kp = det["keypoints"][0]
        print(f"    first keypoint (y, x, score) = {[round(v, 3) for v in first_kp]}")


if __name__ == "__main__":
    main()
