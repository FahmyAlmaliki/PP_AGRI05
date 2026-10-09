"""
Real-time chicken gender detection (webcam + OpenCV + TFLite, FLOAT32 model).

Usage:
    python camera_tflite.py
    python camera_tflite.py --camera 1
    python camera_tflite.py --full-frame
    python camera_tflite.py --model chicken_gender_augmented.tflite --norm raw

Keyboard:
    [Q] / [ESC] : quit
    [S]         : save screenshot
    [C]         : toggle center ROI <-> full frame
"""

import sys
import time
import argparse
import platform
from pathlib import Path

import cv2
import numpy as np


# ── 1. Flexible TFLite interpreter loader ─────────────────────────────────────
def load_tflite_interpreter(model_path: str):
    """Load a TFLite Interpreter from whichever runtime is installed."""
    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter(model_path=model_path), "ai-edge-litert (LiteRT)"
    except ImportError:
        pass

    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter(model_path=model_path), "tflite-runtime"
    except ImportError:
        pass

    try:
        import tensorflow as tf
        return tf.lite.Interpreter(model_path=model_path), "tensorflow.lite"
    except ImportError:
        pass

    print("\n[ERROR] No TFLite runtime found.")
    print("Install one with: pip install ai-edge-litert\n")
    sys.exit(1)


# ── 2. Real-time classifier (float32 only) ────────────────────────────────────
class LiveChickenClassifier:
    CLASSES = ["Female", "Male"]

    # BGR colors
    COLOR_FEMALE = (180, 105, 255)
    COLOR_MALE = (255, 178, 50)

    def __init__(self, model_path: str, norm: str = "raw"):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file '{model_path}' not found!")

        self.norm = norm
        self.interpreter, self.backend = load_tflite_interpreter(str(self.model_path))
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        input_dtype = self.input_details[0]["dtype"]
        if input_dtype != np.float32:
            raise ValueError(
                f"This script is for FLOAT32 models, but the model input is "
                f"{input_dtype}. Use the int8 version of the script instead."
            )

        shape = self.input_details[0]["shape"]  # [1, H, W, 3]
        self.input_h = int(shape[1])
        self.input_w = int(shape[2])

        print(f"[INFO] Input shape  : {shape.tolist()}")
        print(f"[INFO] Input dtype  : {input_dtype.__name__}")
        print(f"[INFO] Output dtype : {self.output_details[0]['dtype'].__name__}")
        print(f"[INFO] Normalization: {self.norm}")

    def _preprocess(self, bgr_image: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR
        )
        data = resized.astype(np.float32)

        # Must match what the model was trained with:
        #   raw   -> keep 0..255 (model has a Rescaling/preprocess layer inside)
        #   zero1 -> scale to 0..1
        #   neg1  -> scale to -1..1 (MobileNet-style)
        if self.norm == "zero1":
            data = data / 255.0
        elif self.norm == "neg1":
            data = data / 127.5 - 1.0

        return np.expand_dims(data, axis=0)

    def predict_cv_image(self, bgr_image: np.ndarray) -> dict:
        input_data = self._preprocess(bgr_image)

        t0 = time.perf_counter()
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]["index"])
        latency_ms = (time.perf_counter() - t0) * 1000

        probs = output[0].astype(np.float32)

        # Sigmoid output (single neuron) -> convert to [P(female), P(male)]
        if probs.size == 1:
            p_male = float(probs[0])
            if p_male < 0 or p_male > 1:  # logit -> sigmoid
                p_male = 1.0 / (1.0 + np.exp(-p_male))
            probs = np.array([1.0 - p_male, p_male], dtype=np.float32)

        # Softmax fallback if the output looks like logits
        elif (
            np.any(probs < 0)
            or np.any(probs > 1)
            or abs(probs.sum() - 1.0) > 0.05
        ):
            exp_p = np.exp(probs - np.max(probs))
            probs = exp_p / exp_p.sum()

        top_idx = int(np.argmax(probs))
        return {
            "class": self.CLASSES[top_idx],
            "confidence": float(probs[top_idx]) * 100,
            "prob_female": float(probs[0]) * 100,
            "prob_male": float(probs[1]) * 100,
            "latency_ms": latency_ms,
        }


# ── 3. UI drawing ─────────────────────────────────────────────────────────────
def draw_prob_bar(frame, label, x, y, percent, color, bar_w=160, bar_h=14):
    cv2.putText(frame, label, (x, y + 11), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    bx = x + 70
    cv2.rectangle(frame, (bx, y), (bx + bar_w, y + bar_h), (60, 60, 60), -1)
    fill = int((percent / 100) * bar_w)
    if fill > 0:
        cv2.rectangle(frame, (bx, y), (bx + fill, y + bar_h), color, -1)
    cv2.putText(frame, f"{percent:.1f}%", (bx + bar_w + 10, y + 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_ui(frame, result, roi_box, use_roi, fps):
    h, w, _ = frame.shape
    cls = result["class"]
    color = (LiveChickenClassifier.COLOR_FEMALE if cls == "Female"
             else LiveChickenClassifier.COLOR_MALE)

    # ROI box with bold corners
    if use_roi and roi_box is not None:
        x1, y1, x2, y2 = roi_box
        L, T = 25, 3
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        for (cx, cy, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                                 (x1, y2, 1, -1), (x2, y2, -1, -1)]:
            cv2.line(frame, (cx, cy), (cx + dx * L, cy), color, T)
            cv2.line(frame, (cx, cy), (cx, cy + dy * L), color, T)
        cv2.putText(frame, "FOKUS OBJEK", (x1 + 10, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # Translucent header & footer panels
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 65), (20, 20, 20), -1)
    cv2.rectangle(overlay, (0, h - 80), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, f"PREDIKSI: {cls.upper()} ({result['confidence']:.1f}%)",
                (20, 42), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f} | Latency: {result['latency_ms']:.1f} ms",
                (w - 240, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (220, 220, 220), 1, cv2.LINE_AA)

    draw_prob_bar(frame, "Female:", 20, h - 52, result["prob_female"],
                  LiveChickenClassifier.COLOR_FEMALE)
    draw_prob_bar(frame, "Male  :", 20, h - 24, result["prob_male"],
                  LiveChickenClassifier.COLOR_MALE)

    mode_text = "Mode: Kotak Fokus" if use_roi else "Mode: Seluruh Layar"
    cv2.putText(frame, mode_text, (w - 320, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 230, 150), 1, cv2.LINE_AA)
    cv2.putText(frame, "[Q] Keluar | [S] Simpan | [C] Mode Layar",
                (w - 320, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (180, 180, 180), 1, cv2.LINE_AA)


# ── 4. Camera main loop ───────────────────────────────────────────────────────
def open_camera(camera_id: int):
    # CAP_V4L2 only exists on Linux; use the default backend elsewhere
    if platform.system() == "Linux":
        cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(camera_id)

    # MJPG reduces USB bandwidth
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def run_camera(camera_id: int, model_path: str, default_roi: bool, norm: str):
    print("\n=== REAL-TIME CHICKEN GENDER DETECTION (TFLite float32) ===")

    classifier = LiveChickenClassifier(model_path=model_path, norm=norm)
    print(f"[OK] Model '{model_path}' loaded")
    print(f"[OK] Runtime backend: {classifier.backend}")

    cap = open_camera(camera_id)
    if not cap.isOpened():
        print(f"\n[ERROR] Cannot open camera ID {camera_id}.")
        print("Check the connection / permission, or try: --camera 1\n")
        sys.exit(1)

    print(f"[INFO] Camera: {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x"
          f"{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f} @ "
          f"{cap.get(cv2.CAP_PROP_FPS):.0f} FPS")

    use_roi = default_roi
    fps, frame_count, fps_timer = 0.0, 0, time.time()
    snapshot_counter = 1

    window_name = "Deteksi Jenis Kelamin Ayam - TFLite"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARN] Failed to read frame from camera.")
                time.sleep(0.05)
                continue

            h, w, _ = frame.shape

            # Centered square ROI (70% of the shorter side)
            roi_size = int(min(h, w) * 0.70)
            x1 = (w - roi_size) // 2
            y1 = (h - roi_size) // 2
            x2, y2 = x1 + roi_size, y1 + roi_size
            roi_box = (x1, y1, x2, y2)

            input_crop = frame[y1:y2, x1:x2] if use_roi else frame
            result = classifier.predict_cv_image(input_crop)

            # FPS (updated every 0.3 s)
            frame_count += 1
            now = time.time()
            if now - fps_timer >= 0.3:
                fps = frame_count / (now - fps_timer)
                frame_count, fps_timer = 0, now

            # Keep a clean copy so screenshots have no overlay
            clean_frame = frame.copy()
            draw_ui(frame, result, roi_box, use_roi, fps)
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                print("\n[INFO] Closing...")
                break
            elif key in (ord("c"), ord("C")):
                use_roi = not use_roi
                print(f"[INFO] Mode: {'ROI center' if use_roi else 'Full frame'}")
            elif key in (ord("s"), ord("S")):
                filename = (f"capture_{result['class'].lower()}_"
                            f"{snapshot_counter:03d}.jpg")
                cv2.imwrite(filename, clean_frame)
                print(f"[OK] Saved: {filename} "
                      f"({result['class']} {result['confidence']:.1f}%)")
                snapshot_counter += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera and windows closed.")


# ── 5. CLI entry point ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Real-time chicken gender detection (TFLite float32)"
    )
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default: 0)")
    parser.add_argument("--model", type=str,
                        default="chicken_gender_augmented.tflite",
                        help="Path to float32 TFLite model")
    parser.add_argument("--full-frame", action="store_true",
                        help="Infer on the full frame instead of the center ROI")
    parser.add_argument("--norm", choices=["raw", "zero1", "neg1"], default="raw",
                        help="Input normalization: raw=0..255, zero1=0..1, "
                             "neg1=-1..1 (default: raw)")
    args = parser.parse_args()

    run_camera(
        camera_id=args.camera,
        model_path=args.model,
        default_roi=not args.full_frame,
        norm=args.norm,
    )


if __name__ == "__main__":
    main()
