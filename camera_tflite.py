"""
=============================================================================
  Program Deteksi Real-Time Kamera / Webcam Menggunakan OpenCV & TFLite
  Model: chicken_gender_augmented.tflite
=============================================================================

Petunjuk Penggunaan:
  Jalankan perintah:
    python camera_tflite.py

  Opsi Tambahan:
    python camera_tflite.py --camera 0          # Ganti ID kamera (0, 1, dst)
    python camera_tflite.py --full-frame        # Deteksi seluruh layar (bukan kotak tengah)

  Kontrol Keyboard saat Kamera Terbuka:
    - [Q] atau [ESC] : Keluar dari program
    - [S]            : Simpan foto/screenshot hasil deteksi
    - [C]            : Ganti mode (Kotak Fokus Tengah <-> Seluruh Layar)
=============================================================================
"""

import os
import sys
import time
import argparse
from pathlib import Path
import cv2
import numpy as np

# ── 1. Loader Interpreter Fleksibel ──────────────────────────────────────────
def load_tflite_interpreter(model_path: str):
    """Memuat TFLite Interpreter dari library yang tersedia."""
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

    print("\n[ERROR] Library TFLite tidak ditemukan.")
    print("Silakan jalankan: pip install ai-edge-litert\n")
    sys.exit(1)


# ── 2. Class Realtime Classifier ─────────────────────────────────────────────
class LiveChickenClassifier:
    CLASSES = ["Female", "Male"]

    # Warna UI BGR: Female (Merah Muda/Magenta), Male (Biru Cerah)
    COLOR_FEMALE = (180, 105, 255)   # Pink / Magenta
    COLOR_MALE = (255, 178, 50)      # Biru / Cyan
    COLOR_NEUTRAL = (200, 200, 200)

    def __init__(self, model_path: str = "chicken_gender_augmented.tflite"):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file '{model_path}' tidak ditemukan!")

        self.interpreter, self.backend = load_tflite_interpreter(str(self.model_path))
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_shape = self.input_details[0]['shape']   # [1, 224, 224, 3]
        self.input_dtype = self.input_details[0]['dtype']   # float32
        self.input_h = int(self.input_shape[1])
        self.input_w = int(self.input_shape[2])

    def predict_cv_image(self, bgr_image: np.ndarray):
        """
        Melakukan inferensi langsung dari array BGR OpenCV.
        """
        # Konversi BGR -> RGB
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        # Resize ke (224, 224)
        resized = cv2.resize(rgb_image, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)

        # Ubah dtype sesuai input model (float32)
        input_data = np.array(resized, dtype=self.input_dtype)
        input_data = np.expand_dims(input_data, axis=0)

        # Inferensi TFLite
        t0 = time.perf_counter()
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        latency_ms = (time.perf_counter() - t0) * 1000

        probabilities = output_data[0]
        # Softmax fallback jika logits murni
        if np.any(probabilities < 0) or np.any(probabilities > 1) or abs(probabilities.sum() - 1.0) > 0.05:
            exp_p = np.exp(probabilities - np.max(probabilities))
            probabilities = exp_p / exp_p.sum()

        top_idx = int(np.argmax(probabilities))
        top_class = self.CLASSES[top_idx]
        confidence = float(probabilities[top_idx]) * 100

        return {
            "class": top_class,
            "confidence": confidence,
            "prob_female": float(probabilities[0]) * 100,
            "prob_male": float(probabilities[1]) * 100,
            "latency_ms": latency_ms
        }


# ── 3. Gambar Tampilan UI Modern pada Frame ──────────────────────────────────
def draw_ui(frame: np.ndarray, result: dict, roi_box: tuple, use_roi: bool, fps: float):
    h, w, _ = frame.shape
    cls = result["class"]
    conf = result["confidence"]
    female_p = result["prob_female"]
    male_p = result["prob_male"]
    latency = result["latency_ms"]

    primary_color = LiveChickenClassifier.COLOR_FEMALE if cls == "Female" else LiveChickenClassifier.COLOR_MALE

    # 1. Gambar Kotak ROI (Jika mode ROI aktif)
    if use_roi and roi_box is not None:
        x1, y1, x2, y2 = roi_box
        # Kotak sudut modern
        thickness = 2
        corner_len = 25
        cv2.rectangle(frame, (x1, y1), (x2, y2), primary_color, 1)

        # Sudut tebal
        t_bold = 3
        # Kiri Atas
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), primary_color, t_bold)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), primary_color, t_bold)
        # Kanan Atas
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), primary_color, t_bold)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), primary_color, t_bold)
        # Kiri Bawah
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), primary_color, t_bold)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), primary_color, t_bold)
        # Kanan Bawah
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), primary_color, t_bold)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), primary_color, t_bold)

        # Label kotak fokus
        cv2.putText(
            frame, "FOKUS OBJEK", (x1 + 10, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, primary_color, 1, cv2.LINE_AA
        )

    # 2. Header Bar Transparan
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 65), (20, 20, 20), -1)
    # Panel Bawah Transparan
    cv2.rectangle(overlay, (0, h - 80), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # 3. Teks Header & Prediksi Utama
    title_text = f"PREDIKSI: {cls.upper()} ({conf:.1f}%)"
    cv2.putText(
        frame, title_text, (20, 42),
        cv2.FONT_HERSHEY_DUPLEX, 1.0, primary_color, 2, cv2.LINE_AA
    )

    info_right = f"FPS: {fps:.1f} | Latency: {latency:.1f} ms"
    cv2.putText(
        frame, info_right, (w - 240, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA
    )

    # 4. Bar Probabilitas di Panel Bawah
    bar_width = 160
    bar_height = 14

    # --- Bar Female ---
    y_fem = h - 52
    cv2.putText(frame, "Female:", (20, y_fem + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (90, y_fem), (90 + bar_width, y_fem + bar_height), (60, 60, 60), -1)
    fill_fem = int((female_p / 100) * bar_width)
    if fill_fem > 0:
        cv2.rectangle(frame, (90, y_fem), (90 + fill_fem, y_fem + bar_height), LiveChickenClassifier.COLOR_FEMALE, -1)
    cv2.putText(frame, f"{female_p:.1f}%", (90 + bar_width + 10, y_fem + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.5, LiveChickenClassifier.COLOR_FEMALE, 1, cv2.LINE_AA)

    # --- Bar Male ---
    y_male = h - 24
    cv2.putText(frame, "Male  :", (20, y_male + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (90, y_male), (90 + bar_width, y_male + bar_height), (60, 60, 60), -1)
    fill_male = int((male_p / 100) * bar_width)
    if fill_male > 0:
        cv2.rectangle(frame, (90, y_male), (90 + fill_male, y_male + bar_height), LiveChickenClassifier.COLOR_MALE, -1)
    cv2.putText(frame, f"{male_p:.1f}%", (90 + bar_width + 10, y_male + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.5, LiveChickenClassifier.COLOR_MALE, 1, cv2.LINE_AA)

    # Petunjuk tombol di kanan bawah
    help_text = "[Q] Keluar | [S] Simpan | [C] Mode Layar"
    cv2.putText(
        frame, help_text, (w - 320, h - 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA
    )

    mode_text = "Mode: Kotak Fokus" if use_roi else "Mode: Seluruh Layar"
    cv2.putText(
        frame, mode_text, (w - 320, h - 50),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 230, 150), 1, cv2.LINE_AA
    )


# ── 4. Main Loop Kamera ──────────────────────────────────────────────────────
def run_camera(camera_id: int = 0, model_path: str = "chicken_gender_augmented.tflite", default_roi: bool = True):
    print("\n=======================================================")
    print(" MENYIAPKAN DETEKSI KAMERA (OPENCV + TFLITE)")
    print("=======================================================")

    classifier = LiveChickenClassifier(model_path=model_path)
    print(f"[OK] Model '{model_path}' berhasil dimuat!")
    print(f"[OK] Backend Runtime : {classifier.backend}")
    print(f"[OK] Membuka Kamera ID: {camera_id}...")

    # Buka webcam
    cap = cv2.VideoCapture(camera_id)

    # Set resolusi kamera yang umum (640x480 atau 1280x720)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print(f"\n[ERROR] Tidak dapat membuka kamera dengan ID {camera_id}.")
        print("Silakan pastikan:")
        print("  1. Webcam terhubung dan izin kamera Windows aktif.")
        print("  2. Coba ganti ID kamera: python camera_tflite.py --camera 1\n")
        sys.exit(1)

    print("\n[INFO] Kamera aktif!")
    print("Tekan [Q] pada jendela video untuk keluar.")
    print("Tekan [S] untuk mengambil dan menyimpan tangkapan layar.")
    print("Tekan [C] untuk berganti antara mode Kotak Tengah / Seluruh Layar.\n")

    use_roi = default_roi
    prev_time = time.time()
    fps = 0.0
    frame_count = 0
    fps_update_interval = 0.3  # update FPS setiap 0.3 detik
    fps_timer = time.time()

    window_name = "Deteksi Jenis Kelamin Ayam - TFLite"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    snapshot_counter = 1

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARN] Gagal membaca frame dari kamera.")
                time.sleep(0.05)
                continue

            h, w, _ = frame.shape

            # Hitung kotak ROI di tengah frame (persegi simetris)
            roi_size = int(min(h, w) * 0.70)
            x1 = (w - roi_size) // 2
            y1 = (h - roi_size) // 2
            x2 = x1 + roi_size
            y2 = y1 + roi_size
            roi_box = (x1, y1, x2, y2)

            # Tentukan gambar input untuk prediksi
            if use_roi:
                input_crop = frame[y1:y2, x1:x2]
            else:
                input_crop = frame

            # Jalankan prediksi
            result = classifier.predict_cv_image(input_crop)

            # Hitung FPS
            frame_count += 1
            now = time.time()
            if now - fps_timer >= fps_update_interval:
                fps = frame_count / (now - fps_timer)
                frame_count = 0
                fps_timer = now

            # Gambar overlay UI
            draw_ui(frame, result, roi_box, use_roi, fps)

            # Tampilkan frame
            cv2.imshow(window_name, frame)

            # Keyboard handler (1 ms)
            key = cv2.waitKey(1) & 0xFF

            # [Q] atau [ESC] -> Keluar
            if key == ord('q') or key == ord('Q') or key == 27:
                print("\n[INFO] Menutup program kamera...")
                break

            # [C] -> Ganti mode ROI / Layar penuh
            elif key == ord('c') or key == ord('C'):
                use_roi = not use_roi
                mode_str = "Kotak Fokus Tengah" if use_roi else "Seluruh Layar"
                print(f"[INFO] Mode deteksi diubah: {mode_str}")

            # [S] -> Simpan Screenshot
            elif key == ord('s') or key == ord('S'):
                filename = f"capture_{result['class'].lower()}_{snapshot_counter:03d}.jpg"
                cv2.imwrite(filename, frame)
                print(f"[OK] Tangkapan layar disimpan: {filename} (Prediksi: {result['class']} {result['confidence']:.1f}%)")
                snapshot_counter += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Kamera dan jendela telah ditutup.")


# ── 5. CLI Entry Point ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Program Real-Time Kamera Deteksi Jenis Kelamin Ayam (TFLite)"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="ID indeks kamera (default: 0 untuk kamera internal/webcam utama)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="chicken_gender_augmented.tflite",
        help="Path ke file model TFLite (default: chicken_gender_augmented.tflite)"
    )
    parser.add_argument(
        "--full-frame",
        action="store_true",
        help="Gunakan seluruh layar untuk inferensi (default menggunakan kotak fokus tengah)"
    )

    args = parser.parse_args()
    run_camera(
        camera_id=args.camera,
        model_path=args.model,
        default_roi=not args.full_frame
    )


if __name__ == "__main__":
    main()
