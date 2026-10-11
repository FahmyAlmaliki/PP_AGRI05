"""
=============================================================================
  Program Deteksi Real-Time Kamera / Webcam Menggunakan OpenCV & TFLite
  Model: chicken_gender_augmented.tflite

  += Fitur Upload ke Web DOCX (tombol Capture & Upload di layar) =========
=============================================================================

Petunjuk Penggunaan:
  Jalankan perintah:
    python camera_tflite.py

  Opsi Tambahan:
    python camera_tflite.py --camera 0            # Ganti ID kamera (0, 1, dst)
    python camera_tflite.py --full-frame          # Deteksi seluruh layar (bukan kotak tengah)
    python camera_tflite.py --api-key docx_xxx    # API key device (atau set DOCX_API_KEY)
    python camera_tflite.py --api-url URL         # Endpoint ingest kustom
    python camera_tflite.py --no-upload           # Nonaktifkan tombol upload

Kontrol Keyboard saat Kamera Terbuka:
    - [Q] atau [ESC] : Keluar dari program
    - [S]            : Simpan foto/screenshot hasil deteksi
    - [U]            : Upload tangkapan terakhir (atau frame ini) ke web
    - [C]            : Ganti mode (Kotak Fokus Tengah <-> Seluruh Layar)

Kontrol Tombol di Layar (klik kiri):
    - [CAPTURE] : Simpan screenshot + simpan frame bersih sebagai kandidat upload
    - [UPLOAD]  : Kirim gambar ke DOCX API (label & confidence dari prediksi)

Environment:
    DOCX_API_KEY  : API key device (wajib untuk upload)
    DOCX_API_URL  : default https://docx.fahmyalmaliki.uk/api/v1/detections
=============================================================================
"""

import os
import sys
import time
import queue
import argparse
import threading
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np

try:
    import requests
except ImportError:
    requests = None

# ── Konfigurasi Upload ───────────────────────────────────────────────────────
DEFAULT_API_URL = os.environ.get(
    "DOCX_API_URL", "https://docx.fahmyalmaliki.uk/api/v1/detections"
)
PERMANENT_ERRORS = {400, 401, 403, 413, 422}
CAPTURE_DIR = Path("captures")


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


# ── 2. Uploader Background (tanpa memblokir kamera) ──────────────────────────
class DocxUploader:
    """Worker thread yang mengirim gambar ke DOCX API secara non-blocking."""

    def __init__(self, url: str, api_key: str, timeout: float = 15.0, retries: int = 3):
        self.url = url
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries
        self.enabled = bool(api_key) and requests is not None
        self.status = "Upload siap" if self.enabled else "Upload nonaktif (API key/requests)"
        self.ok = True
        self._queue: "queue.Queue" = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def submit(self, image_bgr: np.ndarray, label: str, confidence: float, captured_at: str):
        if not self.enabled:
            self.status = "Upload nonaktif: set DOCX_API_KEY"
            self.ok = False
            return

        ok, buffer = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            self.status = "Gagal encode gambar"
            self.ok = False
            return

        self.status = "Mengunggah..."
        self.ok = True
        self._queue.put((buffer.tobytes(), label, confidence, captured_at))

    def _worker(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            self._upload(*item)

    def _upload(self, data: bytes, label: str, confidence: float, captured_at: str):
        form = {"label": label}
        if confidence is not None:
            form["confidence"] = str(confidence)
        if captured_at:
            form["capturedAt"] = captured_at

        for attempt in range(1, self.retries + 1):
            try:
                response = requests.post(
                    self.url,
                    headers={"x-api-key": self.api_key},
                    data=form,
                    files={"image": ("capture.jpg", data, "image/jpeg")},
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                if attempt == self.retries:
                    self.status = f"Upload gagal (jaringan): {exc.__class__.__name__}"
                    self.ok = False
                    return
                time.sleep(min(8, 2 ** attempt))
                continue

            if response.status_code == 201:
                try:
                    det_id = response.json().get("id", "")
                except ValueError:
                    det_id = ""
                self.status = f"Upload OK: {det_id[:8]}"
                self.ok = True
                print(f"[OK] Upload sukses -> {det_id}")
                return

            if response.status_code in PERMANENT_ERRORS:
                self.status = f"Upload ditolak ({response.status_code}): {response.text[:60]}"
                self.ok = False
                print(f"[WARN] {self.status}")
                return

            # 429 / 5xx -> retry
            if attempt == self.retries:
                self.status = f"Upload gagal ({response.status_code})"
                self.ok = False
                return
            sleep_s = min(8, 2 ** attempt)
            self.status = f"Retry {attempt}/{self.retries} ({response.status_code})..."
            time.sleep(sleep_s)


# ── 3. Class Realtime Classifier ─────────────────────────────────────────────
class LiveChickenClassifier:
    CLASSES = ["Female", "Male"]

    # Warna UI BGR: Female (Merah Muda/Magenta), Male (Biru Cerah)
    COLOR_FEMALE = (180, 105, 255)   # Pink / Magenta
    COLOR_MALE = (255, 178, 50)      # Biru / Cyan
    COLOR_NEUTRAL = (200, 200, 200)

    def __init__(self, model_path: str = "chicken_gender_int8.tflite"):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file '{model_path}' tidak ditemukan!")

        self.interpreter, self.backend = load_tflite_interpreter(str(self.model_path))
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_shape = self.input_details[0]['shape']   # [1, 224, 224, 3]
        self.input_dtype = self.input_details[0]['dtype']   # float32
        self.input_scale, self.input_zero_point = \
        self.input_details[0]['quantization']

        self.output_scale, self.output_zero_point = \
            self.output_details[0]['quantization']

        print(f"[INFO] Input dtype       : {self.input_dtype}")
        print(f"[INFO] Input quant      : scale={self.input_scale}, "
            f"zero_point={self.input_zero_point}")

        print(f"[INFO] Output dtype      : {self.output_details[0]['dtype']}")
        print(f"[INFO] Output quant     : scale={self.output_scale}, "
            f"zero_point={self.output_zero_point}")
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

        # =========================
        # PREPROCESSING
        # =========================

        if self.input_dtype == np.float32:

            # Model FLOAT32
            input_data = resized.astype(np.float32)

        elif self.input_dtype == np.int8:

            # Model INT8
            if self.input_scale <= 0:
                raise ValueError(
                    f"Quantization scale tidak valid: {self.input_scale}"
                )

            input_data = resized.astype(np.float32)

            input_data = (
                input_data / self.input_scale
                + self.input_zero_point
            )

            input_data = np.round(input_data)

            input_data = np.clip(
                input_data,
                -128,
                127
            ).astype(np.int8)

        elif self.input_dtype == np.uint8:

            # Model UINT8
            if self.input_scale <= 0:
                raise ValueError(
                    f"Quantization scale tidak valid: {self.input_scale}"
                )

            input_data = resized.astype(np.float32)

            input_data = (
                input_data / self.input_scale
                + self.input_zero_point
            )

            input_data = np.round(input_data)

            input_data = np.clip(
                input_data,
                0,
                255
            ).astype(np.uint8)

        else:
            raise ValueError(
                f"Unsupported input dtype: {self.input_dtype}"
            )

        input_data = np.expand_dims(input_data, axis=0)
        # ============================================================
        # INFERENSI TFLITE
        # ============================================================
        t0 = time.perf_counter()

        # Kirim input ke model
        self.interpreter.set_tensor(
            self.input_details[0]['index'],
            input_data
        )

        # Jalankan inference
        self.interpreter.invoke()

        # Ambil output
        output_data = self.interpreter.get_tensor(
            self.output_details[0]['index']
        )

        # ============================================================
        # DEQUANTIZATION OUTPUT
        # ============================================================
        if self.output_details[0]['dtype'] == np.int8:

            output_data = (
                output_data.astype(np.float32)
                - self.output_zero_point
            ) * self.output_scale

        elif self.output_details[0]['dtype'] == np.uint8:

            output_data = (
                output_data.astype(np.float32)
                - self.output_zero_point
            ) * self.output_scale

        else:

            output_data = output_data.astype(np.float32)

        # ============================================================
        # HITUNG LATENCY
        # ============================================================
        latency_ms = (time.perf_counter() - t0) * 1000

        # ============================================================
        # AMBIL PROBABILITAS
        # ============================================================
        probabilities = output_data[0]

        # Softmax fallback jika output berupa logits
        if (
            np.any(probabilities < 0)
            or np.any(probabilities > 1)
            or abs(probabilities.sum() - 1.0) > 0.05
        ):
            exp_p = np.exp(
                probabilities - np.max(probabilities)
            )

            probabilities = exp_p / exp_p.sum()

        # ============================================================
        # HASIL PREDIKSI
        # ============================================================
        top_idx = int(np.argmax(probabilities))

        top_class = self.CLASSES[top_idx]

        confidence = float(
            probabilities[top_idx]
        ) * 100

        # ============================================================
        # RETURN
        # ============================================================
        return {
            "class": top_class,
            "confidence": confidence,
            "prob_female": float(probabilities[0]) * 100,
            "prob_male": float(probabilities[1]) * 100,
            "latency_ms": latency_ms
        }


# ── 4. Helper: Tombol di Layar ───────────────────────────────────────────────
def point_in_rect(x: int, y: int, rect) -> bool:
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def on_mouse(event, x, y, flags, state):
    """Callback klik/hover untuk tombol on-screen."""
    if event == cv2.EVENT_MOUSEMOVE:
        state["mouse"] = (x, y)
    elif event == cv2.EVENT_LBUTTONDOWN:
        for name, rect in state.get("buttons", {}).items():
            if point_in_rect(x, y, rect):
                if name == "capture":
                    state["do_capture"] = True
                elif name == "upload":
                    state["do_upload"] = True


def draw_button(frame, rect, text, hovered, enabled=True):
    x1, y1, x2, y2 = rect
    if not enabled:
        fill = (70, 70, 70)
        border = (110, 110, 110)
        text_color = (160, 160, 160)
    else:
        fill = (0, 200, 220) if hovered else (0, 160, 180)
        border = (0, 235, 255) if hovered else (0, 200, 220)
        text_color = (20, 20, 20)

    cv2.rectangle(frame, (x1, y1), (x2, y2), fill, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), border, 2)

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
    tx = x1 + (x2 - x1 - tw) // 2
    ty = y1 + (y2 - y1 + th) // 2
    cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 0.6,
                text_color, 1, cv2.LINE_AA)


# ── 5. Gambar Tampilan UI Modern pada Frame ──────────────────────────────────
def draw_ui(frame, result, roi_box, use_roi, fps, state, uploader):
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
    cv2.rectangle(overlay, (0, 0), (w, 72), (20, 20, 20), -1)
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
        frame, info_right, (w - 240, 34),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA
    )

    help_text = "[S] Simpan | [U] Upload | [C] Mode | [Q] Keluar"
    cv2.putText(
        frame, help_text, (w - 360, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA
    )

    mode_text = "Mode: Kotak Fokus" if use_roi else "Mode: Seluruh Layar"
    cv2.putText(
        frame, mode_text, (20, 62),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 230, 150), 1, cv2.LINE_AA
    )

    # 3b. Status upload
    status_color = (120, 230, 120) if uploader.ok else (90, 90, 240)
    cv2.putText(
        frame, uploader.status, (20, 92),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1, cv2.LINE_AA
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

    # 5. Tombol on-screen: [CAPTURE] [UPLOAD]
    btn_w, btn_h, gap, margin = 150, 44, 12, 20
    y1 = h - btn_h - 20
    y2 = y1 + btn_h
    upload_x2 = w - margin
    upload_x1 = upload_x2 - btn_w
    capture_x2 = upload_x1 - gap
    capture_x1 = capture_x2 - btn_w

    capture_rect = (capture_x1, y1, capture_x2, y2)
    upload_rect = (upload_x1, y1, upload_x2, y2)
    state["buttons"] = {"capture": capture_rect, "upload": upload_rect}

    mx, my = state.get("mouse", (0, 0))
    draw_button(frame, capture_rect, "CAPTURE",
                point_in_rect(mx, my, capture_rect), enabled=True)
    draw_button(frame, upload_rect, "UPLOAD",
                point_in_rect(mx, my, upload_rect), enabled=uploader.enabled)


# ── 6. Main Loop Kamera ──────────────────────────────────────────────────────
def run_camera(
    camera_id: int = 0,
    model_path: str = "chicken_gender_int8.tflite",
    default_roi: bool = True,
    api_url: str = DEFAULT_API_URL,
    api_key: str = "",
    upload_enabled: bool = True,
):
    print("\n=======================================================")
    print(" MENYIAPKAN DETEKSI KAMERA (OPENCV + TFLITE)")
    print("=======================================================")

    classifier = LiveChickenClassifier(model_path=model_path)
    print(f"[OK] Model '{model_path}' berhasil dimuat!")
    print(f"[OK] Backend Runtime : {classifier.backend}")

    uploader = DocxUploader(
        url=api_url,
        api_key=api_key if upload_enabled else "",
    )
    print(f"[OK] Endpoint upload : {api_url}")
    print(f"[INFO] Status upload : {uploader.status}")

    print(f"[OK] Membuka Kamera ID: {camera_id}...")

    # Buka webcam
    cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)

    # Gunakan MJPG agar USB bandwidth lebih ringan
    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*'MJPG')
    )

    # Resolusi
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Target 30 FPS
    cap.set(cv2.CAP_PROP_FPS, 30)

    print(f"[INFO] Camera width  : {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    print(f"[INFO] Camera height : {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"[INFO] Camera FPS    : {cap.get(cv2.CAP_PROP_FPS)}")

    if not cap.isOpened():
        print(f"\n[ERROR] Tidak dapat membuka kamera dengan ID {camera_id}.")
        print("Silakan pastikan:")
        print("  1. Webcam terhubung dan izin kamera Windows aktif.")
        print("  2. Coba ganti ID kamera: python camera_tflite.py --camera 1\n")
        sys.exit(1)

    print("\n[INFO] Kamera aktif!")
    print("Tekan [Q] pada jendela video untuk keluar.")
    print("Tekan [S] atau klik tombol CAPTURE untuk mengambil tangkapan layar.")
    print("Tekan [U] atau klik tombol UPLOAD untuk mengirim ke web DOCX.")
    print("Tekan [C] untuk berganti antara mode Kotak Tengah / Seluruh Layar.\n")

    use_roi = default_roi
    frame_count = 0
    fps = 0.0
    fps_update_interval = 0.3  # update FPS setiap 0.3 detik
    fps_timer = time.time()

    window_name = "Deteksi Jenis Kelamin Ayam - TFLite"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    state = {"mouse": (0, 0), "buttons": {}, "do_capture": False, "do_upload": False}
    cv2.setMouseCallback(window_name, on_mouse, state)

    CAPTURE_DIR.mkdir(exist_ok=True)
    snapshot_counter = 1
    last_upload_image = None      # frame bersih (tanpa overlay) untuk diupload
    last_upload_result = None     # hasil prediksi saat capture
    last_upload_time = None       # waktu capture (ISO 8601)

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

            # Salinan bersih (sebelum overlay digambar) untuk kandidat upload
            clean_input = input_crop.copy()

            # Jalankan prediksi
            result = classifier.predict_cv_image(input_crop)

            # Hitung FPS
            frame_count += 1
            now = time.time()
            if now - fps_timer >= fps_update_interval:
                fps = frame_count / (now - fps_timer)
                frame_count = 0
                fps_timer = now

            # Gambar overlay UI + tombol
            draw_ui(frame, result, roi_box, use_roi, fps, state, uploader)

            # Tampilkan frame
            cv2.imshow(window_name, frame)

            # Keyboard handler (1 ms)
            key = cv2.waitKey(1) & 0xFF

            # [Q] atau [ESC] -> Keluar
            if key == ord('q') or key == ord('Q') or key == 27:
                print("\n[INFO] Menutup program kamera...")
                break

            # [C] -> Ganti mode ROI / Layar penuh
            if key == ord('c') or key == ord('C'):
                use_roi = not use_roi
                mode_str = "Kotak Fokus Tengah" if use_roi else "Seluruh Layar"
                print(f"[INFO] Mode deteksi diubah: {mode_str}")

            # [S] -> minta capture (diproses di bawah agar frame sudah beroverlay)
            if key == ord('s') or key == ord('S'):
                state["do_capture"] = True

            # [U] -> minta upload
            if key == ord('u') or key == ord('U'):
                state["do_upload"] = True

            # ── Aksi Capture ────────────────────────────────────────────
            if state["do_capture"]:
                state["do_capture"] = False
                timestamp = datetime.now()
                filename = (
                    CAPTURE_DIR
                    / f"capture_{result['class'].lower()}_{snapshot_counter:03d}.jpg"
                )
                cv2.imwrite(str(filename), frame)
                last_upload_image = clean_input
                last_upload_result = result
                last_upload_time = timestamp.astimezone().isoformat()
                snapshot_counter += 1
                uploader.status = f"Tersimpan: {filename.name}"
                uploader.ok = True
                print(f"[OK] Tangkapan layar disimpan: {filename} "
                      f"(Prediksi: {result['class']} {result['confidence']:.1f}%)")

            # ── Aksi Upload ─────────────────────────────────────────────
            if state["do_upload"]:
                state["do_upload"] = False
                if last_upload_image is None:
                    # Belum ada capture -> upload frame ini
                    img = clean_input
                    res = result
                    captured_at = datetime.now().astimezone().isoformat()
                    print("[INFO] Belum ada capture, mengunggah frame saat ini.")
                else:
                    img = last_upload_image
                    res = last_upload_result
                    captured_at = last_upload_time

                label = "MALE" if res["class"] == "Male" else "FEMALE"
                confidence = round(res["confidence"] / 100.0, 4)
                print(f"[INFO] Mengunggah {label} (conf={confidence:.2f}) -> {api_url}")
                uploader.submit(img, label, confidence, captured_at)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Kamera dan jendela telah ditutup.")


# ── 7. CLI Entry Point ───────────────────────────────────────────────────────
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
        default="chicken_gender_int8.tflite",
        help="Path ke file model TFLite (default: chicken_gender_int8.tflite)"
    )
    parser.add_argument(
        "--full-frame",
        action="store_true",
        help="Gunakan seluruh layar untuk inferensi (default menggunakan kotak fokus tengah)"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=DEFAULT_API_URL,
        help=f"Endpoint ingest DOCX (default: {DEFAULT_API_URL})"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("DOCX_API_KEY", ""),
        help="API key device (atau set env DOCX_API_KEY)"
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Nonaktifkan tombol upload ke web"
    )

    args = parser.parse_args()
    run_camera(
        camera_id=args.camera,
        model_path=args.model,
        default_roi=not args.full_frame,
        api_url=args.api_url,
        api_key=args.api_key,
        upload_enabled=not args.no_upload,
    )


if __name__ == "__main__":
    main()
