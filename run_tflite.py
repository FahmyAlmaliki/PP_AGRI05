"""
=============================================================================
  Program Inferensi / Running Model TFLite - Deteksi Jenis Kelamin Ayam
  Model: chicken_gender_augmented.tflite
=============================================================================

Penggunaan:
  1. Uji satu gambar:
     python run_tflite.py --image "Female/001.jpg"

  2. Uji seluruh gambar dalam satu folder:
     python run_tflite.py --dir "augmented_dataset/Male"

  3. Mode Demo / Cek Otomatis (menguji sampel yang ada):
     python run_tflite.py

  4. Mode Benchmark Kecepatan (FPS & Latency):
     python run_tflite.py --benchmark
=============================================================================
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

# ── 1. Loader Interpreter Fleksibel ──────────────────────────────────────────
# Mendukung: ai-edge-litert (Google LiteRT), tflite-runtime, atau tensorflow
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

    print("\n[ERROR] Tidak ditemukan library untuk menjalankan TFLite.")
    print("Silakan install salah satu library berikut:")
    print("  pip install ai-edge-litert    (Rekomendasi untuk LiteRT / Python 3.14)")
    print("  pip install tflite-runtime    (Untuk perangkat embedded / Raspberry Pi)")
    print("  pip install tensorflow        (Untuk environment lengkap)\n")
    sys.exit(1)


# ── 2. Class Predictor ───────────────────────────────────────────────────────
class ChickenGenderClassifier:
    DEFAULT_CLASSES = ["Female", "Male"]

    def __init__(self, model_path: str = "chicken_gender_augmented.tflite", class_names=None):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"File model '{model_path}' tidak ditemukan! Pastikan file .tflite ada di direktori yang benar."
            )

        self.class_names = class_names or self.DEFAULT_CLASSES
        self.interpreter, self.backend_name = load_tflite_interpreter(str(self.model_path))
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Ambil input shape & type
        self.input_shape = self.input_details[0]['shape']       # e.g. [1, 224, 224, 3]
        self.input_dtype = self.input_details[0]['dtype']       # e.g. float32
        self.img_height = int(self.input_shape[1])
        self.img_width = int(self.input_shape[2])

    def preprocess_image(self, img_input):
        """
        Preprocess gambar:
        - Resize ke ukuran input model (224x224)
        - Format RGB
        - Convert ke float32 [0.0 - 255.0] (EfficientNet memiliki rescale internal)
        """
        if isinstance(img_input, (str, Path)):
            img = Image.open(img_input).convert('RGB')
        elif isinstance(img_input, Image.Image):
            img = img_input.convert('RGB')
        else:
            raise ValueError("Input harus berupa path gambar (str/Path) atau objek PIL.Image")

        # Resize gambar sesuai input model
        img_resized = img.resize((self.img_width, self.img_height), Image.Resampling.BILINEAR)

        # Ubah ke numpy array
        img_array = np.array(img_resized, dtype=self.input_dtype)

        # Tambahkan dimensi batch -> (1, 224, 224, 3)
        img_batch = np.expand_dims(img_array, axis=0)
        return img_batch

    def predict(self, img_input):
        """
        Menjalankan prediksi pada 1 gambar.
        Mengembalikan dictionary berisi kelas terprediksi, confidence, dan rincian probabilitas.
        """
        input_data = self.preprocess_image(img_input)

        start_time = time.perf_counter()
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        latency_ms = (time.perf_counter() - start_time) * 1000

        probabilities = output_data[0]
        # Jika output bukan probabilitas (misal logits dengan nilai di luar [0, 1]), terapkan softmax
        if np.any(probabilities < 0) or np.any(probabilities > 1) or abs(probabilities.sum() - 1.0) > 0.05:
            exp_p = np.exp(probabilities - np.max(probabilities))
            probabilities = exp_p / exp_p.sum()

        top_index = int(np.argmax(probabilities))
        top_class = self.class_names[top_index]
        confidence = float(probabilities[top_index]) * 100

        prob_dict = {
            self.class_names[i]: float(probabilities[i]) * 100
            for i in range(len(self.class_names))
        }

        return {
            "class": top_class,
            "confidence": confidence,
            "probabilities": prob_dict,
            "latency_ms": latency_ms
        }


# ── 3. Fungsi Tampilan Hasil ─────────────────────────────────────────────────
def print_result_card(img_path: str, result: dict):
    """Mencetak kartu hasil prediksi dengan tampilan rapi."""
    cls = result["class"]
    conf = result["confidence"]
    latency = result["latency_ms"]

    icon = "[FEMALE]" if cls == "Female" else "[MALE]"
    color_bar = "=" * 55

    print(f"\n{color_bar}")
    print(f" Gambar      : {img_path}")
    print(f" Prediksi    : {icon} {cls.upper()}")
    print(f" Keyakinan   : {conf:.2f}%")
    print(f" Latensi     : {latency:.2f} ms")
    print(f"{'-' * 55}")
    print(" Rincian Probabilitas:")
    for cname, prob in result["probabilities"].items():
        bar_len = int(prob / 2.5)  # bar panjang maks 40 char
        bar_visual = "#" * bar_len + "-" * (40 - bar_len)
        print(f"   - {cname:7s} : [{bar_visual}] {prob:6.2f}%")
    print(f"{color_bar}\n")


# ── 4. Fitur Pengujian Direktori / Batch ──────────────────────────────────────
def predict_directory(classifier: ChickenGenderClassifier, dir_path: str):
    """Memproses semua file gambar di dalam direktori."""
    dir_p = Path(dir_path)
    if not dir_p.is_dir():
        print(f"[ERROR] Folder '{dir_path}' tidak ditemukan!")
        return

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    img_files = [f for f in dir_p.iterdir() if f.suffix.lower() in exts]

    if not img_files:
        print(f"[INFO] Tidak ditemukan file gambar di folder '{dir_path}'.")
        return

    print(f"\nMemproses {len(img_files)} gambar dari folder: {dir_path}")
    print(f"{'-' * 70}")
    print(f"{'No':<4} {'File':<30} {'Prediksi':<10} {'Confidence':<12} {'Latensi':<10}")
    print(f"{'-' * 70}")

    stats = {"Female": 0, "Male": 0}
    total_time = 0.0

    for idx, fpath in enumerate(img_files, 1):
        res = classifier.predict(fpath)
        stats[res["class"]] += 1
        total_time += res["latency_ms"]
        print(
            f"{idx:<4} {fpath.name[:28]:<30} {res['class']:<10} {res['confidence']:6.2f}%     {res['latency_ms']:6.2f} ms"
        )

    print(f"{'=' * 70}")
    print(f" Ringkasan Pengujian:")
    print(f"   Total gambar   : {len(img_files)}")
    print(f"   Prediksi Female: {stats['Female']} ({stats['Female']/len(img_files)*100:.1f}%)")
    print(f"   Prediksi Male  : {stats['Male']} ({stats['Male']/len(img_files)*100:.1f}%)")
    print(f"   Rata-rata waktu: {total_time/len(img_files):.2f} ms per gambar")
    print(f"{'=' * 70}\n")


# ── 5. Fitur Benchmark Kecepatan (FPS & Latency) ──────────────────────────────
def benchmark_model(classifier: ChickenGenderClassifier, iterations: int = 50):
    """Mengukur kecepatan inferensi model (FPS & Latensi rata-rata)."""
    dummy_input = np.random.uniform(0.0, 255.0, size=(1, 224, 224, 3)).astype(np.float32)

    # Warmup
    for _ in range(5):
        classifier.interpreter.set_tensor(classifier.input_details[0]['index'], dummy_input)
        classifier.interpreter.invoke()

    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        classifier.interpreter.set_tensor(classifier.input_details[0]['index'], dummy_input)
        classifier.interpreter.invoke()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    avg_lat = np.mean(latencies)
    min_lat = np.min(latencies)
    max_lat = np.max(latencies)
    fps = 1000.0 / avg_lat

    print("\n" + "=" * 50)
    print(" HASIL BENCHMARK KECEPATAN TFLITE")
    print("=" * 50)
    print(f" Backend Engine : {classifier.backend_name}")
    print(f" Model Path     : {classifier.model_path.name}")
    print(f" Input Shape    : {classifier.input_shape}")
    print(f" Jumlah Iterasi : {iterations}")
    print(f" Rata-rata      : {avg_lat:.2f} ms")
    print(f" Minimum        : {min_lat:.2f} ms")
    print(f" Maksimum       : {max_lat:.2f} ms")
    print(f" Throughput     : {fps:.1f} FPS (Frame per Detik)")
    print("=" * 50 + "\n")


# ── 6. Mode Demo Otomatis ────────────────────────────────────────────────────
def run_default_demo(classifier: ChickenGenderClassifier):
    """Menjalankan demo otomatis dengan sampel yang ada di folder dataset."""
    print("\n[MODE DEMO] Menguji sampel gambar otomatis dari direktori dataset...")

    test_samples = [
        "augmented_dataset/Female/Female_aug_0001.jpg",
        "augmented_dataset/Male/Male_aug_0001.jpg",
        "Female/Female_001.jpg",
        "Male/Male_001.jpg"
    ]

    found = 0
    for sample in test_samples:
        if os.path.exists(sample):
            res = classifier.predict(sample)
            print_result_card(sample, res)
            found += 1

    if found == 0:
        # Cari file gambar apa saja di workspace
        for folder in ["Female", "Male", "augmented_dataset/Female", "augmented_dataset/Male"]:
            p = Path(folder)
            if p.is_dir():
                for f in p.glob("*.jpg"):
                    res = classifier.predict(f)
                    print_result_card(str(f), res)
                    found += 1
                    break
            if found >= 2:
                break

    print("Tip: Gunakan opsi argumen:")
    print("  python run_tflite.py --image <path_ke_file_gambar>")
    print("  python run_tflite.py --dir <path_ke_folder>")
    print("  python run_tflite.py --benchmark\n")


# ── Main Entry Point ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Program Inferensi Model TFLite Klasifikasi Jenis Kelamin Ayam"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="chicken_gender_augmented.tflite",
        help="Path ke file model .tflite (default: chicken_gender_augmented.tflite)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path ke file gambar tunggal yang ingin diprediksi"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Path ke folder berisi gambar untuk diproses batch"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Jalankan pengujian kecepatan (latensi & FPS)"
    )
    parser.add_argument(
        "--camera",
        nargs="?",
        const=0,
        type=int,
        default=None,
        help="Jalankan mode kamera real-time dengan OpenCV (default camera ID: 0)"
    )

    args = parser.parse_args()

    # Jika mode kamera dipilih
    if args.camera is not None:
        try:
            from camera_tflite import run_camera
            run_camera(camera_id=args.camera, model_path=args.model)
            return
        except ImportError:
            print("[ERROR] File 'camera_tflite.py' tidak ditemukan atau opencv-python belum terpasang.")
            sys.exit(1)

    # Inisialisasi model
    try:
        classifier = ChickenGenderClassifier(model_path=args.model)
        print(f"\n[OK] Model '{args.model}' berhasil dimuat!")
        print(f"[OK] Backend runtime: {classifier.backend_name}")
    except Exception as e:
        print(f"\n[ERROR] Gagal memuat model: {e}")
        sys.exit(1)

    # Eksekusi sesuai argumen
    if args.benchmark:
        benchmark_model(classifier)
    elif args.image:
        if not os.path.exists(args.image):
            print(f"[ERROR] File '{args.image}' tidak ditemukan!")
            sys.exit(1)
        res = classifier.predict(args.image)
        print_result_card(args.image, res)
    elif args.dir:
        predict_directory(classifier, args.dir)
    else:
        run_default_demo(classifier)


if __name__ == "__main__":
    main()
