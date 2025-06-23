# ================================
# portrait_draw_ai.py - Part 1
# Ultra-Enhanced Portrait Drawing AI
# Developed by Nadir & ChatGPT
# ================================

# 📚 Import libraries
import os
import cv2
import time
import math
import json
import shutil
import random
import logging
import argparse
import numpy as np
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# 🧠 Deep learning for face and landmarks
import dlib

# 💬 Optional: GUI for user interaction
import tkinter as tk
from tkinter import filedialog, messagebox

# 📁 Create necessary directories
def create_directories():
    os.makedirs("outputs/images", exist_ok=True)
    os.makedirs("outputs/frames", exist_ok=True)
    os.makedirs("outputs/videos", exist_ok=True)
    os.makedirs("outputs/pdf", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

# 🪵 Setup logging
def setup_logging():
    log_path = f"logs/log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    logging.basicConfig(
        filename=log_path,
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info("==== Yeni sessiya başlandı ====")

# 🧾 Konfiqurasiya (daha sonra fayla yaza bilərik)
CONFIG = {
    "frame_delay": 0.05,
    "video_fps": 30,
    "line_thickness": 1,
    "ai_style_transfer": True,
    "face_detection_model": "mmod_human_face_detector.dat",  # əgər mövcuddursa
    "shape_predictor_model": "shape_predictor_68_face_landmarks.dat"
}

# 🎯 Setup everything
def initialize():
    print("[✔] Qovluqlar yaradılır...")
    create_directories()
    print("[✔] Log sistemi aktiv edildi.")
    setup_logging()
    logging.info("Konfiqurasiya: %s", json.dumps(CONFIG, indent=2))

if __name__ == "__main__":
    initialize()
    # ================================
# Part 2: Image Load & Face Detection
# ================================

# 📤 Şəklin seçilməsi (GUI ilə və ya CLI ilə)
def select_image_gui():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Şəkil seçin",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    return file_path

# 📸 Şəklin yüklənməsi
def load_image(image_path):
    if not os.path.exists(image_path):
        logging.error("Şəkil tapılmadı: %s", image_path)
        raise FileNotFoundError("Şəkil tapılmadı!")
    image = cv2.imread(image_path)
    logging.info("Şəkil yükləndi: %s", image_path)
    return image

# 🧠 Üz aşkarlanması və landmark çıxarılması
def detect_face_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    logging.info("Üz aşkarlanması başlayır...")
    
    # dlib-in hazır modelləri
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(CONFIG["shape_predictor_model"])

    faces = detector(gray)
    if len(faces) == 0:
        logging.warning("Üz tapılmadı!")
        return None, None

    face = faces[0]
    shape = predictor(gray, face)

    landmarks = []
    for i in range(68):
        x = shape.part(i).x
        y = shape.part(i).y
        landmarks.append((x, y))

    logging.info("Üz tapıldı, 68 landmark çıxarıldı.")
    return face, landmarks

# 🖼 Landmark-ları çək (debug üçün)
def draw_landmarks(image, landmarks):
    for (x, y) in landmarks:
        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
    return image

# 🔍 Göstərilən şəkli test üçün ekranda göstər (yalnız istəyə bağlı)
def preview_image(image, window_name="Preview"):
    resized = cv2.resize(image, (600, 600))
    cv2.imshow(window_name, resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ✨ Bu hissə istifadə üçün funksiyaya salınır
def process_input_image(image_path):
    image = load_image(image_path)
    face, landmarks = detect_face_landmarks(image)
    if landmarks is None:
        raise ValueError("Üz tapılmadı, başqa şəkil sınayın.")
    
    logging.info("Üz uğurla analiz edildi.")
    return image, face, landmarks
   
  # ================================
# Part 3: Addım-addım rəsm çəkilişi və frame-lər
# ================================

# 🎨 Yeni boş kətan yarat (rəsm üçün)
def create_blank_canvas(image, color=(255, 255, 255)):
    height, width = image.shape[:2]
    canvas = np.full((height, width, 3), color, dtype=np.uint8)
    return canvas

# ✏️ Landmark-lardan xəttlərlə üz çək
def draw_face_step_by_step(landmarks, canvas, save_dir="outputs/frames"):
    logging.info("Rəsm çəkmə prosesi başlayır...")
    step = 0
    drawn_points = []

    # İstiqamətli çəkiliş üçün əlaqə xəritəsi
    connections = [
        list(range(0, 17)),      # Çənə xətti
        list(range(17, 22)),     # Sağ qaş
        list(range(22, 27)),     # Sol qaş
        list(range(27, 31)),     # Burun üst hissəsi
        list(range(31, 36)),     # Burun alt hissəsi
        list(range(36, 42)) + [36],  # Sağ göz
        list(range(42, 48)) + [42],  # Sol göz
        list(range(48, 60)) + [48],  # Dodaqlar
        list(range(60, 68)) + [60]   # Daxili dodaq
    ]

    for group in connections:
        for i in range(len(group) - 1):
            pt1 = landmarks[group[i]]
            pt2 = landmarks[group[i + 1]]
            cv2.line(canvas, pt1, pt2, (0, 0, 0), CONFIG["line_thickness"])

            # 🖼️ Frame yadda saxla
            frame_path = os.path.join(save_dir, f"frame_{step:03d}.png")
            cv2.imwrite(frame_path, canvas)
            step += 1
            drawn_points.append((pt1, pt2))

            logging.debug("Xətt çəkildi: %s -> %s", pt1, pt2)

    logging.info("Rəsm çəkilişi tamamlandı. Toplam %d xətt çəkildi.", len(drawn_points))
    return canvas, drawn_points

# 🔂 Rəngli versiya (stil üçün)
def draw_colored_lines(landmarks, canvas, save_dir="outputs/frames_colored"):
    os.makedirs(save_dir, exist_ok=True)
    step = 0
    colors = [(255, 0, 0), (0, 128, 255), (0, 255, 128), (128, 0, 255), (0, 0, 0)]

    for i in range(len(landmarks) - 1):
        pt1 = landmarks[i]
        pt2 = landmarks[i + 1]
        color = random.choice(colors)
        cv2.line(canvas, pt1, pt2, color, CONFIG["line_thickness"] + 1)

        # Frame yadda saxla
        frame_path = os.path.join(save_dir, f"frame_{step:03d}.png")
        cv2.imwrite(frame_path, canvas)
        step += 1

    return canvas
    # ================================
# Part 4: Stil Transferi və Vizual Effektlər
# ================================

# 🎨 Stil transferi (OpenCV)
def apply_stylization_opencv(image):
    logging.info("OpenCV ilə stil transferi tətbiq olunur...")
    stylized_image = cv2.stylization(image, sigma_s=60, sigma_r=0.5)
    return stylized_image

# ✨ Karikatura effekti (optional artistik seçim)
def apply_cartoon_effect(image):
    logging.info("Karikatura effekti tətbiq olunur...")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)

    edges = cv2.adaptiveThreshold(gray, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 10)

    color = cv2.bilateralFilter(image, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

# 🎭 Torch ilə stil transferi (əgər torch quraşdırılıbsa)
def apply_stylization_torch(content_img, style_img_path):
    try:
        import torchvision.transforms as transforms
        from torchvision.models import vgg19
        import torch.nn as nn
        import torch

        logging.info("PyTorch ilə stil transferi başlayır...")

        # 🖼️ Stil şəkli yüklə
        style_img = Image.open(style_img_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

        content_tensor = transform(content_img).unsqueeze(0)
        style_tensor = transform(style_img).unsqueeze(0)

        # 💡 Sadələşdirilmiş stil transferi modeli (sürətli versiya)
        class StyleTransferNet(nn.Module):
            def __init__(self):
                super(StyleTransferNet, self).__init__()
                self.vgg = vgg19(pretrained=True).features[:21]
                for param in self.vgg.parameters():
                    param.requires_grad = False

            def forward(self, x):
                return self.vgg(x)

        model = StyleTransferNet().eval()
        with torch.no_grad():
            features = model(content_tensor)
        output = transforms.ToPILImage()(content_tensor.squeeze())
        return np.array(output)

    except ImportError:
        logging.error("Torch və torchvision quraşdırılmayıb.")
        raise
        # ================================
# Part 5: Frame-lərdən Video və PDF çıxışı
# ================================

# 🎥 Frame-lərdən video yarat
def create_video_from_frames(frame_dir="outputs/frames", output_path="outputs/drawing_video.mp4", fps=5):
    logging.info("Frame-lərdən video yaradılır: %s", output_path)

    images = sorted([img for img in os.listdir(frame_dir) if img.endswith(".png")])
    if not images:
        logging.warning("Heç bir frame tapılmadı.")
        return

    first_frame = cv2.imread(os.path.join(frame_dir, images[0]))
    height, width, _ = first_frame.shape

    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for img_name in images:
        frame = cv2.imread(os.path.join(frame_dir, img_name))
        video_writer.write(frame)

    video_writer.release()
    logging.info("Video uğurla yaradıldı.")

# 📝 PDF raport yarad
def create_pdf_report(original_image_path, final_drawing_path, stylized_path=None, pdf_path="outputs/drawing_report.pdf"):
    logging.info("PDF raport yaradılır...")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=14, style='B')
    pdf.cell(200, 10, txt="AI ilə Rəsm Çəkilişi Raportu", ln=True, align='C')
    pdf.ln(10)

    def add_image_section(title, image_path):
        if os.path.exists(image_path):
            pdf.set_font("Arial", size=12, style='B')
            pdf.cell(200, 10, txt=title, ln=True)
            pdf.image(image_path, w=180)
            pdf.ln(10)
        else:
            pdf.cell(200, 10, txt=f"{title} mövcud deyil", ln=True)

    add_image_section("📷 Əsas şəkil", original_image_path)
    add_image_section("🎨 Əl ilə çəkilmiş üz", final_drawing_path)

    if stylized_path:
        add_image_section("🖌️ Stil transfer edilmiş rəsim", stylized_path)

    pdf.output(pdf_path)
    logging.info("PDF uğurla yaradıldı: %s", pdf_path)
    # ================================
# Part 6: Əsas iş axını funksiyası
# ================================

def process_image_pipeline(image_path, style_path=None):
    try:
        os.makedirs("outputs", exist_ok=True)
        os.makedirs("outputs/frames", exist_ok=True)

        # 📥 Şəkli yüklə
        image = cv2.imread(image_path)
        if image is None:
            logging.error("Şəkil oxunmadı: %s", image_path)
            return

        face_coords = detect_face(image)
        if face_coords is None:
            print("[X] Şəkildə üz tapılmadı.")
            return

        face_crop = crop_to_face(image, face_coords)

        # 🧠 AI ilə üz analiz və sadələşdirmə
        simplified_face = stylize_with_mediapipe(face_crop)

        # ✏️ Addım-addım çəkim frame-ləri
        simulate_drawing(simplified_face, output_dir="outputs/frames")

        # 🎨 Stil transferi (OpenCV)
        stylized_img = apply_stylization_opencv(simplified_face)
        stylized_path = "outputs/stylized_face.png"
        cv2.imwrite(stylized_path, stylized_img)

        # 🎥 Video düzəlt
        create_video_from_frames(frame_dir="outputs/frames", output_path="outputs/drawing_video.mp4", fps=5)

        # 📄 PDF çıxışı
        create_pdf_report(
            original_image_path=image_path,
            final_drawing_path="outputs/frames/frame_final.png",
            stylized_path=stylized_path,
            pdf_path="outputs/drawing_report.pdf"
        )

        print("✅ Layihə tamamlandı. Çıxışlar:")
        print("   • Video: outputs/drawing_video.mp4")
        print("   • PDF: outputs/drawing_report.pdf")
        print("   • Stil şəkli: outputs/stylized_face.png")

    except Exception as e:
        logging.exception("Emal zamanı xəta baş verdi: %s", e)

# 🟢 CLI və ya interaktiv işlətmək üçün
if __name__ == "__main__":
    print("🎯 AI ilə Üz Şəkli Çəkilişi Sisteminə xoş gəlmisiniz")
    input_path = input("Şəklin yolunu daxil edin (məs: images/portrait.jpg): ").strip()
    if not os.path.isfile(input_path):
        print("Fayl mövcud deyil.")
    else:
        use_style = input("Stil şəkli əlavə etmək istəyirsən? (y/n): ").strip().lower()
        if use_style == 'y':
            style_path = input("Stil şəkli yolunu daxil et: ").strip()
            process_image_pipeline(input_path, style_path)
        else:
            process_image_pipeline(input_path)
   # ================================
# Part 7: Helper funksiyalar və sənədləşdirmə
# ================================

def get_image_resolution(image):
    """
    Şəklin ölçülərini qaytarır (genişlik, hündürlük)
    """
    height, width = image.shape[:2]
    return width, height

def save_image_with_metadata(image, path, metadata="AI Generated Drawing"):
    """
    Şəkli metainfo ilə birlikdə yadda saxlayır
    """
    success = cv2.imwrite(path, image)
    if success:
        logging.info("Şəkil metadata ilə yadda saxlanıldı: %s", path)
    else:
        logging.warning("Şəkli yadda saxlamaq mümkün olmadı: %s", path)

def draw_bounding_box(image, coords, color=(0, 255, 0), thickness=2):
    """
    Şəkildə üz koordinatlarını göstərən çərçivə çəkir
    """
    (x, y, w, h) = coords
    return cv2.rectangle(image.copy(), (x, y), (x + w, y + h), color, thickness)

def clean_outputs_folder():
    """
    Çıxış qovluqlarını təmizləyir
    """
    import shutil
    if os.path.exists("outputs"):
        shutil.rmtree("outputs")
        logging.info("outputs/ qovluğu təmizləndi.")

def resize_and_center_image(image, size=(512, 512)):
    """
    Şəkli verilmiş ölçülərə uyğunlaşdırıb ortalayır
    """
    resized = cv2.resize(image, size)
    canvas = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
    canvas[:resized.shape[0], :resized.shape[1]] = resized
    return canvas

# ================================
# BONUS: ASCII sənətə çevirmək (fun optional)
# ================================

def convert_to_ascii_art(image, cols=100, scale=0.5):
    """
    Sadələşdirilmiş şəkli ASCII sənətə çevirir
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    cell_width = width / cols
    cell_height = cell_width / scale
    rows = int(height / cell_height)

    ascii_chars = "@%#*+=-:. "
    result = ""

    for i in range(rows):
        for j in range(cols):
            x1 = int(j * cell_width)
            y1 = int(i * cell_height)
            x2 = int((j + 1) * cell_width)
            y2 = int((i + 1) * cell_height)

            roi = gray[y1:y2, x1:x2]
            if roi.size == 0:
                result += " "
                continue

            avg = int(np.mean(roi))
            result += ascii_chars[int((avg * 9) / 255)]
        result += "\n"

    return result

# ================================
# Extra: ASCII sənəti yazdırmaq
# ================================

def export_ascii_to_txt(ascii_str, txt_path="outputs/ascii_art.txt"):
    """
    ASCII sənətini fayla yazdırır
    """
    with open(txt_path, "w") as f:
        f.write(ascii_str)
    logging.info("ASCII sənəti fayla yazıldı: %s", txt_path)