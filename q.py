"""
Ultra Ətraflı AI əsaslı Portret Şəkli Çəkən və Addım-addım Video Yaradıcı
Yaradıcı: ChatGPT + Nadir Layihəsi
Təsviredici: Verilən portret şəklini analiz edib, üz hissələrini tanıyaraq addım-addım çəkir,
sonda isə bu mərhələləri birləşdirərək video yaradır.
"""

# == 1. Kitabxanaların daxil edilməsi ==
import os
import cv2
import dlib
import uuid
import time
import numpy as np
import face_recognition
from PIL import Image, ImageDraw
from moviepy.editor import ImageSequenceClip
import matplotlib.pyplot as plt

# == 2. Fayl və qovluq strukturu ==
OUTPUT_DIR = "portrait_drawing_output"
FRAMES_DIR = os.path.join(OUTPUT_DIR, "frames")
FINAL_IMAGE_PATH = os.path.join(OUTPUT_DIR, "final.png")
VIDEO_PATH = os.path.join(OUTPUT_DIR, "drawing_timelapse.mp4")
LOG_PATH = os.path.join(OUTPUT_DIR, "draw_log.txt")

os.makedirs(FRAMES_DIR, exist_ok=True)

# == 3. Giriş şəklinin yüklənməsi ==
def load_input_image(image_path):
    image = face_recognition.load_image_file(image_path)
    return image

# == 4. Üz aşkarlama və landmark çıxarılması ==
def detect_face_landmarks(image):
    face_landmarks_list = face_recognition.face_landmarks(image)
    if not face_landmarks_list:
        raise ValueError("Üz aşkarlanmadı!")
    return face_landmarks_list[0]

# == 5. Çəkiliş mərhələlərinin tərifi ==
DRAW_ORDER = [
    "chin", "left_eyebrow", "right_eyebrow", "nose_bridge", "nose_tip",
    "left_eye", "right_eye", "top_lip", "bottom_lip"
]

# == 6. Boş kətan yaradılması ==
def create_blank_canvas(image):
    h, w = image.shape[:2]
    return Image.new("RGB", (w, h), "white")

# == 7. Addım-addım çəkiliş funksiyası ==
def draw_landmarks_step_by_step(landmarks, canvas, save_dir):
    draw = ImageDraw.Draw(canvas)
    frame_paths = []
    with open(LOG_PATH, "w") as log_file:
        for part in DRAW_ORDER:
            if part in landmarks:
                points = landmarks[part]
                for i in range(1, len(points)):
                    draw.line([points[i-1], points[i]], fill="black", width=2)
                    frame_path = os.path.join(save_dir, f"frame_{uuid.uuid4().hex[:8]}.png")
                    canvas.save(frame_path)
                    frame_paths.append(frame_path)
                    log_file.write(f"Drawn {part} point {i}\n")
    return frame_paths, canvas

# == 8. Video yaradılması ==
def create_timelapse_video(frame_paths, output_path, fps=5):
    clip = ImageSequenceClip(frame_paths, fps=fps)
    clip.write_videofile(output_path, codec="libx264")

# == 9. Əsas funksiya ==
def main(image_path):
    print("Şəkil yüklənir...")
    image = load_input_image(image_path)

    print("Üz tanınır...")
    landmarks = detect_face_landmarks(image)

    print("Boş kətan yaradılır...")
    canvas = create_blank_canvas(image)

    print("Addım-addım çəkilir...")
    frame_paths, final_image = draw_landmarks_step_by_step(landmarks, canvas, FRAMES_DIR)
    final_image.save(FINAL_IMAGE_PATH)

    print("Video yaradılır...")
    create_timelapse_video(frame_paths, VIDEO_PATH)

    print("Tamamlandı!")
    print(f"Yekun şəkil: {FINAL_IMAGE_PATH}")
    print(f"Video: {VIDEO_PATH}")
    print(f"Log: {LOG_PATH}")

# == 10. İşə salma ==
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI əsaslı portret çəkilişi")
    parser.add_argument("--image", type=str, required=True, help="Portret şəklinin yolu")
    args = parser.parse_args()

    main(args.image)
