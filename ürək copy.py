import cv2
import os
import glob

def prepare_folders():
    # Addım şəkilləri üçün qovluq
    os.makedirs("steps", exist_ok=True)

def step1_convert_to_gray(img, step_path):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"{step_path}/step1_gray.jpg", gray)
    return gray

def step2_edge_detection(gray_img, step_path):
    edges = cv2.Canny(gray_img, 100, 200)
    cv2.imwrite(f"{step_path}/step2_edges.jpg", edges)
    return edges

def step3_overlay_edges(img, edges, step_path):
    # Konturları rəngli şəkil üzərinə yerləşdir (sadə overlay)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(img, 0.7, edges_bgr, 0.3, 0)
    cv2.imwrite(f"{step_path}/step3_overlay.jpg", overlay)
    return overlay

def create_video_from_steps(step_path, output_video_path, fps=1):
    img_array = []
    files = sorted(glob.glob(f'{step_path}/*.jpg'))

    for filename in files:
        img = cv2.imread(filename)
        height, width, _ = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for img in img_array:
        out.write(img)
    out.release()
    print(f"[INFO] Video yaradıldı: {output_video_path}")

def main(image_path):
    prepare_folders()
    step_path = "steps"

    img = cv2.imread(image_path)
    if img is None:
        print("[ERROR] Şəkil tapılmadı və ya düzgün formatda deyil.")
        return

    gray = step1_convert_to_gray(img, step_path)
    edges = step2_edge_detection(gray, step_path)
    overlay = step3_overlay_edges(img, edges, step_path)

    create_video_from_steps(step_path, "drawing_steps.avi", fps=1)

if __name__ == "__main__":
    image_path = "portrait.jpg"  # Buraya çəkiləcək şəkilin adını yazın
    main(image_path)