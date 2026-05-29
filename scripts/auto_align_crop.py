import cv2
import os
import numpy as np

def smart_crop_face(input_path, output_path, target_size=512):
    img = cv2.imread(input_path)
    if img is None: return False
    
    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Haar Cascade
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    
    if len(faces) > 0:
        x, y, fw, fh = max(faces, key=lambda f: f[2]*f[3])
        cx, cy = x + fw//2, y + fh//2
        box = int(max(fw, fh) * 1.6) # Padding 60%
        x1, y1 = max(0, cx - box//2), max(0, cy - box//2)
        x2, y2 = min(w, cx + box//2), min(h, cy + box//2)
        crop = img[y1:y2, x1:x2]
    else:
        size = min(h, w)
        x1, y1 = (w-size)//2, (h-size)//2
        crop = img[y1:y1+size, x1:x1+size]
        
    crop = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, crop)
    return True

def main():
    input_dir = "data/raw_inputs"
    target_dir = "data/raw_targets"
    final_dir = "data/pairs"
    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
    
    print(f"🔍 Đang xử lý {len(files)} cặp ảnh...")
    for idx, fname in enumerate(files):
        if os.path.exists(os.path.join(target_dir, fname)):
            new_name = f"{idx:03d}"
            smart_crop_face(os.path.join(input_dir, fname), os.path.join(final_dir, f"{new_name}_input.png"))
            smart_crop_face(os.path.join(target_dir, fname), os.path.join(final_dir, f"{new_name}_target.png"))
            print(f"✅ Đã cắt cặp {idx}")
    print("🎉 Crop hoàn tất.")

if __name__ == "__main__":
    main()
