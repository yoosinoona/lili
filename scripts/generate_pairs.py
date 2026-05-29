import cv2
import numpy as np
import os

def create_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 20, 70])
    upper = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    return mask.astype(np.float32)/255.0

def degrade(img, mask, seed):
    np.random.seed(seed)
    out = img.copy().astype(np.float32)
    # 1. Tối
    out = np.power(out/255.0, np.random.uniform(1.4, 2.2)) * 255.0
    # 2. Xỉn
    sat = np.random.uniform(0.6, 0.85)
    hsv = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_BGR2HSV)
    hsv[:,:,1] = np.clip(hsv[:,:,1]*sat, 0, 255)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)
    # 3. Mờ & Noise
    out = cv2.GaussianBlur(out, (5,5), 1.0)
    out += np.random.normal(0, 4, out.shape)
    # 4. Áp lên da
    m3 = np.stack([mask]*3, axis=2)
    out = out*m3 + img.astype(np.float32)*(1-m3)
    return np.clip(out, 0, 255).astype(np.uint8)

def main():
    pairs_dir = "data/pairs"
    train_dir = "data/training"
    os.makedirs(train_dir, exist_ok=True)
    
    files = sorted([f for f in os.listdir(pairs_dir) if f.endswith('_target.png')])
    idx = 0
    
    for f in files:
        base = f.replace('_target.png', '')
        target = cv2.imread(os.path.join(pairs_dir, f))
        mask = create_mask(target)
        target = cv2.resize(target, (128, 128))
        mask = cv2.resize(mask, (128, 128))
        
        for i in range(15): # 15 biến thể / ảnh
            inp = degrade(target, mask, idx)
            cv2.imwrite(f"{train_dir}/{idx:03d}_input.png", inp)
            cv2.imwrite(f"{train_dir}/{idx:03d}_target.png", target)
            cv2.imwrite(f"{train_dir}/{idx:03d}_mask.png", (mask*255).astype(np.uint8))
            idx += 1
    print(f"✅ Sinh xong {idx} cặp training.")

if __name__ == "__main__":
    main()
