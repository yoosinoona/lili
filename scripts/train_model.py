import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os

class SkinDataset(Dataset):
    def __init__(self, dir):
        self.dir = dir
        self.files = sorted([f.replace('_input.png','') for f in os.listdir(dir) if f.endswith('_input.png')])
    
    def __len__(self): return len(self.files)
    
    def __getitem__(self, i):
        inp = cv2.imread(f"{self.dir}/{self.files[i]}_input.png")
        tgt = cv2.imread(f"{self.dir}/{self.files[i]}_target.png")
        msk = cv2.imread(f"{self.dir}/{self.files[i]}_mask.png", 0)
        
        # Convert to Tensor
        x = torch.from_numpy(inp).float().permute(2,0,1)/255.0
        y = torch.from_numpy(tgt).float().permute(2,0,1)/255.0
        m = torch.from_numpy(msk).float().unsqueeze(0)/255.0
        
        # Ghép Mask vào Input (thành 4 kênh)
        x = torch.cat([x, m], dim=0)
        return x, y, m

class TinySkinNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, padding=1), nn.LeakyReLU(0.2)
        )
        # Head 1: Màu sắc (Delta RGB)
        self.head_color = nn.Sequential(
            nn.Conv2d(32, 16, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(16, 3, 1), nn.Tanh() 
        )
        # Head 2: Độ mịn (Weight)
        self.head_weight = nn.Sequential(
            nn.Conv2d(32, 8, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(8, 1, 1), nn.Sigmoid()
        )
    
    def forward(self, x):
        feat = self.enc(x)
        return self.head_color(feat), self.head_weight(feat)

def train():
    device = torch.device('cpu')
    ds = SkinDataset("data/training")
    dl = DataLoader(ds, batch_size=4, shuffle=True)
    
    model = TinySkinNet().to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    crit = nn.L1Loss()
    
    print(f"🚀 Bắt đầu train {len(ds)} samples...")
    for epoch in range(25):
        model.train()
        total_loss = 0
        for x, y, m in dl:
            x, y, m = x.to(device), y.to(device), m.to(device)
            delta, weight = model(x)
            
            # Công thức: Output = Input + Delta * Weight
            out = x[:, :3] + delta * weight
            out = torch.clamp(out, 0, 1)
            
            # Loss chỉ tính trên vùng da (mask)
            loss = crit(out * m, y * m)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/25 - Loss: {total_loss/len(dl):.5f}")
    
    torch.save(model.state_dict(), "best_model.pth")
    print("✅ Train xong! Đã lưu best_model.pth")

if __name__ == "__main__":
    train()
