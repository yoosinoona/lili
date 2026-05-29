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
        
        x = torch.from_numpy(inp).float().permute(2,0,1)/255.0
        y = torch.from_numpy(tgt).float().permute(2,0,1)/255.0
        m = torch.from_numpy(msk).float().unsqueeze(0)/255.0
        x = torch.cat([x, m], dim=0)
        return x, y, m

class TinySkinNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, padding=1), nn.LeakyReLU(0.2)
        )
        self.head_color = nn.Sequential(nn.Conv2d(32, 16, 1), nn.LeakyReLU(0.2), nn.Conv2d(16, 3, 1), nn.Tanh())
        self.head_weight = nn.Sequential(nn.Conv2d(32, 8, 1), nn.LeakyReLU(0.2), nn.Conv2d(8, 1, 1), nn.Sigmoid())
    
    def forward(self, x):
        feat = self.enc(x)
        return self.head_color(feat), self.head_weight(feat)

def train():
    device = torch.device('cpu')
    ds = SkinDataset("data/training")
    
    if len(ds) == 0:
        print("❌ DỮ LIỆU RỖNG: Không có ảnh để train.")
        return

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
            out = x[:, :3] + delta * weight
            out = torch.clamp(out, 0, 1)
            loss = crit(out * m, y * m)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/25 - Loss: {total_loss/len(dl):.5f}")
    
    # Lưu weights chuẩn
    torch.save(model.state_dict(), "best_model.pth")
    
    # ✅ Export TorchScript cho PNNX (quan trọng)
    dummy = torch.randn(1, 4, 128, 128)
    traced = torch.jit.trace(model, dummy)
    traced.save("best_model.pt")
    print("✅ Train xong! Đã lưu best_model.pth & best_model.pt")

if __name__ == "__main__":
    train()
