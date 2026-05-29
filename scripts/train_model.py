# ... (giữ nguyên phần class SkinDataset và TinySkinNet ở trên) ...

def train():
    device = torch.device('cpu')
    ds = SkinDataset("data/training")
    
    if len(ds) == 0:
        print("❌ DỮ LIỆU RỖNG: Không có ảnh để train.")
        print("👉 Kiểm tra lại bước 1 và 2.")
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
    
    torch.save(model.state_dict(), "best_model.pth")
    print("✅ Train xong!")

if __name__ == "__main__":
    train()
