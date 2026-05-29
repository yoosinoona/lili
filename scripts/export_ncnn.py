import torch
import torch.nn as nn

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
        f = self.enc(x)
        return self.head_color(f), self.head_weight(f)

def export():
    model = TinySkinNet()
    model.load_state_dict(torch.load("best_model.pth", map_location='cpu'))
    model.eval()
    
    # Export ONNX
    torch.onnx.export(model, torch.randn(1, 4, 128, 128), "skin_enhancer.onnx", 
                      input_names=['input'], output_names=['delta', 'weight'], opset_version=12)
    print("✅ Đã xuất skin_enhancer.onnx")

if __name__ == "__main__":
    export()
