import torch
import torch.optim as optim
from modules import ImprovedUNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = ImprovedUNet(num_classes=2).to(device)
optimizer = optim.Adam(net.parameters(), lr=1e-3)

for epoch in range(5):
    print(f"Epoch {epoch+1}")