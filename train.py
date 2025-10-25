import torch
import torch.optim as optim
from modules import ImprovedUNet
from dataset import create_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 32

# Dataset paths
train_images = "HipMRI_Study_open/keras_slices_data/keras_slices_train"
train_masks = "HipMRI_Study_open/keras_slices_data/keras_slices_seg_train"
test_images = "HipMRI_Study_open/keras_slices_data/keras_slices_test"
test_masks = "HipMRI_Study_open/keras_slices_data/keras_slices_seg_test"

train_loader = create_dataloaders(train_images, train_masks, batch_size, normImage=True)
val_loader = create_dataloaders(test_images, test_masks, batch_size, normImage=True)

net = ImprovedUNet(num_classes=2).to(device)
optimizer = optim.Adam(net.parameters(), lr=1e-3)

for epoch in range(5):
    print(f"Epoch {epoch+1}")