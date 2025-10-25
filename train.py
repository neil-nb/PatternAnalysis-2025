import torch
import torch.nn as nn
from modules import ImprovedUNet
from dataset import create_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 32
num_classes = 6

# Dataset paths
train_images = "HipMRI_Study_open/keras_slices_data/keras_slices_train"
train_masks = "HipMRI_Study_open/keras_slices_data/keras_slices_seg_train"
test_images = "HipMRI_Study_open/keras_slices_data/keras_slices_test"
test_masks = "HipMRI_Study_open/keras_slices_data/keras_slices_seg_test"

train_loader = create_dataloaders(train_images, train_masks, batch_size, normImage=True)
val_loader = create_dataloaders(test_images, test_masks, batch_size, normImage=True)

net = ImprovedUNet(num_classes=num_classes, base_ch=64, dropout_p=0.1, deep_supervision=True).to(device)

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

net.apply(init_weights)