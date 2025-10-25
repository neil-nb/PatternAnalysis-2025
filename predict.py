import torch
from dataset import create_dataloaders
from modules import ImprovedUNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64
num_classes = 6
model_path = "model.pth"

validate_images = "HipMRI_Study_open/keras_slices_data/keras_slices_validate"
validate_masks = "HipMRI_Study_open/keras_slices_data/keras_slices_seg_validate"

validation_loader = create_dataloaders(validate_images, validate_masks, batch_size, normImage=True)

net = ImprovedUNet(num_classes=num_classes, deep_supervision=False).to(device)
net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
net.eval()

if __name__ == "__main__":
    print("Hello, World!")