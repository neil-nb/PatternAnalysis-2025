import torch
from dataset import create_dataloaders
from modules import ImprovedUNet
import torch.nn.functional as F

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

def dice_loss_per_class(pred, target, num_classes, smooth=1):
    pred = F.softmax(pred, dim=1)
    target = target.squeeze(dim=1) if target.dim() == 4 else target
    target_flat = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).contiguous().view(pred.shape[0], num_classes, -1)
    pred_flat = pred.view(pred.shape[0], num_classes, -1)
    intersection = (pred_flat * target_flat).sum(dim=2)
    pred_sum = pred_flat.sum(dim=2)
    target_sum = target_flat.sum(dim=2)
    dice_score = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)
    per_class_loss = 1 - dice_score.mean(dim=0)
    return per_class_loss

if __name__ == "__main__":
    print("Hello, World!")