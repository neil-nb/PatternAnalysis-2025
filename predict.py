import torch
from dataset import create_dataloaders
from modules import ImprovedUNet
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64
num_classes = 6
model_path = "model.pth"

validate_images = "HipMRI_Study_open/keras_slices_data/keras_slices_validate"
validate_masks = "HipMRI_Study_open/keras_slices_data/keras_slices_seg_validate"

validation_loader = create_dataloaders(validate_images, validate_masks, batch_size, standardize=True)

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

def evaluate_model(model, data_loader, device, num_classes):
    total_dice_per_class = np.zeros(num_classes)
    batch_count = 0

    with torch.no_grad():
        for images, true_masks in data_loader:
            images = images.to(device)
            true_masks = true_masks.to(device)
            pred_masks = model(images)
            per_class_loss = dice_loss_per_class(pred_masks, true_masks, num_classes)
            per_class_dice = 1 - per_class_loss.cpu().numpy()
            total_dice_per_class += per_class_dice
            batch_count += 1

    average_dice_per_class = total_dice_per_class / batch_count
    print(f"Average Dice Score per Class:")
    for i, dice_score in enumerate(average_dice_per_class):
        print(f"Class {i}: {dice_score:.4f}")
    print(f"Overall Average Dice Score: {average_dice_per_class.mean():.4f}")


def visualize_predictions(model, data_loader, device, num_classes):
    dataset_size = len(data_loader.dataset)
    indices = random.sample(range(dataset_size), 3)
    samples = [data_loader.dataset[i] for i in indices]

    for idx, (image, true_mask) in enumerate(samples):
        image = image.unsqueeze(0).to(device)
        true_mask = true_mask.squeeze().cpu().numpy()

        with torch.no_grad():
            pred_mask = model(image)
            if isinstance(pred_mask, tuple):
                pred_mask = pred_mask[0]
            pred_mask = torch.argmax(pred_mask, dim=1).squeeze().cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image.squeeze().cpu().numpy(), cmap='gray')
        axes[0].set_title("Original Image")
        axes[1].imshow(true_mask, cmap='gray')
        axes[1].set_title("True Segmentation")
        axes[2].imshow(pred_mask, cmap='gray')
        axes[2].set_title("Predicted Segmentation")

        for ax in axes:
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(f'validation_{idx + 1}.png')
        plt.close()

if __name__ == "__main__":
    evaluate_model(net, validation_loader, device, num_classes)
    visualize_predictions(net, validation_loader, device, num_classes)