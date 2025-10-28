import os
import torch
from dataset import create_dataloaders
from modules import ImprovedUNet
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 64
num_classes = 6
model_path = "2D_Improved_UNET_47205145/model.pth"

# Dataset paths
validate_images = "2D_Improved_UNET_47205145/HipMRI_Study_open/keras_slices_data/keras_slices_validate"
validate_masks = "2D_Improved_UNET_47205145/HipMRI_Study_open/keras_slices_data/keras_slices_seg_validate"

# Load validation data
validation_loader = create_dataloaders(validate_images, validate_masks, batch_size, standardize=True)

# Load the trained model
net = ImprovedUNet(num_classes=num_classes, deep_supervision=False).to(device)
net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
net.eval()

def compute_loss(pred, target, num_classes, smooth=1):
    """Calculate Dice loss for each class separately.
    Args:
        pred (torch.Tensor): Predicted logits from the model of shape [batch_size, num_classes, H, W].
        target (torch.Tensor): Ground truth masks of shape [batch_size, H, W].
        num_classes (int): Number of segmentation classes.
        smooth (float): Smoothing factor to avoid division by zero.
    """
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

def model_evaluation(model, data_loader, device, num_classes, save_path="2D_Improved_UNET_47205145/images/dice_scores.png"):
    """
    Evaluate the segmentation model and plot average Dice scores per anatomical class.

    Args:
        model (torch.nn.Module): Trained segmentation model.
        data_loader (torch.utils.data.DataLoader): Validation dataset loader.
        device (torch.device): Computation device (CPU or CUDA).
        num_classes (int): Total number of segmentation classes.
        save_path (str): File path for saving the bar chart.
    """
    class_labels = ["Background", "Body Outline", "Bone", "Bladder", "Rectum", "Prostate"]

    model.eval()
    dice_sums = np.zeros(num_classes, dtype=np.float64)
    num_batches = 0

    with torch.no_grad():
        for images, masks in data_loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            if isinstance(outputs, tuple):  # handle deep supervision
                outputs = outputs[0]

            per_class_dice = 1 - compute_loss(outputs, masks, num_classes).cpu().numpy()
            dice_sums += per_class_dice
            num_batches += 1

    mean_dice_per_class = dice_sums / max(num_batches, 1)
    overall_dice = mean_dice_per_class.mean()

    # Create directory if missing
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Plot bar chart
    plt.figure(figsize=(8, 5))
    bars = plt.bar(class_labels, mean_dice_per_class, color="skyblue", edgecolor="black")
    plt.axhline(y=overall_dice, color="red", linestyle="--", linewidth=1.2, label=f"Mean Dice: {overall_dice:.4f}")

    # Annotate values on bars
    for bar, score in zip(bars, mean_dice_per_class):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{score:.3f}",
                 ha="center", va="bottom", fontsize=9)

    plt.title("Average Dice Score per Anatomical Class", fontsize=13, fontweight="bold")
    plt.xlabel("Anatomical Structure", fontsize=11)
    plt.ylabel("Dice Score", fontsize=11)
    plt.ylim(0, 1.05)
    plt.legend(frameon=False)
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return mean_dice_per_class, overall_dice

def save_predictions(model, data_loader, device, num_samples=3, save_dir="2D_Improved_UNET_47205145/images"):
    """
    Generate and save visualisations of model predictions alongside input images and ground truth masks.

    Args:
        model (torch.nn.Module): The trained segmentation network.
        data_loader (torch.utils.data.DataLoader): DataLoader containing validation data.
        device (torch.device): Hardware device used for inference.
        num_samples (int): Number of random examples to generate visualisations for.
        save_dir (str): Directory path where output figures will be saved.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    dataset = data_loader.dataset
    total_items = len(dataset)
    chosen_indices = random.sample(range(total_items), min(num_samples, total_items))

    for idx, data_index in enumerate(chosen_indices, start=1):
        img_tensor, mask_tensor = dataset[data_index]
        img_batch = img_tensor.unsqueeze(0).to(device)

        # Run model inference
        with torch.no_grad():
            prediction = model(img_batch)
            if isinstance(prediction, tuple):  # for deep supervision models
                prediction = prediction[0]
            prediction_mask = torch.argmax(prediction, dim=1).squeeze().cpu().numpy()

        # Convert tensors to NumPy arrays for display
        img_array = img_tensor.squeeze().cpu().numpy()
        true_array = mask_tensor.squeeze().cpu().numpy()

        # Save images
        plt.imsave(os.path.join(save_dir, f"mri_slice_{idx}.png"), img_array, cmap="gray")
        plt.imsave(os.path.join(save_dir, f"ground_truth_{idx}.png"), true_array, cmap="gray")
        plt.imsave(os.path.join(save_dir, f"model_prediction_{idx}.png"), prediction_mask, cmap="gray")


# Main
if __name__ == "__main__":
    model_evaluation(net, validation_loader, device, num_classes)
    save_predictions(net, validation_loader, device, num_samples=3, save_dir="2D_Improved_UNET_47205145/images")