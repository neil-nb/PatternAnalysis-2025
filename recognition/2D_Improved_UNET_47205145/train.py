import torch
import torch.nn as nn
from modules import ImprovedUNet
from dataset import create_dataloaders
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import numpy as np

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 32
num_classes = 6
learning_rate = 3e-3
num_epochs = 25

# Dataset paths
train_images = "HipMRI_Study_open/keras_slices_data/keras_slices_train"
train_masks = "HipMRI_Study_open/keras_slices_data/keras_slices_seg_train"
test_images = "HipMRI_Study_open/keras_slices_data/keras_slices_test"
test_masks = "HipMRI_Study_open/keras_slices_data/keras_slices_seg_test"

# Load data
train_loader = create_dataloaders(train_images, train_masks, batch_size, standardize=True)
val_loader = create_dataloaders(test_images, test_masks, batch_size, standardize=True)

# Model setup
net = ImprovedUNet(num_classes=num_classes, base_ch=64, dropout_p=0.1, deep_supervision=True).to(device)

def initialize_weights(m):
    """Initialize weights of convolutional layers using Kaiming normal initialization.
     Args:
         m (nn.Module): A module in the neural network.
     """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# Apply weight initialization
net.apply(initialize_weights)


# Loss functions
class GeneralizedDiceLoss(nn.Module):
    """Generalized Dice Loss with inverse volume weighting for class imbalance."""
    def __init__(self, eps=1e-6):
        """Initialize the GeneralizedDiceLoss."""
        super().__init__()
        self.eps = eps

    def forward(self, logits, target):
        """Compute the Generalized Dice Loss."""
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        target_1h = F.one_hot(target.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        w = 1.0 / (torch.clamp(target_1h.sum(dim=dims), min=self.eps) ** 2)
        intersection = (probs * target_1h).sum(dim=dims)
        union = probs.sum(dim=dims) + target_1h.sum(dim=dims)
        dice = (2.0 * intersection + self.eps) / (union + self.eps)
        gdice = 1.0 - (w * dice).sum() / torch.clamp(w.sum(), min=self.eps)
        return gdice
    
class ComboLoss(nn.Module):
    """Hybrid loss combining Generalized Dice and Cross-Entropy."""
    def __init__(self, ce_weight=None, alpha=0.5):
        """Initialize the ComboLoss.
        Args:
            ce_weight (torch.Tensor, optional): Class weights for CrossEntropyLoss.
            alpha (float): Weighting factor for the Dice loss.
        """
        super().__init__()
        self.alpha = alpha
        self.dice = GeneralizedDiceLoss()
        self.ce = nn.CrossEntropyLoss(weight=ce_weight)

    def forward(self, logits, target):
        """Compute the Combo Loss."""
        return self.alpha * self.dice(logits, target) + (1.0 - self.alpha) * self.ce(logits, target)
    
# Optimizer and scheduler
optimizer = optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-2)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
criterion = ComboLoss(alpha=0.5)

# Training and validation loops
def train():
    """
    Executes the training loop for the Improved U-Net model, recording training and validation losses.
    """
    history_train, history_val = [], []
    print("Beginning model optimisation...\n")

    for epoch_idx in range(num_epochs):
        net.train()
        running_total, batch_counter = 0.0, 0

        for batch_idx, (x_batch, y_batch) in enumerate(train_loader, start=1):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            # Forward and backward passes with mixed precision
            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                predictions = net(x_batch)
                if isinstance(predictions, tuple):
                    primary, aux_1, aux_2 = predictions
                    main_loss = criterion(primary, y_batch.squeeze(1))
                    aux_loss_1 = criterion(aux_1, y_batch.squeeze(1))
                    aux_loss_2 = criterion(aux_2, y_batch.squeeze(1))
                    total_loss = (0.6 * main_loss) + (0.25 * aux_loss_1) + (0.15 * aux_loss_2)
                else:
                    total_loss = criterion(predictions, y_batch.squeeze(1))

            # Gradient computation and update
            scaler.scale(total_loss).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_total += total_loss.item()
            batch_counter += 1

            # Periodic console feedback
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch_idx+1}/{num_epochs} | Batch {batch_idx}/{len(train_loader)} "
                      f"| Current Loss: {total_loss.item():.4f}")

        # End-of-epoch metrics
        mean_train_loss = running_total / max(batch_counter, 1)
        history_train.append(mean_train_loss)
        print(f"Epoch {epoch_idx+1}: Average Training Loss = {mean_train_loss:.4f}")

        # Validation
        val_loss = validate()
        history_val.append(val_loss)
        scheduler.step()

    # Save final model weights and training curve
    torch.save(net.state_dict(), "model.pth")
    loss_plot(history_train, history_val, smooth_window=3, save_path="images/loss_curve.png")


def validate():
    """Validate the model on the validation dataset."""
    net.eval()
    val_loss = 0.0

    # Validation loop with mixed precision
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = net(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, masks.squeeze(1))
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    return avg_val_loss

def loss_plot(train_losses, val_losses, smooth_window=3, save_path="images/loss.png"):
    """Plot and save smoothed training and validation losses over epochs."""
    
    def smooth(values, window):
        """Apply simple moving average smoothing."""
        if window < 2 or len(values) < window:
            return values
        kernel = np.ones(window) / window
        return np.convolve(values, kernel, mode="valid")

    # Prepare data
    epochs = np.arange(1, len(train_losses) + 1)
    train_smooth = smooth(train_losses, smooth_window)
    val_smooth = smooth(val_losses, smooth_window)
    offset = len(epochs) - len(train_smooth)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs[offset:], train_smooth, color="steelblue", linewidth=2, label="Training Loss")
    ax.plot(epochs[offset:], val_smooth, color="indianred", linewidth=2, label="Validation Loss")

    # Format plot
    ax.set_title("Training and Validation Loss", fontsize=14, weight="bold")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    ax.xaxis.get_major_locator().set_params(integer=True)

    # Save and close
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

# Main
if __name__ == "__main__":
    train()
