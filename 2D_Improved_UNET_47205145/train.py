import torch
import torch.nn as nn
from modules import ImprovedUNet
from dataset import create_dataloaders
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import matplotlib.pyplot as plt

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 32
num_classes = 6
learning_rate = 3e-3
num_epochs = 20

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

def init_weights(m):
    """Initialize weights of convolutional layers using Kaiming normal initialization.
     Args:
         m (nn.Module): A module in the neural network.
     """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# Apply weight initialization
net.apply(init_weights)


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
def train_model():
    """Train the Improved UNet model."""
    train_losses, val_losses = [], []
    print("Starting training\n")
    start_time = time.time()

    for epoch in range(num_epochs):
        net.train()
        epoch_loss = 0.0

        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)

            # Forward pass with mixed precision
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = net(images)
                if isinstance(outputs, tuple):
                    main, aux2, aux3 = outputs
                    loss_main = criterion(main, masks.squeeze(1))
                    loss_aux2 = criterion(aux2, masks.squeeze(1))
                    loss_aux3 = criterion(aux3, masks.squeeze(1))
                    loss = 0.6 * loss_main + 0.25 * loss_aux2 + 0.15 * loss_aux3
                else:
                    loss = criterion(outputs, masks.squeeze(1))

            # Backward pass and optimization
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if (i + 1) % batch_size == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1}, Average Training Loss: {avg_train_loss:.4f}")

        avg_val_loss = validate_model()
        val_losses.append(avg_val_loss)
        scheduler.step()

    end_time = time.time()
    print("\nFinished Training")
    print(f"Total training time: {end_time - start_time:.2f} seconds")

    torch.save(net.state_dict(), "model.pth")
    plot_losses(train_losses, val_losses)

def validate_model():
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

def plot_losses(train_losses, val_losses):
    """Plot training and validation losses."""
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, "b", label="Training Loss")
    plt.plot(epochs, val_losses, "r", label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("images/loss_graph.png")
    plt.close()

# Main
if __name__ == "__main__":
    train_model()
