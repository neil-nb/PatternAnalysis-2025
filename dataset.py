import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import torch.nn.functional as F


class ProstateSegmentationDataset(Dataset):
    """
    A PyTorch Dataset for prostate segmentation using NIfTI (.nii/.nii.gz) files.

    This class loads 3D medical images and corresponding segmentation masks,
    applies optional preprocessing (standardisation and resizing), and returns
    tensors suitable for model input.
    """
    def __init__(self, img_dir, mask_dir, apply_transform=None, standardize=False, output_shape=(256, 128)):
        """
        Args:
            img_dir (str): Directory containing image files.
            mask_dir (str): Directory containing mask files.
            transform (callable, optional): Optional transform applied to image-mask pairs.
            standardize (bool, optional): Whether to standardise image intensities.
            output_shape (tuple, optional): Target spatial dimensions (height, width).
        """
        # Collect sorted lists of image and mask file paths
        self.images = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(('.nii', '.nii.gz'))])
        self.masks = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(('.nii', '.nii.gz'))])

        # Store parameters
        self.apply_transform = apply_transform
        self.standardize = standardize
        self.output_shape = output_shape

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.images)

    def _read_nifti(self, path):
        """Load a NIfTI file and return it as a float32 NumPy array."""
        return nib.load(path).get_fdata(dtype=np.float32)
    
    def _prepare_tensor(self, array, is_mask=False):
        """
        Convert a NumPy array to a PyTorch tensor and resize it.

        Args:
            array (np.ndarray): Input array.
            is_mask (bool): Whether the array represents a segmentation mask.

        Returns:
            torch.Tensor: Resized tensor.
        """
        tensor = torch.as_tensor(array, dtype=torch.float32).unsqueeze(0)

        # Resize the tensor to the desired output shape  
        if is_mask:
            tensor = F.interpolate(tensor.unsqueeze(0), size=self.output_shape, mode='nearest').squeeze(0).long()
        else:
            tensor = F.interpolate(tensor.unsqueeze(0), size=self.output_shape, mode='bilinear', align_corners=False).squeeze(0)

        return tensor
    
    def __getitem__(self, idx):
        """Load and preprocess a sample from the dataset.
        
        Args:
            idx (int): Index of the sample.

        Returns:
            (torch.Tensor, torch.Tensor): Tuple of (image, mask) tensors.
        """

        # Load image and mask
        image_np = self._read_nifti(self.images[idx])
        mask_np = self._read_nifti(self.masks[idx])

        # Standardise image intensities if required
        if self.standardize:
            mean, std = image_np.mean(), image_np.std()
            image_np = (image_np - mean) / (std + 1e-6)

        # Ensure mask is integer type
        mask_np = np.rint(mask_np).astype(np.int64)

        # Convert to tensors and resize
        image_tensor = self._prepare_tensor(image_np)
        mask_tensor = self._prepare_tensor(mask_np, is_mask=True)

        # Apply any additional transformations
        if self.apply_transform:
            image_tensor, mask_tensor = self.apply_transform(image_tensor, mask_tensor)

        return image_tensor, mask_tensor

def create_dataloaders(image_dir, mask_dir, batch_size, standardize=False):
    """
    Create a PyTorch DataLoader for prostate segmentation.

    Args:
        image_dir (str): Directory containing image files.
        mask_dir (str): Directory containing mask files.
        batch_size (int): Number of samples per batch.
        standardize (bool, optional): Whether to standardise image intensities.

    Returns:
        torch.utils.data.DataLoader: DataLoader instance.
    """
    dataset = ProstateSegmentationDataset(image_dir, mask_dir, standardize=standardize)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

