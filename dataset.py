import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import torch.nn.functional as F

class HipMRIDataset(Dataset):
    def __init__(self, img_dir, mask_dir, apply_transform=None, standardise=False, output_shape=(256, 128)):
        self.images = sorted(
            [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(('.nii', '.nii.gz'))]
        )
        self.masks = sorted(
            [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(('.nii', '.nii.gz'))]
        )
        self.apply_transform = apply_transform
        self.standardise = standardise
        self.output_shape = output_shape

    def __len__(self):
        return len(self.images)

    def _read_nifti(self, path):
        return nib.load(path).get_fdata(dtype=np.float32)
    
    def _prepare_tensor(self, array, is_mask=False):
        tensor = torch.as_tensor(array, dtype=torch.float32).unsqueeze(0)
        if is_mask:
            tensor = F.interpolate(tensor.unsqueeze(0), size=self.output_shape, mode='nearest').squeeze(0).long()
        else:
            tensor = F.interpolate(tensor.unsqueeze(0), size=self.output_shape, mode='bilinear', align_corners=False).squeeze(0)
        return tensor