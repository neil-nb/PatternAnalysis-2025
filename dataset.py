import os
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
import torch.nn.functional as F


class ProstateSegmentationDataset(Dataset):
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
    
    def __getitem__(self, idx):
        image_np = self._read_nifti(self.images[idx])
        mask_np = self._read_nifti(self.masks[idx])

        if self.standardise:
            mean, std = image_np.mean(), image_np.std()
            image_np = (image_np - mean) / (std + 1e-6)

        mask_np = np.rint(mask_np).astype(np.int64)

        image_tensor = self._prepare_tensor(image_np)
        mask_tensor = self._prepare_tensor(mask_np, is_mask=True)

        if self.apply_transform:
            image_tensor, mask_tensor = self.apply_transform(image_tensor, mask_tensor)

        return image_tensor, mask_tensor


def build_loader(img_dir, mask_dir, batch_size=4, standardise=False, output_shape=(256, 128), shuffle=True, num_workers=0):
    dataset = ProstateSegmentationDataset(img_dir=img_dir, mask_dir=mask_dir, standardise=standardise, output_shape=output_shape)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
