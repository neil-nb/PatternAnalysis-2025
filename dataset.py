import os
from torch.utils.data import Dataset

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

