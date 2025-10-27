# 2D Improved U-Net for Prostate MRI Segmentation
Student Number: 47205145 \
Name: Neil Barigye

## Overview
This project implements an Improved U-Net for 2D segmentation of the HipMRI Prostate Cancer dataset, which corresponds to Project 3 (Normal Difficulty).  
The model enhances the original U-Net architecture by incorporating residual connections, spatial and channel squeeze-and-excitation (scSE) modules, and attention gates to improve feature representation and segmentation accuracy.  


## Algorithm Description

### Model
The Improved U-Net builds upon the standard U-Net by adding architectural improvements for better gradient flow, attention, and feature recalibration:
- Residual Blocks for enhanced gradient propagation and stable training.  
- Spatial and Channel Squeeze-and-Excitation (scSE) Blocks to adaptively emphasise informative features.  
- Attention Gates to suppress irrelevant features from encoder layers before concatenation.  
- Deep Supervision via auxiliary outputs that stabilise learning during early epochs.  

The model is implemented in `modules.py` under the class `ImprovedUNet`.

### Data
The model is trained on 2D MRI slices and corresponding segmentation masks from the HipMRI Study on Prostate Cancer.  
Preprocessing and loading are handled in `dataset.py`:
- Images and masks are read from `.nii` or `.nii.gz` files using NiBabel.  
- Images are standardised and resized to 256×128.  
- Data is returned as tensors compatible with PyTorch.  
