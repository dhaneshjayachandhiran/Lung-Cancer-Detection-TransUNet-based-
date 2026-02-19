import os
import random
import numpy as np
import torch
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from glob import glob
from scipy.ndimage import zoom
from TransUNet_model import UltimateTransUNet, TransUNetConfig

# =============================================================================
# 1. DATA SYNCHRONIZATION & GEOMETRY
# =============================================================================
def get_synchronized_data(selected_path, config, candidates_df):
    filename = os.path.basename(selected_path)
    parts = filename.replace(".npy", "").split("_")
    series_uid = parts[1]
    row_idx = int(parts[2])
    
    row = candidates_df.iloc[row_idx]
    world_coords = np.array([row['coordX'], row['coordY'], row['coordZ']])

    mhd_path = glob(os.path.join(config.ROOT_DIR, 'Subsets', 'subset*', f"{series_uid}.mhd"))[0]
    itk_img = sitk.ReadImage(mhd_path)
    full_img_array = sitk.GetArrayFromImage(itk_img)
    origin, spacing = np.array(itk_img.GetOrigin()), np.array(itk_img.GetSpacing())

    # Map world coordinates to voxel indices
    voxel_coords = np.round(np.abs(world_coords - origin) / spacing).astype(int)
    full_img_array = np.clip(full_img_array, -1000, 400)
    
    return full_img_array, voxel_coords, series_uid

# =============================================================================
# 2. ORTHOGONAL RENDERER (512x512 Perfect Squares)
# =============================================================================
def plot_orthogonal_suite(img_full, voxel_coords, confidence, uid):
    vx, vy, vz = voxel_coords
    target_size = 512
    
    # Axial (Top View)
    axial = zoom(img_full[vz, :, :], target_size / img_full.shape[1], order=1)
    
    # Coronal (Front View) - Rescaled for anatomy
    cor_raw = img_full[:, vy, :]
    coronal = zoom(cor_raw, (target_size / cor_raw.shape[0], target_size / cor_raw.shape[1]), order=1)
    
    # Sagittal (Side View) - Rescaled for anatomy
    sag_raw = img_full[:, :, vx]
    sagittal = zoom(sag_raw, (target_size / sag_raw.shape[0], target_size / sag_raw.shape[1]), order=1)

    # UI Construction
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), facecolor='black')
    plt.subplots_adjust(top=0.82, bottom=0.1, wspace=0.05)

    # Clean Header
    plt.figtext(0.5, 0.92, "TRANSUNET 3D ARCHITECTURAL ANALYSIS", color='yellow', 
                fontsize=22, fontweight='bold', ha='center')
    plt.figtext(0.5, 0.88, f"Patient UID: {uid} | Risk: {confidence:.2f}%", 
                color='white', fontsize=14, ha='center')

    # Data mapping for loop
    views = [axial, coronal, sagittal]
    titles = ["AXIAL (TOP)", "CORONAL (FRONT)", "SAGITTAL (SIDE)"]
    
    # Calculate Coordinate Centers for the 512 space
    cy = (vy / img_full.shape[1]) * 512
    cx = (vx / img_full.shape[2]) * 512
    cz = (vz / img_full.shape[0]) * 512

    for i in range(3):
        axes[i].imshow(views[i], cmap='gray', extent=[0, 512, 512, 0])
        
        # Draw targeted indicators based on plane
        if i == 0: # Top: X, Y
            circ = plt.Circle((cx, cy), 18, color='cyan', fill=False, lw=2)
        elif i == 1: # Front: X, Z
            circ = plt.Circle((cx, cz), 18, color='yellow', fill=False, lw=2)
        elif i == 2: # Side: Y, Z
            circ = plt.Circle((cy, cz), 18, color='magenta', fill=False, lw=2)
            
        axes[i].add_artist(circ)
        axes[i].set_title(titles[i], color='white', fontsize=12, pad=10)
        axes[i].axis('off')

    plt.show()

# =============================================================================
# 3. EXECUTION
# =============================================================================
if __name__ == '__main__':
    config = TransUNetConfig()
    # Loading path context
    candidates_df = pd.read_csv(os.path.join(config.ROOT_DIR, 'Common CSV files', 'candidates_V2.csv'))
    
    model = UltimateTransUNet(in_channels=config.SLICES).to(config.DEVICE)
    model.load_state_dict(torch.load("transunet_ULTIMATE_best.pth", map_location=config.DEVICE, weights_only=True))
    model.eval()

    # Pick sample from preprocessed subsets
    pos_paths = glob(os.path.join(config.PREPROCESSED_PATH, 'trans_pre_subset*', 'images', 'pos*.npy'))
    sample_path = random.choice(pos_paths)
    
    img_full, v_coords, uid = get_synchronized_data(sample_path, config, candidates_df)
    patch_3d = np.load(sample_path)
    
    # Use center-weighted slices for inference
    img_tensor = torch.from_numpy(patch_3d[24:40, :, :]).float().unsqueeze(0).to(config.DEVICE)
    
    with torch.no_grad():
        _, p_clf = model(img_tensor)
        conf = torch.sigmoid(p_clf).item() * 100

    plot_orthogonal_suite(img_full, v_coords, conf, uid)