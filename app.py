import streamlit as st
import os
import random
import numpy as np
import torch
import pandas as pd
import SimpleITK as sitk
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from TransUNet_model import UltimateTransUNet, TransUNetConfig, MultiTaskDataset

# =============================================================================
# 1. CORE LOGIC (Load Model & Data)
# =============================================================================
@st.cache_resource
def load_resources():
    config = TransUNetConfig()
    device = config.DEVICE
    model = UltimateTransUNet(in_channels=config.SLICES).to(device)
    # Ensure this file is in your root directory
    model.load_state_dict(torch.load("transunet_ULTIMATE_best.pth", map_location=device, weights_only=True))
    model.eval()
    
    csv_path = os.path.join(config.ROOT_DIR, 'Common CSV files', 'candidates_V2.csv')
    candidates_df = pd.read_csv(csv_path)
    return model, config, candidates_df

def get_prediction_data(selected_path, config, candidates_df):
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

    voxel_coords = np.round(np.abs(world_coords - origin) / spacing).astype(int)
    full_img_array = np.clip(full_img_array, -1000, 400)
    return full_img_array, voxel_coords, series_uid

# =============================================================================
# 2. UI LAYOUT
# =============================================================================
st.set_page_config(page_title="TransUNet Analysis", layout="wide")
model, config, candidates_df = load_resources()

st.title("🔬 TransUNet: Hybrid Transformer-CNN for Lung Cancer")
st.markdown("---")

# Section 1: Technical Explanation
st.header("1. How it's Trained")
col_text, col_img = st.columns([2, 1])
with col_text:
    st.write("""
    The **Ultimate TransUNet** model is a multi-task framework trained on the LUNA16 dataset for simultaneous **Nodule Segmentation** and **Malignancy Classification**.
    
    * **The Encoder**: A CNN backbone extracts high-resolution local features (edges, textures).
    * **The Transformer Bottleneck**: 6 Transformer blocks model long-range dependencies, allowing the model to understand the nodule's position relative to the entire lung anatomy.
    * **Hybrid Loss**: We use a combination of **Dice Loss** (for precise mask overlap) and **Focal Cross-Entropy** (to handle the class imbalance of small nodules).
    * **Integrity**: Validated with a Brier Score of **0.0523**, ensuring diagnostic "honesty" over raw overconfidence.
    """)
with col_img:
    st.info("**Model Stats**\n- Accuracy: 93.63%\n- AUC-ROC: 0.9807\n- Precision: 0.96")

st.markdown("---")

# Section 2: Fixed Examples from Dataset
st.header("2. Dataset Examples (Fixed Samples)")
ex_paths = glob(os.path.join(config.PREPROCESSED_PATH, 'trans_pre_subset*', 'images', 'pos*.npy'))[:3]
e_cols = st.columns(3)

for i, path in enumerate(ex_paths):
    with e_cols[i]:
        patch = np.load(path)
        fig, ax = plt.subplots(facecolor='black')
        ax.imshow(patch[32, :, :], cmap='bone')
        ax.set_title(f"Example {i+1}", color='white')
        ax.axis('off')
        st.pyplot(fig)

st.markdown("---")

# Section 3: Random Prediction Generator
st.header("3. Real-Time Diagnostic Generator")
if st.button("🚀 Randomly Pick Patient & Generate Prediction"):
    all_pos = glob(os.path.join(config.PREPROCESSED_PATH, 'trans_pre_subset*', 'images', 'pos*.npy'))
    sample = random.choice(all_pos)
    
    img_full, v_coords, uid = get_prediction_data(sample, config, candidates_df)
    patch_3d = np.load(sample)
    vx, vy, vz = v_coords[0], v_coords[1], v_coords[2]

    # Inference
    img_tensor = torch.from_numpy(patch_3d[24:40, :, :]).float().unsqueeze(0).to(config.DEVICE)
    with torch.no_grad():
        p_mask, p_clf = model(img_tensor)
        mask = torch.sigmoid(p_mask).cpu().numpy()[0, 0]
        conf = torch.sigmoid(p_clf).item() * 100

    # UI Results
    st.subheader(f"Patient UID: {uid}")
    r_col1, r_col2 = st.columns(2)
    
    with r_col1:
        st.write("### AI Localization")
        fig_loc, ax_loc = plt.subplots(figsize=(10, 5), facecolor='black')
        ax_loc.imshow(img_full[vz, :, :], cmap='gray')
        rect = patches.Rectangle((vx-25, vy-25), 50, 50, lw=2, edgecolor='red', facecolor='none')
        ax_loc.add_patch(rect)
        ax_loc.set_title(f"Malignancy Risk: {conf:.2f}%", color='red' if conf > 50 else 'green')
        ax_loc.axis('off')
        st.pyplot(fig_loc)

    with r_col2:
        st.write("### High-Res Segmentation")
        fig_seg, ax_seg = plt.subplots(figsize=(5, 5), facecolor='black')
        ax_seg.imshow(patch_3d[32, :, :], cmap='bone')
        ax_seg.imshow(mask, cmap='jet', alpha=0.4) # Overlaying predicted mask
        ax_seg.set_title("TransUNet Predicted Mask", color='white')
        ax_seg.axis('off')
        st.pyplot(fig_seg)