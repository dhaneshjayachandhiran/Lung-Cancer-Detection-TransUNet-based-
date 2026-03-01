import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import random
from glob import glob
import matplotlib.pyplot as plt
import SimpleITK as sitk
from scipy import ndimage

# Import your architectures
from TransUNet_model import UltimateTransUNet, TransUNetConfig
from Resnet_model import ResNetMultiTaskModel
from simpleCNN_model import SimpleMultiTaskCNN

# =============================================================================
# STREAMLIT SETUP & STYLING
# =============================================================================
st.set_page_config(page_title="Lung Nodule AI Diagnostics", layout="wide", page_icon="🫁")

st.markdown("""
<style>
body, .stApp { background-color: #050505; color: #e0e0e0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
h1, h2, h3 { color: #ffffff; }
.stButton button {
    background: #1e1e2f; border: 1px solid #00d2ff;
    color: #00d2ff; font-size: 16px; font-weight: bold; padding: 12px 32px;
    border-radius: 6px; width: 100%; transition: 0.3s;
}
.stButton button:hover { background: #00d2ff; color: #000; box-shadow: 0 0 15px #00d2ff; }
.stSelectbox > div > div { background: #1a1a1a !important; color: #e0e0e0 !important; }
.metric-box { background: #111; padding: 20px; border-radius: 10px; border-left: 5px solid #00d2ff; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# MASTER BRAIN ARCHITECTURE
# =============================================================================
class UltimateEnsembleBrain(nn.Module):
    def __init__(self, in_channels=16):
        super().__init__()
        self.simple_cnn = SimpleMultiTaskCNN(in_channels=in_channels)
        self.resnet_18  = ResNetMultiTaskModel(in_channels=in_channels)
        self.trans_unet = UltimateTransUNet(in_channels=in_channels)
        self.channel_compressor = nn.Conv2d(128, 96, kernel_size=1)
        self.fusion_gate = nn.Sequential(
            nn.Linear(256 + 512 + 256, 512), nn.BatchNorm1d(512),
            nn.ReLU(), nn.Dropout(0.4), nn.Linear(512, 1)
        )

    def forward(self, x):
        sc1  = self.simple_cnn.enc1(x)
        sc4  = self.simple_cnn.enc4(self.simple_cnn.pool(self.simple_cnn.enc3(self.simple_cnn.pool(self.simple_cnn.enc2(self.simple_cnn.pool(sc1))))))
        feat_cnn = torch.flatten(self.simple_cnn.avgpool(sc4), 1)
        r    = self.resnet_18.relu(self.resnet_18.bn1(self.resnet_18.first_conv(x)))
        l4   = self.resnet_18.layer4(self.resnet_18.layer3(self.resnet_18.layer2(self.resnet_18.layer1(self.resnet_18.maxpool(r)))))
        feat_res = torch.flatten(self.resnet_18.avgpool(l4), 1)
        ts1  = self.trans_unet.enc1(x); ts2 = self.trans_unet.enc2(ts1); ts3 = self.trans_unet.enc3(ts2)
        tb   = self.trans_unet.bottleneck_pool(ts3)
        tb_flat = tb.flatten(2).transpose(1, 2) + self.trans_unet.pos_embed
        t_out   = self.trans_unet.transformers(tb_flat)
        feat_trans = torch.mean(t_out, dim=1)
        tb_out = t_out.transpose(1, 2).reshape(tb.shape)
        td1  = self.trans_unet.dec1(torch.cat([self.trans_unet.up1(tb_out), ts3], dim=1))
        td2  = self.trans_unet.dec2(torch.cat([self.trans_unet.up2(td1), ts2], dim=1))
        fi   = torch.cat([self.trans_unet.up3(td2), ts1, sc1], dim=1)
        mask = self.trans_unet.seg_final(self.trans_unet.dec3(self.channel_compressor(fi)))
        return mask, self.fusion_gate(torch.cat((feat_cnn, feat_res, feat_trans), dim=1))

@st.cache_resource(show_spinner=False)
def load_model(path):
    cfg    = TransUNetConfig()
    device = cfg.DEVICE
    model  = UltimateEnsembleBrain(16).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model, device

# =============================================================================
# HELPER FUNCTIONS 
# =============================================================================
def load_ct_scan(path):
    itk_img = sitk.ReadImage(path)
    img_array = sitk.GetArrayFromImage(itk_img)
    origin = np.array(itk_img.GetOrigin())
    spacing = np.array(itk_img.GetSpacing())
    return img_array, origin, spacing

def world_to_voxel(world_coords, origin, spacing):
    stretched_voxel_coords = np.absolute(world_coords - origin)
    return np.round(stretched_voxel_coords / spacing).astype(int)

# =============================================================================
# PATHS & CONFIGURATION (UPDATED)
# =============================================================================
# 1. Main Project Folder
PROJECT_ROOT = r"I:\Lung Cancer Project (Simple CNN)"

# 2. CSV Folder (Outside the image data folder)
CSV_PATH = os.path.join(PROJECT_ROOT, "Common CSV files", "candidates_V2.csv") 

# 3. Image Data Folder
BASE_DIR = os.path.join(PROJECT_ROOT, "LUNA16_High_Volume_Data")
SUBSETS_DIR = os.path.join(BASE_DIR, "Subsets") 

# ONLY the Scratch model is loaded now
WEIGHTS = {
    "SCRATCH FINAL (Baseline)": "ultimate_ensemble_brain_SCRATCH_FINAL.pth"
}

# =============================================================================
# MAIN UI
# =============================================================================
st.markdown("<h1>🫁 AI-Powered Lung Nodule Diagnostics</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #888;'>Clinical Grade Full-Scan Analysis & Segmentation Engine</p><br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    weight_key = st.selectbox("Select Diagnostic Model", list(WEIGHTS.keys()))
with col2:
    filter_opt = st.selectbox("Patient Scenario (For Demo)", ["Random Candidate", "Known Nodule (Malignant)", "Healthy Tissue (Benign)"])
with col3:
    st.markdown("<br>", unsafe_allow_html=True)
    scan_btn = st.button("▶ INITIALIZE SCAN")

st.markdown("---")

if scan_btn:
    # 1. Verification
    weights_path = WEIGHTS[weight_key]
    if not os.path.exists(weights_path):
        st.error(f"Model weights not found: `{weights_path}`")
        st.stop()
    if not os.path.exists(CSV_PATH):
        st.error(f"CSV Database not found: `{CSV_PATH}`")
        st.stop()

    with st.spinner("Accessing Patient Database..."):
        # 2. Load CSV and pick a candidate
        df = pd.read_csv(CSV_PATH)
        if filter_opt == "Known Nodule (Malignant)":
            df = df[df['class'] == 1]
        elif filter_opt == "Healthy Tissue (Benign)":
            df = df[df['class'] == 0]
        
        candidate = df.sample(1).iloc[0]
        seriesuid = candidate['seriesuid']
        world_coords = np.array([candidate['coordX'], candidate['coordY'], candidate['coordZ']])
        gt_label = int(candidate['class'])

        # 3. Locate the .mhd file recursively in Subsets
        mhd_files = glob(os.path.join(SUBSETS_DIR, "**", f"{seriesuid}.mhd"), recursive=True)
        if not mhd_files:
            st.error(f"Could not find raw scan `{seriesuid}.mhd` in `{SUBSETS_DIR}`")
            st.stop()
        mhd_path = mhd_files[0]

    with st.spinner("Extracting 3D Volume & Mapping Coordinates..."):
        # 4. Extract Real CT Scan & Coordinates
        img_array, origin, spacing = load_ct_scan(mhd_path)
        voxel_coords = world_to_voxel(world_coords, origin, spacing)
        vx, vy, vz = voxel_coords[0], voxel_coords[1], voxel_coords[2]
        
        # Guard against boundaries
        half_ps = 32
        if (vz < 8 or vz > img_array.shape[0]-8 or 
            vy < half_ps or vy > img_array.shape[1]-half_ps or 
            vx < half_ps or vx > img_array.shape[2]-half_ps):
            st.warning("Candidate too close to scan boundary. Please scan again.")
            st.stop()

        # Extract Full Slice and Deep Learning Patch
        full_slice = img_array[vz, :, :]
        
        # Extract 16 slices for the 3D context
        patch_3d_16 = img_array[vz-8:vz+8, vy-half_ps:vy+half_ps, vx-half_ps:vx+half_ps]
        
        # 5. Preprocess exactly like training
        img_tensor = torch.from_numpy(patch_3d_16).float()
        img_tensor = (img_tensor - img_tensor.mean()) / (img_tensor.std() + 1e-6)

    with st.spinner("Running Neural Inference (CNN + ResNet + TransUNet)..."):
        # 6. Model Prediction
        model, device = load_model(weights_path)
        with torch.no_grad():
            p_mask, p_clf = model(img_tensor.unsqueeze(0).to(device))
            
        prob = torch.sigmoid(p_clf).item()
        
        # Scratch model evaluation uses standard 0.5 boundary
        threshold = 0.5 
        pred_label = 1 if prob >= threshold else 0
        
        mask_2d = torch.sigmoid(p_mask).squeeze().cpu().numpy()

    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    # Normalize images for display
    lo, hi = np.percentile(full_slice, 1), np.percentile(full_slice, 99)
    disp_full = np.clip((full_slice - lo) / (hi - lo + 1e-6), 0, 1)
    
    patch_center = patch_3d_16[8, :, :]
    lo_p, hi_p = np.percentile(patch_center, 1), np.percentile(patch_center, 99)
    disp_patch = np.clip((patch_center - lo_p) / (hi_p - lo_p + 1e-6), 0, 1)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5), facecolor="#050505", gridspec_kw={'width_ratios': [2, 1, 1]})
    
    # [Panel 1] FULL CT SCAN
    ax1.imshow(disp_full, cmap="gray")
    # Draw a bounding box around the candidate area
    rect = plt.Rectangle((vx - half_ps, vy - half_ps), 64, 64, linewidth=2, edgecolor='#00d2ff', facecolor='none', linestyle='--')
    ax1.add_patch(rect)
    ax1.set_title("Full Axial CT Slice", color="white", pad=10)
    ax1.axis("off")

    # [Panel 2] ZOOMED PATCH (What the AI sees)
    ax2.imshow(disp_patch, cmap="gray")
    ax2.set_title("Isolated Nodule Region (64x64)", color="#aaaaaa", pad=10)
    ax2.axis("off")

    # [Panel 3] AI SEGMENTATION MASK
    ax3.imshow(disp_patch, cmap="gray")
    # Overlay the mask
    mask_overlay = np.ma.masked_where(mask_2d < 0.4, mask_2d)
    ax3.imshow(mask_overlay, cmap="autumn", alpha=0.6, interpolation='none')
    ax3.set_title("AI Segmentation Boundary", color="#ff4444", pad=10)
    ax3.axis("off")

    fig.patch.set_facecolor("#050505")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # =========================================================================
    # CLINICAL VERDICT DASHBOARD
    # =========================================================================
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    
    c1.markdown("<div class='metric-box'>", unsafe_allow_html=True)
    c1.metric("Patient ID (Series UID)", f"...{seriesuid[-8:]}")
    c1.markdown("</div>", unsafe_allow_html=True)

    c2.markdown("<div class='metric-box'>", unsafe_allow_html=True)
    c2.metric("Ground Truth (Biopsy)", "🚨 Malignant" if gt_label == 1 else "✅ Benign Tissue")
    c2.markdown("</div>", unsafe_allow_html=True)

    c3.markdown("<div class='metric-box'>", unsafe_allow_html=True)
    c3.metric("AI Diagnosis", "🚨 Malignant" if pred_label == 1 else "✅ Benign Tissue")
    c3.markdown("</div>", unsafe_allow_html=True)

    c4.markdown("<div class='metric-box'>", unsafe_allow_html=True)
    confidence = prob if pred_label == 1 else (1 - prob)
    c4.metric("AI Confidence Level", f"{confidence * 100:.1f} %")
    c4.markdown("</div>", unsafe_allow_html=True)

    # Final Verification Toast
    if pred_label == gt_label:
        st.success("✔️ AI Diagnosis matches Ground Truth. Assessment complete.")
    else:
        st.error("⚠️ AI Diagnosis conflicts with Ground Truth. Manual radiologist review required.")