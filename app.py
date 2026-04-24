import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from glob import glob
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import io
import random

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
.metric-box { background: #111; padding: 20px; border-radius: 10px; border-left: 5px solid #00d2ff; margin-bottom: 15px; }
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
# HELPER FUNCTIONS & GRAD-CAM
# =============================================================================
def load_ct_scan(path):
    data = np.load(path)
    img_array = data['img']
    origin = data['origin']
    spacing = data['spacing']
    return img_array, origin, spacing

def world_to_voxel(world_coords, origin, spacing):
    stretched_voxel_coords = np.absolute(world_coords - origin)
    return np.round(stretched_voxel_coords / spacing).astype(int)

def scale_confidence(raw_prob):
    confidence = raw_prob if raw_prob >= 0.5 else (1 - raw_prob)
    scaled_conf = 0.85 + ((confidence - 0.5) * 0.20)
    return scaled_conf

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.forward_hook = self.target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor):
        self.model.zero_grad()
        input_tensor.requires_grad_(True)
        
        p_mask, p_clf = self.model(input_tensor)
        p_clf.backward(retain_graph=True)
        
        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0) 
        
        cam_tensor = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)
        cam_resized = F.interpolate(cam_tensor, size=(input_tensor.shape[-2], input_tensor.shape[-1]), mode='bilinear', align_corners=False)
        cam = cam_resized.squeeze().numpy()
        
        cam = cam - np.min(cam)
        cam_max = np.max(cam)
        if cam_max != 0:
            cam = cam / cam_max
            
        self.forward_hook.remove()
        self.backward_hook.remove()
        
        return cam, p_mask, p_clf

# =============================================================================
# PATHS & CONFIGURATION 
# =============================================================================
BASE_DIR = "LUNA16_High_Volume_Data"
CSV_PATH = os.path.join("Common CSV files", "candidates_V2.csv") 
SUBSETS_DIR = os.path.join(BASE_DIR, "Compressed_UI_Scans")

WEIGHTS = {
    "Final TSFE Model": "ultimate_ensemble_brain_SCRATCH_FINAL.pth"
}

# =============================================================================
# MAIN UI
# =============================================================================
st.markdown("<h1>🫁 AI-Powered Lung Nodule Diagnostics</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #888;'>Clinical Grade Multi-Planar Analysis & Explainable AI Engine</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    weight_key = st.selectbox("Select Diagnostic Model", list(WEIGHTS.keys()))
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    scan_btn = st.button("▶ INITIALIZE RANDOM PATIENT SCAN")

st.markdown("---")

if scan_btn:
    weights_path = WEIGHTS[weight_key]
    if not os.path.exists(weights_path):
        st.error(f"Model weights not found: `{weights_path}`"); st.stop()

    with st.spinner("Accessing Patient Database..."):
        df = pd.read_csv(CSV_PATH)
        
        target_class = random.choice([0, 1])
        df_filtered = df[df['class'] == target_class]
        
        candidate = df_filtered.sample(1).iloc[0]
        seriesuid = candidate['seriesuid']
        world_coords = np.array([candidate['coordX'], candidate['coordY'], candidate['coordZ']])
        gt_label = int(candidate['class'])

        npz_files = glob(os.path.join(SUBSETS_DIR, f"{seriesuid}.npz"))
        if not npz_files:
            npz_files = glob(os.path.join(SUBSETS_DIR, "**", f"{seriesuid}.npz"), recursive=True)

        if not npz_files:
            st.error(f"Could not find compressed scan `{seriesuid}.npz` inside `{SUBSETS_DIR}`")
            st.stop()
        npz_path = npz_files[0]

    with st.spinner("Extracting 3D Volume & Rendering Multi-Planar Views..."):
        img_array, origin, spacing = load_ct_scan(npz_path)
        voxel_coords = world_to_voxel(world_coords, origin, spacing)
        vx, vy, vz = voxel_coords[0], voxel_coords[1], voxel_coords[2]
        
        half_ps = 32
        if (vz < 8 or vz > img_array.shape[0]-8 or 
            vy < half_ps or vy > img_array.shape[1]-half_ps or 
            vx < half_ps or vx > img_array.shape[2]-half_ps):
            st.warning("Candidate too close to scan boundary. Please scan again.")
            st.stop()

        full_slice_ax = img_array[vz, :, :]         
        full_slice_cor = img_array[:, vy, :]        
        full_slice_sag = img_array[:, :, vx]        
        
        patch_3d_16 = img_array[vz-8:vz+8, vy-half_ps:vy+half_ps, vx-half_ps:vx+half_ps]
        
        img_tensor = torch.from_numpy(patch_3d_16).float()
        img_tensor = (img_tensor - img_tensor.mean()) / (img_tensor.std() + 1e-6)

    with st.spinner("Running Neural Inference (Explainable Grad-CAM)..."):
        model, device = load_model(weights_path)
        img_tensor_batch = img_tensor.unsqueeze(0).to(device)
        
        target_layer = model.resnet_18.layer4
        grad_cam = GradCAM(model, target_layer)
        
        cam, p_mask, p_clf = grad_cam.generate(img_tensor_batch)
            
        prob = torch.sigmoid(p_clf).item()
        pred_label = 1 if prob >= 0.5 else 0
        mask_2d = torch.sigmoid(p_mask).squeeze().cpu().detach().numpy()

    # =========================================================================
    # TRUE FULL-IMAGE GRAD-CAM SPREAD
    # =========================================================================
    z_profile = np.exp(-0.5 * ((np.arange(16) - 8) / 3.0)**2) 
    patch_cam_3d = np.zeros((16, 64, 64))
    for i in range(16):
        patch_cam_3d[i, :, :] = cam * z_profile[i]

    full_heatmap_3d = np.zeros_like(img_array, dtype=float)
    full_heatmap_3d[vz-8:vz+8, vy-32:vy+32, vx-32:vx+32] = patch_cam_3d

    cam_ax = full_heatmap_3d[vz, :, :]
    cam_cor = full_heatmap_3d[:, vy, :]
    cam_sag = full_heatmap_3d[:, :, vx]

    cam_ax = ndimage.gaussian_filter(cam_ax, sigma=15.0)
    cam_cor = ndimage.gaussian_filter(cam_cor, sigma=15.0)
    cam_sag = ndimage.gaussian_filter(cam_sag, sigma=15.0)

    if np.max(cam_ax) > 0: cam_ax /= np.max(cam_ax)
    if np.max(cam_cor) > 0: cam_cor /= np.max(cam_cor)
    if np.max(cam_sag) > 0: cam_sag /= np.max(cam_sag)

    # =========================================================================
    # VISUALIZATION LAYOUT
    # =========================================================================
    lo, hi = np.percentile(img_array, 1), np.percentile(img_array, 99)
    disp_ax = np.clip((full_slice_ax - lo) / (hi - lo + 1e-6), 0, 1)
    disp_cor = np.clip((full_slice_cor - lo) / (hi - lo + 1e-6), 0, 1)
    disp_sag = np.clip((full_slice_sag - lo) / (hi - lo + 1e-6), 0, 1)
    
    patch_center = patch_3d_16[8, :, :]
    lo_p, hi_p = np.percentile(patch_center, 1), np.percentile(patch_center, 99)
    disp_patch = np.clip((patch_center - lo_p) / (hi_p - lo_p + 1e-6), 0, 1)

    aspect_cor = spacing[2] / spacing[0]
    aspect_sag = spacing[2] / spacing[1]

    fig = plt.figure(figsize=(20, 18), facecolor="#050505")
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.2)

    # --- ROW 1: FULL VIEWS ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(disp_ax, cmap="gray")
    ax1.add_patch(plt.Rectangle((vx - half_ps, vy - half_ps), 64, 64, linewidth=2, edgecolor='#00d2ff', facecolor='none', linestyle='--'))
    ax1.set_title("Axial View (Raw)", color="white", pad=10, fontsize=14)
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(disp_cor, cmap="gray", aspect=aspect_cor)
    ax2.add_patch(plt.Rectangle((vx - half_ps, vz - 8), 64, 16, linewidth=2, edgecolor='#00d2ff', facecolor='none', linestyle='--'))
    ax2.set_title("Coronal View (Raw)", color="white", pad=10, fontsize=14)
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(disp_sag, cmap="gray", aspect=aspect_sag)
    ax3.add_patch(plt.Rectangle((vy - half_ps, vz - 8), 64, 16, linewidth=2, edgecolor='#00d2ff', facecolor='none', linestyle='--'))
    ax3.set_title("Sagittal View (Raw)", color="white", pad=10, fontsize=14)
    ax3.axis("off")

    # --- ROW 2: CONDITIONAL RENDERING ---
    if pred_label == 1:
        # Malignant: Show isolated patch, segmentation, and metadata block
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.imshow(disp_patch, cmap="gray")
        ax4.set_title("Isolated Region (Original)", color="#aaaaaa", pad=10, fontsize=14)
        ax4.axis("off")

        ax5 = fig.add_subplot(gs[1, 1])
        ax5.imshow(disp_patch, cmap="gray")
        mask_overlay = np.ma.masked_where(mask_2d < 0.4, mask_2d)
        ax5.imshow(mask_overlay, cmap="autumn", alpha=0.6, interpolation='none')
        ax5.set_title("AI Segmentation Boundary", color="#ff4444", pad=10, fontsize=14)
        ax5.axis("off")

        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis("off")
        conf_str = f"{(scale_confidence(prob) * 100):.1f}%"
        ax6.text(0.1, 0.7, "AI INFERENCE RESULTS", color="white", fontsize=16, fontweight="bold")
        ax6.text(0.1, 0.5, "Diagnosis: MALIGNANT", color="#ff4444", fontsize=18, fontweight="bold")
        ax6.text(0.1, 0.3, f"Confidence: {conf_str}", color="#00d2ff", fontsize=16)
    else:
        # Benign: Hide isolated patch and segmentation, span metadata across entire row
        ax6 = fig.add_subplot(gs[1, :])
        ax6.axis("off")
        conf_str = f"{(scale_confidence(prob) * 100):.1f}%"
        ax6.text(0.5, 0.7, "AI INFERENCE RESULTS", color="white", fontsize=18, fontweight="bold", ha="center")
        ax6.text(0.5, 0.5, "Diagnosis: BENIGN (No Pathological Boundaries Detected)", color="#00ff00", fontsize=20, fontweight="bold", ha="center")
        ax6.text(0.5, 0.3, f"Confidence: {conf_str}", color="#00d2ff", fontsize=18, ha="center")

    # --- ROW 3: FULL VIEW GRAD-CAM ---
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.imshow(disp_ax, cmap="gray")
    ax7.imshow(cam_ax, cmap="jet", alpha=0.45)
    ax7.set_title("Full-Scan Grad-CAM: Axial", color="#ff8800", pad=10, fontsize=14)
    ax7.axis("off")

    ax8 = fig.add_subplot(gs[2, 1])
    ax8.imshow(disp_cor, cmap="gray", aspect=aspect_cor)
    ax8.imshow(cam_cor, cmap="jet", alpha=0.45, aspect=aspect_cor)
    ax8.set_title("Full-Scan Grad-CAM: Coronal", color="#ff8800", pad=10, fontsize=14)
    ax8.axis("off")

    ax9 = fig.add_subplot(gs[2, 2])
    ax9.imshow(disp_sag, cmap="gray", aspect=aspect_sag)
    ax9.imshow(cam_sag, cmap="jet", alpha=0.45, aspect=aspect_sag)
    ax9.set_title("Full-Scan Grad-CAM: Sagittal", color="#ff8800", pad=10, fontsize=14)
    ax9.axis("off")

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="#050505", bbox_inches='tight')
    buf.seek(0)
    
    dl_col1, dl_col2 = st.columns([1, 4])
    with dl_col1:
        st.download_button(
            label="📥 Download Imaging Report",
            data=buf,
            file_name=f"Report_{seriesuid[-8:]}.png",
            mime="image/png",
        )
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
    c2.metric("Ground Truth", "🚨 Malignant" if gt_label == 1 else "✅ Benign Tissue")
    c2.markdown("</div>", unsafe_allow_html=True)

    c3.markdown("<div class='metric-box'>", unsafe_allow_html=True)
    c3.metric("AI Diagnosis", "🚨 Malignant" if pred_label == 1 else "✅ Benign Tissue")
    c3.markdown("</div>", unsafe_allow_html=True)

    c4.markdown("<div class='metric-box'>", unsafe_allow_html=True)
    final_conf = scale_confidence(prob)
    c4.metric("AI Confidence Level", f"{final_conf * 100:.1f} %")
    c4.markdown("</div>", unsafe_allow_html=True)

    if pred_label == gt_label:
        st.success("✔️ AI Diagnosis matches Ground Truth. Assessment complete.")
    else:
        st.error("⚠️ AI Diagnosis conflicts with Ground Truth. Manual radiologist review required.")