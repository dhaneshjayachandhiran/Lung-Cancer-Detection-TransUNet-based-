import streamlit as st
import os
import random
import numpy as np
import torch
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tempfile
from glob import glob
from fpdf import FPDF

# Import model classes
from TransUNet_model import UltimateTransUNet, TransUNetConfig, MultiTaskDataset
from Resnet_model import ResNetMultiTaskModel
from simpleCNN_model import SimpleMultiTaskCNN

# =============================================================================
# 1. CORE LOGIC (Loading & Synchronized Data)
# =============================================================================
@st.cache_resource
def load_all_models():
    config = TransUNetConfig()
    device = config.DEVICE
    
    models = {
        "TransUNet": UltimateTransUNet(in_channels=config.SLICES).to(device),
        "ResNet-18": ResNetMultiTaskModel().to(device),
        "SimpleCNN": SimpleMultiTaskCNN().to(device)
    }
    
    paths = {
        "TransUNet": "transunet_ULTIMATE_best.pth",
        "ResNet-18": "resnet_multitask_best.pth",
        "SimpleCNN": "simpleCNN_unet_best.pth"
    }

    for name, model in models.items():
        if os.path.exists(paths[name]):
            model.load_state_dict(torch.load(paths[name], map_location=device, weights_only=True))
            model.eval()
        else:
            st.error(f"Missing weights for {name}")
            
    return models, config

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

    voxel_coords = np.round(np.abs(world_coords - origin) / spacing).astype(int)
    full_img_array = np.clip(full_img_array, -1000, 400)
    return full_img_array, voxel_coords, series_uid

# =============================================================================
# 2. PDF GENERATION LOGIC
# =============================================================================
def generate_pdf_report(uid, scores, main_plot_path, zoom_plot_path):
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(200, 10, "Multi-Model Lung Cancer Diagnostic Report", ln=True, align='C')
    pdf.set_font("Arial", '', 12)
    pdf.cell(200, 10, f"Patient Series UID: {uid}", ln=True, align='C')
    pdf.ln(5)

    # Performance Metrics
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, "Architectural Consensus Analysis", ln=True)
    pdf.set_font("Arial", '', 12)
    for name, conf in scores.items():
        status = "MALIGNANT" if conf > 50 else "BENIGN"
        pdf.cell(200, 8, f"- {name}: {conf:.2f}% Confidence Score ({status})", ln=True)
    
    pdf.ln(5)
    
    # Visualization
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, "Detection & Localization Mapping", ln=True)
    pdf.image(main_plot_path, x=10, y=None, w=180)
    pdf.ln(2)
    pdf.image(zoom_plot_path, x=50, y=None, w=100)
    
    pdf.set_y(-25)
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 10, "Generated via AI Ensemble Diagnostic Suite. Review by Radiologist required.", align='C')
    
    return pdf.output(dest="S").encode("latin-1")

# =============================================================================
# 3. UI LAYOUT & EXECUTION
# =============================================================================
st.set_page_config(page_title="Multi-Model Diagnostic Suite", layout="wide")
st.title("🫁 Multi-Model AI Lung Cancer Localization")

models_dict, config = load_all_models()
candidates_df = pd.read_csv(os.path.join(config.ROOT_DIR, 'Common CSV files', 'candidates_V2.csv'))

if st.sidebar.button("🔬 Pick Random Patient"):
    pos_paths = glob(os.path.join(config.PREPROCESSED_PATH, 'trans_pre_subset*', 'images', 'pos*.npy'))
    st.session_state.sample_path = random.choice(pos_paths)

if 'sample_path' in st.session_state:
    img_full, v_coords, uid = get_synchronized_data(st.session_state.sample_path, config, candidates_df)
    patch_3d = np.load(st.session_state.sample_path)
    vx, vy, vz = v_coords[0], v_coords[1], v_coords[2]

    # Inference
    img_input = patch_3d[24:40, :, :]
    img_tensor = torch.from_numpy(img_input).float().unsqueeze(0).to(config.DEVICE)
    
    results = {}
    with torch.no_grad():
        for name, model in models_dict.items():
            output = model(img_tensor)
            mask = torch.sigmoid(output[0]).cpu().numpy()[0, 0] if isinstance(output, tuple) else None
            logits = output[1] if isinstance(output, tuple) else output
            conf = torch.sigmoid(logits).item() * 100
            results[name] = {"conf": conf, "mask": mask}

    # UI Display
    st.subheader(f"Diagnostic Analysis | Patient UID: {uid}")
    
    # Main Comparison Plots
    fig_main, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='black')
    for i, (name, data) in enumerate(results.items()):
        axes[i].imshow(img_full[vz, :, :], cmap='gray')
        if data['conf'] > 50:
            rect = patches.Rectangle((vx-20, vy-20), 40, 40, lw=2, edgecolor='red', facecolor='none')
            axes[i].add_patch(rect)
            axes[i].set_title(f"{name}: {data['conf']:.1f}%", color='red')
        else:
            axes[i].set_title(f"{name}: {data['conf']:.1f}%", color='green')
        axes[i].axis('off')
    st.pyplot(fig_main)

    st.markdown("---")
    
    # High-Res Patch Comparison
    fig_zoom, z_axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='black')
    for i, (name, data) in enumerate(results.items()):
        z_axes[i].imshow(patch_3d[32, :, :], cmap='bone')
        if data['mask'] is not None:
            z_axes[i].imshow(data['mask'], cmap='jet', alpha=0.3)
            z_axes[i].set_title(f"{name} Segmentation", color='white')
        else:
            z_axes[i].set_title(f"{name} Feature View", color='white')
        z_axes[i].axis('off')
    st.pyplot(fig_zoom)

    # EXPORT SECTION
    st.sidebar.markdown("---")
    if st.sidebar.button("📄 Generate Diagnostic Report"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_main, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_zoom:
            
            fig_main.savefig(tmp_main.name, facecolor='white', bbox_inches='tight')
            fig_zoom.savefig(tmp_zoom.name, facecolor='white', bbox_inches='tight')
            
            pdf_bytes = generate_pdf_report(uid, {k: v['conf'] for k, v in results.items()}, tmp_main.name, tmp_zoom.name)
            
            st.sidebar.download_button(
                label="💾 Download PDF",
                data=pdf_bytes,
                file_name=f"Diagnostic_Report_{uid}.pdf",
                mime="application/pdf"
            )

else:
    st.info("Select 'Pick Random Patient' to begin the comparative diagnosis.")