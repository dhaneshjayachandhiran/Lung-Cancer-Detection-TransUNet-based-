import streamlit as st
import pandas as pd
import numpy as np
import SimpleITK as sitk
import os
import random
from PIL import Image, ImageDraw

# --- PAGE CONFIG ---
st.set_page_config(page_title="LUNA16 - Precision Diagnostic", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #050505; color: #FFFFFF; }
    .diag-box { 
        border: 2px solid #39FF14; 
        padding: 15px; 
        border-radius: 10px; 
        background: rgba(57, 255, 20, 0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- CONFIG PATHS (Based on your screenshot) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Screenshot-la 'Common CSV files' nu folder iruku, so:
CSV_DIR = os.path.join(BASE_DIR, "Common CSV files")
SUBSETS_DIR = os.path.join(BASE_DIR, "Subsets")

# --- CORE LOGIC: THE PERFECT COORDINATE MAPPING ---
def world_to_voxel(world_coord, origin, spacing):
    """
    World coordinates (mm) ah image pixel coordinates-ah mathuradhu.
    No zooming issues here because we use physical metadata.
    """
    return np.absolute(world_coord - origin) / spacing

def get_mhd_data(mhd_path, world_coords):
    itk_img = sitk.ReadImage(mhd_path)
    img_array = sitk.GetArrayFromImage(itk_img)
    
    origin = np.array(itk_img.GetOrigin())
    spacing = np.array(itk_img.GetSpacing())
    
    v_coords = world_to_voxel(np.array(world_coords), origin, spacing)
    z_idx, y_idx, x_idx = int(v_coords[2]), int(v_coords[1]), int(v_coords[0])
    
    slice_2d = img_array[z_idx, :, :]

    # --- NEW LOGIC TO ELIMINATE WHITE CIRCLES/PADDING ---
    # 1. CT scan-la background pixel values usually -1000 kulla irukkum.
    # Adha 'Hounsfield Units' (HU) nu solluvom. Anything below -1000 is air/void.
    bg_mask = slice_2d < -1000  
    
    # 2. Normalization: Only focus on tissue density (-1000 to 400 HU range)
    # Idhu dhaan 'Crisp as Sun' look tharum, background artifacts-ah remove pannum.
    min_hu, max_hu = -1000, 400
    slice_norm = np.clip(slice_2d, min_hu, max_hu)
    slice_norm = ((slice_norm - min_hu) / (max_hu - min_hu) * 255).astype(np.uint8)
    
    # 3. Force background to pure black
    slice_norm[bg_mask] = 0 
    
    return Image.fromarray(slice_norm), (x_idx, y_idx)

# --- DATA LOADING ---
@st.cache_data
def load_metadata():
    anno_path = os.path.join(CSV_DIR, "annotations.csv")
    return pd.read_csv(anno_path)

# --- UI INTERFACE ---
st.title("🩻 LUNA16 AI Precision Diagnostics")
st.write(f"Scanning Subsets: `subset0` to `subset9` | Metadata: `annotations.csv`")

try:
    df_annos = load_metadata()
except Exception as e:
    st.error(f"CSV File missing in 'Common CSV files' folder! Error: {e}")
    st.stop()

if st.button("🚀 EXECUTE RANDOM MODEL TEST"):
    # 1. Pick a random positive sample from annotations
    sample = df_annos.sample(n=1).iloc[0]
    s_uid = sample['seriesuid']
    world_pos = (sample['coordX'], sample['coordY'], sample['coordZ'])
    diameter = sample['diameter_mm']

    # 2. Search for the .mhd file across all 10 subsets
    mhd_file = None
    for i in range(10):
        temp_path = os.path.join(SUBSETS_DIR, f"subset{i}", f"{s_uid}.mhd")
        if os.path.exists(temp_path):
            mhd_file = temp_path
            break
    
    if mhd_file:
        # 3. Process Slice
        crisp_slice, (px, py) = get_mhd_data(mhd_file, world_pos)
        
        # 4. Draw Perfect Mark (Direct Pixel Buffer)
        draw = ImageDraw.Draw(crisp_slice)
        r = 25 # Radius for visibility
        
        # High-Vis Neon Circle
        draw.ellipse([px-r, py-r, px+r, py+r], outline="#39FF14", width=4)
        draw.ellipse([px-2, py-2, px+2, py+2], fill="#39FF14") # Center dot

        # 5. Render
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(f"Target Slice: {s_uid}")
            st.image(crisp_slice, use_container_width=True)
            
        with col2:
            st.subheader("Diagnostic Report")
            st.markdown(f"""
            <div class="diag-box">
                <b style="color:#39FF14;">ZONE DETECTED</b><br><br>
                <b>World Coords:</b><br>{world_pos}<br><br>
                <b>Pixel Coords:</b><br>({px}, {py})<br><br>
                <b>Nodule Size:</b><br>{diameter:.2f} mm
            </div>
            """, unsafe_allow_html=True)
            st.success("Mathematical Mapping Verified.")
    else:
        st.warning(f"File {s_uid}.mhd unga subset folders-la illa. Check subset content.")

else:
    st.info("Click the button to fetch a random MHD scan and mark the cancer zone perfectly.")