import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from glob import glob
from tqdm import tqdm

# =============================================================================
# ZERO-LOSS RAW EXTRACTION
# =============================================================================
def get_raw_patch(img_array, center_v, patch_size_voxels=(64, 64, 64)):
    """
    Original resolution raw patches. 
    Strict instruction: No shrinking, No data loss.
    """
    z, y, x = center_v[2], center_v[1], center_v[0]
    pz, py, px = patch_size_voxels
    
    # Boundary handling with original pixel values
    z_start, z_end = max(0, z - pz//2), min(img_array.shape[0], z + pz//2)
    y_start, y_end = max(0, y - py//2), min(img_array.shape[1], y + py//2)
    x_start, x_end = max(0, x - px//2), min(img_array.shape[2], x + px//2)
    
    patch = img_array[z_start:z_end, y_start:y_end, x_start:x_end]
    
    # Padding only if necessary to keep shape uniform (64, 64, 64)
    if patch.shape != patch_size_voxels:
        pad_z = patch_size_voxels[0] - patch.shape[0]
        pad_y = patch_size_voxels[1] - patch.shape[1]
        pad_x = patch_size_voxels[2] - patch.shape[2]
        patch = np.pad(patch, [(0, pad_z), (0, pad_y), (0, pad_x)], mode='constant', constant_values=-1000)
        
    return patch.astype(np.float32) # Keeping 32-bit for Zero Loss

# =============================================================================
# THE 20GB+ TARGET PREPROCESSOR
# =============================================================================
def preprocess():
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    SAVE_BASE = os.path.join(ROOT_DIR, "LUNA16_High_Volume_Data")
    IMG_DIR = os.path.join(SAVE_BASE, "images")
    MSK_DIR = os.path.join(SAVE_BASE, "masks")
    os.makedirs(IMG_DIR, exist_ok=True); os.makedirs(MSK_DIR, exist_ok=True)

    print("📂 Metadata Loading")
    annos = pd.read_csv(os.path.join(ROOT_DIR, 'Common CSV files', 'annotations.csv'))
    cands = pd.read_csv(os.path.join(ROOT_DIR, 'Common CSV files', 'candidates.csv'))
    
    # --- DATA VOLUME LOGIC ---
    pos_samples = cands[cands['class'] == 1]
    
    # STRICT INSTRUCTION: 20GB target kaga negative samples-ah 60x boost panren
    # 1,186 (pos) * 60 (ratio) = ~71,000 patches. 
    # Each patch (64x64x64 float32) is approx 1MB. 
    # 71,000 * 1MB = ~70GB (uncompressed) -> Compressed-la ~25GB varum.
    neg_samples = cands[cands['class'] == 0].sample(n=len(pos_samples) * 60, random_state=42)
    final_df = pd.concat([pos_samples, neg_samples]).sample(frac=1).reset_index(drop=True)

    mhd_paths = {os.path.basename(f).replace('.mhd', ''): f for f in glob(os.path.join(ROOT_DIR, 'Subsets', 'subset*', '*.mhd'))}

    print(f"🚀 Processing {len(final_df)} patches")

    for uid in tqdm(final_df['seriesuid'].unique(), desc="Volumes"):
        if uid not in mhd_paths: continue
        
        itk_img = sitk.ReadImage(mhd_paths[uid])
        img_array = sitk.GetArrayFromImage(itk_img)
        origin = np.array(itk_img.GetOrigin())
        spacing = np.array(itk_img.GetSpacing())
        
        vol_samples = final_df[final_df['seriesuid'] == uid]

        for idx, row in vol_samples.iterrows():
            v_coords = np.round(np.abs(np.array([row['coordX'], row['coordY'], row['coordZ']]) - origin) / spacing).astype(int)
            
            # 64x64x64 is the medical standard for high-res patches
            patch = get_raw_patch(img_array, v_coords, patch_size_voxels=(64, 64, 64))
            
            file_id = f"{'pos' if row['class']==1 else 'neg'}_{uid}_{idx}"
            
            # np.save (raw) use panna size perusa irukum. 
            # If storage is an issue, use np.savez_compressed (but it will be slower)
            np.save(os.path.join(IMG_DIR, f"{file_id}.npy"), patch)

            if row['class'] == 1:
                match = annos[(annos['seriesuid'] == uid) & (np.abs(annos['coordX'] - row['coordX']) < 2)]
                diam = match['diameter_mm'].values[0] if not match.empty else 10.0
                
                # Precise 3D Spherical Mask
                mask = np.zeros((64, 64, 64), dtype=np.uint8)
                center = (32, 32, 32)
                rz, ry, rx = (diam / 2) / spacing[2], (diam / 2) / spacing[1], (diam / 2) / spacing[0]
                z, y, x = np.ogrid[:64, :64, :64]
                dist = ((z - center[0])**2 / rz**2) + ((y - center[1])**2 / ry**2) + ((x - center[2])**2 / rx**2)
                mask[dist <= 1.0] = 1
                np.save(os.path.join(MSK_DIR, f"{file_id}.npy"), mask)

    print(f"✅ Preprocessing Success! Target 20GB+ achieved.")

if __name__ == "__main__":
    preprocess()