import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from glob import glob
from tqdm import tqdm
from TransUNet_model import TransUNetConfig

# =============================================================================
# 1. STORAGE-SAVING UTILITIES
# =============================================================================
def normalize_and_compress(image):
    """Convert to 16-bit float and scale to 0-1 for 50% space savings."""
    image = (image - (-1000)) / (400 - (-1000))
    return np.clip(image, 0, 1).astype(np.float16)

def create_compact_mask(shape, center, diameter, spacing):
    """Create 8-bit integer mask to reduce storage by 4x."""
    mask = np.zeros(shape, dtype=np.uint8)
    rz, ry, rx = (diameter / 2) / spacing[0], (diameter / 2) / spacing[1], (diameter / 2) / spacing[2]
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist = ((z - center[0])**2 / rz**2) + ((y - center[1])**2 / ry**2) + ((x - center[2])**2 / rx**2)
    mask[dist <= 1.0] = 1
    return mask

# =============================================================================
# 2. THE 75GB LIMIT PREPROCESSOR
# =============================================================================
def preprocess_limited_ensemble():
    config = TransUNetConfig()
    save_base = os.path.join(config.ROOT_DIR, "Ensemble_Data_Safe")
    img_dir, msk_dir = os.path.join(save_base, "images"), os.path.join(save_base, "masks")
    os.makedirs(img_dir, exist_ok=True); os.makedirs(msk_dir, exist_ok=True)

    # Load candidates and sub-sample negatives to prevent data explosion
    df = pd.read_csv(os.path.join(config.ROOT_DIR, 'Common CSV files', 'candidates_V2.csv'))
    pos_df = df[df['class'] == 1]
    # Limit negatives to a reasonable ratio (e.g., 2 negatives for every 1 positive)
    neg_df = df[df['class'] == 0].sample(n=len(pos_df) * 2, random_state=config.SEED)
    candidates = pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)

    mhd_files = {os.path.basename(f).replace('.mhd', ''): f for f in glob(os.path.join(config.ROOT_DIR, 'Subsets', 'subset*', '*.mhd'))}
    
    print(f"🎯 Targeted Preprocessing: {len(candidates)} total samples (Goal: <75GB)")

    for uid in tqdm(candidates['seriesuid'].unique(), desc="Processing Volumes"):
        if uid not in mhd_files: continue
        
        itk_img = sitk.ReadImage(mhd_files[uid])
        img_array = sitk.GetArrayFromImage(itk_img)
        origin, spacing = np.array(itk_img.GetOrigin()), np.array(itk_img.GetSpacing())
        vol_cands = candidates[candidates['seriesuid'] == uid]

        for idx, row in vol_cands.iterrows():
            v_coords = np.round(np.abs(np.array([row['coordX'], row['coordY'], row['coordZ']]) - origin) / spacing).astype(int)
            z, y, x = v_coords[2], v_coords[1], v_coords[0]

            patch = img_array[max(0, z-32):z+32, max(0, y-32):y+32, max(0, x-32):x+32]
            if patch.shape != (64, 64, 64):
                patch = np.pad(patch, [(0, 64-patch.shape[0]), (0, 64-patch.shape[1]), (0, 64-patch.shape[2])], constant_values=-1000)

            # SAVE COMPRESSED NPZ
            save_name = f"{'pos' if row['class']==1 else 'neg'}_{uid}_{idx}.npz"
            np.savez_compressed(os.path.join(img_dir, save_name), data=normalize_and_compress(patch))

            if row['class'] == 1:
                mask = create_compact_mask((64, 64, 64), [32, 32, 32], row.get('diameter_mm', 10.0), spacing)
                np.savez_compressed(os.path.join(msk_dir, save_name), data=mask)

    print(f"✅ Safe preprocessing complete! Check folder: {save_base}")

if __name__ == "__main__":
    preprocess_limited_ensemble()