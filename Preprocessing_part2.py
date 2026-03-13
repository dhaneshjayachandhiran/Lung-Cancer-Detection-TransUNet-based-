import SimpleITK as sitk
import numpy as np
import os
from glob import glob
from tqdm import tqdm

# Paths
INPUT_DIR = "Subsets"
OUTPUT_DIR = "LUNA16_High_Volume_Data/Compressed_UI_Scans"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Find all .mhd files
mhd_files = glob(os.path.join(INPUT_DIR, "**", "*.mhd"), recursive=True)

print(f"Found {len(mhd_files)} scans. Starting compression...")

for mhd_path in tqdm(mhd_files):
    # Extract filename
    seriesuid = os.path.basename(mhd_path).replace('.mhd', '')
    
    # Read original scan
    itk_img = sitk.ReadImage(mhd_path)
    img_array = sitk.GetArrayFromImage(itk_img)
    origin = np.array(itk_img.GetOrigin())
    spacing = np.array(itk_img.GetSpacing())
    
    # Downcast to int16 to save memory (HU values easily fit in int16)
    img_array_int16 = img_array.astype(np.int16)
    
    # Save as a highly compressed .npz file
    out_path = os.path.join(OUTPUT_DIR, f"{seriesuid}.npz")
    np.savez_compressed(
        out_path, 
        img=img_array_int16, 
        origin=origin, 
        spacing=spacing
    )

print("Compression Complete! You can now use the new folder for the UI.")