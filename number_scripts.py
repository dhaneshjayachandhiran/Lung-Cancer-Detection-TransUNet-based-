import os
from glob import glob

# Path to your subsets folder
subsets_path = r'I:\Lung Cancer Project (Simple CNN)\Subsets'

def count_luna16_scans(root_path):
    # LUNA16 scans are identified by .mhd files
    # We search recursively through all subset0, subset1, etc. folders
    scan_files = glob(os.path.join(root_path, 'subset*', '*.mhd'))
    
    total_scans = len(scan_files)
    
    print("="*40)
    print(f"📊 LUNA16 DATASET SCAN COUNT")
    print("="*40)
    print(f"📂 Location: {root_path}")
    print(f"✅ Total Scans Found: {total_scans}")
    print("="*40)
    
    return total_scans

if __name__ == "__main__":
    if os.path.exists(subsets_path):
        count_luna16_scans(subsets_path)
    else:
        print(f"❌ Error: Path not found: {subsets_path}")