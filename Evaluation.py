import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split  # Added missing import
from tqdm import tqdm
import os
from glob import glob # Changed this to fix the TypeError

# Import your existing architecture and dataset
from TransUNet_model import TransUNetConfig
from ensemble_evaluation import UltimateEnsembleBrain, EnsembleEvalDataset

# =============================================================================
# 1. THE RELIABILITY ENGINE
# =============================================================================
def run_calibration_audit(model, loader, device):
    """
    Analyzes model honesty by comparing predicted confidence to actual accuracy.
    Objective: Prove the 0.7995+ Dice results are clinically grounded.
    """
    model.eval()
    all_probs = []
    all_labels = []

    print("🔬 Executing Clinical Honesty Audit...")
    with torch.no_grad():
        for img, _, lbl in tqdm(loader, desc="Calibrating"):
            img = img.to(device)
            # Forward pass to get malignancy logits
            _, p_clf = model(img)
            
            # Map logits to probability space [0, 1]
            prob = torch.sigmoid(p_clf).cpu().numpy().flatten()
            all_probs.extend(prob)
            all_labels.extend(lbl.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # 1. Brier Score
    brier = brier_score_loss(all_labels, all_probs)
    
    # 2. Calibration Curve (Reliability data in 10 confidence bins)
    prob_true, prob_pred = calibration_curve(all_labels, all_probs, n_bins=10)

    # 3. Expected Calibration Error (ECE) Calculation
    ece = np.abs(prob_pred - prob_true).mean()

    # 4. Generate Visualization for IEEE Report
    plt.figure(figsize=(10, 8), facecolor='white')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Ideal (Perfectly Honest)')
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, color='blue', 
             label=f'Brain V4 (ECE: {ece:.4f})')
    
    plt.xlabel('Mean Predicted Malignancy Confidence')
    plt.ylabel('Empirical Accuracy (Fraction of Positives)')
    plt.title('Reliability Analysis: Master Brain V4 Calibration')
    plt.legend(loc='upper left')
    plt.grid(alpha=0.3)
    
    plt.savefig('calibration_reliability_report.png', dpi=300)
    print(f"\n✅ Audit Complete. Reliability Graph saved to disk.")
    return brier, ece

# =============================================================================
# 2. MAIN EXECUTION FUNCTION
# =============================================================================
def main():
    config = TransUNetConfig()
    device = config.DEVICE
    
    # Configuration for 1.57GB 16-channel dataset
    print(f"🚀 Initializing Evaluation for 1.57GB Grayscale Pipeline...")
    
    # Instantiate Master Brain with 16 input channels
    model = UltimateEnsembleBrain(in_channels=16).to(device)
    
    # Load your current weights
    save_path = "ultimate_ensemble_brain_v4_FINAL.pth"
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
        print(f"⭐ V4 weights loaded successfully (Current Dice: 0.7995).")
    else:
        print(f"❌ Error: {save_path} not found.")
        return

    # Prepare Validation Data
    # Ensure this path matches where your 1.57GB .npz files are stored
    data_path = os.path.join(config.ROOT_DIR, "Ensemble_Data_Safe")
    img_dir = os.path.join(data_path, "images")
    msk_dir = os.path.join(data_path, "masks")
    
    # Logic to fetch and split files
    all_files = glob(os.path.join(img_dir, "*.npz"))
    if not all_files:
        print(f"❌ Error: No .npz files found in {img_dir}")
        return

    file_list = [(os.path.basename(f), 1 if "pos" in f else 0) for f in all_files]
    _, val_files = train_test_split(file_list, test_size=0.2, random_state=42)
    
    val_loader = DataLoader(EnsembleEvalDataset(val_files, img_dir, msk_dir), 
                            batch_size=8, shuffle=False)

    # Execute Honesty Audit
    brier, ece = run_calibration_audit(model, val_loader, device)
    
    print("\n" + "="*50)
    print("🏆 FINAL HONESTY VERIFICATION REPORT 🏆")
    print("="*50)
    print(f"{'Metric':<25} | {'Value':<10}")
    print("-" * 40)
    print(f"{'Brier Score (Calibration)':<25} | {brier:.4f}")
    print(f"{'Exp. Calibration Error':<25} | {ece:.4f}")
    print("-" * 40)
    
    if ece < 0.05:
        print("💡 VERDICT: MODEL IS CLINICALLY SENSIBLE")
    else:
        print("💡 VERDICT: MODEL EXHIBITS OVERCONFIDENCE BIAS")
    print("="*50)

if __name__ == "__main__":
    main()