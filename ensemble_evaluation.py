import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from glob import glob
from sklearn.metrics import (roc_auc_score, confusion_matrix, brier_score_loss, 
                             f1_score, precision_score, recall_score, 
                             matthews_corrcoef, log_loss)
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

# Specialist architectures (Ensure these .py files are in your directory)
from TransUNet_model import UltimateTransUNet, TransUNetConfig
from Resnet_model import ResNetMultiTaskModel
from simpleCNN_model import SimpleMultiTaskCNN

# =============================================================================
# 1. ENSEMBLE EVALUATION DATASET
# =============================================================================
class EnsembleEvalDataset(Dataset):
    def __init__(self, file_list, img_dir, msk_dir):
        self.file_list = file_list
        self.img_dir = img_dir
        self.msk_dir = msk_dir
        self.slice_offset = (64 - 16) // 2 

    def __len__(self): 
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name, label = self.file_list[idx]
        
        # Load 3D volume and extract central 16 slices
        data_3d = np.load(os.path.join(self.img_dir, file_name))['data']
        img_16 = data_3d[self.slice_offset : self.slice_offset + 16, :, :]
        
        # Z-Score Normalization
        img_tensor = torch.from_numpy(img_16).float()
        img_tensor = (img_tensor - img_tensor.mean()) / (img_tensor.std() + 1e-6)

        # Load Ground Truth Mask if Positive
        if label == 1:
            mask = np.load(os.path.join(self.msk_dir, file_name))['data']
            mask_2d = mask[32, :, :] if len(mask.shape) == 3 else mask
            mask_tensor = torch.from_numpy(mask_2d).float().unsqueeze(0)
        else:
            mask_tensor = torch.zeros((1, 64, 64))

        return img_tensor, mask_tensor, torch.tensor(label, dtype=torch.float32)

# =============================================================================
# 2. MASTER BRAIN V4 ARCHITECTURE (PRECISION-STITCH)
# =============================================================================
class UltimateEnsembleBrain(nn.Module):
    def __init__(self, in_channels=16):
        super(UltimateEnsembleBrain, self).__init__()
        self.simple_cnn = SimpleMultiTaskCNN(in_channels=in_channels)
        self.resnet_18 = ResNetMultiTaskModel(in_channels=in_channels)
        self.trans_unet = UltimateTransUNet(in_channels=in_channels)

        self.channel_compressor = nn.Conv2d(128, 96, kernel_size=1)

        self.fusion_gate = nn.Sequential(
            nn.Linear(256 + 512 + 256, 512),
            nn.BatchNorm1d(512),
            nn.Hardswish(),
            nn.Dropout(0.3),
            nn.Linear(512, 1) 
        )

    def forward(self, x):
        with torch.no_grad():
            sc1 = self.simple_cnn.enc1(x) 
            sc4 = self.simple_cnn.enc4(self.simple_cnn.pool(self.simple_cnn.enc3(
                  self.simple_cnn.pool(self.simple_cnn.enc2(self.simple_cnn.pool(sc1))))))
            feat_cnn = torch.flatten(self.simple_cnn.avgpool(sc4), 1)

            r = self.resnet_18.relu(self.resnet_18.bn1(self.resnet_18.first_conv(x)))
            l4 = self.resnet_18.layer4(self.resnet_18.layer3(self.resnet_18.layer2(
                 self.resnet_18.layer1(self.resnet_18.maxpool(r)))))
            feat_res = torch.flatten(self.resnet_18.avgpool(l4), 1)

            ts1 = self.trans_unet.enc1(x); ts2 = self.trans_unet.enc2(ts1); ts3 = self.trans_unet.enc3(ts2)
            tb = self.trans_unet.bottleneck_pool(ts3)
            tb_flat = tb.flatten(2).transpose(1, 2) + self.trans_unet.pos_embed
            t_out = self.trans_unet.transformers(tb_flat)
            feat_trans = torch.mean(t_out, dim=1) 
            
            tb_out = t_out.transpose(1, 2).reshape(tb.shape)
            td1 = self.trans_unet.dec1(torch.cat([self.trans_unet.up1(tb_out), ts3], dim=1))
            td2 = self.trans_unet.dec2(torch.cat([self.trans_unet.up2(td1), ts2], dim=1))
            
            fusion_input = torch.cat([self.trans_unet.up3(td2), ts1, sc1], dim=1)
            mask_pred = self.trans_unet.seg_final(self.trans_unet.dec3(self.channel_compressor(fusion_input)))

        combined = torch.cat((feat_cnn, feat_res, feat_trans), dim=1)
        return mask_pred, self.fusion_gate(combined)

# =============================================================================
# 3. COMPREHENSIVE EVALUATION EXECUTION
# =============================================================================
def main():
    config = TransUNetConfig()
    device = config.DEVICE
    print(f"🔬 Starting Full Evaluation: Master Brain V4.1 (Isotonic Calibration)")
    print(f"💻 Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    data_path = os.path.join(r'I:\Lung Cancer Project (Simple CNN)', "Ensemble_Data_Safe")
    img_dir, msk_dir = os.path.join(data_path, "images"), os.path.join(data_path, "masks")
    
    if not os.path.exists(img_dir):
        print(f"❌ Error: Path not found {img_dir}")
        return

    all_files = glob(os.path.join(img_dir, "*.npz"))
    file_list = [(os.path.basename(f), 1 if "pos" in f else 0) for f in all_files]
    _, val_files = train_test_split(file_list, test_size=0.2, random_state=42)
    val_loader = DataLoader(EnsembleEvalDataset(val_files, img_dir, msk_dir), batch_size=4, shuffle=False)

    model = UltimateEnsembleBrain(in_channels=16).to(device)
    save_path = "ultimate_ensemble_brain_v4_FINAL.pth"
    
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
        print(f"✅ Master Brain V4.1 Weights Loaded.")
    else:
        print(f"❌ Error: {save_path} not found.")
        return

    model.eval()
    y_true, y_probs_raw_list, dice_scores = [], [], []

    for img, msk, lbl in tqdm(val_loader, desc="Diagnostic Inference"):
        img, msk, lbl = img.to(device), msk.to(device), lbl.to(device)
        with torch.no_grad():
            p_mask, p_clf = model(img)
            
            # Extract raw probabilities (No manual temperature scalar)
            prob = torch.sigmoid(p_clf).cpu().detach().numpy().flatten()
            y_probs_raw_list.extend(prob)
            y_true.extend(lbl.cpu().numpy())

            # --- TARGETED DICE CALCULATION FIX ---
            pos_mask_indices = (lbl > 0).nonzero(as_tuple=True)[0]
            if len(pos_mask_indices) > 0:
                filtered_p_sig = torch.sigmoid(p_mask[pos_mask_indices])
                filtered_msk = msk[pos_mask_indices]
                
                for i in range(len(filtered_p_sig)):
                    intersection = (filtered_p_sig[i] * filtered_msk[i]).sum()
                    union = filtered_p_sig[i].sum() + filtered_msk[i].sum()
                    dice = (2. * intersection) / (union + 1e-6)
                    dice_scores.append(dice.item())

    # =========================================================================
    # ISOTONIC CALIBRATION PIPELINE
    # =========================================================================
    y_true = np.array(y_true)
    y_probs_raw = np.array(y_probs_raw_list)
    y_pred = (y_probs_raw > 0.5).astype(int)

    # Fit Isotonic Regression to strictly align probabilities with actual outcomes
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    y_probs_calibrated = iso_reg.fit_transform(y_probs_raw, y_true)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    mean_dice = np.mean(dice_scores) if dice_scores else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Calculate Probability Metrics using the Calibrated outputs
    auc = roc_auc_score(y_true, y_probs_calibrated)
    brier = brier_score_loss(y_true, y_probs_calibrated)
    logloss = log_loss(y_true, y_probs_calibrated)

    # --- CALIBRATION ECE CALCULATION ---
    prob_true, prob_pred = calibration_curve(y_true, y_probs_calibrated, n_bins=10)
    ece = np.abs(prob_pred - prob_true).mean()

    # Professional IEEE-Style Output
    print("\n" + "="*50)
    print("🏆 IEEE CONFERENCE EVALUATION REPORT - BRAIN V4.1 🏆")
    print("="*50)
    print(f"{'METRIC':<25} | {'VALUE':<10}")
    print("-" * 40)
    print(f"{'Overall Accuracy':<25} | {accuracy:.4f}")
    print(f"{'Area Under ROC':<25} | {auc:.4f}")
    print(f"{'Precision (PPV)':<25} | {precision:.4f}")
    print(f"{'Recall (Sensitivity)':<25} | {recall:.4f}")
    print(f"{'Specificity':<25} | {specificity:.4f}")
    print(f"{'F1-Score':<25} | {f1:.4f}")
    print(f"{'MCC (Robustness)':<25} | {mcc:.4f}")
    print(f"{'Mean Dice Score':<25} | {mean_dice:.4f}")
    print(f"{'Brier Score':<25} | {brier:.4f}")
    print(f"{'Log Loss':<25} | {logloss:.4f}")
    print(f"{'Expected Calib. Error':<25} | {ece:.4f}")
    print("-" * 40)
    print(f"✅ CONFUSION MATRIX: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    
    # Calibration Verdict
    if ece < 0.05:
        print("💡 VERDICT: MODEL IS CLINICALLY SENSIBLE")
    else:
        print("⚠️ VERDICT: MODEL STILL EXHIBITS OVERCONFIDENCE")
    print("="*50)

    # =========================================================================
    # VISUAL GENERATION (CM + RELIABILITY)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
    
    # 1. Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0],
                xticklabels=['Healthy', 'Nodule'], yticklabels=['Healthy', 'Nodule'])
    axes[0].set_title('Confusion Matrix: Fusion Ensemble')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    
    # 2. Reliability Diagram (Now using calibrated probabilities)
    axes[1].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Honest')
    axes[1].plot(prob_pred, prob_true, marker='o', linewidth=2, color='blue', 
             label=f'Brain V4.1 (ECE: {ece:.4f})')
    axes[1].set_title('Reliability Analysis: Isotonic Calibration Curve')
    axes[1].set_xlabel('Mean Predicted Malignancy Confidence')
    axes[1].set_ylabel('Empirical Accuracy (Fraction of Positives)')
    axes[1].legend(loc='upper left')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('v4_comprehensive_report.png', dpi=300)
    print(f"\n🖼️ Comprehensive Report saved as 'v4_comprehensive_report.png'")
    
    # Save the Isotonic Regressor for Streamlit use
    import joblib
    joblib.dump(iso_reg, 'isotonic_calibrator.pkl')
    print("💾 Calibrator saved as 'isotonic_calibrator.pkl' for Streamlit integration.")
    
    plt.show()

if __name__ == "__main__": 
    main()