import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from glob import glob
from sklearn.metrics import (roc_auc_score, confusion_matrix, 
                             f1_score, precision_score, recall_score, 
                             precision_recall_curve, roc_curve)
from sklearn.model_selection import train_test_split

# Importing specialist architectures
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
        self.z_start, self.z_end = 24, 40 

    def __len__(self): return len(self.file_list)

    def __getitem__(self, idx):
        file_name, label = self.file_list[idx]
        
        # Load raw 3D array (Zero Loss HU values)
        img_3d = np.load(os.path.join(self.img_dir, file_name))
        img_16 = img_3d[self.z_start:self.z_end, :, :]
        
        # Robust Normalization
        img_tensor = torch.from_numpy(img_16).float()
        img_tensor = (img_tensor - img_tensor.mean()) / (img_tensor.std() + 1e-6)

        if label == 1:
            mask_3d = np.load(os.path.join(self.msk_dir, file_name))
            mask_2d = mask_3d[32, :, :] 
            mask_tensor = torch.from_numpy(mask_2d).float().unsqueeze(0)
        else:
            mask_tensor = torch.zeros((1, 64, 64))

        return img_tensor, mask_tensor, torch.tensor(label, dtype=torch.float32)

# =============================================================================
# 2. ENSEMBLE BRAIN ARCHITECTURE (Exact match to your training script)
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
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 1) 
        )

    def forward(self, x):
        sc1 = self.simple_cnn.enc1(x)
        sc4 = self.simple_cnn.enc4(self.simple_cnn.pool(self.simple_cnn.enc3(self.simple_cnn.pool(self.simple_cnn.enc2(self.simple_cnn.pool(sc1))))))
        feat_cnn = torch.flatten(self.simple_cnn.avgpool(sc4), 1)

        r = self.resnet_18.relu(self.resnet_18.bn1(self.resnet_18.first_conv(x)))
        l4 = self.resnet_18.layer4(self.resnet_18.layer3(self.resnet_18.layer2(self.resnet_18.layer1(self.resnet_18.maxpool(r)))))
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
# 3. COMPREHENSIVE EVALUATION ENGINE
# =============================================================================
def main():
    config = TransUNetConfig()
    device = config.DEVICE
    print(f"🔬 FULLY CONFIGURED EVALUATION: Scratch Model Baseline")

    data_path = os.path.join(r'I:\Lung Cancer Project (Simple CNN)', "LUNA16_High_Volume_Data")
    img_dir, msk_dir = os.path.join(data_path, "images"), os.path.join(data_path, "masks")
    
    all_files = glob(os.path.join(img_dir, "*.npy"))
    file_list = [(os.path.basename(f), 1 if "pos" in f else 0) for f in all_files]
    
    # Must use same random state as training to evaluate on unseen validation data
    _, val_files = train_test_split(file_list, test_size=0.15, random_state=config.SEED)
    val_loader = DataLoader(EnsembleEvalDataset(val_files, img_dir, msk_dir), batch_size=4, shuffle=False)

    model = UltimateEnsembleBrain(in_channels=16).to(device)
    
    # Load the scratch trained weights
    save_path = "ultimate_ensemble_brain_SCRATCH_FINAL.pth"
    
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
        print(f"✅ Baseline Scratch Weights Loaded Successfully.")
    else:
        print(f"❌ Error: {save_path} not found. Check filename.")
        return

    model.eval()
    y_true, y_probs_raw_list, dice_scores = [], [], []

    print("🏃 Diagnostic Inference in progress...")
    for img, msk, lbl in tqdm(val_loader):
        img, msk, lbl = img.to(device), msk.to(device), lbl.to(device)
        with torch.no_grad():
            p_mask, p_clf = model(img)
            
            prob = torch.sigmoid(p_clf).cpu().numpy().flatten()
            y_probs_raw_list.extend(prob)
            y_true.extend(lbl.cpu().numpy())

            # Targeted Dice Score calculation (Positives Only)
            pos_idx = (lbl > 0).nonzero(as_tuple=True)[0]
            if len(pos_idx) > 0:
                p_sig = torch.sigmoid(p_mask[pos_idx])
                gt_msk = msk[pos_idx]
                for i in range(len(p_sig)):
                    inter = (p_sig[i] * gt_msk[i]).sum()
                    uni = p_sig[i].sum() + gt_msk[i].sum()
                    dice_scores.append(((2. * inter) / (uni + 1e-6)).item())

    # --- METRICS & BASELINE THRESHOLDING ---
    y_true = np.array(y_true)
    y_probs_raw = np.array(y_probs_raw_list)
    
    # Standard 0.5 Threshold for raw baseline performance
    threshold = 0.5
    y_pred = (y_probs_raw > threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / len(y_true)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mean_dice = np.mean(dice_scores) if dice_scores else 0.0
    auc = roc_auc_score(y_true, y_probs_raw)

    # --- REPORT ---
    print("\n" + "="*15 + " BASELINE CLINICAL REPORT " + "="*15)
    print(f"Accuracy: {accuracy:.4f} | AUC: {auc:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f} | Precision: {precision:.4f}")
    print(f"Mean Dice Score: {mean_dice:.4f} | F1-Score: {f1:.4f}")
    print("-" * 47)
    print(f"Conf Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print("="*47)

    # --- VISUALIZATIONS ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Healthy', 'Nodule'], yticklabels=['Healthy', 'Nodule'])
    axes[0].set_title('Confusion Matrix: Baseline Model')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_probs_raw)
    axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate (Recall)')
    axes[1].set_title('Receiver Operating Characteristic')
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig('baseline_full_report.png')
    print("🖼️ Full Evaluation Plot saved as 'baseline_full_report.png'")
    plt.show()

if __name__ == "__main__": main()