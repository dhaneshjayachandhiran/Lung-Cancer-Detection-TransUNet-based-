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
        
        # Z-Score Normalization (Synced with V4 Training Pipeline)
        img_tensor = torch.from_numpy(img_16).float()
        img_tensor = (img_tensor - img_tensor.mean()) / (img_tensor.std() + 1e-6)

        # Load Ground Truth Mask if Positive
        if label == 1:
            mask = np.load(os.path.join(self.msk_dir, file_name))['data']
            # Extract central slice for 2D segmentation comparison
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
        # Initialize the specialist streams
        self.simple_cnn = SimpleMultiTaskCNN(in_channels=in_channels)
        self.resnet_18 = ResNetMultiTaskModel(in_channels=in_channels)
        self.trans_unet = UltimateTransUNet(in_channels=in_channels)

        # V4 Skip-Fusion Compressor: Merges high-res CNN edges with TransUNet Decoder
        self.channel_compressor = nn.Conv2d(128, 96, kernel_size=1)

        # Final Malignancy Decision Head
        self.fusion_gate = nn.Sequential(
            nn.Linear(256 + 512 + 256, 512),
            nn.BatchNorm1d(512),
            nn.Hardswish(),
            nn.Dropout(0.3),
            nn.Linear(512, 1) 
        )

    def forward(self, x):
        with torch.no_grad():
            # 1. SimpleCNN Edge Stream
            sc1 = self.simple_cnn.enc1(x) 
            sc4 = self.simple_cnn.enc4(self.simple_cnn.pool(self.simple_cnn.enc3(
                  self.simple_cnn.pool(self.simple_cnn.enc2(self.simple_cnn.pool(sc1))))))
            feat_cnn = torch.flatten(self.simple_cnn.avgpool(sc4), 1)

            # 2. ResNet-18 Texture Stream
            r = self.resnet_18.relu(self.resnet_18.bn1(self.resnet_18.first_conv(x)))
            l4 = self.resnet_18.layer4(self.resnet_18.layer3(self.resnet_18.layer2(
                 self.resnet_18.layer1(self.resnet_18.maxpool(r)))))
            feat_res = torch.flatten(self.resnet_18.avgpool(l4), 1)

            # 3. TransUNet Global Context Stream
            ts1 = self.trans_unet.enc1(x); ts2 = self.trans_unet.enc2(ts1); ts3 = self.trans_unet.enc3(ts2)
            tb = self.trans_unet.bottleneck_pool(ts3)
            tb_flat = tb.flatten(2).transpose(1, 2) + self.trans_unet.pos_embed
            t_out = self.trans_unet.transformers(tb_flat)
            feat_trans = torch.mean(t_out, dim=1) 
            
            # V4 Skip-Fusion Decoder
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
    print(f"🔬 Starting Full Evaluation: Master Brain V4 (Precision-Stitch)")
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
        print(f"✅ Master Brain V4 Weights Loaded.")
    else:
        print(f"❌ Error: {save_path} not found.")
        return

    model.eval()
    y_true, y_probs, dice_scores = [], [], []

    for img, msk, lbl in tqdm(val_loader, desc="Diagnostic Inference"):
        img, msk, lbl = img.to(device), msk.to(device), lbl.to(device)
        with torch.no_grad():
            p_mask, p_clf = model(img)
            prob = torch.sigmoid(p_clf).cpu().detach().numpy().flatten()
            y_probs.extend(prob)
            y_true.extend(lbl.cpu().numpy())

            if lbl.sum() > 0:
                p_sig = torch.sigmoid(p_mask)
                intersection = (p_sig * msk).sum()
                union = p_sig.sum() + msk.sum()
                dice = (2. * intersection) / (union + 1e-6)
                dice_scores.append(dice.item())

    # Final Metric Calculations
    y_true, y_probs = np.array(y_true), np.array(y_probs)
    y_pred = (y_probs > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_probs)
    specificity = tn / (tn + fp)
    mcc = matthews_corrcoef(y_true, y_pred)
    brier = brier_score_loss(y_true, y_probs)
    logloss = log_loss(y_true, y_probs)
    mean_dice = np.mean(dice_scores) if dice_scores else 0.0

    # Professional IEEE-Style Output
    print("\n" + "="*50)
    print("🏆 IEEE CONFERENCE EVALUATION REPORT - BRAIN V4 🏆")
    print("="*50)
    print(f"{'METRIC':<20} | {'VALUE':<10}")
    print("-" * 35)
    print(f"{'Overall Accuracy':<20} | {accuracy:.4f}")
    print(f"{'Area Under ROC':<20} | {auc:.4f}")
    print(f"{'Precision (PPV)':<20} | {precision:.4f}")
    print(f"{'Recall (Sensitivity)':<20} | {recall:.4f}")
    print(f"{'Specificity':<20} | {specificity:.4f}")
    print(f"{'F1-Score':<20} | {f1:.4f}")
    print(f"{'MCC (Robustness)':<20} | {mcc:.4f}")
    print(f"{'Mean Dice Score':<20} | {mean_dice:.4f}")
    print(f"{'Brier Score':<20} | {brier:.4f}")
    print(f"{'Log Loss':<20} | {logloss:.4f}")
    print("-" * 35)
    print(f"✅ CONFUSION MATRIX: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print("="*50)

    # =========================================================================
    # VISUAL CONFUSION MATRIX GENERATION
    # =========================================================================
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Healthy', 'Nodule'], 
                yticklabels=['Healthy', 'Nodule'])
    plt.title('Confusion Matrix: Lung Nodule Detection (Tri-Stream Fusion Ensemble)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Save for your paper
    plt.savefig('confusion_matrix_v4.png', dpi=300)
    print(f"\n🖼️ Confusion Matrix saved as 'confusion_matrix_v4.png'")
    plt.show()

if __name__ == "__main__": 
    main()