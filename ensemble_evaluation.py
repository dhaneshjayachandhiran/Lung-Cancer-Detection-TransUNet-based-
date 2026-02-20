import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
from glob import glob
from sklearn.metrics import roc_auc_score, confusion_matrix, brier_score_loss, f1_score
from sklearn.model_selection import train_test_split

# Import specialist architectures from your existing files
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
        # Your models expect 16 slices as input
        self.slice_offset = (64 - 16) // 2 

    def __len__(self): return len(self.file_list)

    def __getitem__(self, idx):
        file_name, label = self.file_list[idx]
        
        # Load compressed 3D patch data
        data_3d = np.load(os.path.join(self.img_dir, file_name))['data']
        img_16 = data_3d[self.slice_offset : self.slice_offset + 16, :, :]
        img_tensor = torch.from_numpy(img_16).float()

        if label == 1:
            mask = np.load(os.path.join(self.msk_dir, file_name))['data']
            # Using middle slice (32) for 2D ground truth comparison
            mask_2d = mask[32, :, :] if len(mask.shape) == 3 else mask
            mask_tensor = torch.from_numpy(mask_2d).float().unsqueeze(0)
        else:
            mask_tensor = torch.zeros((1, 64, 64))

        return img_tensor, mask_tensor, torch.tensor(label, dtype=torch.float32)

# =============================================================================
# 2. ENSEMBLE BRAIN ARCHITECTURE (V2 STructure)
# =============================================================================
class UltimateEnsembleBrain(nn.Module):
    def __init__(self, in_channels=16):
        super(UltimateEnsembleBrain, self).__init__()
        self.simple_cnn = SimpleMultiTaskCNN(in_channels=in_channels)
        self.resnet_18 = ResNetMultiTaskModel(in_channels=in_channels)
        self.trans_unet = UltimateTransUNet(in_channels=in_channels)

        # Fusion Head for classification
        self.fusion_gate = nn.Sequential(
            nn.Linear(256 + 512 + 256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 1) 
        )

    def forward(self, x):
        # Extract features from specialists
        with torch.no_grad():
            # SimpleCNN Path
            s1 = self.simple_cnn.enc1(x)
            s4 = self.simple_cnn.enc4(self.simple_cnn.pool(self.simple_cnn.enc3(self.simple_cnn.pool(self.simple_cnn.enc2(self.simple_cnn.pool(s1))))))
            feat_cnn = torch.flatten(self.simple_cnn.avgpool(s4), 1)

            # ResNet Path
            r = self.resnet_18.maxpool(self.resnet_18.relu(self.resnet_18.bn1(self.resnet_18.first_conv(x))))
            l4 = self.resnet_18.layer4(self.resnet_18.layer3(self.resnet_18.layer2(self.resnet_18.layer1(r))))
            feat_res = torch.flatten(self.resnet_18.avgpool(l4), 1)

            # TransUNet Path
            ts1 = self.trans_unet.enc1(x); ts2 = self.trans_unet.enc2(ts1); ts3 = self.trans_unet.enc3(ts2)
            tb = self.trans_unet.bottleneck_pool(ts3)
            tb_flat = tb.flatten(2).transpose(1, 2) + self.trans_unet.pos_embed
            t_out = self.trans_unet.transformers(tb_flat)
            feat_trans = torch.mean(t_out, dim=1)
            
            # Full Decoder Path for Mask Prediction
            tb_out = t_out.transpose(1, 2).reshape(tb.shape)
            td3 = self.trans_unet.dec3(torch.cat([self.trans_unet.up3(self.trans_unet.dec2(torch.cat([self.trans_unet.up2(self.trans_unet.dec1(torch.cat([self.trans_unet.up1(tb_out), ts3], dim=1))), ts2], dim=1))), ts1], dim=1))
            mask_pred = self.trans_unet.seg_final(td3)

        combined = torch.cat((feat_cnn, feat_res, feat_trans), dim=1)
        return mask_pred, self.fusion_gate(combined)

# =============================================================================
# 3. EVALUATION MAIN SCRIPT
# =============================================================================
def main():
    config = TransUNetConfig()
    device = config.DEVICE
    print(f"🔬 Evaluating Optimized Master Brain (V2) on {device}...")

    # Data Path Management
    data_path = os.path.join(r'I:\Lung Cancer Project (Simple CNN)', "Ensemble_Data_Safe")
    img_dir, msk_dir = os.path.join(data_path, "images"), os.path.join(data_path, "masks")
    
    all_files = glob(os.path.join(img_dir, "*.npz"))
    file_list = [(os.path.basename(f), 1 if "pos" in f else 0) for f in all_files]
    
    # Use the same split as training for validity
    _, val_files = train_test_split(file_list, test_size=0.2, random_state=42)
    val_loader = DataLoader(EnsembleEvalDataset(val_files, img_dir, msk_dir), batch_size=4, shuffle=False)

    # Load the Optimized Model
    model = UltimateEnsembleBrain(in_channels=16).to(device)
    save_path = "ultimate_ensemble_brain_v2.pth"
    
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
        print(f"✅ Master Brain V2 loaded from {save_path}")
    else:
        print(f"❌ Error: {save_path} not found. Ensure training finished!")
        return

    model.eval()
    y_true, y_probs, dice_scores = [], [], []

    # Testing Loop
    for img, msk, lbl in tqdm(val_loader, desc="Testing"):
        img, msk, lbl = img.to(device), msk.to(device), lbl.to(device)
        
        with torch.no_grad():
            p_mask, p_clf = model(img)
            
            # Classification Probability
            prob = torch.sigmoid(p_clf).cpu().detach().numpy().flatten()
            y_probs.extend(prob)
            y_true.extend(lbl.cpu().numpy())

            # Dice Score Calculation for Positive Cases
            if lbl.sum() > 0:
                p_sig = torch.sigmoid(p_mask)
                intersection = (p_sig * msk).sum()
                union = p_sig.sum() + msk.sum()
                dice = (2. * intersection) / (union + 1e-6)
                dice_scores.append(dice.item())

    # Final Metric Output
    y_true, y_probs = np.array(y_true), np.array(y_probs)
    y_pred = (y_probs > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print("\n" + "="*40)
    print("🚀 FINAL ENSEMBLE V2 PERFORMANCE 🚀")
    print("="*40)
    print(f"📊 ACCURACY:    {(tp+tn)/(tp+tn+fp+fn):.4f}")
    print(f"📊 AUC-ROC:     {roc_auc_score(y_true, y_probs):.4f}")
    print(f"📊 SENSITIVITY: {tp/(tp+fn):.4f}")
    print(f"📊 SPECIFICITY: {tn/(tn+fp):.4f}")
    print(f"📊 F1-SCORE:    {f1_score(y_true, y_pred):.4f}")
    print(f"🔥 MEAN DICE:   {np.mean(dice_scores):.4f}")
    print(f"🎯 BRIER SCORE: {brier_score_loss(y_true, y_probs):.4f}")
    print("="*40)

if __name__ == "__main__":
    main()