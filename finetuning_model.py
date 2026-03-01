import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split

# Specialist architectures
from TransUNet_model import UltimateTransUNet, TransUNetConfig
from Resnet_model import ResNetMultiTaskModel
from simpleCNN_model import SimpleMultiTaskCNN

# =============================================================================
# 1. ENSEMBLE DATASET
# =============================================================================
class EnsembleDataset(Dataset):
    def __init__(self, file_list, img_dir, msk_dir):
        self.file_list = file_list
        self.img_dir = img_dir
        self.msk_dir = msk_dir
        self.z_start, self.z_end = 24, 40 

    def __len__(self): return len(self.file_list)

    def __getitem__(self, idx):
        file_name, label = self.file_list[idx]
        img_3d = np.load(os.path.join(self.img_dir, file_name))
        img_tensor = torch.from_numpy(img_3d[self.z_start:self.z_end, :, :]).float()
        img_tensor = (img_tensor - img_tensor.mean()) / (img_tensor.std() + 1e-6)

        if label == 1:
            mask_3d = np.load(os.path.join(self.msk_dir, file_name))
            mask_tensor = torch.from_numpy(mask_3d[32, :, :]).float().unsqueeze(0)
        else:
            mask_tensor = torch.zeros((1, 64, 64))

        return img_tensor, mask_tensor, torch.tensor(label, dtype=torch.float32)

# =============================================================================
# 2. ENSEMBLE BRAIN ARCHITECTURE
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
        tb = self.trans_unet.bottleneck_pool(ts3); tb_flat = tb.flatten(2).transpose(1, 2) + self.trans_unet.pos_embed
        t_out = self.trans_unet.transformers(tb_flat); feat_trans = torch.mean(t_out, dim=1) 
        tb_out = t_out.transpose(1, 2).reshape(tb.shape)
        td1 = self.trans_unet.dec1(torch.cat([self.trans_unet.up1(tb_out), ts3], dim=1))
        td2 = self.trans_unet.dec2(torch.cat([self.trans_unet.up2(td1), ts2], dim=1))
        
        fusion_input = torch.cat([self.trans_unet.up3(td2), ts1, sc1], dim=1)
        mask_pred = self.trans_unet.seg_final(self.trans_unet.dec3(self.channel_compressor(fusion_input)))

        combined = torch.cat((feat_cnn, feat_res, feat_trans), dim=1)
        return mask_pred, self.fusion_gate(combined)

# =============================================================================
# 3. HOLY GRAIL LOSS (DYNAMIC ASYMMETRIC PENALTY)
# =============================================================================
def holy_grail_loss(p_clf, label, p_mask, target_mask):
    # Base BCE Loss without reduction so we can manipulate individual samples
    bce = nn.functional.binary_cross_entropy_with_logits(p_clf, label.unsqueeze(1), reduction='none')
    p_clf_sig = torch.sigmoid(p_clf)
    
    # 1. PRECISION ANCHOR: If model predicts > 0.5 on a Healthy lung (FP risk), apply 15x Penalty
    fp_mask = (p_clf_sig > 0.5) & (label.unsqueeze(1) == 0)
    
    # 2. RECALL BOOSTER: If model predicts < 0.5 on a Cancer lung (FN risk), apply 20x Penalty
    fn_mask = (p_clf_sig < 0.5) & (label.unsqueeze(1) == 1)
    
    # Apply dynamic weights
    weights = torch.ones_like(bce)
    weights[fp_mask] = 15.0 # DON'T drop Precision
    weights[fn_mask] = 20.0 # DO fix Recall
    
    clf_loss = (bce * weights).mean()

    # Tversky Loss matching the exact FN/FP pressure
    p_mask_sig = torch.sigmoid(p_mask)
    tp = (p_mask_sig * target_mask).sum()
    fp = ((1 - target_mask) * p_mask_sig).sum()
    fn = (target_mask * (1 - p_mask_sig)).sum()
    
    # alpha=0.9 (Heavy FP push), beta=0.9 (Heavy FN push)
    tversky = (tp + 1e-6) / (tp + 0.9*fp + 0.9*fn + 1e-6)
    seg_loss = 1 - tversky

    return clf_loss + (10.0 * seg_loss)

def main():
    config = TransUNetConfig()
    device = config.DEVICE
    random.seed(config.SEED); np.random.seed(config.SEED); torch.manual_seed(config.SEED)

    data_path = os.path.join(r'I:\Lung Cancer Project (Simple CNN)', "LUNA16_High_Volume_Data")
    img_dir, msk_dir = os.path.join(data_path, "images"), os.path.join(data_path, "masks")
    
    all_files = glob(os.path.join(img_dir, "*.npy"))
    pos_files = [f for f in all_files if "pos" in f]
    
    # We need a large chunk of negatives so the model doesn't forget how to reject healthy scans
    neg_files = list(np.random.choice([f for f in all_files if "neg" in f], len(pos_files) * 5, replace=False))
    
    train_list = [(os.path.basename(f), 1 if "pos" in f else 0) for f in (pos_files + neg_files)]
    train_loader = DataLoader(EnsembleDataset(train_list, img_dir, msk_dir), batch_size=8, shuffle=True)

    print("🔥 LOADING PRECISION BEAST FOR HOLY GRAIL TUNING...")
    model = UltimateEnsembleBrain(in_channels=16).to(device)
    
    checkpoint_path = "ultimate_ensemble_brain_PRECISION_BEAST.pth"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("✅ Base weights loaded. Commencing absolute boundary freezing.")

    # Micro-surgical learning rate. We want tiny adjustments.
    optimizer = optim.AdamW(model.parameters(), lr=1e-6, weight_decay=1e-2)

    print(f"🚀 Targeting ~80% Recall while freezing Precision at ~90%...")

    for epoch in range(3): 
        model.train()
        loop = tqdm(train_loader, desc=f"Holy Grail Epoch {epoch+1}/3")
        
        for img, msk, lbl in loop:
            img, msk, lbl = img.to(device), msk.to(device), lbl.to(device)
            optimizer.zero_grad()
            
            p_mask, p_clf = model(img)
            
            total_loss = holy_grail_loss(p_clf, lbl, p_mask, msk)
            
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5) # Very strict gradient clipping to prevent metric collapse
            optimizer.step()
            
            loop.set_postfix(loss=total_loss.item())

    torch.save(model.state_dict(), "ultimate_ensemble_brain_HOLY_GRAIL.pth")
    print("⭐ Holy Grail Tuning Complete! Weights saved as HOLY_GRAIL.pth.")

if __name__ == "__main__": main()