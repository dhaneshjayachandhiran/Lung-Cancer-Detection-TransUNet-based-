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

# Importing specialist architectures
from TransUNet_model import UltimateTransUNet, TransUNetConfig
from Resnet_model import ResNetMultiTaskModel
from simpleCNN_model import SimpleMultiTaskCNN

# =============================================================================
# 1. ENSEMBLE DATASET (Optimized for Raw 64x64x64 .npy)
# =============================================================================
class EnsembleDataset(Dataset):
    def __init__(self, file_list, img_dir, msk_dir):
        self.file_list = file_list
        self.img_dir = img_dir
        self.msk_dir = msk_dir
        # Extracting the central 16 slices to provide 3D context to 2D models
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
            # Segmentation Target: Central slice of the nodule
            mask_2d = mask_3d[32, :, :] 
            mask_tensor = torch.from_numpy(mask_2d).float().unsqueeze(0)
        else:
            mask_tensor = torch.zeros((1, 64, 64))

        return img_tensor, mask_tensor, torch.tensor(label, dtype=torch.float32)

# =============================================================================
# 2. ENSEMBLE BRAIN (Fresh Initialization)
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
        # 1. SimpleCNN (Fast Screening)
        sc1 = self.simple_cnn.enc1(x)
        sc4 = self.simple_cnn.enc4(self.simple_cnn.pool(self.simple_cnn.enc3(self.simple_cnn.pool(self.simple_cnn.enc2(self.simple_cnn.pool(sc1))))))
        feat_cnn = torch.flatten(self.simple_cnn.avgpool(sc4), 1)

        # 2. ResNet (Residual Detail Extraction)
        r = self.resnet_18.relu(self.resnet_18.bn1(self.resnet_18.first_conv(x)))
        l4 = self.resnet_18.layer4(self.resnet_18.layer3(self.resnet_18.layer2(self.resnet_18.layer1(self.resnet_18.maxpool(r)))))
        feat_res = torch.flatten(self.resnet_18.avgpool(l4), 1)

        # 3. TransUNet (Global Context via Transformers)
        ts1 = self.trans_unet.enc1(x); ts2 = self.trans_unet.enc2(ts1); ts3 = self.trans_unet.enc3(ts2)
        tb = self.trans_unet.bottleneck_pool(ts3)
        tb_flat = tb.flatten(2).transpose(1, 2) + self.trans_unet.pos_embed
        t_out = self.trans_unet.transformers(tb_flat)
        feat_trans = torch.mean(t_out, dim=1) 
        
        tb_out = t_out.transpose(1, 2).reshape(tb.shape)
        td1 = self.trans_unet.dec1(torch.cat([self.trans_unet.up1(tb_out), ts3], dim=1))
        td2 = self.trans_unet.dec2(torch.cat([self.trans_unet.up2(td1), ts2], dim=1))
        
        # Skip-Fusion for Segmentation
        fusion_input = torch.cat([self.trans_unet.up3(td2), ts1, sc1], dim=1)
        mask_pred = self.trans_unet.seg_final(self.trans_unet.dec3(self.channel_compressor(fusion_input)))

        combined = torch.cat((feat_cnn, feat_res, feat_trans), dim=1)
        return mask_pred, self.fusion_gate(combined)

# =============================================================================
# 3. TRAINING ENGINE (Starting from Scratch)
# =============================================================================
def focal_tversky_loss(pred, target):
    pred = torch.sigmoid(pred)
    tp = (pred * target).sum()
    fp = ((1 - target) * pred).sum()
    fn = (target * (1 - pred)).sum()
    tversky = (tp + 1e-6) / (tp + 0.3*fp + 0.7*fn + 1e-6)
    return torch.pow((1 - tversky), 1/1.5) 

def main():
    config = TransUNetConfig()
    device = config.DEVICE
    random.seed(config.SEED); np.random.seed(config.SEED); torch.manual_seed(config.SEED)

    # Path to your custom 20GB+ Raw Dataset
    data_path = os.path.join(r'I:\Lung Cancer Project (Simple CNN)', "LUNA16_High_Volume_Data")
    img_dir, msk_dir = os.path.join(data_path, "images"), os.path.join(data_path, "masks")
    
    all_files = glob(os.path.join(img_dir, "*.npy"))
    file_list = [(os.path.basename(f), 1 if "pos" in f else 0) for f in all_files]
    
    train_files, val_files = train_test_split(file_list, test_size=0.15, random_state=config.SEED)
    
    train_loader = DataLoader(EnsembleDataset(train_files, img_dir, msk_dir), batch_size=12, shuffle=True)

    # Initialize Fresh Model
    print("🔥 INITIALIZING FRESH ENSEMBLE BRAIN (NO CHECKPOINTS)...")
    model = UltimateEnsembleBrain(in_channels=16).to(device)

    # Optimizer & Scheduler for large data
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-4, steps_per_epoch=len(train_loader), epochs=20)
    
    print(f"🚀 Training on {len(train_files)} high-res samples...")

    for epoch in range(20): # Increased epochs for full training
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/20")
        epoch_loss = 0
        
        for img, msk, lbl in loop:
            img, msk, lbl = img.to(device), msk.to(device), lbl.to(device)
            optimizer.zero_grad()
            
            p_mask, p_clf = model(img)
            
            # Classification Loss (BCE)
            loss_clf = nn.functional.binary_cross_entropy_with_logits(p_clf, lbl.unsqueeze(1))
            
            # Segmentation Loss (Focal Tversky)
            pos_idx = (lbl > 0).nonzero(as_tuple=True)[0]
            if len(pos_idx) > 0:
                loss_seg = torch.log(torch.cosh(focal_tversky_loss(p_mask[pos_idx], msk[pos_idx])))
                total_loss = (10.0 * loss_seg) + loss_clf
            else:
                total_loss = loss_clf
            
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += total_loss.item()
            loop.set_postfix(loss=total_loss.item())

        # Save model every few epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"ensemble_brain_scratch_epoch_{epoch+1}.pth")

    torch.save(model.state_dict(), "ultimate_ensemble_brain_SCRATCH_FINAL.pth")
    print("⭐ Fresh Training Complete! Weights saved as SCRATCH_FINAL.")

if __name__ == "__main__": main()