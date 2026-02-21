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
# 1. ENSEMBLE DATASET (Stabilized)
# =============================================================================
class EnsembleDataset(Dataset):
    def __init__(self, file_list, img_dir, msk_dir):
        self.file_list = file_list
        self.img_dir = img_dir
        self.msk_dir = msk_dir
        self.slice_offset = (64 - 16) // 2 

    def __len__(self): return len(self.file_list)

    def __getitem__(self, idx):
        file_name, label = self.file_list[idx]
        data_3d = np.load(os.path.join(self.img_dir, file_name))['data']
        img_16 = data_3d[self.slice_offset : self.slice_offset + 16, :, :]
        img_tensor = torch.from_numpy(img_16).float()
        img_tensor = (img_tensor - img_tensor.mean()) / (img_tensor.std() + 1e-6)

        if label == 1:
            mask = np.load(os.path.join(self.msk_dir, file_name))['data']
            mask_2d = mask[32, :, :] if len(mask.shape) == 3 else mask
            mask_tensor = torch.from_numpy(mask_2d).float().unsqueeze(0)
        else:
            mask_tensor = torch.zeros((1, 64, 64))

        return img_tensor, mask_tensor, torch.tensor(label, dtype=torch.float32)

# =============================================================================
# 2. ENSEMBLE BRAIN (V3 Architecture with Skip-Fusion)
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
# 3. V4 LOSS: BOUNDARY-AWARE FOCAL DICE
# =============================================================================
def focal_tversky_loss(pred, target, alpha=0.3, beta=0.7, gamma=1.5):
    pred = torch.sigmoid(pred)
    tp = (pred * target).sum()
    fp = ((1 - target) * pred).sum()
    fn = (target * (1 - pred)).sum()
    tversky = (tp + 1e-6) / (tp + alpha*fp + beta*fn + 1e-6)
    return torch.pow((1 - tversky), 1/gamma) 

def main():
    config = TransUNetConfig()
    device = config.DEVICE
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    data_path = os.path.join(r'I:\Lung Cancer Project (Simple CNN)', "Ensemble_Data_Safe")
    img_dir, msk_dir = os.path.join(data_path, "images"), os.path.join(data_path, "masks")
    
    all_files = glob(os.path.join(img_dir, "*.npz"))
    file_list = [(os.path.basename(f), 1 if "pos" in f else 0) for f in all_files]
    train_files, _ = train_test_split(file_list, test_size=0.2, random_state=42)
    
    # Corrected variable name here
    train_dataset = EnsembleDataset(train_files, img_dir, msk_dir)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    model = UltimateEnsembleBrain(in_channels=16).to(device)
    
    # Load V3 checkpoint with security fix
    v3_path = "ultimate_ensemble_brain_v3.pth"
    if os.path.exists(v3_path):
        model.load_state_dict(torch.load(v3_path, map_location=device, weights_only=True))
        print("✅ Resume from V3 (0.72 Dice) checkpoint...")

    # Precision learning rate
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=5e-3)
    
    print("🧠 BRAIN V4: Final Precision Stitch for 0.80+ Dice...")
    for epoch in range(10):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/10")
        for img, msk, lbl in loop:
            img, msk, lbl = img.to(device), msk.to(device), lbl.to(device)
            optimizer.zero_grad()
            p_mask, p_clf = model(img)
            
            # Weighted V4 Loss
            loss_clf = nn.functional.binary_cross_entropy_with_logits(p_clf, lbl.unsqueeze(1))
            loss_seg = torch.log(torch.cosh(focal_tversky_loss(p_mask, msk))) if lbl.sum() > 0 else 0
            (10.0 * loss_seg + loss_clf).backward() # High Seg priority
            
            # Clip gradients for stability
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loop.set_postfix(loss=(10.0*loss_seg + loss_clf).item() if isinstance(loss_seg, torch.Tensor) else loss_clf.item())

    torch.save(model.state_dict(), "ultimate_ensemble_brain_v4_FINAL.pth")
    print("⭐ V4 Final Precision Stitching Complete!")

if __name__ == "__main__": main()