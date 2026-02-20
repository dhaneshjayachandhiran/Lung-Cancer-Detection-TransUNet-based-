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

# Importing specialist architectures from your files
from TransUNet_model import UltimateTransUNet, TransUNetConfig
from Resnet_model import ResNetMultiTaskModel
from simpleCNN_model import SimpleMultiTaskCNN

# =============================================================================
# 1. ENSEMBLE DATASET (Handles .npz Compressed Files)
# =============================================================================
class EnsembleDataset(Dataset):
    def __init__(self, file_list, img_dir, msk_dir):
        self.file_list = file_list
        self.img_dir = img_dir
        self.msk_dir = msk_dir
        # specialist models need 16 slices
        self.slice_offset = (64 - 16) // 2 

    def __len__(self): 
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name, label = self.file_list[idx]
        
        # Load compressed 3D patch
        data_3d = np.load(os.path.join(self.img_dir, file_name))['data']
        
        # Extract 16 slices for the input
        img_16_slices = data_3d[self.slice_offset : self.slice_offset + 16, :, :]
        img_tensor = torch.from_numpy(img_16_slices).float()

        if label == 1:
            mask = np.load(os.path.join(self.msk_dir, file_name))['data']
            # Using middle slice for 2D mask as per your model scripts
            mask_2d = mask[32, :, :] if len(mask.shape) == 3 else mask
            mask_tensor = torch.from_numpy(mask_2d).float().unsqueeze(0)
        else:
            mask_tensor = torch.zeros((1, 64, 64))

        return img_tensor, mask_tensor, torch.tensor(label, dtype=torch.float32)

# =============================================================================
# 2. ENSEMBLE BRAIN ARCHITECTURE
# =============================================================================
class UltimateEnsembleBrain(nn.Module):
    def __init__(self, in_channels=16):
        super(UltimateEnsembleBrain, self).__init__()
        
        # Initialize specialists
        self.simple_cnn = SimpleMultiTaskCNN(in_channels=in_channels)
        self.resnet_18 = ResNetMultiTaskModel(in_channels=in_channels)
        self.trans_unet = UltimateTransUNet(in_channels=in_channels)

        # Fusion Gate: Combines the knowledge of all three
        # Layers: SimpleCNN(256) + ResNet(512) + TransUNet(256) = 1024
        self.fusion_gate = nn.Sequential(
            nn.Linear(256 + 512 + 256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 1) 
        )

    def forward(self, x):
        # 1. SimpleCNN Feature Extraction
        s1 = self.simple_cnn.enc1(x)
        s2 = self.simple_cnn.enc2(self.simple_cnn.pool(s1))
        s3 = self.simple_cnn.enc3(self.simple_cnn.pool(s2))
        s4 = self.simple_cnn.enc4(self.simple_cnn.pool(s3))
        feat_cnn = torch.flatten(self.simple_cnn.avgpool(s4), 1)

        # 2. ResNet Feature Extraction
        r = self.resnet_18.relu(self.resnet_18.bn1(self.resnet_18.first_conv(x)))
        r = self.resnet_18.maxpool(r)
        l1 = self.resnet_18.layer1(r); l2 = self.resnet_18.layer2(l1)
        l3 = self.resnet_18.layer3(l2); l4 = self.resnet_18.layer4(l3)
        feat_res = torch.flatten(self.resnet_18.avgpool(l4), 1)

        # 3. TransUNet Feature Extraction & Mask Prediction
        ts1 = self.trans_unet.enc1(x); ts2 = self.trans_unet.enc2(ts1); ts3 = self.trans_unet.enc3(ts2)
        tb = self.trans_unet.bottleneck_pool(ts3)
        tb_flat = tb.flatten(2).transpose(1, 2) + self.trans_unet.pos_embed
        t_out = self.trans_unet.transformers(tb_flat)
        feat_trans = torch.mean(t_out, dim=1) 
        
        # TransUNet Decoder for Segmentation Mask
        tb_out = t_out.transpose(1, 2).reshape(tb.shape)
        td1 = self.trans_unet.dec1(torch.cat([self.trans_unet.up1(tb_out), ts3], dim=1))
        td2 = self.trans_unet.dec2(torch.cat([self.trans_unet.up2(td1), ts2], dim=1))
        td3 = self.trans_unet.dec3(torch.cat([self.trans_unet.up3(td2), ts1], dim=1))
        mask_pred = self.trans_unet.seg_final(td3)

        # THE BRAIN DECISION
        combined = torch.cat((feat_cnn, feat_res, feat_trans), dim=1)
        final_malignancy = self.fusion_gate(combined)

        return mask_pred, final_malignancy

# =============================================================================
# 3. ADVANCED LOSS FUNCTIONS (DICE FOCUS)
# =============================================================================
def dice_loss(pred, target, smooth=1e-6):
    """Calculates overlap loss to boost Mean Dice."""
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))

def hybrid_ensemble_loss(p_mask, gt_mask, p_clf, gt_clf):
    # Classification stays the same
    clf_loss = nn.functional.binary_cross_entropy_with_logits(p_clf, gt_clf.unsqueeze(1))
    
    seg_loss = 0
    pos_mask = (gt_clf == 1)
    if pos_mask.any():
        # LOG-COSH DICE: Much more stable for small nodules
        d_loss = dice_loss(p_mask[pos_mask], gt_mask[pos_mask])
        seg_loss = torch.log(torch.cosh(d_loss)) 
    
    # Balance: 2.0 weight is the 'sweet spot' to keep 96% Acc while lifting Dice
    return (2.0 * seg_loss) + clf_loss

# =============================================================================
# 4. MAIN TRAINING LOGIC
# =============================================================================
def main():
    config = TransUNetConfig()
    device = config.DEVICE
    
    # Setup Randomness for Reproducibility
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed(42)

    # 1. Setup Data (1.57GB Safe Dataset)
    data_path = os.path.join(r'I:\Lung Cancer Project (Simple CNN)', "Ensemble_Data_Safe")
    img_dir, msk_dir = os.path.join(data_path, "images"), os.path.join(data_path, "masks")
    
    all_files = glob(os.path.join(img_dir, "*.npz"))
    file_list = [(os.path.basename(f), 1 if "pos" in f else 0) for f in all_files]
    train_files, val_files = train_test_split(file_list, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(EnsembleDataset(train_files, img_dir, msk_dir), batch_size=8, shuffle=True)
    val_loader = DataLoader(EnsembleDataset(val_files, img_dir, msk_dir), batch_size=8, shuffle=False)

    # 2. Init Ensemble model
    model = UltimateEnsembleBrain(in_channels=16).to(device)

    # 3. Load Specialist Weights
    weights = {
        model.simple_cnn: "simpleCNN_unet_best.pth",
        model.resnet_18: "resnet_multitask_best.pth",
        model.trans_unet: "transunet_ULTIMATE_best.pth"
    }

    print("🛠️ Loading specialist weights...")
    for mod, path in weights.items():
        if os.path.exists(path):
            mod.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            print(f"✅ Loaded: {path}")

    # 4. Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    # 5. Training Loop (Optimized for 15 Epochs)
    best_loss = float('inf')
    for epoch in range(15):
        model.train()
        epoch_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/15")
        
        for img, msk, lbl in loop:
            img, msk, lbl = img.to(device), msk.to(device), lbl.to(device)
            
            optimizer.zero_grad()
            p_mask, p_clf = model(img)
            
            # Weighted Hybrid Loss calculation
            loss = hybrid_ensemble_loss(p_mask, msk, p_clf, lbl)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "ultimate_ensemble_brain_v2.pth")
            print(f"⭐ Best Model Saved (Epoch {epoch+1})")

    print("\n🔥 Dice-Optimized Training Finished! Master Brain V2 Ready.")

if __name__ == "__main__":
    main()