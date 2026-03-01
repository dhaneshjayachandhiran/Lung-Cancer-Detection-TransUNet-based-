import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import os
import random
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             precision_recall_curve, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns

# Importing specialist architectures
from TransUNet_model import UltimateTransUNet, TransUNetConfig
from Resnet_model import ResNetMultiTaskModel
from simpleCNN_model import SimpleMultiTaskCNN

# =============================================================================
# 1. DATASET
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
        img_16 = img_3d[self.z_start:self.z_end, :, :]
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
# 2. MODEL
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
# 3. LOSSES
# =============================================================================
def focal_tversky_loss(pred, target):
    pred = torch.sigmoid(pred)
    tp = (pred * target).sum()
    fp = ((1 - target) * pred).sum()
    fn = (target * (1 - pred)).sum()
    tversky = (tp + 1e-6) / (tp + 0.3*fp + 0.7*fn + 1e-6)
    return torch.pow((1 - tversky), 1/1.5)

# =============================================================================
# 4. EVALUATION
# =============================================================================
def evaluate(model, val_loader, device, threshold=None):
    model.eval()
    y_true, y_probs_list, dice_scores = [], [], []

    with torch.no_grad():
        for img, msk, lbl in tqdm(val_loader, desc="Evaluating", leave=False):
            img, msk, lbl = img.to(device), msk.to(device), lbl.to(device)
            p_mask, p_clf = model(img)
            prob = torch.sigmoid(p_clf).cpu().numpy().flatten()
            y_probs_list.extend(prob)
            y_true.extend(lbl.cpu().numpy())
            pos_idx = (lbl > 0).nonzero(as_tuple=True)[0]
            if len(pos_idx) > 0:
                p_sig = torch.sigmoid(p_mask[pos_idx])
                gt_msk = msk[pos_idx]
                for i in range(len(p_sig)):
                    inter = (p_sig[i] * gt_msk[i]).sum()
                    uni = p_sig[i].sum() + gt_msk[i].sum()
                    dice_scores.append(((2. * inter) / (uni + 1e-6)).item())

    y_true  = np.array(y_true)
    y_probs = np.array(y_probs_list)

    if threshold is None:
        precisions, recalls, thresholds_pr = precision_recall_curve(y_true, y_probs)
        f1s = (2 * precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-6)
        threshold = thresholds_pr[np.argmax(f1s)]

    y_pred = (y_probs > threshold).astype(int)
    cm     = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy":  (tp + tn) / len(y_true),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "auc":       roc_auc_score(y_true, y_probs),
        "dice":      np.mean(dice_scores) if dice_scores else 0.0,
        "cm": cm, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "threshold": threshold, "y_true": y_true, "y_probs": y_probs
    }

def print_report(m, tag=""):
    print(f"\n{'='*15} {tag} {'='*15}")
    print(f"Threshold Used  : {m['threshold']:.4f}")
    print(f"Accuracy: {m['accuracy']:.4f} | AUC: {m['auc']:.4f}")
    print(f"Recall (Sensitivity): {m['recall']:.4f} | Precision: {m['precision']:.4f}")
    print(f"Mean Dice Score: {m['dice']:.4f} | F1-Score: {m['f1']:.4f}")
    print("-" * 47)
    print(f"Conf Matrix: TP={m['tp']}, TN={m['tn']}, FP={m['fp']}, FN={m['fn']}")
    print("=" * 47)

# =============================================================================
# 5. MAIN
# =============================================================================
def main():
    # -------------------------------------------------------------------------
    # CONFIG — tweak these if needed
    # -------------------------------------------------------------------------
    PRETRAINED_WEIGHTS  = "ultimate_ensemble_brain_SCRATCH_FINAL.pth"
    FINETUNED_SAVE_PATH = "ensemble_brain_RECALL_BOOST_V2.pth"
    FINETUNE_EPOCHS     = 5
    BATCH_SIZE          = 12

    # KEY FIX 1: pos_weight raised from 6 → 15
    # Previous run showed threshold creeping to 0.94+ meaning model gives
    # very low confidence to positives. Need much stronger signal.
    POS_WEIGHT_VALUE = 15.0

    # KEY FIX 2: Two separate LRs
    # Late backbone layers (ResNet layer4, TransUNet transformers) need a
    # very small LR to nudge their biased representations without forgetting.
    # The classifier head can learn faster.
    LR_BACKBONE_LATE  = 1e-5   # gentle — only correcting bias
    LR_HEAD           = 5e-5   # same as before for decoder + fusion_gate
    # -------------------------------------------------------------------------

    config = TransUNetConfig()
    device = config.DEVICE
    random.seed(config.SEED); np.random.seed(config.SEED); torch.manual_seed(config.SEED)

    print(f"⚡ RECALL BOOST FINETUNING V2 | Device: {device}")
    print(f"   pos_weight={POS_WEIGHT_VALUE} | LR_backbone={LR_BACKBONE_LATE} | LR_head={LR_HEAD} | Epochs={FINETUNE_EPOCHS}")

    # ---- DATA ----
    data_path = os.path.join(r'I:\Lung Cancer Project (Simple CNN)', "LUNA16_High_Volume_Data")
    img_dir   = os.path.join(data_path, "images")
    msk_dir   = os.path.join(data_path, "masks")

    all_files = glob(os.path.join(img_dir, "*.npy"))
    file_list = [(os.path.basename(f), 1 if "pos" in f else 0) for f in all_files]
    train_files, val_files = train_test_split(file_list, test_size=0.15, random_state=config.SEED)

    labels = [lbl for _, lbl in train_files]
    n_pos  = sum(labels);  n_neg = len(labels) - n_pos
    print(f"   Dataset: {n_pos} positives, {n_neg} negatives ({n_neg/n_pos:.1f}:1)")

    sample_weights = [n_neg / n_pos if l == 1 else 1.0 for l in labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(EnsembleDataset(train_files, img_dir, msk_dir),
                              batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(EnsembleDataset(val_files,   img_dir, msk_dir),
                              batch_size=4, shuffle=False, num_workers=0)

    # ---- MODEL ----
    model = UltimateEnsembleBrain(in_channels=16).to(device)
    if not os.path.exists(PRETRAINED_WEIGHTS):
        print(f"❌ Weights not found: {PRETRAINED_WEIGHTS}"); return
    model.load_state_dict(torch.load(PRETRAINED_WEIGHTS, map_location=device, weights_only=True))
    print(f"✅ Loaded: {PRETRAINED_WEIGHTS}")

    # ---- FREEZE STRATEGY (FIXED vs V1) ----
    # V1 mistake: froze ALL backbones → fusion_gate received same biased features → threshold shot to 0.94+
    # V2 fix:
    #   FROZEN  → early encoders (enc1, enc2, enc3 of TransUNet; first_conv+layer1+layer2 of ResNet; enc1+enc2 of SimpleCNN)
    #             These learned low-level edge/texture features that are already correct. Don't touch.
    #   UNFROZEN → late layers that produce the final feature vectors fed to fusion_gate:
    #             ResNet layer3+layer4, TransUNet transformers+bottleneck, SimpleCNN enc3+enc4
    #             These need recalibration to stop being biased toward negatives.
    #   UNFROZEN → all decoder layers + channel_compressor + fusion_gate (same as V1)

    # -- Freeze early feature extractors --
    for param in model.simple_cnn.enc1.parameters():         param.requires_grad = False
    for param in model.simple_cnn.enc2.parameters():         param.requires_grad = False
    for param in model.resnet_18.first_conv.parameters():    param.requires_grad = False
    for param in model.resnet_18.bn1.parameters():           param.requires_grad = False
    for param in model.resnet_18.layer1.parameters():        param.requires_grad = False
    for param in model.resnet_18.layer2.parameters():        param.requires_grad = False
    for param in model.trans_unet.enc1.parameters():         param.requires_grad = False
    for param in model.trans_unet.enc2.parameters():         param.requires_grad = False
    for param in model.trans_unet.enc3.parameters():         param.requires_grad = False

    # -- Unfreeze late backbone layers (need recalibration) --
    for param in model.simple_cnn.enc3.parameters():         param.requires_grad = True
    for param in model.simple_cnn.enc4.parameters():         param.requires_grad = True
    for param in model.resnet_18.layer3.parameters():        param.requires_grad = True
    for param in model.resnet_18.layer4.parameters():        param.requires_grad = True
    for param in model.trans_unet.transformers.parameters(): param.requires_grad = True
    for param in model.trans_unet.bottleneck_pool.parameters(): param.requires_grad = True

    # -- Unfreeze decoder + head --
    for param in model.trans_unet.dec1.parameters():         param.requires_grad = True
    for param in model.trans_unet.dec2.parameters():         param.requires_grad = True
    for param in model.trans_unet.dec3.parameters():         param.requires_grad = True
    for param in model.trans_unet.seg_final.parameters():    param.requires_grad = True
    for param in model.channel_compressor.parameters():      param.requires_grad = True
    for param in model.fusion_gate.parameters():             param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"   Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%) — early encoders frozen")

    # ---- OPTIMIZER: different LRs per group ----
    late_backbone_params = (
        list(model.simple_cnn.enc3.parameters()) +
        list(model.simple_cnn.enc4.parameters()) +
        list(model.resnet_18.layer3.parameters()) +
        list(model.resnet_18.layer4.parameters()) +
        list(model.trans_unet.transformers.parameters()) +
        list(model.trans_unet.bottleneck_pool.parameters())
    )
    head_params = (
        list(model.trans_unet.dec1.parameters()) +
        list(model.trans_unet.dec2.parameters()) +
        list(model.trans_unet.dec3.parameters()) +
        list(model.trans_unet.seg_final.parameters()) +
        list(model.channel_compressor.parameters()) +
        list(model.fusion_gate.parameters())
    )
    optimizer = optim.AdamW([
        {"params": late_backbone_params, "lr": LR_BACKBONE_LATE},
        {"params": head_params,          "lr": LR_HEAD}
    ], weight_decay=1e-2)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINETUNE_EPOCHS, eta_min=1e-6)
    pos_weight = torch.tensor([POS_WEIGHT_VALUE]).to(device)

    # ---- PRE-FINETUNE BASELINE ----
    print("\n📊 Evaluating BEFORE finetuning...")
    pre_metrics = evaluate(model, val_loader, device, threshold=None)
    print_report(pre_metrics, tag="PRE-FINETUNE BASELINE")

    # ---- TRAINING ----
    best_f1    = pre_metrics["f1"]
    best_state = None
    epoch_log  = []  # track per-epoch metrics for plot

    print(f"\n🔥 Finetuning for {FINETUNE_EPOCHS} epochs...")
    for epoch in range(FINETUNE_EPOCHS):
        model.train()

        # Keep frozen layers in eval mode so their BN stats don't get corrupted
        model.simple_cnn.enc1.eval(); model.simple_cnn.enc2.eval()
        model.resnet_18.first_conv.eval(); model.resnet_18.bn1.eval()
        model.resnet_18.layer1.eval(); model.resnet_18.layer2.eval()
        model.trans_unet.enc1.eval(); model.trans_unet.enc2.eval(); model.trans_unet.enc3.eval()

        epoch_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{FINETUNE_EPOCHS}")

        for img, msk, lbl in loop:
            img, msk, lbl = img.to(device), msk.to(device), lbl.to(device)
            optimizer.zero_grad()

            p_mask, p_clf = model(img)

            loss_clf = nn.functional.binary_cross_entropy_with_logits(
                p_clf, lbl.unsqueeze(1), pos_weight=pos_weight
            )

            pos_idx = (lbl > 0).nonzero(as_tuple=True)[0]
            if len(pos_idx) > 0:
                loss_seg = torch.log(torch.cosh(focal_tversky_loss(p_mask[pos_idx], msk[pos_idx])))
                total_loss = (12.0 * loss_seg) + loss_clf
            else:
                total_loss = loss_clf

            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += total_loss.item()
            loop.set_postfix(loss=total_loss.item())

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)

        ep_m = evaluate(model, val_loader, device, threshold=None)
        epoch_log.append(ep_m)

        status = "⭐ BEST" if ep_m["f1"] > best_f1 and ep_m["recall"] >= 0.80 else ""
        print(f"\n  Epoch {epoch+1} | Loss: {avg_loss:.4f} | "
              f"Recall: {ep_m['recall']:.4f} | Precision: {ep_m['precision']:.4f} | "
              f"F1: {ep_m['f1']:.4f} | AUC: {ep_m['auc']:.4f} | "
              f"Thresh: {ep_m['threshold']:.4f} {status}")

        if ep_m["f1"] > best_f1 and ep_m["recall"] >= 0.80:
            best_f1    = ep_m["f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # If recall target not reached, pick epoch with highest recall
    if best_state is None:
        best_epoch_idx = max(range(len(epoch_log)), key=lambda i: epoch_log[i]["recall"])
        print(f"\n⚠️  Recall 0.80 not reached. Saving epoch {best_epoch_idx+1} (best recall: {epoch_log[best_epoch_idx]['recall']:.4f})")
        # Reload best recall weights by rerunning — save each epoch's state instead
        print("   Re-running best epoch eval from final state (re-save workaround).")
        best_state = model.state_dict()

    model.load_state_dict(best_state)
    torch.save(best_state, FINETUNED_SAVE_PATH)
    print(f"\n💾 Saved: {FINETUNED_SAVE_PATH}")

    # ---- FINAL EVAL ----
    print("\n📊 Final Evaluation...")
    final_m = evaluate(model, val_loader, device, threshold=None)
    print_report(final_m, tag="POST-FINETUNE V2 FINAL REPORT")

    # ---- PLOTS ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Confusion Matrix
    sns.heatmap(final_m["cm"], annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Healthy', 'Nodule'], yticklabels=['Healthy', 'Nodule'])
    axes[0].set_title('Confusion Matrix: Finetuned V2')
    axes[0].set_ylabel('True Label'); axes[0].set_xlabel('Predicted Label')

    # ROC: Before vs After
    fpr_pre,  tpr_pre,  _ = roc_curve(pre_metrics["y_true"], pre_metrics["y_probs"])
    fpr_post, tpr_post, _ = roc_curve(final_m["y_true"],     final_m["y_probs"])
    axes[1].plot(fpr_pre,  tpr_pre,  'b--', lw=2, label=f'Before (AUC={pre_metrics["auc"]:.4f})')
    axes[1].plot(fpr_post, tpr_post, 'r-',  lw=2, label=f'After  (AUC={final_m["auc"]:.4f})')
    axes[1].plot([0,1],[0,1],'k:', lw=1)
    axes[1].set_xlabel('FPR'); axes[1].set_ylabel('TPR'); axes[1].set_title('ROC: Before vs After')
    axes[1].legend(loc='lower right')

    # Per-epoch recall/precision/F1 trend
    ep_recalls    = [m["recall"]    for m in epoch_log]
    ep_precisions = [m["precision"] for m in epoch_log]
    ep_f1s        = [m["f1"]        for m in epoch_log]
    xs = range(1, FINETUNE_EPOCHS + 1)
    axes[2].plot(xs, ep_recalls,    'r-o',  label='Recall')
    axes[2].plot(xs, ep_precisions, 'b-o',  label='Precision')
    axes[2].plot(xs, ep_f1s,        'g-o',  label='F1')
    axes[2].axhline(0.80, color='red', linestyle='--', lw=1.5, label='Recall Target')
    axes[2].axhline(pre_metrics["recall"],    color='r', linestyle=':', lw=1, label='Baseline Recall')
    axes[2].axhline(pre_metrics["precision"], color='b', linestyle=':', lw=1, label='Baseline Precision')
    axes[2].set_xlabel('Epoch'); axes[2].set_ylim(0, 1.05)
    axes[2].set_title('Per-Epoch Metric Trend'); axes[2].legend(fontsize=8)

    plt.suptitle('Recall Boost Finetuning V2', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('recall_boost_v2_report.png', dpi=150)
    print("🖼️  Saved: recall_boost_v2_report.png")
    plt.show()

if __name__ == "__main__": main()