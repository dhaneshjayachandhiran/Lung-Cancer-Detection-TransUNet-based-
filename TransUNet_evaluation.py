import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from glob import glob
import os, random
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, 
    roc_curve, precision_recall_curve, average_precision_score,
    matthews_corrcoef, f1_score, recall_score, precision_score
)

# Import the updated architecture and config
from TransUNet_model import UltimateTransUNet, MultiTaskDataset, TransUNetConfig

# =============================================================================
# 1. ADVANCED METRIC CALCULATORS
# =============================================================================
def calculate_segmentation_metrics(pred, target, threshold=0.5):
    """Calculates Dice and IoU for a single sample."""
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    dice = (2. * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    return dice.item(), iou.item()

def run_comprehensive_evaluation():
    config = TransUNetConfig()
    print("🚀 Initializing Full-Scale Architectural Evaluation...")
    
    # Data Setup
    pos_paths = glob(os.path.join(config.PREPROCESSED_PATH, 'trans_pre_subset*', 'images', 'pos*.npy'))
    neg_paths = glob(os.path.join(config.PREPROCESSED_PATH, 'trans_pre_subset*', 'images', 'neg*.npy'))
    all_files = [(f, 1) for f in pos_paths] + [(f, 0) for f in neg_paths]
    
    random.seed(config.SEED)
    random.shuffle(all_files)
    
    val_files = all_files[int(len(all_files) * 0.8):]
    val_loader = DataLoader(MultiTaskDataset(val_files), batch_size=config.BATCH_SIZE, shuffle=False)

    # Load Model
    model = UltimateTransUNet(in_channels=config.SLICES).to(config.DEVICE)
    if os.path.exists(config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE, weights_only=True))
        model.eval()
        print(f"✅ Loaded: {config.MODEL_SAVE_PATH}")
    else:
        print("❌ Error: Model weights missing."); return

    y_true, y_prob, dice_list, iou_list = [], [], [], []

    # Inference Loop
    with torch.no_grad():
        for imgs, masks, labels in tqdm(val_loader, desc="Evaluating"):
            imgs, masks, labels = imgs.to(config.DEVICE), masks.to(config.DEVICE), labels.to(config.DEVICE)
            p_mask, p_clf = model(imgs)
            
            y_prob.extend(torch.sigmoid(p_clf).cpu().numpy())
            y_true.extend(labels.cpu().numpy())
            
            # Segment only positive cases
            for i in range(len(labels)):
                if labels[i] == 1:
                    d, iou = calculate_segmentation_metrics(p_mask[i], masks[i])
                    dice_list.append(d)
                    iou_list.append(iou)

    y_prob, y_true = np.array(y_prob).flatten(), np.array(y_true).flatten()
    y_pred = (y_prob > 0.5).astype(int)

    # =============================================================================
    # 2. CALCULATION OF EVERY SINGLE RELEVANT METRIC
    # =============================================================================
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        "Accuracy": (tp + tn) / (tp + tn + fp + fn),
        "Sensitivity (Recall)": tp / (tp + fn),
        "Specificity": tn / (tn + fp),
        "Precision (PPV)": tp / (tp + fp),
        "F1-Score": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "AUC-ROC": roc_auc_score(y_true, y_prob),
        "Avg Precision (mAP)": average_precision_score(y_true, y_prob),
        "Mean Dice (Seg)": np.mean(dice_list),
        "Mean IoU (Seg)": np.mean(iou_list)
    }

    # =============================================================================
    # 3. CONSOLIDATED OUTPUT REPORT
    # =============================================================================
    print("\n" + "="*70)
    print("           TRANSUNET ARCHITECTURAL MASTER PERFORMANCE REPORT")
    print("="*70)
    for k, v in metrics.items():
        print(f"{k:<25}: {v:.4f}")
    print("-" * 70)
    print(classification_report(y_true, y_pred, target_names=['Healthy', 'Nodule']))

    # =============================================================================
    # 4. TRIPLE-VISUALIZATION SUITE
    # =============================================================================
    fig, ax = plt.subplots(1, 3, figsize=(22, 6))
    
    # Heatmap CM
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0], xticklabels=['H', 'N'], yticklabels=['H', 'N'])
    ax[0].set_title('Confusion Matrix')

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ax[1].plot(fpr, tpr, color='darkorange', label=f'AUC: {metrics["AUC-ROC"]:.4f}')
    ax[1].plot([0, 1], [0, 1], '--', color='navy')
    ax[1].set_title('ROC Characteristics')
    ax[1].legend()

    # Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ax[2].plot(rec, prec, color='green', label=f'mAP: {metrics["Avg Precision (mAP)"]:.4f}')
    ax[2].set_title('Precision-Recall Curve')
    ax[2].legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_comprehensive_evaluation()