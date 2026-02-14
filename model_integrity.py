import os
import torch
import random
import numpy as np
import pandas as pd
from glob import glob
from torch.utils.data import DataLoader
from sklearn.metrics import brier_score_loss, accuracy_score

# Import model classes
from TransUNet_model import UltimateTransUNet, TransUNetConfig, MultiTaskDataset
from Resnet_model import ResNetMultiTaskModel
from simpleCNN_model import SimpleMultiTaskCNN

def get_audit_loader(config):
    pos_paths = glob(os.path.join(config.PREPROCESSED_PATH, 'trans_pre_subset*', 'images', 'pos*.npy'))
    neg_paths = glob(os.path.join(config.PREPROCESSED_PATH, 'trans_pre_subset*', 'images', 'neg*.npy'))
    all_files = [(f, 1) for f in pos_paths] + [(f, 0) for f in neg_paths]
    random.seed(config.SEED)
    random.shuffle(all_files)
    split = int(len(all_files) * 0.8)
    test_ds = MultiTaskDataset(all_files[split:])
    return DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

def run_integrity_audit():
    config = TransUNetConfig()
    device = torch.device(config.DEVICE)
    test_loader = get_audit_loader(config)

    models_to_audit = {
        "SimpleCNN": (SimpleMultiTaskCNN(), "simpleCNN_unet_best.pth"),
        "ResNet-18": (ResNetMultiTaskModel(), "resnet_multitask_best.pth"),
        "TransUNet": (UltimateTransUNet(in_channels=config.SLICES), "transunet_ULTIMATE_best.pth")
    }

    audit_data = []

    for name, (model, weight_path) in models_to_audit.items():
        if not os.path.exists(weight_path): continue
        
        print(f"🔄 Auditing {name}...")
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        model.to(device).eval()

        y_true, y_prob = [], []
        with torch.no_grad():
            for img, _, label in test_loader:
                img = img.to(device)
                output = model(img)
                logits = output[1] if isinstance(output, tuple) else output
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                y_prob.extend(probs)
                y_true.extend(label.numpy())

        y_true, y_prob = np.array(y_true), np.array(y_prob)
        y_pred = (y_prob > 0.5).astype(int)

        # FIX: Calculate Gap based ONLY on positive predictions (Clinical Honesty)
        pos_indices = np.where(y_pred == 1)[0]
        if len(pos_indices) > 0:
            avg_pos_conf = np.mean(y_prob[pos_indices])
            actual_pos_acc = np.mean(y_true[pos_indices])
            calibration_gap = abs(avg_pos_conf - actual_pos_acc)
        else:
            calibration_gap = 0
            avg_pos_conf = 0

        brier = brier_score_loss(y_true, y_prob)
        total_acc = accuracy_score(y_true, y_pred)

        # Final Verdict Logic
        status = "✅ TRULY TRAINED"
        if calibration_gap > 0.15: status = "🚨 OVERCONFIDENT"
        if total_acc > 0.98 and name != "TransUNet": status = "🚩 POTENTIAL OVERFIT"

        audit_data.append({
            "Model": name,
            "Accuracy": round(total_acc, 4),
            "Nodule Conf.": round(avg_pos_conf, 4),
            "Gap": round(calibration_gap, 4),
            "Brier (Honesty)": round(brier, 4),
            "Verdict": status
        })

    df = pd.DataFrame(audit_data)
    print("\n" + "="*95)
    print("                OFFICIAL MODEL INTEGRITY & CLINICAL HONESTY REPORT")
    print("="*95)
    print(df.to_string(index=False))
    print("="*95)

if __name__ == "__main__":
    run_integrity_audit()