import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from glob import glob
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Importing your model classes
from TransUNet_model import UltimateTransUNet, TransUNetConfig, MultiTaskDataset
from Resnet_model import ResNetMultiTaskModel
from simpleCNN_model import SimpleMultiTaskCNN

def get_test_loader(config):
    pos_paths = glob(os.path.join(config.PREPROCESSED_PATH, 'trans_pre_subset*', 'images', 'pos*.npy'))
    neg_paths = glob(os.path.join(config.PREPROCESSED_PATH, 'trans_pre_subset*', 'images', 'neg*.npy'))
    all_files = [(f, 1) for f in pos_paths] + [(f, 0) for f in neg_paths]
    all_files.sort() 
    split = int(len(all_files) * 0.8)
    test_ds = MultiTaskDataset(all_files[split:])
    return DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

def run_comprehensive_comparison():
    config = TransUNetConfig()
    device = torch.device(config.DEVICE)
    test_loader = get_test_loader(config)

    models_to_test = {
        "SimpleCNN": (SimpleMultiTaskCNN(), "simpleCNN_unet_best.pth"),
        "ResNet-18": (ResNetMultiTaskModel(), "resnet_multitask_best.pth"),
        "TransUNet": (UltimateTransUNet(in_channels=config.SLICES), "transunet_ULTIMATE_best.pth")
    }

    results = []
    for name, (model, weight_path) in models_to_test.items():
        if os.path.exists(weight_path):
            print(f"🔄 Evaluating {name}...")
            model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
            model.to(device).eval()
            
            y_true, y_pred, y_prob = [], [], []
            with torch.no_grad():
                for img, _, label in test_loader:
                    img, label = img.to(device), label.to(device)
                    output = model(img)
                    # Extract classification logits
                    logits = output[1] if isinstance(output, tuple) else output
                    probs = torch.sigmoid(logits).cpu().numpy().flatten()
                    
                    y_prob.extend(probs)
                    y_pred.extend((probs > 0.5).astype(int))
                    y_true.extend(label.cpu().numpy())
            
            # Compute Cumulative Metrics
            results.append({
                "Model": name,
                "Accuracy": accuracy_score(y_true, y_pred),
                "Precision": precision_score(y_true, y_pred),
                "Recall": recall_score(y_true, y_pred),
                "F1-Score": f1_score(y_true, y_pred),
                "AUC-ROC": roc_auc_score(y_true, y_prob)
            })
        else:
            print(f"⚠️ Missing weights for {name}")

    # Display Final Report
    df = pd.DataFrame(results)
    print("\n" + "="*70)
    print("           CUMULATIVE ARCHITECTURAL PERFORMANCE REPORT")
    print("="*70)
    print(df.to_string(index=False))

    # Visualization
    df.set_index("Model").plot(kind='bar', figsize=(12, 6))
    plt.title("Multi-Metric Comparison: Baselines vs TransUNet")
    plt.ylabel("Score")
    plt.ylim(0.7, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":
    run_comprehensive_comparison()