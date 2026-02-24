import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from ensemble_evaluation import UltimateEnsembleBrain, EnsembleEvalDataset
from TransUNet_model import TransUNetConfig
from glob import glob
from sklearn.model_selection import train_test_split
import os

def calibrate_timid_model():
    config = TransUNetConfig()
    device = config.DEVICE
    
    # 1. Load Data
    data_path = os.path.join(config.ROOT_DIR, "Ensemble_Data_Safe")
    all_files = glob(os.path.join(data_path, "images", "*.npz"))
    file_list = [(os.path.basename(f), 1 if "pos" in f else 0) for f in all_files]
    _, val_files = train_test_split(file_list, test_size=0.2, random_state=42)
    loader = DataLoader(EnsembleEvalDataset(val_files, os.path.join(data_path, "images"), os.path.join(data_path, "masks")), batch_size=8, shuffle=False)

    # 2. Load Model
    model = UltimateEnsembleBrain(in_channels=16).to(device)
    model.load_state_dict(torch.load("ultimate_ensemble_brain_v4_FINAL.pth", map_location=device, weights_only=True))
    model.eval()
    
    logits_list, labels_list = [], []
    
    print("Gathering logits for Calibration...")
    with torch.no_grad():
        for img, _, lbl in loader:
            _, logits = model(img.to(device))
            logits_list.append(logits)
            labels_list.append(lbl.to(device))
            
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list).unsqueeze(1)
    
    # 3. Learn the Temperature (T)
    # Start at 1.0. Because model is underconfident, T should shrink < 1.0 to amplify probabilities
    temperature = nn.Parameter(torch.ones(1).to(device))
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=100)
    
    def eval_loss():
        optimizer.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(logits / temperature, labels)
        loss.backward()
        return loss
    
    optimizer.step(eval_loss)
    print(f"\n⭐ Optimal Clinical Temperature Found: {temperature.item():.4f}")
    print("In your Streamlit/Evaluation scripts, change 'torch.sigmoid(p_clf)' to 'torch.sigmoid(p_clf / T)'")

if __name__ == "__main__":
    calibrate_timid_model()