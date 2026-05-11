# 🫁 Lung Cancer Diagnostic Suite: Clinical-Grade AI Ensemble
### Featuring True 3D Volumetric Grad-CAM & Multi-Planar Inference

## 📌 Project Overview
This research project focuses on the automated detection, segmentation, and explainability of lung nodules from CT scans using the **LUNA16 dataset**. The study evaluates the evolution of deep learning architectures—moving from basic convolutional networks to hybrid Transformers—culminating in a **Master Ensemble Brain** that mathematically fuses multiple neural networks for unprecedented clinical accuracy.

---

## 🏗️ Model Architectures

### 1. SimpleCNN (The Baseline)
* **Architecture**: A lightweight Encoder-Decoder U-Net.
* **Purpose**: Established the initial benchmark for localized nodule segmentation.
* **Limitation**: Struggled with "Hard Samples" and complex lung parenchyma textures due to a limited receptive field.

### 2. Multi-Task ResNet-18
* **Architecture**: ResNet-18 backbone with a custom Transpose Convolution decoder.
* **Purpose**: Utilized deep residual learning to enhance feature extraction.
* **Strength**: Achieved high spatial accuracy and robust performance on varying tumor morphologies.

### 3. Ultimate TransUNet 
* **Architecture**: Hybrid CNN-Transformer.
* **Feature**: Employs a **6-layer Vision Transformer Bottleneck** to capture global context and long-range dependencies.
* **Innovation**: Uses **Hybrid Loss** (Dice Loss + Focal Loss) to handle severe class imbalance (Malignant vs. Benign).

### 4. 🧠 The Ensemble Model - Tri Fusion Ensemble Model (Final Proposed Model)
* **Architecture**: A multi-modal fusion engine.
* **Mechanism**: Simultaneously passes the 3D voxel patch through the SimpleCNN, ResNet-18, and TransUNet. The outputs are flattened, concatenated, and fed into a **Dense Fusion Gate**.
* **Clinical Advantage**: Eliminates individual model biases by forcing a consensus among textural, spatial, and global attention features, drastically reducing False Positives.

---

## 📊 Comparative Performance Leaderboard

| Metric | SimpleCNN | ResNet-18 | Ultimate TransUNet | **TSFE** |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Dice Score** | 0.8665 | 0.9401 | 0.9191 | **0.95+** |
| **AUC (Detection)** | 0.9816 | 0.9712 | 0.9807 | **0.99+** |
| **Overall Accuracy** | 93.00% | 93.63% | 94.10% | **99.34%** |
| **Nodule Precision** | 0.93 | 0.92 | 0.96 | **0.98** |

*(Note: Ensemble metrics represent final optimized pipeline reduction of false alarms).*

---

## 🖥️ Streamlit Diagnostic Dashboard (Clinical UI)
The project features a **High-Tech Diagnostic HUD** built with Streamlit, designed to mimic enterprise hospital software (like Siemens or GE Healthcare).
* **Coordinate-Neutral Inference**: Extracts physical patient coordinates (World-to-Voxel) but performs mathematically blind predictions to ensure zero data leakage.
* **Lightning-Fast `.npz` Pipeline**: Massive 120GB+ `.mhd` files are pre-compressed into localized `.npz` arrays, allowing the dashboard to render full patient volumes instantly.
* **True 3D Volumetric Grad-CAM**: Features a completely explainable AI overlay. The 2D attention weights are spherically projected and blurred via a Gaussian depth-profile, rendering a clinical "heatmap" seamlessly across the Axial, Coronal, and Sagittal planes.
* **Auto-Reporting**: Automatically generates and downloads professional multi-planar PNG imaging reports for clinical documentation.

---

## 📂 Project Directory Structure 
```bash
├── Common CSV files/          # candidates_V2.csv (Neutral Ground Truth Coordinates)
├── Compressed_UI_Scans/       # Highly optimized .npz CT Volumes for instant UI loading
├── TransUNet_Preprocessed/    # Specialized 16x64x64 voxel patches for training
├── simpleCNN_model.py         # Baseline Architecture
├── Resnet_model.py            # Residual Architecture
├── TransUNet_model.py         # Transformer Architecture
├── app.py                     # Streamlit Diagnostic UI & Grad-CAM visualizer
└── ultimate_ensemble_brain_SCRATCH_FINAL.pth # Final optimized fusion weights
```

---

## 🚀 Execution Guide

1. **Virtual Environment Setup**:
* Python 3.12.X is must needed for this to run.
```bash
py -3.12 -m venv venv
# Activate the environment (Windows)
venv\Scripts\activate
```

2. **Install Required Libraries**:
```bash
pip install -r requirements.txt
```
*(Note: `scipy` is strictly required for the 3D Volumetric Grad-CAM Gaussian projections).*

3. **Data Preparation**:
* Download the LUNA16 CT Scan Dataset.
* Ensure original `.mhd` files have been routed through the compression script to generate `.npz` files inside the `Compressed_UI_Scans/` directory.

4. **Launching UI Interface**:
```bash
streamlit run app.py
```
*Navigate to `http://localhost:8501` to access the clinical dashboard.*

### Authors - Dhanesh J, Akash Krishnan
