# OLP-AI-HCMC-2025 🚀

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![OLP-AI](https://img.shields.io/badge/Competition-OLP--AI--2025-red)](https://oai.hutech.edu.vn/)

## 📌 Introduction
This repository contains the source code, pre-trained models, and experimental results of our team's solution. The project focuses on leveraging state-of-the-art Machine Learning and Deep Learning architectures to tackle the specific challenges defined in the OLP-AI 2025 competition.

## 📂 Project Structure
```text
.
├── data/               # Raw and processed datasets (train, test, val)
├── libs/               # Custom utility modules, loss functions, and model architectures
├── output/             # Model checkpoints, training logs, and submission files
├── main.py             # Main entry point for training and inference
├── requirements.txt    # Python environment dependencies
└── README.md           # Project documentation
```

## 📦 Prerequisites
- Python 3.12
- Conda (optional)
- CUDA (GPU version, not CPU)

## 🛠 Installation & Setup
### 1. Clone the repository

```bash
git clone https://github.com/ThanhDanh1510/OLP-AI-HCMC-2025.git
cd OLP-AI-HCMC-2025
```

### 2. Create and activate a virtual environment (Optional)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install required dependencies

```bash
pip install -r requirements.txt
```
## 🚀 Usage Guide
### 1. Data preparation
Place your dataset files into the data/ directory. Ensure the folder structure follows the competition's official requirements.

### 2. Execution

```bash
python main.py --config configs/default.yaml
```

## 🌟 Features & Methodology
### Data Preprocessing
- Input size: 3x32x32 (standard for competition benchmarks)
- Advanced augmentation (Normalization, Padding, and Feature Scaling)

### Model Architecture: SimpleCNN
- Depth: 8-layer deep Convolutional Neural Network
- Activation: LeakyReLU
- Normalization: Integrated BatchNorm2d layers for stable and faster training
- Residual Connections: Implementation of custom Shortcut Blocks (using 5x5 Conv) 
  to mitigate vanishing gradients and improve deep feature extraction

### Optimization & Training
- Weight Init: He initialization
- Optimizer: AdamW / Adam with weight decay

## 📝 Results
- Epochs          : 30
- Batch Size      : 64
- Accuracy        : 68.00%
- F1-Score        : 67.61%
