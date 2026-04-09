# 🍎 AI Food Freshness Inspector

> Classify fruits & vegetables as **Fresh or Rotten** using CNNs — with confidence scores and Grad-CAM explainability.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-red)](https://streamlit.io)
[![HuggingFace](https://img.shields.io/badge/Deploy-HuggingFace%20Spaces-yellow)](https://huggingface.co/spaces)

---

## 🎯 Project Overview

This project builds an end-to-end deep learning pipeline to detect whether a fruit or vegetable is **fresh or rotten**, using:

- **Custom CNN** (built from scratch to learn every layer)
- **Transfer Learning** (EfficientNet-B0, MobileNetV2, ResNet-50)
- **Grad-CAM** explainability (visualise what the CNN actually looks at)
- **Streamlit App** (upload image → get prediction + heatmap)

### 🗂️ Dataset
- **Kaggle**: ["Fresh and Rotten Classification"](https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-images-of-fruits-and-vegetables)
- 13,000+ images · 6 categories (fresh/rotten × apple, banana, orange)

---

## 📁 Project Structure

```
food-freshness-inspector/
│
├── notebooks/                  # Jupyter / Colab notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_custom_cnn.ipynb
│   ├── 03_transfer_learning.ipynb
│   ├── 04_grad_cam.ipynb
│   └── 05_model_comparison.ipynb
│
├── src/                        # Reusable Python modules
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py          # Dataset loading & splitting
│   │   └── augmentation.py     # Data augmentation pipelines
│   ├── models/
│   │   ├── __init__.py
│   │   ├── custom_cnn.py       # CNN built from scratch
│   │   └── transfer_models.py  # EfficientNet, MobileNet, ResNet wrappers
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py          # Training loop, callbacks, logging
│   ├── explainability/
│   │   ├── __init__.py
│   │   └── grad_cam.py         # Grad-CAM implementation
│   └── utils/
│       ├── __init__.py
│       ├── visualise.py        # Plotting utilities
│       └── metrics.py          # Evaluation helpers
│
├── app/                        # Streamlit deployment app
│   ├── app.py
│   ├── requirements.txt
│   └── assets/
│       └── demo.gif
│
├── configs/                    # Hyperparameter configs
│   ├── custom_cnn_config.yaml
│   └── transfer_config.yaml
│
├── models/                     # Saved model weights (gitignored)
│   └── .gitkeep
│
├── results/                    # Training plots, metrics (gitignored)
│   └── .gitkeep
│
├── requirements.txt            # Full dev requirements
├── requirements-colab.txt      # Minimal Colab requirements
└── README.md
```

---

## 🚀 Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data Preparation | ⬜ |
| 2 | Custom CNN (from scratch) | ⬜ |
| 3 | Transfer Learning | ⬜ |
| 4 | Grad-CAM Explainability | ⬜ |
| 5 | Streamlit Deployment | ⬜ |

---

## ⚡ Quick Start (Local Development)

```bash
# Clone and enter directory
cd food-freshness-inspector

# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate        # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/app.py
```

## ☁️ Training on Google Colab

Open any notebook in `notebooks/` — each is self-contained with dataset download and training code.

```python
# At the top of each Colab notebook:
!pip install -r requirements-colab.txt
```

---

## 📊 Results

*(To be filled after training)*

| Model | Accuracy | Inference Time |
|-------|----------|----------------|
| Custom CNN | - | - |
| MobileNetV2 | - | - |
| EfficientNet-B0 | - | - |
| ResNet-50 | - | - |

---

## 🧠 Key Concepts Covered

- Convolution & receptive fields
- Batch Normalisation & Dropout
- Data Augmentation strategies
- Learning Rate Scheduling
- Transfer Learning & Fine-tuning
- Grad-CAM gradient-weighted class activation maps
- Model export (`.h5`, `.pt`, ONNX)

---

## 📸 Demo

*(Grad-CAM heatmap overlay will go here)*

---

## 👤 Author

Built as a portfolio project demonstrating end-to-end CNN development for computer vision.
