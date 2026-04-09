# 🍎 FreshVision AI: Deep Learning Fruit Freshness Inspector

DEPLOYED LINK: https://fruit-checkup-vaibhav.streamlit.app/

> An end-to-end computer vision system to classify fruits as **Fresh or Rotten** using a fine-tuned MobileNetV2 architecture, featuring real-time Grad-CAM explainability via Streamlit.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-red)](https://streamlit.io)

---

## 🎯 Architecture Overview

FreshVision AI is designed as an end-to-end deep learning pipeline. The primary objective is not just accurate classification of fresh vs. rotten fruit, but providing **transparency in model decision-making** using XAI (Explainable AI) techniques.

The system is separated into three core components:
1. **Model Training & Fine-Tuning Pipeline**: Handles data ingestion, augmentation, transfer learning, and hyperparameter tuning.
2. **Explainability Engine**: A Grad-CAM implementation that generates class activation maps specifically layered onto the final spatial convolution operations.
3. **Inference UI (Streamlit)**: A robust, low-latency web interface allowing users to upload images or take live photos for immediate analysis.

### 🗂️ Dataset & Augmentation Strategy
- **Base Dataset**: ["Fresh and Rotten Classification" on Kaggle](https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-images-of-fruits-and-vegetables).
- **Categories**: 6 distinct classes (`freshapples`, `freshbanana`, `freshoranges`, `rottenapples`, `rottenbanana`, `rottenoranges`).
- **Data Augmentation Strategies**: To combat overfitting and improve generalization across diverse lighting conditions and angles, we applied robust real-time augmentations using TensorFlow's `ImageDataGenerator`.
  - Random rotations (up to 20°)
  - Width & Height shifts
  - Horizontal flipping & Random zooming

---

## 🧠 Deep Learning Architecture: MobileNetV2

We employ **Transfer Learning**, specifically utilizing the **MobileNetV2** architecture pre-trained on ImageNet.

### Why MobileNetV2?
MobileNetV2 provides an optimal trade-off between **latency and accuracy**. By leveraging **depthwise separable convolutions** and **inverted residual blocks with linear bottlenecks**, the model significantly reduces the parameter count compared to traditional networks like ResNet or VGG, making it ideal for real-time web deployment without sacrificing predictive performance.

### Custom Head & Fine-Tuning Strategy
1. **Base Extraction**: The foundational layers (ImageNet weights) are initially frozen. We extract features up to the last convolutional block.
2. **Custom Classification Head**:
   - `GlobalAveragePooling2D`: Replaces standard flattening to drastically reduce parameters and minimize overfitting.
   - `Dense(128, ReLU)`: Adds required non-linear transformations specific to the fruit dataset.
   - `Dropout(0.5)`: High regularization to ensure the network relies on diverse feature distributions.
   - `Dense(6, Softmax)`: Outputs normalized probability distributions over the 6 classes.
3. **Fine-Tuning**: After training the top layers, we unfreeze the later stages of the base MobileNetV2 architecture. We drastically reduce the learning rate (`1e-5`) and fine-tune using the `Adam` optimizer, allowing the model to refine its abstract representations (e.g., specific rot texture maps) rather than just edges and colors.

---

## 🔬 Explainable AI: Grad-CAM

A key feature of FreshVision AI is its Explainable AI (XAI) component. Neural networks are often treated as "black boxes". We utilize **Gradient-weighted Class Activation Mapping (Grad-CAM)** to break this paradigm.

### How Grad-CAM Works in This Project
1. **Gradient Extraction**: During inference, we capture the gradient of the winning class's score with respect to the feature maps of the final convolutional layer (in MobileNetV2, this is the `out_relu` layer).
2. **Global Average Pooling**: The gradients are globally average-pooled to obtain "neuron importance weights".
3. **Weighted Feature Maps**: We compute a weighted sum of the layer's forward-activation maps.
4. **ReLU Activation**: A ReLU is applied to the combination to isolate features that have a *positive* influence on the predicted class.
5. **Upsampling & Overlay**: The resulting coarse heatmap (often 7x7) is upsampled back to the original image dimensions (224x224) using bicubic interpolation and overlaid using a `jet` colormap.

**Interpretation:**
- 🔴 **Red/Warm regions**: High attention. These are the specific textures or color gradients (like a bruised spot on an apple) that convinced the model of its prediction.
- 🔵 **Blue/Cool regions**: Low attention. These are background elements or uninformative pixels that the model actively ignored.

---

## 🚀 Streamlit Deployment

The frontend is constructed using Streamlit to offer a direct, seamless interface for inference.

### Key Capabilities
- **Dual Input Modes**: Supports standard file uploads (`JPG`, `PNG`, `WEBP`) or live capture via a webcam.
- **Preprocessing Pipeline**: Integrates the identical image resizing and `[0, 1]` normalization steps used during training to prevent data drift between train/inference.
- **Real-Time UI Rerendering**: Uses Streamlit's reactive framework and custom injected CSS for a professional, glassmorphism-inspired "app" feel.
- **Full Transparency**: Outputs the exact confidence metric alongside a probability breakdown for all 6 classes.

---

## ⚡ Quick Start (Local Development)

### Prerequisites
- Python 3.9+
- TensorFlow 2.x

```bash
# 1. Clone the repository
git clone https://github.com/7vaibhav31/Fruit_Checkup.git
cd Fruit_Checkup

# 2. Setup Virtual Environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# 3. Install Requirements
pip install -r requirements.txt

# 4. Launch the AI Inspector
cd pro
streamlit run app.py
```
*(The Streamlit app will automatically initialize locally at `http://localhost:8501`)*
