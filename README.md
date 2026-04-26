# Chest X-Ray Pneumonia Detector

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-ResNet50-orange)
![Streamlit](https://img.shields.io/badge/Demo-Streamlit-red)
![HuggingFace](https://img.shields.io/badge/Model-HuggingFace-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A deep learning system that detects pneumonia from chest X-ray images using a fine-tuned ResNet50 model. Achieves 0.970 AUC-ROC with 99.7% sensitivity on the pneumonia class.
 **[Live Demo on Streamlit](https://chest-xray-pneumonia-detector-je9uqlucpuekamvnoljend.streamlit.app/)**
 **[Model on HuggingFace](https://huggingface.co/alioubguindo/resnet50-pneumonia-detector)**

---

##  Project Overview

This project applies transfer learning to medical imaging — fine-tuning a ResNet50 pretrained on ImageNet to classify chest X-rays as Normal or Pneumonia.

Key design decision: **optimize recall over precision** for the Pneumonia class. In a medical context, missing a pneumonia case (false negative) is far more dangerous than flagging a healthy patient for further review (false positive).

---

##  Results

| Metric | Value |
|--------|-------|
| AUC-ROC | **0.970** |
| Recall (Pneumonia) | **99.7%** |
| Accuracy | 81.0% |
| False Negatives | 1 / 390 |


---

##  Project Structure
```
chest-xray-pneumonia-detector/
├── notebooks/
│   ├── 01_EDA_XRay.ipynb             # Exploratory Data Analysis
│   └── 02_Modeling.ipynb             # ResNet50 fine-tuning
├── app/
│   └── streamlit_app.py              # Streamlit web application
├── requirements.txt
└── README.md

```



---

## Dataset

- **Source**: [Chest X-Ray Images (Pneumonia) — Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Size**: 5,216 training images + 624 test images
- **Classes**: NORMAL (1,341) vs PNEUMONIA (3,875)
- **Imbalance ratio**: 1:3 → handled with WeightedRandomSampler

---

## Model Architecture

- **Base**: ResNet50 pretrained on ImageNet (24M parameters)
- **Fine-tuned layers**: Last residual block (layer4) + custom classifier head
- **Classifier head**: Dropout(0.5) → Linear(2048→256) → ReLU → Dropout(0.3) → Linear(256→2)
- **Trainable parameters**: 15.5M / 24M (64.5%)

---

##  Training Details

- **Epochs**: 10
- **Batch size**: 32
- **Optimizer**: Adam (lr=0.0001)
- **Scheduler**: StepLR (step=3, gamma=0.5)
- **Class imbalance**: WeightedRandomSampler
- **Augmentation**: RandomHorizontalFlip, RandomRotation(±10°), ColorJitter
- **Hardware**: NVIDIA T4 GPU (Google Colab)

---

##  Run Locally

```bash
git clone https://github.com/badaraaliouguindo/chest-xray-pneumonia-detector
cd chest-xray-pneumonia-detector
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

---

##  Key Learnings

- Transfer learning dramatically reduces data requirements for medical imaging
- Class imbalance handling is critical — WeightedRandomSampler outperforms simple oversampling
- In medical AI, optimizing recall > precision for pathology detection
- Data augmentation prevents overfitting on small medical datasets
- ResNet's skip connections solve the vanishing gradient problem in deep networks

---

##  Medical Disclaimer

This tool is for **educational purposes only** and should not replace professional medical diagnosis. Always consult qualified healthcare professionals for medical decisions.

---

## Tech Stack

`Python` `PyTorch` `torchvision` `ResNet50` `Streamlit` `HuggingFace Hub` `scikit-learn` `Google Colab`

---

##  Author

**Badara Aliou Guindo** — Master's student in Data Science & AI
[GitHub](https://github.com/badaraaliouguindo) • [HuggingFace](https://huggingface.co/alioubguindo)
