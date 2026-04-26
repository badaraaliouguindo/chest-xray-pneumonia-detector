import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# --- Configuration ---
st.set_page_config(
    page_title="Pneumonia Detector",
    page_icon="💊",
    layout="wide"
)

# --- Style CSS ---
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
    }
    
    [data-testid="stMainBlockContainer"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        min-height: 100vh;
        padding: 40px 20px;
    }
    
    .main-header {
        text-align: center;
        margin-bottom: 40px;
        color: #1a202c;
    }
    
    .main-header h1 {
        font-size: 48px;
        font-weight: 700;
        margin-bottom: 12px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .main-header p {
        font-size: 18px;
        color: #4a5568;
        font-weight: 400;
    }
    
    .upload-section, .prediction-section {
        background: white;
        border-radius: 16px;
        padding: 32px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(0, 0, 0, 0.05);
    }
    
    .upload-section h2, .prediction-section h2 {
        font-size: 24px;
        color: #2d3748;
        margin-bottom: 20px;
        font-weight: 600;
    }
    
    .result-normal {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        color: #2d3748;
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 24px;
    }
    
    .result-pneumonia {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: #2d3748;
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 24px;
    }
    
    .result-text {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 12px;
    }
    
    .confidence-breakdown {
        background: #f7fafc;
        padding: 20px;
        border-radius: 12px;
        margin: 20px 0;
    }
    
    .confidence-breakdown h3 {
        font-size: 16px;
        color: #2d3748;
        margin-bottom: 16px;
        font-weight: 600;
    }
    
    .metric-card {
        background: #f7fafc;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        border-left: 4px solid #667eea;
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 8px;
    }
    
    .metric-label {
        font-size: 14px;
        color: #718096;
        font-weight: 500;
    }
    
    .info-box {
        background: linear-gradient(135deg, #e0e7ff 0%, #f3e8ff 100%);
        color: #3f2c70;
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin: 20px 0;
    }
    
    .divider {
        border: 0;
        height: 1px;
        background: linear-gradient(to right, transparent, #cbd5e0, transparent);
        margin: 32px 0;
    }
    
    .footer-text {
        text-align: center;
        color: #718096;
        font-size: 12px;
        margin-top: 32px;
    }
    
    .model-info {
        background: #f7fafc;
        padding: 24px;
        border-radius: 12px;
        border-left: 4px solid #667eea;
    }
    
    .model-info h3 {
        color: #2d3748;
        margin-bottom: 16px;
        font-weight: 600;
    }
    
    .model-info ul {
        color: #4a5568;
        line-height: 1.8;
    }
    
    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed #cbd5e0 !important;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# --- Modèle ---
class ResNet50Classifier(nn.Module):
    def __init__(self):
        super(ResNet50Classifier, self).__init__()
        self.model = models.resnet50(weights=None)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.model(x)

@st.cache_resource
@st.cache_resource
def load_model():
    from huggingface_hub import hf_hub_download
    model_path = hf_hub_download(
        repo_id="alioubguindo/resnet50-pneumonia-detector",
        filename="resnet50_pneumonia.pth"
    )
    
    # On charge directement avec weights_only=False pour compatibilité
    state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
    
    model = ResNet50Classifier()
    
    # Nettoyage des clés si nécessaire
    new_state_dict = {}
    for k, v in state_dict.items():
        # Supprime le préfixe "model." si présent
        new_key = k.replace("model.", "") if k.startswith("model.") else k
        new_state_dict[new_key] = v
    
    # Essai avec le state_dict original d'abord
    try:
        model.model.load_state_dict(new_state_dict)
    except:
        model.load_state_dict(state_dict)
    
    model.eval()
    return model

# --- Transformation ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Chargement ---
model = load_model()

# --- Interface ---
st.markdown('<div class="main-header"><h1>Pneumonia Detector</h1><p>Advanced Medical Imaging Analysis</p></div>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="upload-section"><h2>Upload X-Ray Image</h2></div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Choose a chest X-ray image",
        type=['jpg', 'jpeg', 'png']
    )

    if uploaded:
        image = Image.open(uploaded).convert('RGB')
        st.image(image, caption="Uploaded X-Ray Image", use_column_width=True)

with col2:
    if uploaded:
        st.markdown('<div class="prediction-section"><h2>Analysis Results</h2></div>', unsafe_allow_html=True)

        # Prédiction
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs   = torch.softmax(outputs, dim=1)[0]
            pred    = torch.argmax(probs).item()

        prob_normal    = probs[0].item() * 100
        prob_pneumonia = probs[1].item() * 100

        if pred == 1:
            st.markdown('<div class="result-pneumonia"><div class="result-text">PNEUMONIA DETECTED</div><p>Proceed with caution - Medical review recommended</p></div>', unsafe_allow_html=True)
            st.metric("Pneumonia Probability", f"{prob_pneumonia:.1f}%")
        else:
            st.markdown('<div class="result-normal"><div class="result-text">NORMAL</div><p>No signs of pneumonia detected</p></div>', unsafe_allow_html=True)
            st.metric("Normal Probability", f"{prob_normal:.1f}%")

        # Barre de probabilité
        st.markdown('<div class="confidence-breakdown"><h3>Confidence Breakdown</h3></div>', unsafe_allow_html=True)
        st.progress(prob_normal / 100)
        st.caption(f"Normal: {prob_normal:.1f}%")
        st.progress(prob_pneumonia / 100)
        st.caption(f"Pneumonia: {prob_pneumonia:.1f}%")

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # Métriques du modèle
        st.subheader("Model Performance")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown('<div class="metric-card"><div class="metric-value">0.970</div><div class="metric-label">AUC-ROC Score</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown('<div class="metric-card"><div class="metric-value">99.7%</div><div class="metric-label">Sensitivity</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown('<div class="metric-card"><div class="metric-value">81.0%</div><div class="metric-label">Accuracy</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="info-box"><strong>Medical Disclaimer</strong><br>This tool is for educational purposes only and should not replace professional medical diagnosis. Always consult qualified healthcare professionals.</div>', unsafe_allow_html=True)

    else:
        st.markdown('<div class="info-box"><strong>Getting Started</strong><br>Upload a chest X-ray image to begin analysis. The system will provide instant detection results with confidence metrics.</div>', unsafe_allow_html=True)

        # Model Info
        st.markdown('<div class="model-info"><h3>Model Information</h3><ul><li><strong>Architecture:</strong> ResNet50 (fine-tuned)</li><li><strong>Training Data:</strong> 5,216 chest X-rays</li><li><strong>Classes:</strong> Normal vs Pneumonia</li><li><strong>Performance:</strong> AUC-ROC 0.970</li><li><strong>Framework:</strong> PyTorch + Streamlit</li></ul></div>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)
