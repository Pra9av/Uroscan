import streamlit as st
import torch
import segmentation_models_pytorch as smp
import numpy as np
import os
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- Page Config ---
st.set_page_config(page_title="UroScan Pro AI", layout="wide", page_icon="🏥")

# Custom CSS for a cleaner look - FIXED the argument name here
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; font-weight: bold; }
    .stAlert { border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏥 UroScan: Advanced Kidney Stone Diagnostic")
st.markdown("---")

# --- 1. Model Loading with Auto-Path Handling ---
@st.cache_resource
def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'backend', 'best_stone_model_final.pth')
    
    # Architecture must match your training (ResNet34)
    model = smp.Unet(encoder_name='resnet34', classes=3, in_channels=3)
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model file missing at: {model_path}")
        st.stop()
        
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

model = load_model()

# --- 2. Enhanced Inference Function ---
def get_prediction(image, model):
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    input_tensor = transform(image=np.array(image))['image'].unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        # Get Probabilities for heatmap and final mask
        probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()
        mask = np.argmax(probs, axis=0)
    
    return mask, probs[2] # Return mask and Stone Confidence Map (Class 2)

# --- 3. Sidebar Information ---
with st.sidebar:
    st.header("About UroScan")
    st.info("AI Analysis using ResNet34 U-Net.")
    st.write("**Color Key:**")
    st.write("🔵 **Blue**: Kidney")
    st.write("🔴 **Red**: Stone")
    st.divider()
    st.warning("⚠️ For Research Use Only.")

# --- 4. User Interface ---
uploaded_file = st.file_uploader("Upload Medical Scan", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Scan")
        st.image(img, use_container_width=True)

    if st.button("🚀 Run AI Analysis"):
        with st.spinner("Analyzing pixels..."):
            mask, stone_heatmap = get_prediction(img, model)
            
            with col2:
                st.subheader("AI Segmentation Map")
                # Prepare colored mask
                mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
                mask_rgb[mask == 1] = [0, 120, 255] # Kidney
                mask_rgb[mask == 2] = [255, 0, 0]   # Stone
                
                # Resize mask back to original image size
                mask_resized = cv2.resize(mask_rgb, (img.size[0], img.size[1]), interpolation=cv2.INTER_NEAREST)
                st.image(mask_resized, use_container_width=True)
            
            # --- Diagnostic Report Section ---
            st.divider()
            res_col1, res_col2 = st.columns([1, 1])
            
            with res_col1:
                st.subheader("Diagnostic Summary")
                if 2 in mask:
                    st.error("🚨 POSITIVE: Stone Detected")
                    stone_size = np.sum(mask == 2)
                    st.metric("Detected Stone Area", f"{stone_size} px")
                else:
                    st.success("✅ NEGATIVE: No Stones Detected")
                    st.metric("Detected Stone Area", "0 px")

            with res_col2:
                st.subheader("Stone Confidence Heatmap")
                # Normalize heatmap for visibility
                if stone_heatmap.max() > 0:
                    heatmap_vis = (stone_heatmap / stone_heatmap.max())
                else:
                    heatmap_vis = stone_heatmap
                
                st.image(heatmap_vis, use_container_width=True)
                st.caption("Bright spots indicate where the AI is suspicious of a stone.")

else:
    st.info("Please upload a kidney scan to begin.")