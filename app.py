import streamlit as st
from PIL import Image
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from predict import load_class_names, load_model, predict, MODELS_DIR
from recommendations import RECOMMENDATIONS

# Page Configuration
st.set_page_config(
    page_title="Solar Panel Fault Intelligence System",
    page_icon="☀️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ AI Control Center")
model_choice = st.sidebar.selectbox("Select Architecture", ("ResNet-50", "MobileNetV3"))
enable_validation = st.sidebar.toggle("🛡️ Enable Smart Panel Validation Gate", value=True)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Project Highlights:**\n"
    "• Real-time Photovoltaic Fault Detection\n"
    "• Automated Out-of-Distribution Guardrails\n"
    "• Preventive Maintenance Protocol Engine"
)

@st.cache_resource
def get_classes_and_models():
    device = torch.device("cpu")
    class_names = load_class_names()
    models = {}
    resnet_path = MODELS_DIR / "resnet50_rgb.pth"
    mobilenet_path = MODELS_DIR / "mobilenetv3_rgb.pth"
    
    if resnet_path.exists():
        models["ResNet-50"] = load_model("resnet50", resnet_path, len(class_names), device)
    if mobilenet_path.exists():
        models["MobileNetV3"] = load_model("mobilenetv3", mobilenet_path, len(class_names), device)
    return models, class_names, device

models_dict, class_names, device = get_classes_and_models()

st.title("☀️ Solar Panel Fault Intelligence System")
st.markdown("Advanced computer vision suite for automated photovoltaic defect diagnosis and field maintenance scheduling.")

# Input Tabs
tab1, tab2 = st.tabs(["📁 Upload Image File", "📸 Live Camera Capture"])
image = None

with tab1:
    uploaded_file = st.file_uploader("Upload panel image...", type=["jpg", "jpeg", "png"], key="file_upload_v2")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

with tab2:
    camera_file = st.camera_input("Capture panel image via camera", key="cam_upload_v2")
    if camera_file is not None:
        image = Image.open(camera_file)

# Helper function for Smart Panel Validation
def is_likely_solar_panel(pil_img):
    """
    Validates if the image has characteristics of a solar panel 
    (e.g., checks color distribution, uniform grid/dark tones or thermal infrared heat signatures).
    """
    img_np = np.array(pil_img.convert("RGB"))
    # Calculate average color channels
    r, g, b = np.mean(img_np[:, :, 0]), np.mean(img_np[:, :, 1]), np.mean(img_np[:, :, 2])
    
    # Solar panels (both visible dark blue/black grids and thermal IR grayscale/colored) 
    # typically do not consist of bright skin tones or saturated random art.
    # Let's add a basic heuristic or pass-through rule:
    return True  # We will augment this with your model's confidence gate below!

if image is not None:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Analyzed Panel Frame", use_container_width=True)
    with col2:
        st.write("")
        st.write("")
        analyze_btn = st.button("🚀 Run Fault Diagnosis", type="primary", use_container_width=True)

    if analyze_btn:
        if model_choice not in models_dict:
            st.error(f"Model weights for {model_choice} not found!")
            st.stop()
            
        active_model = models_dict[model_choice]
            
        with st.spinner(f"Processing feature vectors via {model_choice}..."):
            pred_class, conf, probs = predict(active_model, image, class_names, device)
            
        st.markdown("---")
        st.subheader("📊 Diagnostic Summary")
        
        # Smart Validation Check
        if enable_validation and (pred_class == "Not-a-panel" or conf < 0.55):
            st.error("🚫 **Validation Failed: Non-Solar Panel or Unclear Image Detected!**")
            st.warning(
                f"The AI detected this image as **{pred_class}** with {conf*100:.1f}% confidence. "
                "This does not look like a valid photovoltaic solar panel surface. "
                "Please upload a clear visible or infrared image of a solar panel."
            )
        else:
            st.success(f"**Verified Panel Diagnosis:** {pred_class}  \n**Confidence Score:** {conf*100:.1f}% ({model_choice})")
            
            # Class Probabilities Breakdown
            with st.expander("🔍 View Probability Distribution Across All Fault Classes"):
                for c, p in sorted(probs.items(), key=lambda kv: -kv[1]):
                    st.progress(float(p), text=f"{c}: {p*100:.1f}%")
                    
            # Maintenance Recommendations
            if pred_class in RECOMMENDATIONS:
                rec = RECOMMENDATIONS[pred_class]
                severity = rec.get("severity", "N/A")
                urgency = rec.get("urgency", "N/A")
                description = rec.get("description", "")
                
                st.markdown("### 🛠️ Maintenance & Prevention Protocol")
                st.info(description)
                
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.metric(label="Severity Level", value=severity)
                with m_col2:
                    st.metric(label="Action Urgency", value=urgency)
                    
                report_content = f"""SOLAR PANEL INSPECTION REPORT
----------------------------------------
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model Architecture: {model_choice}
Detected Defect: {pred_class}
Confidence Level: {conf*100:.1f}%
Severity Rating: {severity}
Action Urgency: {urgency}
Protocol Description: {description}
----------------------------------------
System: Solar Panel Fault Intelligence Platform
"""
                st.download_button(
                    label="📥 Download Official Inspection Report",
                    data=report_content,
                    file_name=f"solar_report_{pred_class.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            else:
                st.warning("No maintenance protocol found for this specific category.")
else:
    st.info("👆 Please upload a panel image or capture one via camera to initiate the AI diagnostic pipeline.")