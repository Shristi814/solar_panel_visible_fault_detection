import streamlit as st
from PIL import Image
import torch
from pathlib import Path
from predict import load_class_names, load_model, predict, EVAL_TF, MODELS_DIR
from recommendations import RECOMMENDATIONS  # Your existing recommendation dictionary

st.set_page_config(
    page_title="Solar Panel Fault Detection",
    page_icon="☀️",
    layout="centered"
)

@st.cache_resource
def get_model_and_classes():
    device = torch.device("cpu")
    class_names = load_class_names()
    model = load_model("resnet50", MODELS_DIR / "resnet50_rgb.pth", len(class_names), device)
    return model, class_names, device

model, class_names, device = get_model_and_classes()

st.title("☀️ Solar Panel Fault & Defect Detection")
st.write("Upload an infrared or visible image of a photovoltaic panel to classify potential faults and view maintenance recommendations.")

# File uploader (supports both uploaded files and camera capture)
uploaded_file = st.file_uploader("Choose a panel image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Panel Image", use_container_width=True)
    
    if st.button("Run Fault Analysis", type="primary"):
        with st.spinner("Analyzing image features..."):
            pred_class, conf, probs = predict(model, image, class_names, device)
            
        st.success(f"**Detected Fault:** {pred_class} (Confidence: {conf*100:.1f}%)")
        
        # Display probabilities breakdown
        with st.expander("See full class probabilities"):
            for c, p in sorted(probs.items(), key=lambda kv: -kv[1]):
                st.progress(p, text=f"{c}: {p*100:.1f}%")
                
        # Fetch and display recommendations from your dictionary
        if pred_class in RECOMMENDATIONS:
            rec = RECOMMENDATIONS[pred_class]
            st.markdown("### 🛠️ Maintenance & Prevention Protocol")
            st.info(rec.get("description", ""))
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Severity Level", value=rec.get("severity", "N/A"))
            with col2:
                st.metric(label="Urgency / Action Time", value=rec.get("urgency", "N/A"))
        else:
            st.warning("No specific recommendation found for this category.")