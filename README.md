# Solar Panel Visible & Thermal Fault Detection System

An end-to-end deep learning and computer vision application designed to detect and classify faults in solar panels using image processing and deep learning architectures (MobileNetV3 and ResNet-50). Accompanied by an interactive Streamlit web application for real-time inference and automated maintenance recommendations.

---

## 🚀 Features
* **Multi-Class Fault Classification:** Detects various types of anomalies and faults in solar panels.
* **Deep Learning Backbones:** Leverages fine-tuned architectures like MobileNetV3 and ResNet-50 for high-accuracy feature extraction.
* **Interactive Web Interface:** Built with Streamlit (`app.py`) for seamless image uploads and instant diagnostic results.
* **Automated Recommendation Engine:** Generates specific maintenance guidelines and operational insights (`recommendations.py`) based on the detected fault type.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Deep Learning Framework:** PyTorch, Torchvision
* **Web Framework:** Streamlit
* **Data Processing & Utilities:** NumPy, Pandas, Scikit-Learn, OpenCV

---

## 📂 Project Structure

```text
solar_panel_visible_fault_detection/
│
├── models/                  # Saved model checkpoints and class labels
├── data/                    # Dataset directory (Excluded from git tracking)
├── app.py                   # Main Streamlit web application
├── model_utils.py           # Model loading and inference helper functions
├── predict.py               # Script for running predictions via terminal
├── recommendations.py       # Automated maintenance action dictionary
├── train_rgb.py             # Model training script
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
