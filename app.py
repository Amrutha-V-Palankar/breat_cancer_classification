import os
import json
import random
import string
from pathlib import Path

import streamlit as st
from streamlit_lottie import st_lottie
import requests
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay

# -------------------------
# Config / Paths
# -------------------------
MODEL_PATH_DENSE = "model/densenet121_idc.h5"
MODEL_PATH_VGG = "model/vgg16_idc.h5"
REPORT_PATH_DENSE = "evaluation/classification_report.json"
REPORT_PATH_VGG = "evaluation/vgg16_classification_report.json"
CM_PATH_DENSE = "evaluation/confusion_matrix.csv"
CM_PATH_VGG = "evaluation/vgg16_confusion_matrix.csv"
USERS_DB = "users.json"
REMEMBER_FILE = "remember.json"

# -------------------------
# Utilities
# -------------------------
st.set_page_config(page_title="Breast Cancer Detection", layout="wide")

def safe_load_json(path):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def safe_save_json(path, data):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def load_users_normalized():
    raw = safe_load_json(USERS_DB)
    normalized = {}
    for username, value in raw.items():
        if isinstance(value, dict):
            pwd = value.get("password", "")
            email = value.get("email", "")
            normalized[username] = {"password": pwd, "email": email}
        elif isinstance(value, str):
            normalized[username] = {"password": value, "email": ""}
    return normalized

def save_users(users_dict):
    safe_save_json(USERS_DB, users_dict)

def load_remember():
    return safe_load_json(REMEMBER_FILE) or {}

def save_remember(data):
    safe_save_json(REMEMBER_FILE, data)

# -------------------------
# Lottie animation loader
# -------------------------
LOTTIE_URL = "https://assets1.lottiefiles.com/packages/lf20_jcikwtux.json"
def load_lottie_url(url):
    try:
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None

lottie_json = load_lottie_url(LOTTIE_URL)

# -------------------------
# Session defaults
# -------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "page" not in st.session_state:
    st.session_state.page = "Login/Register"

remember_data = load_remember()
prefill_username = remember_data.get("last_user", "") if isinstance(remember_data, dict) else ""

# -------------------------
# Pages
# -------------------------
def auth_page():
    st.markdown("<div style='display:flex; gap: 24px;'>", unsafe_allow_html=True)
    left, right = st.columns([1, 1])

    # Left side → Animation
    with left:
        if lottie_json:
            st_lottie(lottie_json, height=340, key="login_anim")
        else:
            st.image("https://cdn-icons-png.flaticon.com/512/2910/2910796.png", width=260)

    # Right side → Form
    with right:
        st.header("🔐 Login / Register")
        mode = st.radio("Mode", ["Login", "Register"], horizontal=True)

        username = st.text_input("Username", value=prefill_username, key="input_username")
        password = st.text_input("Password", type="password", key="input_password")

        users = load_users_normalized()

        if mode == "Login":
            if st.button("Login"):
                user = users.get(username)
                if user and user.get("password") == password:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.page = "Home"
                    save_remember({"last_user": username})
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials.")

        elif mode == "Register":
            email = st.text_input("Email (for reset)")
            if st.button("Register"):
                if username in users:
                    st.warning("Username already exists.")
                else:
                    users[username] = {"password": password, "email": email}
                    save_users(users)
                    st.success("✅ Registered successfully! Please login.")

    st.markdown("</div>", unsafe_allow_html=True)

def home_page():
    st.title("🏠 Home")
    st.subheader("Welcome to the Breast Cancer Detection System")

    st.markdown("""
    ## 🧬 About Breast Cancer
    Breast cancer occurs when cells in the breast grow and multiply in an unregulated way.
    It is one of the most researched areas in medical science, and advancements in technology have made early detection more reliable and accessible.

    Our application uses a state-of-the-art deep learning model (**DenseNet121**) to analyze histopathological images of breast tissue
    and assist in identifying potential abnormalities at the microscopic level.

    --- 

    ## 🔍 Common Symptoms
    - A lump or thickening in the breast
    - Change in size or shape of the breast
    - Skin changes such as dimpling or redness
    - Nipple discharge other than breast milk
    - Inversion or retraction of the nipple
    - Persistent pain in the breast or underarm area

    > Regular self-examinations and screening are helpful in identifying changes early.

    --- 

    ## 💡 Precautions & Health Advice
    - Perform monthly breast self-examinations
    - Schedule regular clinical screenings (e.g., mammograms)
    - Maintain a balanced diet and healthy body weight
    - Limit alcohol consumption and avoid smoking
    - Stay physically active and manage stress
    - Discuss family history with a healthcare provider

    --- 

    🧠 **Remember:** Early detection can make a significant difference in treatment outcomes.
    """)

def predict_uploaded_image():
    st.title("🩺 Upload Image for Prediction")
    uploaded = st.file_uploader("Upload an image (jpg/png/jpeg)", type=["jpg", "png", "jpeg"])
    if not uploaded:
        st.info("Upload an image to get prediction.")
        return
    try:
        img = Image.open(uploaded).convert("RGB").resize((224,224))
        st.image(img, caption="Uploaded Image")
        arr = img_to_array(img) / 255.0
        arr = np.expand_dims(arr, 0)
        if not os.path.exists(MODEL_PATH_DENSE):
            st.warning(f"Model file not found at `{MODEL_PATH_DENSE}`. Predictions disabled.")
            return
        model = load_model(MODEL_PATH_DENSE)
        pred = model.predict(arr)
        try:
            val = float(pred[0][0])
        except Exception:
            val = float(np.asarray(pred).reshape(-1)[0])
        label = "Cancer" if val > 0.5 else "No Cancer"
        confidence = val if val > 0.5 else 1 - val
        st.markdown(f"### 🔍 Prediction: **{label}**")
        st.markdown(f"### 📊 Confidence: **{confidence*100:.2f}%**")
    except Exception as e:
        st.error(f"Prediction error: {e}")

# -------------------------
# Custom Confusion Matrix (Exact Format)
# -------------------------
# def plot_confusion_matrix_custom(cm, model_name="Confusion matrix"):
#     fig, ax = plt.subplots(figsize=(5, 4))
#     cax = ax.matshow(cm, cmap="Blues")
#     fig.colorbar(cax)

#     ax.set_title(model_name, pad=20)
#     ax.set_xlabel("Predicted")
#     ax.set_ylabel("Expected")

#     ax.set_xticks(np.arange(cm.shape[1]))
#     ax.set_yticks(np.arange(cm.shape[0]))
#     ax.set_xticklabels(range(cm.shape[1]))
#     ax.set_yticklabels(range(cm.shape[0]))

#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

#     plt.tight_layout()
#     return fig

def plot_confusion_matrix_custom(cm, model_name="Confusion matrix"):
    fig, ax = plt.subplots(figsize=(5, 4))
    cax = ax.matshow(cm, cmap="Blues")
    fig.colorbar(cax)

    # Title and labels
    ax.set_title(model_name, pad=20)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Expected")

    # Proper tick placement
    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))
    ax.set_xticklabels([str(i) for i in range(cm.shape[1])])
    ax.set_yticklabels([str(i) for i in range(cm.shape[0])])

    # Annotate with raw counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", color="black")

    plt.tight_layout()
    return fig


# -------------------------
# Evaluation Metrics Page
# -------------------------
def evaluation_metrics():
    st.title("📈 Evaluation Metrics")
    if not os.path.exists(REPORT_PATH_DENSE) or not os.path.exists(REPORT_PATH_VGG):
        st.warning("Evaluation reports not found. Please ensure both JSON files exist.")
        return

    # Load classification reports
    with open(REPORT_PATH_DENSE, "r") as f:
        dense_report = json.load(f)
    with open(REPORT_PATH_VGG, "r") as f:
        vgg_report = json.load(f)

    st.subheader("📑 Classification Reports")
    col1, col2 = st.columns(2)

    def report_to_df(report):
        df = pd.DataFrame(report).T
        return df.round(4)

    with col1:
        st.markdown("### 🟢 DenseNet121")
        st.table(report_to_df(dense_report))

    with col2:
        st.markdown("### 🔵 VGG16")
        st.table(report_to_df(vgg_report))

    # Comparison
    metrics = ["precision", "recall", "f1-score", "accuracy"]
    dense_scores = [dense_report["macro avg"][m] if m != "accuracy" else dense_report["accuracy"] for m in metrics]
    vgg_scores = [vgg_report["macro avg"][m] if m != "accuracy" else vgg_report["accuracy"] for m in metrics]

    st.subheader("📊 DenseNet121 vs VGG16 (Comparison)")
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(metrics))
    width = 0.35
    bars1 = ax.bar(x - width/2, dense_scores, width, label="DenseNet121")
    bars2 = ax.bar(x + width/2, vgg_scores, width, label="VGG16")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Performance Comparison")
    ax.legend()
    ax.bar_label(bars1, labels=[f"{v*100:.1f}%" for v in dense_scores], padding=3)
    ax.bar_label(bars2, labels=[f"{v*100:.1f}%" for v in vgg_scores], padding=3)
    st.pyplot(fig)

    # Confusion Matrices
    st.subheader("🧩 Confusion Matrices")
    cm_paths = {"DenseNet121": CM_PATH_DENSE, "VGG16": CM_PATH_VGG}
    for model_name, path in cm_paths.items():
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0)
            cm = df.values
            fig = plot_confusion_matrix_custom(cm, model_name=f"{model_name} Confusion matrix")
            st.pyplot(fig)
        else:
            st.warning(f"Confusion matrix not found for {model_name}")

# -------------------------
# Sidebar / Routing
# -------------------------
st.sidebar.title("Navigation")
if st.session_state.authenticated:
    st.sidebar.markdown(f"👤 **{st.session_state.username}**")
    page_choice = st.sidebar.selectbox("Choose page", ["Home", "Upload & Predict", "Evaluation Metrics"])
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.session_state.page = "Login/Register"
        st.rerun()
    st.session_state.page = page_choice
else:
    st.session_state.page = "Login/Register"

if st.session_state.page == "Login/Register":
    auth_page()
elif st.session_state.page == "Home":
    home_page()
elif st.session_state.page == "Upload & Predict":
    predict_uploaded_image()
elif st.session_state.page == "Evaluation Metrics":
    evaluation_metrics()
