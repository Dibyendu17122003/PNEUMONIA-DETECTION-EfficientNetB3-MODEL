# app.py
import os
import ssl
import time
import base64
import smtplib
import traceback
import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st
import altair as alt

# TensorFlow / Keras (Assuming these are installed in the environment)
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input 

# Utils
import cv2
from fpdf import FPDF
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders

# --- Define Base Directory for Robust Path Handling ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="🩺",
    layout="wide",
)

# -------------------- GLOBALS ------------------------
MODEL_FILENAME = "pneumonia_final_Dibyendu.h5"
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)
IMAGE_SIZE = (300, 300)
CLASS_NAMES = ["Normal", "Pneumonia"] # index 0 = Normal, 1 = Pneumonia
NORMAL_IDX = 0
PNEUMONIA_IDX = 1

# --- HARDCODED FUTURISTIC DARK THEME COLORS ---
BG_COLOR = "#0A101A"     # Deep Dark Background
CARD_BG_COLOR = "#1B2A3A" # Dark Card Background (Blue-Grey)
ACCENT_COLOR = "#4DEEEA"   # Cyan/Aqua Accent for glow
TEXT_COLOR = "#E0FFFF"  # Off-White/Cyan Text
SUCCESS_COLOR = "#00FF7F"  # Bright Green for Normal
FAILURE_COLOR = "#FF6347"  # Tomato Red for Pneumonia

# -------------------- SESSION STATE INIT -----------------
if "history" not in st.session_state:
    st.session_state.history = []
if "last_overlay_b64" not in st.session_state:
    st.session_state.last_overlay_b64 = None
if "last_original_b64" not in st.session_state:
    st.session_state.last_original_b64 = None
if "last_pdf_bytes" not in st.session_state:
    st.session_state.last_pdf_bytes = None
if "doctor_notes" not in st.session_state:
    st.session_state.doctor_notes = ""
if "latest_batch_results" not in st.session_state:
    st.session_state.latest_batch_results = None # Re-introducing this to display results immediately after batch run
    
# -------------------- THEME / STYLE (Hardcoded Dark Mode) ------------------

def apply_theme():
    """Applies modern, responsive CSS styling for permanent dark theme."""
    st.markdown(f"""
    <style>
        /* General Background and Typography */
        .stApp {{
            background-color: {BG_COLOR};
            color: {TEXT_COLOR};
        }}
        h1, h2, h3, h4, h5, h6, .stMarkdown, label {{
            color: {TEXT_COLOR};
        }}
        
        /* Main Header Glow Effect */
        h1.main-header {{
              text-shadow: 0 0 10px {ACCENT_COLOR}, 0 0 20px rgba(77, 238, 234, 0.6);
              text-align: center;
              padding-bottom: 20px;
        }}
        
        /* Custom Card/Container Style (Glass/Glow Effect) */
        .stContainer, .stTabs {{
            background-color: {CARD_BG_COLOR};
            border-radius: 16px;
            padding: 30px;
            border: 1px solid rgba(77, 238, 234, 0.2);
            box-shadow: 0 0 15px rgba(77, 238, 234, 0.15); /* Subtle border glow */
            margin-bottom: 25px;
        }}
        
        /* Result Banners */
        .result-banner {{
            padding: 18px;
            border-radius: 10px;
            font-weight: 700;
            text-align: center;
            margin-bottom: 20px;
            font-size: 1.1em;
        }}
        .result-normal {{
            background-color: rgba(0, 255, 127, 0.1); 
            border-left: 6px solid {SUCCESS_COLOR};
            color: {SUCCESS_COLOR};
        }}
        .result-pneumonia {{
            background-color: rgba(255, 99, 71, 0.1); 
            border-left: 6px solid {FAILURE_COLOR};
            color: {FAILURE_COLOR};
        }}

        /* Detailed Probability Box */
        .prob-box {{
            padding: 12px;
            border-radius: 8px;
            border: 1px solid {ACCENT_COLOR};
            margin-bottom: 10px;
            font-size: 14px;
            color: {TEXT_COLOR};
        }}
        
        /* Images - Fully Responsive */
        .responsive-img {{
            max-width: 100%;
            height: auto;
            border-radius: 12px; 
            margin-top: 15px;
            border: 3px solid {ACCENT_COLOR};
            box-shadow: 0 0 8px {ACCENT_COLOR};
        }}
        
        /* Gauge Wrapper */
        .gauge-wrapper {{
            display: flex;
            justify-content: center;
            margin-top: 15px;
        }}
        
        /* Button Styling (High Contrast) */
        .stButton>button {{
            background-color: {ACCENT_COLOR};
            border-color: {ACCENT_COLOR};
            color: {BG_COLOR}; 
            font-weight: bold;
        }}
        .stButton>button:hover {{
            background-color: #008CBA; 
            border-color: #008CBA;
        }}
        
        /* Metrics for Summary Table */
        [data-testid="stMetricValue"] {{
            font-size: 24px;
            color: {ACCENT_COLOR};
        }}
        
        /* Footer/Disclaimer */
        .disclaimer {{
            font-size: 11px;
            opacity: 0.6;
            text-align: center;
            margin-top: 40px;
            border-top: 1px solid {CARD_BG_COLOR};
            padding-top: 15px;
        }}
    </style>
    """, unsafe_allow_html=True)

apply_theme()

# -------------------- SIDEBAR CONTROLS (Define variables in global scope) ------------------------
with st.sidebar:
    st.markdown("## ⚙ Settings & Configuration")
    
    st.markdown("---")
    st.markdown("### Model Behavior")
    
    prediction_threshold = st.slider(
        "Pneumonia Classification Threshold (%)",
        min_value=50, max_value=99, value=80, step=1,
        help="Confidence level required for the AI to classify the result as PNEUMONIA. Increasing this reduces false positives but may increase false negatives."
    )
    
    enable_gradcam = st.checkbox(
        "🔥 Grad-CAM heatmap (Explainability)", 
        value=True, 
        help="Generates a heat map overlay showing the regions of the image the AI focused on."
    )

    st.markdown("---")
    st.markdown("### 📧 Email Configuration")
    sender_email = st.text_input("Sender Gmail", placeholder="yourname@gmail.com")
    sender_app_password = st.text_input("Gmail App Password", type="password", help="Use a Gmail App Password, not your regular password.")
    
    st.markdown("---")
    st.markdown(f"*Model:* `{MODEL_FILENAME}`")
    st.markdown("*Input:* 300×300 RGB (EfficientNet preprocess)")


# -------------------- MODEL LOAD HELPERS ------------------------
@st.cache_resource(show_spinner="Loading and warming up AI Model...")
def load_model_cached():
    """Load the Keras model with robust path check and warm-up."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"FATAL ERROR: Model file not found at: {MODEL_PATH}. Please ensure '{MODEL_FILENAME}' is in the app directory.")
        st.stop()
    
    model = load_model(MODEL_PATH, compile=False) 
    
    try:
        # Dummy prediction for warm-up
        _ = model.predict(np.zeros((1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32), verbose=0)
    except Exception as e:
          st.error(f"Model warm-up failed. Check model file integrity. Error: {e}")
          st.stop()
          
    return model

def prepare_image(pil_img: Image.Image) -> np.ndarray:
    """Resize -> RGB -> EfficientNet preprocess -> add batch dimension."""
    img = pil_img.convert("RGB").resize(IMAGE_SIZE)
    arr = np.asarray(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def npimg_to_b64(np_img_bgr: np.ndarray) -> str:
    """Convert OpenCV BGR image (used by Grad-CAM) to base64 PNG string."""
    _, im_png = cv2.imencode(".png", np_img_bgr)
    return base64.b64encode(im_png.tobytes()).decode()

def render_gauge(percent: float, label: str):
    """Draw a clean semicircle gauge using inline SVG for confidence visualization."""
    p = max(0, min(100, percent))
    angle = -90 + (p * 180.0 / 100.0)
    # Use hardcoded theme colors
    gauge_color = FAILURE_COLOR if p > 50 else SUCCESS_COLOR
    
    st.markdown(f"""
    <div class='gauge-wrapper'>
      <svg viewBox="0 0 100 60" style="width: 150px; max-width: 90%;">
        <path d="M10,50 A40,40 0 0,1 90,50" fill="none" stroke="rgba(255,255,255,0.25)" stroke-width="6"/>
        <path d="M10,50 A40,40 0 0,1 90,50" fill="none" stroke="{gauge_color}" stroke-width="6"
              stroke-dasharray="{p*1.256},{125.6}"/>
        <line x1="50" y1="50" x2="{50 + 38 * np.cos(np.radians(angle))}" y2="{50 + 38 * np.sin(np.radians(angle))}"
              stroke="{gauge_color}" stroke-width="2"/>
        <text x="50" y="58" text-anchor="middle" font-size="8" fill="{TEXT_COLOR}">{label}: {p:.1f}%</text>
      </svg>
    </div>
    """, unsafe_allow_html=True)

# -------------------- GRAD-CAM LOGIC ------------------------
def find_last_conv_layer(keras_model: tf.keras.Model):
    """Robustly get the last 4D layer (conv/activation) for Grad-CAM."""
    for layer in reversed(keras_model.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer.name
        except Exception:
            continue
    return None

def gradcam_overlay(pil_img: Image.Image, model: tf.keras.Model, processed_batch: np.ndarray, target_class_idx: int) -> np.ndarray:
    """Compute Grad-CAM heatmap and overlay on the original image."""
    layer_name = find_last_conv_layer(model)
    if layer_name is None:
        st.warning("Could not find a suitable convolution layer for Grad-CAM.")
        return np.asarray(pil_img.convert("RGB").resize(IMAGE_SIZE))[:, :, ::-1]

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(processed_batch)
        loss = preds[:, target_class_idx] 

    grads = tape.gradient(loss, conv_out)
    if grads is None:
          return np.asarray(pil_img.convert("RGB").resize(IMAGE_SIZE))[:, :, ::-1]

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = tf.nn.relu(tf.reduce_sum(tf.multiply(pooled_grads, conv_out), axis=-1))

    denom = tf.reduce_max(heatmap)
    heatmap = heatmap / (denom + 1e-10)
    heatmap = heatmap.numpy()

    heatmap = cv2.resize(heatmap, IMAGE_SIZE[::-1])
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original_rgb = np.asarray(pil_img.convert("RGB").resize(IMAGE_SIZE))
    original_bgr = original_rgb[:, :, ::-1]
    overlay = cv2.addWeighted(original_bgr, 0.6, heatmap_colored, 0.4, 0)
    
    return overlay

# -------------------- REPORT & EMAIL LOGIC ------------------------
def generate_pdf_report(image_name: str, result: str, confidence: float, notes: str, heatmap_b64: str | None) -> bytes:
    """Create a PDF report in memory using FPDF."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 12, "Pneumonia AI Diagnostic Report", ln=True, align="C")

    pdf.set_font("Arial", "", 13)
    pdf.ln(6)
    pdf.cell(0, 8, f"Image: {image_name}", ln=True)
    pdf.cell(0, 8, f"Prediction: {result}", ln=True)
    pdf.cell(0, 8, f"Confidence: {confidence:.2f}%", ln=True)
    pdf.cell(0, 8, f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(4)
    
    pdf.set_font("Arial", "I", 11)
    pdf.set_text_color(100, 100, 100)
    disclaimer = "Disclaimer: This AI tool is for research/educational use only and does not constitute a medical diagnosis. Consult a qualified clinician for interpretation."
    pdf.multi_cell(0, 7, disclaimer)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 6, "Doctor Notes:", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 6, f"{notes if notes else 'N/A'}")
    pdf.ln(3)

    if heatmap_b64:
        temp_file_name = f"temp_gradcam_{os.getpid()}.png"
        try:
            img_bytes = base64.b64decode(heatmap_b64)
            with open(temp_file_name, "wb") as f:
                f.write(img_bytes)
                
            pdf.ln(3)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "Grad-CAM Heatmap:", ln=True)
            pdf.image(temp_file_name, x=(pdf.w - 120) / 2, w=120) 
            
        except Exception:
            pass
        finally:
            if os.path.exists(temp_file_name):
                os.remove(temp_file_name)

    out = pdf.output(dest="S").encode("latin1")
    return out

def send_email_with_attachment(sender_email: str, app_password: str, recipient: str, subject: str, body: str, attachment_bytes: bytes, filename: str) -> bool:
    """Send email via Gmail SMTP with an attachment."""
    try:
        if not (sender_email and app_password and recipient):
            st.warning("Please fill Sender Gmail, App Password, and Recipient in the sidebar settings.", icon="⚠️")
            return False

        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = recipient
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment_bytes)
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename="{filename}"')
        msg.attach(part)

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context, timeout=10) as server:
            server.login(sender_email, app_password)
            server.send_message(msg)
        return True
        
    except smtplib.SMTPAuthenticationError:
        st.error("Email sending failed: Authentication Error. Check your Gmail App Password.", icon="❌")
        return False
    except Exception:
        st.toast("Email sending failed. Check configurations and internet connection.", icon="❌")
        return False


# -------------------- MAIN APP STRUCTURE -------------------------

# --- Header ---
st.markdown("<h1 class='main-header'>🩺 Pneumonia Detection AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>EfficientNet-based X-ray Analysis for Assistive Diagnosis</p>", unsafe_allow_html=True)

# --- Load Model (Cached) ---
model = load_model_cached()

# --- Tabs for Clean Structure ---
tab_single, tab_batch, tab_history, tab_analysis = st.tabs(["📊 Single Prediction", "📦 Batch Processing", "📜 History Log", "📈 Analysis & Trends"])

# -------------------- TAB 1: SINGLE PREDICTION --------------------
with tab_single:
    st.markdown("### Upload and Analyze a Chest X-ray")
    
    col_upload, col_result = st.columns([1.5, 2], gap="large")

    with col_upload:
        # --- Upload Section ---
        with st.container(border=True):
            uploaded = st.file_uploader("📤 Upload Chest X-ray (JPG/PNG)", type=["jpg", "jpeg", "png"])
            
            pil_img = None
            if uploaded:
                try:
                    pil_img = Image.open(uploaded)
                    st.session_state.last_original_b64 = base64.b64encode(uploaded.getvalue()).decode()
                    
                    st.markdown(f"**Original Image: {uploaded.name}**")
                    st.image(pil_img, caption="Uploaded X-ray", use_column_width="always")
                    
                except Exception:
                    st.error("Could not read this image file. Ensure it's a valid JPG/PNG.", icon="❌")
                    pil_img = None
        
        # --- Doctor Notes ---
        with st.expander("📝 Add Doctor Notes", expanded=True):
            st.session_state.doctor_notes = st.text_area(
                "Notes (optional)", 
                value=st.session_state.doctor_notes, 
                height=150,
                help="These notes will be embedded in the downloadable PDF report."
            )

    with col_result:
        # --- Prediction Action ---
        predict_btn = st.button("🚀 Run AI Prediction", use_container_width=True, type="primary")

        if predict_btn:
            # Reset batch results when running single prediction
            st.session_state.latest_batch_results = None 
            
            if pil_img is None:
                st.warning("Please upload an image first to run the prediction.", icon="⚠️")
                st.session_state.last_overlay_b64 = None
                st.session_state.last_pdf_bytes = None
            else:
                st.session_state.last_overlay_b64 = None
                st.session_state.last_pdf_bytes = None
                
                with st.spinner("Analyzing X-ray with EfficientNet…"):
                    try:
                        # 1. Prediction
                        batch = prepare_image(pil_img)
                        probs = model.predict(batch, verbose=0)[0]  
                        
                        pneu_conf = probs[PNEUMONIA_IDX] * 100.0
                        
                        # Determine final label based on the dynamic threshold
                        if pneu_conf >= prediction_threshold:
                            label = CLASS_NAMES[PNEUMONIA_IDX]
                            conf = pneu_conf
                        else:
                            label = CLASS_NAMES[NORMAL_IDX]
                            conf = probs[NORMAL_IDX] * 100.0

                        # 2. Show detailed probabilities and final result
                        col_prob_n, col_prob_p = st.columns(2)
                        with col_prob_n:
                              st.markdown(f"<div class='prob-box'>**Normal Probability:** {probs[NORMAL_IDX]*100:.2f}%</div>", unsafe_allow_html=True)
                        with col_prob_p:
                              st.markdown(f"<div class='prob-box'>**Pneumonia Probability:** {probs[PNEUMONIA_IDX]*100:.2f}%</div>", unsafe_allow_html=True)
                        
                        st.info(f"Classification threshold applied: **{prediction_threshold}%**")

                        style = "result-pneumonia" if label == "Pneumonia" else "result-normal"
                        st.markdown(f"<div class='result-banner {style}'>"
                                        f"FINAL DIAGNOSIS: {'**PNEUMONIA DETECTED**' if label=='Pneumonia' else '**NORMAL**'}<br>"
                                        f"Confidence: {conf:.2f}%</div>", unsafe_allow_html=True)
                        
                        render_gauge(conf, f"{label} Confidence")

                        # 3. Grad-CAM (if enabled)
                        if enable_gradcam:
                            st.subheader("🔥 Grad-CAM Explainability")
                            try:
                                # Use the index of the predicted class for Grad-CAM target
                                target_idx_for_cam = PNEUMONIA_IDX if label == "Pneumonia" else NORMAL_IDX
                                overlay_bgr = gradcam_overlay(pil_img, model, batch, target_idx_for_cam)
                                st.session_state.last_overlay_b64 = npimg_to_b64(overlay_bgr)

                                col_heat_img, col_heat_dl = st.columns([1, 1], gap="small")
                                with col_heat_img:
                                    st.markdown("##### Activation Map")
                                    st.image(base64.b64decode(st.session_state.last_overlay_b64), caption=f"Highlighted regions for '{label}' prediction", use_column_width="always")
                                with col_heat_dl:
                                    st.markdown("##### Download Artifacts")
                                    st.download_button(
                                        "⬇ Heatmap (PNG)",
                                        data=base64.b64decode(st.session_state.last_overlay_b64),
                                        file_name=f"heatmap_{os.path.splitext(uploaded.name)[0]}.png",
                                        mime="image/png",
                                        use_container_width=True
                                    )
                                    
                            except Exception:
                                st.error("Grad-CAM visualization failed. Check TensorFlow and dependencies.", icon="❌")
                                st.session_state.last_overlay_b64 = None
                                
                        # 4. Save to history
                        st.session_state.history.append({
                            "Image": uploaded.name,
                            "Result": label,
                            "Confidence": f"{conf:.2f}%",
                            "Time": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "Confidence_Value": conf # Store numerical value for analysis
                        })
                        
                        # 5. PDF Report Generation
                        pdf_bytes = generate_pdf_report(
                            uploaded.name, label, conf, st.session_state.doctor_notes, st.session_state.last_overlay_b64
                        )
                        st.session_state.last_pdf_bytes = pdf_bytes

                        st.download_button(
                            "📄 Download Full PDF Report",
                            data=pdf_bytes,
                            file_name=f"Report_{os.path.splitext(uploaded.name)[0]}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            type="secondary"
                        )
                        st.success("Prediction complete! See report for details.", icon="✅")

                    except Exception:
                        st.error("An unexpected error occurred during the prediction process.", icon="❌")
                        with st.expander("Show error details"):
                            st.code(traceback.format_exc())

# -------------------- TAB 2: BATCH PROCESSING --------------------
with tab_batch:
    st.markdown("### Process Multiple X-rays")
    
    with st.container(border=True):
        # NOTE: Using a key here is crucial for Streamlit to handle the file list correctly across reruns
        batch_files = st.file_uploader("Upload Multiple X-rays (JPG/PNG)", type=["jpg","jpeg","png"], accept_multiple_files=True, key="batch_uploader")
        
        if batch_files:
            st.info(f"Ready to process {len(batch_files)} images.")
            if st.button("▶ Run Batch Prediction", use_container_width=True, type="primary"):
                
                rows_for_display = []
                history_entries = []
                
                with st.spinner(f"Running batch predictions on {len(batch_files)} files..."):
                    for f in batch_files:
                        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                        
                        try:
                            im = Image.open(f)
                            p = model.predict(prepare_image(im), verbose=0)[0]
                            pneu_conf = p[PNEUMONIA_IDX] * 100.0
                            
                            # Apply the same classification threshold to batch results
                            if pneu_conf >= prediction_threshold:
                                label = CLASS_NAMES[PNEUMONIA_IDX]
                                conf = pneu_conf
                            else:
                                label = CLASS_NAMES[NORMAL_IDX]
                                conf = p[NORMAL_IDX] * 100.0

                            # Data for history (required for Analysis tab)
                            history_entries.append({
                                "Image": f.name,
                                "Result": label,
                                "Confidence": f"{conf:.2f}%",
                                "Time": current_time,
                                "Confidence_Value": conf
                            })
                            
                            # Data for immediate display in Batch tab
                            rows_for_display.append([f.name, label, f"{conf:.2f}%"])
                            
                        except Exception:
                            # Log error case to history as well for completeness
                            history_entries.append({
                                "Image": f.name,
                                "Result": "Error",
                                "Confidence": "—",
                                "Time": current_time,
                                "Confidence_Value": 0.0 # Use 0.0 for error to prevent breaking stats
                            })
                            rows_for_display.append([f.name, "Error", "—"])
                    
                    # 1. Append ALL results to the master history list
                    st.session_state.history.extend(history_entries)
                    
                    # 2. Store results for immediate display in the batch tab
                    df_latest_batch = pd.DataFrame(rows_for_display, columns=["Image", "Result", "Confidence"])
                    st.session_state.latest_batch_results = df_latest_batch
                    
                    st.success(f"Batch prediction complete. Added {len(history_entries)} results to history. Used threshold: {prediction_threshold}%", icon="✅")
                    # Rerun to show new results in Analysis
                    st.rerun() 


    if st.session_state.latest_batch_results is not None:
        st.markdown("### Latest Batch Results Table")
        
        # Display the results stored in session state
        st.dataframe(st.session_state.latest_batch_results, use_container_width=True)
        
        # Add download button for the latest batch
        st.download_button(
            "⬇ Download Latest Batch Results (CSV)",
            data=st.session_state.latest_batch_results.to_csv(index=False).encode("utf-8"),
            file_name="latest_batch_results.csv",
            mime="text/csv",
            use_container_width=True
        )
    elif st.session_state.history:
        st.info("No batch prediction has been run yet in this tab, but history is available. Check the 'History Log' and 'Analysis & Trends' tabs.")
        

# -------------------- TAB 3: HISTORY LOG --------------------
with tab_history:
    st.markdown("### 📜 Session Prediction Log")
    
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        
        # Format the display table nicely (dropping the internal numerical column)
        df_display = df.drop(columns=['Confidence_Value'], errors='ignore')
        df_display.insert(0, 'ID', range(1, 1 + len(df_display)))
        
        st.dataframe(df_display, use_container_width=True, height=350)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download History Log (CSV)", data=csv_bytes, file_name="prediction_history.csv", mime="text/csv", use_container_width=True, type="secondary")
    else:
        st.info("No predictions recorded in this session yet. Run a single prediction or a batch to start logging.")

    st.markdown("---")

    # --- Email Section (Uses last generated PDF) ---
    st.markdown("### 📧 Send Last Report")
    
    colE1, colE2, colE3 = st.columns([2, 2, 1])
    with colE1:
        recipient_email = st.text_input("Recipient email", value="", key="recipient_email_input", placeholder="doctor@hospital.com")
    with colE2:
        subj = st.text_input("Subject", value="Pneumonia AI Report", placeholder="AI Diagnostic Report")

    with colE3:
        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True) # Spacer
        send_btn = st.button("Send Last Report", use_container_width=True, key="send_email_button", type="secondary")
    
    if send_btn:
        recipient = (recipient_email or "").strip()
        
        if not recipient:
            st.warning("Please provide a recipient email address.", icon="⚠️")
        elif st.session_state.last_pdf_bytes is None:
            st.warning("Please run a successful prediction (in the 'Single Prediction' tab) to generate the report first.", icon="⚠️")
        else:
            success = send_email_with_attachment(
                sender_email=sender_email.strip(),
                app_password=sender_app_password.strip(),
                recipient=recipient,
                subject=subj.strip() or "Pneumonia AI Report",
                body="Attached is the AI-generated Pneumonia report.",
                attachment_bytes=st.session_state.last_pdf_bytes,
                filename="report.pdf"
            )
            if success:
                st.success(f"Report successfully emailed to {recipient}! ✅")

# -------------------- TAB 4: ANALYSIS & TRENDS --------------------
with tab_analysis:
    st.markdown("### 📈 Prediction Analysis and Visualization")
    
    if not st.session_state.history:
        st.info("Run predictions in the 'Single Prediction' or 'Batch Processing' tabs to generate data for analysis.")
    else:
        df = pd.DataFrame(st.session_state.history)
        
        # Ensure 'Confidence_Value' is numeric (it is now guaranteed for non-Error entries)
        df['Confidence_Value'] = pd.to_numeric(df['Confidence_Value'], errors='coerce') 
        
        # Filter out "Error" entries for numerical calculations, but keep for total count/display if needed
        df_valid = df[df['Result'] != 'Error'].copy()
        
        if df_valid.empty:
            st.warning("No successful predictions were recorded to perform numerical analysis.")
        else:
            # Re-index only the valid predictions for the time-series chart
            df_valid['Index'] = range(len(df_valid)) 

            ## Statistical Summary
            st.subheader("Statistical Summary")
            stats = {
                'Metric': ['Total Valid Predictions', 'Mean Confidence', 'Std Dev Confidence', 'Max Confidence'],
                'Value': [
                    len(df_valid),
                    f"{df_valid['Confidence_Value'].mean():.2f}%",
                    f"{df_valid['Confidence_Value'].std(skipna=True):.2f}%",
                    f"{df_valid['Confidence_Value'].max():.2f}%"
                ]
            }
            stats_df = pd.DataFrame(stats)
            col_stats, col_gap = st.columns([1, 2])
            with col_stats:
                st.table(stats_df.set_index('Metric'))
            
            st.markdown("---")
            
            ## Confidence Density Plot
            st.subheader("Confidence Score Distribution")
            st.markdown("Visualizes how concentrated the model's confidence scores are.")

            counts_by_result = df_valid['Result'].value_counts()
            has_density_data = all(count >= 2 for count in counts_by_result)

            if has_density_data:
                chart_density = alt.Chart(df_valid).transform_density(
                    'Confidence_Value',
                    as_=['Confidence', 'Density'],
                    groupby=['Result']
                ).mark_area(opacity=0.6, line=True).encode(
                    x=alt.X('Confidence:Q', title='Confidence (%)'),
                    y=alt.Y('Density:Q', title='Density'),
                    color=alt.Color('Result:N', scale=alt.Scale(domain=['Normal', 'Pneumonia'], range=[SUCCESS_COLOR, FAILURE_COLOR])),
                    tooltip=['Result', 'Confidence', 'Density']
                ).properties(
                    title='Confidence Distribution (KDE)'
                ).configure_view(
                    strokeOpacity=0,
                    fill=CARD_BG_COLOR
                ).configure_title(
                    color=TEXT_COLOR
                ).configure_axis(
                    labelColor=TEXT_COLOR,
                    titleColor=TEXT_COLOR
                ).configure_legend(
                    titleColor=TEXT_COLOR,
                    labelColor=TEXT_COLOR
                ).interactive()
                st.altair_chart(chart_density, use_container_width=True)
            else:
                st.warning("Insufficient data points for the Density plot (need at least 2 valid predictions per result type). Showing Histogram fallback.")
                # Fallback to a simple histogram
                chart_hist = alt.Chart(df_valid).mark_bar().encode(
                    x=alt.X('Confidence_Value', title='Confidence (%)', bin=True),
                    y=alt.Y('count()', title='Count'),
                    color=alt.Color('Result:N', scale=alt.Scale(domain=['Normal', 'Pneumonia'], range=[SUCCESS_COLOR, FAILURE_COLOR])),
                    tooltip=['Result', 'count()']
                ).properties(
                    title='Confidence Histogram (Fallback)'
                ).configure_view(
                    strokeOpacity=0,
                    fill=CARD_BG_COLOR
                ).configure_title(
                    color=TEXT_COLOR
                ).configure_axis(
                    labelColor=TEXT_COLOR,
                    titleColor=TEXT_COLOR
                ).configure_legend(
                    titleColor=TEXT_COLOR,
                    labelColor=TEXT_COLOR
                ).interactive()
                st.altair_chart(chart_hist, use_container_width=True)

            st.markdown("---")

            ## Time Series Scatter Plot (Trend Analysis)
            st.subheader("Confidence Trend by Prediction Index")
            st.markdown("Tracks confidence for each diagnosis over the course of the session.")
            
            # Add tooltips for detail on hover
            tooltip_fields = ['Index', 'Image', 'Result', alt.Tooltip('Confidence_Value', format='.2f')]

            chart_scatter = alt.Chart(df_valid).mark_circle(size=120).encode(
                x=alt.X('Index', title='Prediction Number', axis=alt.Axis(format='d')),
                y=alt.Y('Confidence_Value', title='Confidence (%)'),
                color=alt.Color('Result', scale=alt.Scale(domain=['Normal', 'Pneumonia'], range=[SUCCESS_COLOR, FAILURE_COLOR])),
                tooltip=tooltip_fields
            )

            # Add a simple linear trend line for overall confidence 
            chart_trend = chart_scatter.transform_regression('Index', 'Confidence_Value').mark_line(strokeDash=[5,5], color=ACCENT_COLOR).interactive()

            st.altair_chart((chart_scatter + chart_trend).properties(
                title='Prediction Confidence Over Time'
            ).configure_view(
                strokeOpacity=0,
                fill=CARD_BG_COLOR
            ).configure_title(
                color=TEXT_COLOR
            ).configure_axis(
                labelColor=TEXT_COLOR,
                titleColor=TEXT_COLOR
            ).configure_legend(
                titleColor=TEXT_COLOR,
                labelColor=TEXT_COLOR
            ), use_container_width=True)

            st.markdown("---")

            ## Diagnosis Breakdown (Donut Chart)
            st.subheader("Diagnosis Frequency Breakdown (Valid Predictions)")
            counts_df = df_valid["Result"].value_counts().reset_index()
            counts_df.columns = ['Result', 'Count']
            counts_df['Percentage'] = (counts_df['Count'] / counts_df['Count'].sum()) * 100
            
            chart_pie = alt.Chart(counts_df).encode(
                theta=alt.Theta("Count", stack=True),
                color=alt.Color('Result', scale=alt.Scale(domain=['Normal', 'Pneumonia'], range=[SUCCESS_COLOR, FAILURE_COLOR])),
                order=alt.Order('Percentage', sort='descending')
            )

            chart_arc = chart_pie.mark_arc(outerRadius=120, innerRadius=80, stroke=CARD_BG_COLOR, strokeWidth=2).encode(
                tooltip=['Result', 'Count', alt.Tooltip('Percentage', format='.1f')]
            )

            text = chart_pie.mark_text(radius=140).encode(
                text=alt.Text('Percentage', format='.1f'),
                color=alt.value(TEXT_COLOR) 
            )

            st.altair_chart((chart_arc + text).configure_view(
                strokeOpacity=0,
                fill=CARD_BG_COLOR
            ).configure_title(
                color=TEXT_COLOR
            ).configure_legend(
                titleColor=TEXT_COLOR,
                labelColor=TEXT_COLOR
            ), use_container_width=True)


# -------------------- FOOTER ----------------------
st.markdown("<div class='disclaimer'>⚠ This tool is for educational/research use only and is not a certified medical device. Always consult a qualified clinician for diagnosis and treatment.</div>", unsafe_allow_html=True)
