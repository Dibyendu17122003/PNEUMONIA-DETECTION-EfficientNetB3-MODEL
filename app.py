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
# This ensures MODEL_PATH is always correct, regardless of the deployment environment's CWD.
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

# -------------------- SESSION STATE INIT -----------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
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
if "batch_results" not in st.session_state:
    st.session_state.batch_results = None
    
# -------------------- THEME / STYLE (Modern Blue/Grey) ------------------
def apply_theme():
    """Applies modern, responsive CSS styling for Blue/Grey theme."""
    DARK_BG = "#0D1117"  # GitHub dark mode background
    LIGHT_BG = "#161B22" # Card background (Subtle Grey-Blue)
    PRIMARY_BLUE = "#58A6FF" # GitHub primary blue for accent
    TEXT_COLOR = "#C9D1D9" # Light grey text

    st.markdown(f"""
    <style>
        /* General Background and Typography */
        .stApp {{
            background-color: {DARK_BG};
            color: {TEXT_COLOR};
        }}
        h1, h2, h3, h4, h5, h6, .stMarkdown, label {{
            color: {TEXT_COLOR};
        }}
        
        /* Main Header Neon/Glow Effect */
        h1.main-header {{
             text-shadow: 0 0 8px {PRIMARY_BLUE}, 0 0 12px rgba(88, 166, 255, 0.4);
             text-align: center;
             padding-bottom: 20px;
        }}
        
        /* Custom Card/Container Style */
        .stContainer, .stTabs {{
            background-color: {LIGHT_BG};
            border-radius: 12px;
            padding: 25px;
            border: 1px solid rgba(88, 166, 255, 0.2);
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
            margin-bottom: 20px;
        }}
        
        /* Result Banners - Use Streamlit's alert styling with custom border */
        .result-banner {{
            padding: 15px;
            border-radius: 8px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 15px;
        }}
        .result-normal {{
            background-color: rgba(0, 255, 127, 0.1); 
            border-left: 5px solid #00FF7F; /* Green accent */
            color: #00FF7F;
        }}
        .result-pneumonia {{
            background-color: rgba(255, 69, 0, 0.1); 
            border-left: 5px solid #FF4500; /* Red accent */
            color: #FF4500;
        }}

        /* Detailed Probability Box */
        .prob-box {{
            padding: 10px;
            border-radius: 8px;
            border: 1px solid rgba(88, 166, 255, 0.4);
            margin-bottom: 15px;
            font-size: 14px;
        }}
        
        /* Images - Fully Responsive */
        .responsive-img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px; 
            margin-top: 10px;
            border: 2px solid {PRIMARY_BLUE};
            box-shadow: 0 0 5px rgba(88, 166, 255, 0.5);
        }}
        
        /* Gauge Wrapper */
        .gauge-wrapper {{
            display: flex;
            justify-content: center;
            margin-top: 10px;
        }}
        
        /* Button Styling (ensures consistency) */
        .stButton>button {{
            background-color: {PRIMARY_BLUE};
            border-color: {PRIMARY_BLUE};
            color: {DARK_BG}; /* Dark text on bright button */
        }}
        .stButton>button:hover {{
            background-color: #008CBA; 
            border-color: #008CBA;
        }}
        
        /* Footer/Disclaimer */
        .disclaimer {{
            font-size: 11px;
            opacity: 0.6;
            text-align: center;
            margin-top: 30px;
        }}
    </style>
    """, unsafe_allow_html=True)

apply_theme()

# -------------------- SIDEBAR CONTROLS (Define variables in global scope) ------------------------
with st.sidebar:
    st.markdown("## ⚙ Settings & Configuration")
    
    if st.button("🎨 Toggle Theme (Light/Dark)"):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.experimental_rerun()
        
    st.markdown("---")
    st.markdown("### Model Behavior")
    
    # NEW FEATURE: Adjustable threshold
    prediction_threshold = st.slider(
        "Pneumonia Classification Threshold (%)",
        min_value=50, max_value=99, value=80, step=1,
        help="Confidence level required for the AI to classify the result as PNEUMONIA."
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
    gauge_color = "#FF4500" if p > 50 else "#00FF7F"
    
    st.markdown(f"""
    <div class='gauge-wrapper'>
      <svg viewBox="0 0 100 60" style="width: 150px; max-width: 90%;">
        <!-- Track -->
        <path d="M10,50 A40,40 0 0,1 90,50" fill="none" stroke="rgba(255,255,255,0.25)" stroke-width="6"/>
        <!-- Fill -->
        <path d="M10,50 A40,40 0 0,1 90,50" fill="none" stroke="{gauge_color}" stroke-width="6"
              stroke-dasharray="{p*1.256},{125.6}"/>
        <!-- Needle -->
        <line x1="50" y1="50" x2="{50 + 38 * np.cos(np.radians(angle))}" y2="{50 + 38 * np.sin(np.radians(angle))}"
              stroke="{gauge_color}" stroke-width="2"/>
        <!-- Text Label -->
        <text x="50" y="58" text-anchor="middle" font-size="8" fill="currentColor">{label}: {p:.1f}%</text>
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
tab_single, tab_batch, tab_history = st.tabs(["📊 Single Prediction", "📦 Batch Processing", "📜 History & Analytics"])

# -------------------- TAB 1: SINGLE PREDICTION --------------------
with tab_single:
    st.markdown("### Upload and Analyze a Chest X-ray")
    
    # Use fluid columns for responsiveness (adjusts automatically on mobile)
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
                        
                        st.info(f"Using threshold: **{prediction_threshold}%**")

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
                            "Time": time.strftime("%Y-%m-%d %H:%M:%S")
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
                            # This traceback is for the original prediction error, but the NameError is fixed.
                            st.code(traceback.format_exc())

# -------------------- TAB 2: BATCH PROCESSING --------------------
with tab_batch:
    st.markdown("### Process Multiple X-rays")
    
    with st.container(border=True):
        batch_files = st.file_uploader("Upload Multiple X-rays (JPG/PNG)", type=["jpg","jpeg","png"], accept_multiple_files=True, key="batch_uploader")
        
        if batch_files:
            st.info(f"Ready to process {len(batch_files)} images.")
            if st.button("▶ Run Batch Prediction", use_container_width=True, type="primary"):
                with st.spinner(f"Running batch predictions on {len(batch_files)} files..."):
                    rows = []
                    for f in batch_files:
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

                            rows.append([f.name, label, f"{conf:.2f}%"])
                        except Exception:
                            rows.append([f.name, "Error", "—"])
                    
                    dfb = pd.DataFrame(rows, columns=["Image", "Result", "Confidence"])
                    st.session_state.batch_results = dfb
                    st.success(f"Batch prediction complete. Used threshold: {prediction_threshold}%", icon="✅")

    if st.session_state.batch_results is not None:
        st.markdown("### Batch Results Table")
        st.dataframe(st.session_state.batch_results, use_container_width=True)
        st.download_button(
            "⬇ Download Batch Results (CSV)",
            data=st.session_state.batch_results.to_csv(index=False).encode("utf-8"),
            file_name="batch_results.csv",
            mime="text/csv",
            use_container_width=True
        )

# -------------------- TAB 3: HISTORY & ANALYTICS --------------------
with tab_history:
    st.markdown("### 📜 Prediction History")
    
    col_hist, col_dash = st.columns([2, 1], gap="large")

    with col_hist:
        if st.session_state.history:
            df = pd.DataFrame(st.session_state.history)
            st.dataframe(df, use_container_width=True, height=300)

            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Download History Log (CSV)", data=csv_bytes, file_name="prediction_history.csv", mime="text/csv", use_container_width=True, type="secondary")
        else:
            st.info("No predictions recorded in this session yet.")
    
    with col_dash:
        st.markdown("### 📊 Session Analytics")
        if st.session_state.history:
            df = pd.DataFrame(st.session_state.history)
            counts = df["Result"].value_counts().reindex(CLASS_NAMES, fill_value=0)
            
            st.bar_chart(counts, use_container_width=True)
            
            df['Confidence_Val'] = df['Confidence'].str.replace('%', '').astype(float)
            avg_conf = df['Confidence_Val'].mean()
            
            st.metric(label="Average Confidence", value=f"{avg_conf:.2f}%")
        else:
            st.write("Run predictions to populate the dashboard.")
            
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

# -------------------- FOOTER ----------------------
st.markdown("<div class='disclaimer'>⚠ This tool is for educational/research use only and is not a certified medical device. Always consult a qualified clinician for diagnosis and treatment.</div>", unsafe_allow_html=True)
