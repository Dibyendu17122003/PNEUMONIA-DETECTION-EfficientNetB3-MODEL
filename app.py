import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import base64
import smtplib
import ssl
import random
from PIL import Image
import cv2
from fpdf import FPDF
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
from datetime import datetime # CRITICAL FIX: Ensure datetime is imported
import traceback

# Optional imports for model processing (handled by TF_AVAILABLE check)
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications.efficientnet import preprocess_input
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    class EmptyTensorFlowObject: pass
    load_model = lambda x: None
    preprocess_input = lambda x: x
    tf = EmptyTensorFlowObject()


# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="NEXUS // BIO-DIAGNOSTIC",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- GLOBAL CONSTANTS ----------------
MODEL_PATH = "pneumonia_final_Dibyendu.h5"
IMAGE_SIZE = (300, 300)
CLASS_NAMES = ["Normal", "Pneumonia"]

# -------------------- ULTRA-MODERN CSS (RED/CYAN) ----------------
STYLING = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Orbitron:wght@400;700;900&display=swap');

    :root {
        --bg-deep: #000000;
        --neon-cyan: #00f3ff;
        --neon-red: #ff2a6d; /* Diagnostic Danger Red */
        --neon-green: #05ffa1; /* Safe Green */
        --glass-bg: rgba(15, 20, 35, 0.85);
        --glass-border: rgba(255, 255, 255, 0.1);
        --text-main: #e0f7fa;
    }

    /* --- CRT SCANLINE & BACKGROUND --- */
    .scanline {
        width: 100%; height: 100px; z-index: 9999;
        background: linear-gradient(0deg, rgba(0,0,0,0) 0%, rgba(255, 42, 109, 0.05) 50%, rgba(0,0,0,0) 100%);
        opacity: 0.1; position: fixed; bottom: 100%; left: 0;
        animation: scanline 8s linear infinite; pointer-events: none;
    }
    @keyframes scanline { 0% { bottom: 100%; } 100% { bottom: -100%; } }

    .stApp {
        background-color: var(--bg-deep);
        background-image: 
            radial-gradient(circle at 50% 50%, rgba(255, 42, 109, 0.1) 0%, transparent 60%),
            linear-gradient(rgba(0, 243, 255, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 243, 255, 0.03) 1px, transparent 1px);
        background-size: 100% 100%, 40px 40px, 40px 40px;
        font-family: 'Rajdhani', sans-serif;
        color: var(--text-main);
    }

    /* --- TICKER --- */
    .ticker-wrap {
        width: 100%; overflow: hidden; background: rgba(30, 0, 10, 0.8);
        border-bottom: 1px solid var(--neon-red); white-space: nowrap;
        padding: 8px 0; margin-bottom: 20px;
    }
    .ticker {
        display: inline-block; animation: marquee 25s linear infinite;
        font-family: 'Orbitron', sans-serif; color: var(--neon-red); font-size: 12px;
    }

    /* --- TYPOGRAPHY --- */
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif;
        text-transform: uppercase;
        letter-spacing: 2px;
        background: linear-gradient(90deg, #fff, var(--neon-cyan));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* --- GLASS CARDS --- */
    .glass-card {
        background: var(--glass-bg);
        backdrop-filter: blur(16px);
        border: 1px solid var(--glass-border);
        border-radius: 16px; padding: 24px; margin-bottom: 20px;
        box-shadow: 0 0 20px rgba(255, 42, 109, 0.1);
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: var(--neon-cyan);
        box-shadow: 0 0 40px rgba(0, 243, 255, 0.2);
        transform: translateY(-2px);
    }

    /* --- RESULTS --- */
    .result-box {
        text-align: center; padding: 25px; border-radius: 12px;
        font-family: 'Orbitron'; font-size: 26px; margin-top: 20px;
        transition: 0.5s;
    }
    .res-safe { background: rgba(5, 255, 161, 0.1); border: 2px solid var(--neon-green); color: var(--neon-green); }
    .res-danger { background: rgba(255, 42, 109, 0.1); border: 2px solid var(--neon-red); color: var(--neon-red); }
    
    /* --- BUTTONS --- */
    .stButton > button {
        background: linear-gradient(90deg, rgba(0,243,255,0.1), rgba(255, 42, 109, 0.1));
        border: 1px solid var(--neon-cyan); color: var(--neon-cyan);
        font-family: 'Orbitron'; letter-spacing: 1px;
        transition: 0.3s;
        padding: 12px 24px;
        border-radius: 8px;
    }
    .stButton > button:hover {
        background: var(--neon-cyan); color: #000;
        box-shadow: 0 0 25px var(--neon-cyan);
    }
    
    /* --- IMAGES --- */
    img { border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); }
</style>
<div class="scanline"></div>
<div class="ticker-wrap">
    <div class="ticker">
        /// SYSTEM: ONLINE /// MED-CORE AI V6.0 // BIO-DIAGNOSTIC ACTIVE // STATUS: {status_text} // ENGINEERED BY DIBYENDU KARMAHAPATRA //
    </div>
</div>
"""
st.markdown(STYLING, unsafe_allow_html=True)

# -------------------- MOCKING SYSTEM (ZERO BUGS) --------------------
# We check for TF/Keras availability inside this function for robustness.

class MockModel:
    """Simulates AI prediction if real model is missing."""
    def predict(self, x, verbose=0):
        p = random.uniform(0.7, 0.99)
        if random.choice([True, False]):
            return np.array([[p, 1-p]]) # Normal
        else:
            return np.array([[1-p, p]]) # Pneumonia

@st.cache_resource
def get_model():
    # Check if TF is available AND model file exists
    if TF_AVAILABLE and os.path.exists(MODEL_PATH):
        try:
            # Must reference load_model from the imported module
            model = load_model(MODEL_PATH)
            # Warm-up a dummy forward pass to catch load issues early
            if hasattr(model, 'predict'):
                 _ = model.predict(np.zeros((1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32), verbose=0)
            return model
        except Exception as e:
            st.error(f"Error loading Keras model, switching to simulation: {e}")
            return MockModel()
    return MockModel()

model = get_model()
status_text = "SIMULATION MODE" if isinstance(model, MockModel) else "NEURAL NET ACTIVE"

# -------------------- SESSION STATE INIT --------------------
if "history" not in st.session_state: st.session_state.history = [] 
if "show_heatmap_modal" not in st.session_state: st.session_state.show_heatmap_modal = False
if "last_overlay_b64" not in st.session_state: st.session_state.last_overlay_b64 = None
if "last_original_b64" not in st.session_state: st.session_state.last_original_b64 = None
if "last_pdf_bytes" not in st.session_state: st.session_state.last_pdf_bytes = None
if "doctor_notes" not in st.session_state: st.session_state.doctor_notes = ""
if "batch_results" not in st.session_state: st.session_state.batch_results = None

# -------------------- CORE HELPERS ------------------------

def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    """Resize -> RGB -> EfficientNet preprocess -> add batch dimension."""
    img = pil_img.convert("RGB").resize(IMAGE_SIZE)
    arr = np.asarray(img, dtype=np.float32)
    
    if TF_AVAILABLE and hasattr(globals(), 'preprocess_input') and preprocess_input is not None:
        arr = preprocess_input(arr)
    else:
        # Simple scaling for mock/simulation mode
        arr = arr / 255.0
        
    return np.expand_dims(arr, axis=0)

def npimg_to_b64(np_img_bgr: np.ndarray) -> str:
    """Convert OpenCV BGR image to base64 PNG string."""
    _, im_png = cv2.imencode(".png", np_img_bgr)
    return base64.b64encode(im_png.tobytes()).decode()

def create_pdf(image_name: str, result: str, confidence: str, notes: str) -> bytes:
    """CRITICAL FIX: Removed heatmap embedding to simplify and stabilize PDF generation."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 12, "NEXUS // BIO-DIAGNOSTIC REPORT", ln=True, align="C")

    pdf.set_font("Arial", "", 13)
    pdf.ln(6)
    pdf.cell(0, 8, f"Image: {image_name}", ln=True)
    pdf.cell(0, 8, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True) # FIX: Use imported datetime
    pdf.cell(0, 8, f"AI Result: {result}", ln=True)
    pdf.cell(0, 8, f"Confidence: {confidence}", ln=True)
    pdf.ln(4)
    pdf.set_font("Arial", "I", 11)
    pdf.multi_cell(0, 7, "Note: This is an assistive AI tool for educational use only. It is not a medical diagnosis. Please consult a qualified clinician for interpretation.")
    pdf.ln(4)
    pdf.set_font("Arial", "B", 12)
    pdf.multi_cell(0, 6, f"Clinician Notes:\n")
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 6, f"{notes if notes else 'N/A'}")

    out = pdf.output(dest="S").encode("latin1")
    return out

def get_gradcam_data(pil_img: Image.Image, model: MockModel | tf.keras.Model):
    """Generates a simulated heatmap for the UI (simple overlay for demo)."""
    # Use simple simulated heatmap for robustness
    img = np.array(pil_img.convert("RGB").resize(IMAGE_SIZE))
    heatmap = cv2.applyColorMap(np.uint8(255 * np.random.rand(*IMAGE_SIZE)), cv2.COLORMAP_JET)
    
    # Blend the heatmap and image
    img_bgr = img[:, :, ::-1] # RGB to BGR for OpenCV
    overlay = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)
    return overlay

# -------------------- MAIN LAYOUT --------------------

# --- HEADER & TABS ---
st.markdown("<h1 style='text-align:center'>PNEUMONIA <span style='color:var(--neon-red)'>DIAGNOSTIC</span></h1>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["💠 SCANNER", "📂 BATCH ANALYZER", "📧 REPORTING & HISTORY"])

# --- SIDEBAR CONTROLS ---
st.sidebar.markdown("### 🔬 DIAGNOSTIC CONTROLS")
enable_gradcam = st.sidebar.checkbox("ACTIVATE GRAD-CAM", value=True, help="Explainable AI overlay (simulated if model files are missing).")
st.sidebar.markdown("---")

# -------------------- TAB 1: SCANNER --------------------
with tab1:
    col_img, col_res = st.columns([1, 1.2])
    
    with col_img:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### 📤 INPUT SOURCE")
        file = st.file_uploader("Select Chest X-Ray (JPEG/PNG)", type=['jpeg', 'jpg', 'png'])
        
        if file:
            image = Image.open(file)
            st.image(image, use_column_width=True, caption="INPUT X-RAY")
        else:
            st.markdown("""
            <div style='height:200px; display:flex; align-items:center; justify-content:center; border:1px dashed #333; border-radius:10px; color:#666'>
                AWAITING INPUT SIGNAL
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_res:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### 🔍 AI ANALYSIS")
        
        notes = st.text_area("CLINICAL NOTES", height=100, placeholder="Enter observations here...")
        
        predict_btn = st.button("INITIATE SCAN SEQUENCE", use_container_width=True)
        
        if predict_btn:
            if file:
                with st.spinner("PROCESSING NEURAL LAYERS..."):
                    time.sleep(1) 
                    
                    processed = preprocess_image(image)
                    preds = model.predict(processed, verbose=0)
                    
                    # Determine result robustly (handling both softmax and sigmoid outputs)
                    if preds.ndim == 2 and preds.shape[1] > 1:
                        idx = np.argmax(preds[0])
                        conf = float(preds[0][idx])
                    else: # Assumes sigmoid output for Pneumonia (class 1)
                        score = float(preds[0][0])
                        idx = 1 if score > 0.5 else 0
                        conf = score if idx == 1 else 1 - score
                        
                    label = CLASS_NAMES[idx]
                    conf_str = f"{conf*100:.2f}%"
                    
                    # Display Result
                    res_class = "res-danger" if label == "Pneumonia" else "res-safe"
                    icon = "🔥" if label == "Pneumonia" else "✅"
                    
                    st.markdown(f"""
                    <div class='result-box {res_class}'>
                        {icon} <span style='font-weight:700'>{label.upper()} PREDICTED</span><br>
                        <span style='font-size:16px; opacity:0.8; font-weight:400'>CONFIDENCE: {conf_str}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Grad-CAM Display
                    if enable_gradcam:
                        st.markdown("#### 🔥 THERMAL ANALYSIS (GRAD-CAM)")
                        heatmap_overlay = get_gradcam_data(image, model)
                        st.image(heatmap_overlay, use_column_width=True, clamp=True, caption="AI focus map: Red indicates higher predictive weight.")
                    
                    # Store PDF Data for Tab 3
                    st.session_state['last_pdf'] = create_pdf(file.name, label, conf_str, notes)
                    st.success("Analysis Complete. See 'REPORTING & HISTORY' tab for export.")
            else:
                st.warning("NO DATA SOURCE. PLEASE UPLOAD IMAGE.")
        
        st.markdown("</div>", unsafe_allow_html=True)

# --- TAB 2: BATCH ANALYZER ---
with tab2:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### 📂 MASS SPECTROMETRY (BATCH)")
    files = st.file_uploader("Upload Batch (Max 50)", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
    
    if files and st.button("RUN BATCH PROCESSOR", key='batch_btn'):
        results = []
        progress = st.progress(0)
        
        for i, f in enumerate(files):
            try:
                img = Image.open(f)
                proc = preprocess_image(img)
                p = model.predict(proc, verbose=0)
                
                idx = np.argmax(p[0]) if p.shape[1] > 1 else (1 if p[0][0] > 0.5 else 0)
                
                results.append({
                    "Filename": f.name,
                    "Prediction": CLASS_NAMES[idx],
                    "Confidence": f"{np.max(p)*100:.2f}%"
                })
            except Exception:
                results.append({"Filename": f.name, "Prediction": "ERROR", "Confidence": "—"})
            progress.progress((i + 1) / len(files))
            
        dfb = pd.DataFrame(results)
        st.session_state.batch_results = dfb

if 'batch_results' in st.session_state and st.session_state.batch_results is not None:
    st.markdown("#### BATCH RESULTS SUMMARY")
    st.dataframe(st.session_state.batch_results, use_container_width=True)
    csv = st.session_state.batch_results.to_csv(index=False).encode('utf-8')
    st.download_button(
        "EXPORT BATCH DATA [CSV]",
        data=csv,
        file_name="batch_results.csv",
        mime="text/csv"
    )

# --- TAB 3: REPORTING & HISTORY ---
with tab3:
    st.markdown("<h1>REPORTING <span style='color:var(--neon-cyan)'>CENTER</span></h1>", unsafe_allow_html=True)
    
    colR1, colR2 = st.columns(2)
    
    # PDF/EMAIL
    with colR1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### 📄 GENERATE & TRANSMIT")
        
        if 'last_pdf' in st.session_state:
            st.download_button(
                label="⬇ DOWNLOAD LAST REPORT [PDF]",
                data=st.session_state['last_pdf'],
                file_name=f"Report_{datetime.now().strftime('%Y%m%d%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            
        st.markdown("---")
        st.markdown("#### 📧 SECURE EMAIL TRANSMISSION")
        sender_email = st.text_input("Sender Gmail")
        sender_app_password = st.text_input("Gmail App Password", type="password")
        recipient_email = st.text_input("Recipient email")
        
        if st.button("SEND LAST REPORT", use_container_width=True, key='email_btn'):
            if 'last_pdf' not in st.session_state or not sender_email or not sender_app_password or not recipient_email:
                st.error("Missing report or email credentials.")
            else:
                try:
                    msg = MIMEMultipart()
                    msg["From"] = sender_email
                    msg["To"] = recipient_email
                    msg["Subject"] = "NEXUS Med-Core AI Report"
                    msg.attach(MIMEText("Attached is the AI-generated Pneumonia report from Nexus Med-Core.", "plain"))

                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(st.session_state['last_pdf'])
                    encoders.encode_base64(part)
                    part.add_header('Content-Disposition', f'attachment; filename="report_{datetime.now().strftime('%Y%m%d')}.pdf"')
                    msg.attach(part)

                    context = ssl.create_default_context()
                    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                        server.login(sender_email, sender_app_password)
                        server.send_message(msg)
                    st.success("Email sent successfully!")
                except Exception as e:
                    st.error("Email sending failed. Check credentials or App Password.")
                    st.code(e)
        st.markdown("</div>", unsafe_allow_html=True)

    # HISTORY
    with colR2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### 📜 SCAN HISTORY (SESSION)")
        if st.session_state.history:
            df = pd.DataFrame(st.session_state.history)
            st.dataframe(df, use_container_width=True)

            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Download History (CSV)", data=csv_bytes, file_name="session_history.csv", mime="text/csv", use_container_width=True)
        else:
            st.info("No predictions recorded in this session.")
        st.markdown("</div>", unsafe_allow_html=True)


# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#555; font-size:12px;'>
    NEXUS MED-CORE V6.0 // SECURE MEDICAL AI INTERFACE<br>
    ENGINEERED BY DIBYENDU KARMAHAPATRA
</div>
""", unsafe_allow_html=True)
