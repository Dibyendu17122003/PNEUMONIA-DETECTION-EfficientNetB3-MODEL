# app.py
import os
import io
import ssl
import time
import base64
import smtplib
import traceback
import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st

# TensorFlow / Keras
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

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="🩺",
    layout="wide",
)

# -------------------- GLOBALS ------------------------
MODEL_PATH = "pneumonia_final_Dibyendu.h5"
IMAGE_SIZE = (300, 300)
CLASS_NAMES = ["Normal", "Pneumonia"]  # index 0 = Normal, 1 = Pneumonia

# -------------------- THEME / STYLE ------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "history" not in st.session_state:
    st.session_state.history = []  # {Image, Result, Confidence, Time}
if "show_heatmap_modal" not in st.session_state:
    st.session_state.show_heatmap_modal = False
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

def apply_theme():
    dark = st.session_state.theme == "dark"
    bg = "linear-gradient(135deg, #0f2027, #203a43, #2c5364)" if dark else "linear-gradient(135deg, #ffffff, #e6f7ff)"
    fg = "#ffffff" if dark else "#0f1a2b"
    accent = "#00eaff" if dark else "#0077b6"

    st.markdown(f"""
    <style>
        body {{
            background: {bg};
            background-size: 200% 200%;
            animation: gradientMove 10s ease infinite;
            color: {fg};
        }}
        @keyframes gradientMove {{
            0% {{background-position: 0% 50%;}}
            50% {{background-position: 100% 50%;}}
            100% {{background-position: 0% 50%;}}
        }}
        .glass {{
            background: rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 22px;
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255,255,255,0.15);
            box-shadow: 0 0 28px rgba(0, 234, 255, 0.25);
        }}
        h1,h2,h3,h4,p,label,span,div,li,td,th {{
            color: {fg};
        }}
        .result-good {{
            background: rgba(0, 255, 0, 0.16);
            border: 1px solid rgba(0,255,0,0.35);
            padding: 14px;
            border-radius: 12px;
            font-weight: 700;
            text-align: center;
        }}
        .result-bad {{
            background: rgba(255, 0, 0, 0.16);
            border: 1px solid rgba(255,0,0,0.35);
            padding: 14px;
            border-radius: 12px;
            font-weight: 700;
            text-align: center;
        }}
        .neon {{
            text-shadow: 0 0 8px {accent}, 0 0 16px {accent};
        }}
        .small-muted {{
            font-size: 12px;
            opacity: 0.75;
        }}
        /* Modal styling */
        .modal-backdrop {{
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.65);
            z-index: 1000;
        }}
        .modal-card {{
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%,-50%);
            background: rgba(20,20,30,0.95);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 16px;
            padding: 16px;
            z-index: 1001;
            max-width: 92vw;
            max-height: 88vh;
            overflow: auto;
            box-shadow: 0 0 24px rgba(0,0,0,0.6);
        }}
        .modal-img {{
            width: 100%;
            height: auto;
            display: block;
            border-radius: 12px;
        }}
        .img-card {{
            text-align: center;
        }}
        .img-thumb {{
            max-width: 260px; 
            width: 100%;
            height: auto;
            border-radius: 12px; 
            box-shadow: 0 0 12px rgba(0,234,255,0.5);
        }}
        .img-thumb-heat {{
            max-width: 260px; 
            width: 100%;
            height: auto;
            border-radius: 12px; 
            box-shadow: 0 0 12px rgba(255,95,95,0.5);
        }}
        .img-caption {{
            margin-top: 8px;
            font-size: 13px;
            opacity: 0.85;
        }}
        @media (max-width: 768px) {{
            .img-thumb, .img-thumb-heat {{ max-width: 220px; }}
        }}
        /* Gauge */
        .gauge {{
            width: 240px; max-width: 90%;
            margin: 6px auto 0 auto;
        }}
    </style>
    """, unsafe_allow_html=True)

apply_theme()

# -------------------- SIDEBAR ------------------------
st.sidebar.markdown("## ⚙️ Controls")
if st.sidebar.button("🌗 Toggle Light/Dark"):
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
    apply_theme()

enable_gradcam = st.sidebar.checkbox("Grad-CAM heatmap", value=True, help="Explainable AI overlay")
enable_voice = st.sidebar.checkbox("Voice output", value=False, help="Uses your OS TTS (might not work on some servers)")
voice_lang = st.sidebar.selectbox("Voice language", ["English", "Hindi", "Bengali"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Email Settings (Gmail + App Password)**")
with st.sidebar.expander("Configure Email"):
    sender_email = st.text_input("Sender Gmail", placeholder="yourname@gmail.com")
    sender_app_password = st.text_input("Gmail App Password", type="password", placeholder="App password")
    default_recipient = st.text_input("Default Recipient (optional)", placeholder="recipient@example.com")
st.sidebar.markdown("---")
st.sidebar.markdown("**Model file:** `pneumonia_final_Dibyendu.h5`")
st.sidebar.markdown("**Input:** 300×300 RGB (EfficientNetB3 preprocess)")

# -------------------- HELPERS ------------------------
@st.cache_resource(show_spinner=True)
def load_model_cached():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    # Warm-up a dummy forward pass to catch load issues early
    _ = model.predict(np.zeros((1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32), verbose=0)
    return model

def prepare_image(pil_img: Image.Image) -> np.ndarray:
    """Resize -> RGB -> EfficientNet preprocess -> add batch dimension."""
    img = pil_img.convert("RGB").resize(IMAGE_SIZE)
    arr = np.asarray(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def npimg_to_b64(np_img_bgr: np.ndarray) -> str:
    """Convert OpenCV BGR image to base64 PNG string."""
    _, im_png = cv2.imencode(".png", np_img_bgr)
    return base64.b64encode(im_png.tobytes()).decode()

def safe_speak(text: str):
    """Optional local TTS; ignored if unavailable."""
    if not enable_voice:
        return
    try:
        import pyttsx3
        engine = pyttsx3.init()
        # Try to pick a voice that contains the language keyword
        wanted = {"English": "en", "Hindi": "hi", "Bengali": "bn"}[voice_lang]
        try:
            voices = engine.getProperty("voices")
            chosen = None
            for v in voices:
                vid = f"{v.id}".lower()
                vname = f"{getattr(v,'name','')}".lower()
                if wanted in vid or wanted in vname:
                    chosen = v.id
                    break
            if chosen:
                engine.setProperty("voice", chosen)
        except Exception:
            pass
        engine.say(text)
        engine.runAndWait()
    except Exception:
        st.toast("Voice output unavailable on this system.", icon="⚠️")

def render_gauge(percent: float, label: str):
    """Draw a clean semicircle gauge using inline SVG."""
    p = max(0, min(100, percent))
    # map 0..100 to -90..90 degrees
    angle = -90 + (p * 180.0 / 100.0)
    st.markdown(f"""
    <div style="text-align:center;">
      <svg viewBox="0 0 100 60" class="gauge">
        <path d="M10,50 A40,40 0 0,1 90,50" fill="none" stroke="rgba(255,255,255,0.35)" stroke-width="6"/>
        <path d="M10,50 A40,40 0 0,1 90,50" fill="none" stroke="#00eaff" stroke-width="6"
              stroke-dasharray="{p*1.256},{125.6}"/>
        <line x1="50" y1="50" x2="{50 + 38 * np.cos(np.radians(angle))}" y2="{50 + 38 * np.sin(np.radians(angle))}"
              stroke="#00eaff" stroke-width="2"/>
        <text x="50" y="58" text-anchor="middle" font-size="8" fill="currentColor">{label}: {p:.1f}%</text>
      </svg>
    </div>
    """, unsafe_allow_html=True)

def generate_pdf_report(image_name: str, result: str, confidence: float, notes: str, heatmap_b64: str | None) -> bytes:
    """Create a PDF report in memory including notes and optional heatmap."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 12, "Pneumonia Detection Report", ln=True, align="C")

    pdf.set_font("Arial", "", 13)
    pdf.ln(6)
    pdf.cell(0, 8, f"Image: {image_name}", ln=True)
    pdf.cell(0, 8, f"Prediction: {result}", ln=True)
    pdf.cell(0, 8, f"Confidence: {confidence:.2f}%", ln=True)
    pdf.cell(0, 8, f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(4)
    pdf.set_font("Arial", "I", 11)
    pdf.multi_cell(0, 7, "Note: This is an assistive AI tool for educational use only. It is not a medical diagnosis. Please consult a qualified clinician for interpretation.")
    pdf.ln(4)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 6, f"Doctor Notes:\n{notes if notes else 'N/A'}")
    pdf.ln(3)

    # Embed heatmap if available
    if heatmap_b64:
        try:
            img_bytes = base64.b64decode(heatmap_b64)
            tmp = "tmp_gradcam_embed.png"
            with open(tmp, "wb") as f:
                f.write(img_bytes)
            pdf.ln(3)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "Grad-CAM Heatmap:", ln=True)
            pdf.image(tmp, w=120)
            try:
                os.remove(tmp)
            except Exception:
                pass
        except Exception:
            # Continue without heatmap
            pass

    out = pdf.output(dest="S").encode("latin1")
    return out

def find_last_conv_layer(keras_model: tf.keras.Model):
    """Robustly get the last 4D layer (conv/activation) for Grad-CAM."""
    for layer in reversed(keras_model.layers):
        try:
            out = layer.output
            if len(out.shape) == 4:
                return layer.name
        except Exception:
            continue
    return None

def gradcam_overlay(pil_img: Image.Image, model: tf.keras.Model, processed_batch: np.ndarray) -> np.ndarray:
    """Compute Grad-CAM heatmap and overlay on the original image."""
    layer_name = find_last_conv_layer(model)
    if layer_name is None:
        return np.asarray(pil_img.convert("RGB").resize(IMAGE_SIZE))[:, :, ::-1]  # return BGR

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(processed_batch)
        top_class = tf.argmax(preds[0])
        loss = preds[:, top_class]

    grads = tape.gradient(loss, conv_out)
    if grads is None:
        return np.asarray(pil_img.convert("RGB").resize(IMAGE_SIZE))[:, :, ::-1]

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)

    conv_out = conv_out[0]  # H x W x C
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_out), axis=-1)

    heatmap = tf.nn.relu(heatmap)
    denom = tf.reduce_max(heatmap)
    heatmap = heatmap / (denom + 1e-10)
    heatmap = heatmap.numpy()

    # Resize & apply colormap
    heatmap = cv2.resize(heatmap, IMAGE_SIZE[::-1])
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Original RGB -> BGR for OpenCV blend
    original_rgb = np.asarray(pil_img.convert("RGB").resize(IMAGE_SIZE))
    original_bgr = original_rgb[:, :, ::-1]
    overlay = cv2.addWeighted(original_bgr, 0.6, heatmap, 0.4, 0)
    return overlay

def send_email_with_attachment(sender_email: str, app_password: str, recipient: str, subject: str, body: str, attachment_bytes: bytes, filename: str) -> bool:
    """Send email via Gmail SMTP with an attachment; safe fallbacks."""
    try:
        if not (sender_email and app_password and recipient):
            st.warning("Please fill Sender Gmail, App Password and Recipient.", icon="⚠️")
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
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, app_password)
            server.send_message(msg)
        return True
    except Exception:
        st.toast("Email sending failed. Check Gmail & App Password.", icon="⚠️")
        return False

# -------------------- HEADER -------------------------
st.markdown("<h1 class='neon' style='text-align:center;'>🩺 Pneumonia Detection AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;' class='small-muted'>Upload a chest X-ray • AI predicts • Optional Grad-CAM heatmap • Download PDF • Save history</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# -------------------- MODEL LOAD ---------------------
try:
    model = load_model_cached()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# -------------------- UPLOAD & PREDICT ---------------
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    uploaded = st.file_uploader("📤 Upload Chest X-ray (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded:
        try:
            pil_img = Image.open(uploaded)
            # Store original as base64 for consistent display sizing
            st.session_state.last_original_b64 = base64.b64encode(uploaded.getvalue()).decode()
            st.markdown(f"""
                <div class='img-card'>
                    <h4>📤 Original X-ray</h4>
                    <img class='img-thumb' src='data:image/png;base64,{st.session_state.last_original_b64}'/>
                    <div class='img-caption'>Uploaded: {uploaded.name}</div>
                </div>
            """, unsafe_allow_html=True)
        except Exception:
            st.error("Could not read this image file.")
            pil_img = None
    else:
        pil_img = None
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("### 🔍 Prediction")

    # Doctor notes input (kept inside the same right card to preserve layout)
    st.session_state.doctor_notes = st.text_area("Doctor notes (optional)", value=st.session_state.doctor_notes, help="These notes will be saved inside the PDF report.")

    predict_btn = st.button("Run AI Prediction", use_container_width=True, type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

# Perform prediction
if predict_btn:
    if pil_img is None:
        st.warning("Please upload an image first.")
    else:
        with st.spinner("Analyzing X-ray with EfficientNet…"):
            try:
                batch = prepare_image(pil_img)
                probs = model.predict(batch, verbose=0)[0]  # [Normal, Pneumonia]
                normal_p, pneu_p = float(probs[0]), float(probs[1])
                top_idx = int(np.argmax(probs))
                label = CLASS_NAMES[top_idx]
                conf = probs[top_idx] * 100.0

                # Voice (optional)
                try:
                    to_say = "Pneumonia detected" if label == "Pneumonia" else "Result is normal"
                    safe_speak(to_say)
                except Exception:
                    pass

                # Show result
                style = "result-bad" if label == "Pneumonia" else "result-good"
                st.markdown(f"<div class='{style}'>"
                            f"{'⚠️' if label=='Pneumonia' else '✅'} <b>{label}</b><br>"
                            f"Confidence: {conf:.2f}%</div>", unsafe_allow_html=True)

                # Confidence gauge (modern)
                render_gauge(conf, "Confidence")

                # Grad-CAM
                if enable_gradcam:
                    try:
                        overlay_bgr = gradcam_overlay(pil_img, model, batch)
                        # Save overlay to session b64 for both thumbnail & fullscreen modal
                        st.session_state.last_overlay_b64 = npimg_to_b64(overlay_bgr)

                        colA, colB = st.columns([1, 1])
                        with colA:
                            st.markdown(f"""
                                <div class='img-card'>
                                    <h4>📤 Original</h4>
                                    <img class='img-thumb' src='data:image/png;base64,{st.session_state.last_original_b64}'/>
                                </div>
                            """, unsafe_allow_html=True)
                        with colB:
                            st.markdown(f"""
                                <div class='img-card'>
                                    <h4>🔥 Grad-CAM</h4>
                                    <img class='img-thumb-heat' src='data:image/png;base64,{st.session_state.last_overlay_b64}'/>
                                    <div class='img-caption'>Highlighted risk regions</div>
                                </div>
                            """, unsafe_allow_html=True)

                        # Fullscreen button
                        open_modal = st.button("🔎 View Fullscreen Heatmap", use_container_width=True)
                        if open_modal:
                            st.session_state.show_heatmap_modal = True

                        # Download heatmap
                        st.download_button(
                            "⬇ Download Heatmap (PNG)",
                            data=base64.b64decode(st.session_state.last_overlay_b64),
                            file_name=f"heatmap_{os.path.splitext(uploaded.name)[0]}.png",
                            mime="image/png",
                            use_container_width=True
                        )

                    except Exception:
                        st.toast("Grad-CAM failed on this model/TF build.", icon="⚠️")

                # Save to history
                st.session_state.history.append({
                    "Image": uploaded.name,
                    "Result": label,
                    "Confidence": f"{conf:.2f}%",
                    "Time": time.strftime("%Y-%m-%d %H:%M:%S")
                })

                # PDF download (now includes notes + heatmap)
                pdf_bytes = generate_pdf_report(
                    uploaded.name, label, conf, st.session_state.doctor_notes, st.session_state.last_overlay_b64
                )
                st.session_state.last_pdf_bytes = pdf_bytes
                st.download_button(
                    "📄 Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"Report_{os.path.splitext(uploaded.name)[0]}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

            except Exception:
                st.error("Prediction failed.")
                with st.expander("Show error details"):
                    st.code(traceback.format_exc())

st.markdown("<br>", unsafe_allow_html=True)

# -------------------- FULLSCREEN MODAL ---------------
if st.session_state.show_heatmap_modal and st.session_state.last_overlay_b64:
    # Backdrop
    st.markdown("<div class='modal-backdrop'></div>", unsafe_allow_html=True)
    # Card with the big image
    st.markdown(f"""
        <div class='modal-card'>
            <h3 style='margin: 6px 0 10px 0;'>🔎 Fullscreen Grad-CAM</h3>
            <img class='modal-img' src='data:image/png;base64,{st.session_state.last_overlay_b64}' />
            <p class='small-muted' style='margin-top:8px;'>Press the button below to close.</p>
        </div>
    """, unsafe_allow_html=True)
    # Close button (toggles state)
    if st.button("Close Heatmap", type="secondary", use_container_width=True):
        st.session_state.show_heatmap_modal = False

# -------------------- EMAIL REPORT -------------------
st.markdown("## 📧 Email Report")
colE1, colE2 = st.columns([2, 1])
with colE1:
    recipient_email = st.text_input("Recipient email", value=st.session_state.get("default_recipient", ""))
with colE2:
    subj = st.text_input("Subject", value="Pneumonia AI Report")

send_btn = st.button("Send Last Report via Email", use_container_width=True)
if send_btn:
    if st.session_state.last_pdf_bytes is None:
        st.warning("Generate a report first (run a prediction).", icon="⚠️")
    else:
        success = send_email_with_attachment(
            sender_email=sender_email.strip(),
            app_password=sender_app_password.strip(),
            recipient=(recipient_email or default_recipient or "").strip(),
            subject=subj.strip() or "Pneumonia AI Report",
            body="Attached is the AI-generated Pneumonia report.",
            attachment_bytes=st.session_state.last_pdf_bytes,
            filename="report.pdf"
        )
        if success:
            st.success("Email sent ✅")

st.markdown("<br>", unsafe_allow_html=True)

# -------------------- BATCH PREDICTION ----------------
st.markdown("## 📦 Batch Prediction")
batch_files = st.file_uploader("Upload multiple X-rays", type=["jpg","jpeg","png"], accept_multiple_files=True)
if batch_files:
    if st.button("Run Batch Prediction", use_container_width=True):
        with st.spinner("Running batch predictions..."):
            rows = []
            for f in batch_files:
                try:
                    im = Image.open(f)
                    p = model.predict(prepare_image(im), verbose=0)[0]
                    idx = int(np.argmax(p))
                    rows.append([f.name, CLASS_NAMES[idx], f"{p[idx]*100:.2f}%"])
                except Exception:
                    rows.append([f.name, "Error", "—"])
            dfb = pd.DataFrame(rows, columns=["Image", "Result", "Confidence"])
            st.session_state.batch_results = dfb

if st.session_state.batch_results is not None:
    st.dataframe(st.session_state.batch_results, use_container_width=True)
    st.download_button(
        "⬇ Download Batch CSV",
        data=st.session_state.batch_results.to_csv(index=False).encode("utf-8"),
        file_name="batch_results.csv",
        mime="text/csv",
        use_container_width=True
    )

# -------------------- HISTORY / EXPORT ----------------
st.markdown("## 📜 Prediction History")
if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download History (CSV)", data=csv_bytes, file_name="prediction_history.csv", mime="text/csv", use_container_width=True)
else:
    st.info("No predictions yet. Your results will appear here after the first run.")

st.markdown("<br>", unsafe_allow_html=True)

# -------------------- MINI DASHBOARD ------------------
st.markdown("## 📊 Mini Dashboard (Session)")
if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    counts = df["Result"].value_counts().reindex(CLASS_NAMES, fill_value=0)
    st.bar_chart(counts, use_container_width=True)
else:
    st.write("Run a few predictions to see session analytics.")

# -------------------- DISCLAIMER ----------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p class='small-muted'>⚠️ This tool is for educational use only and is not a medical device. "
            "Always consult a qualified clinician for diagnosis and treatment.</p>", unsafe_allow_html=True)
