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
# Use a robust preprocessing function that covers standard inputs
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
CLASS_NAMES = ["Normal", "Pneumonia"] # index 0 = Normal, 1 = Pneumonia

# -------------------- SESSION STATE INIT -----------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark" # Retain theme state
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
    
# Removed: show_heatmap_modal (simplifying modal handling)

# -------------------- THEME / STYLE (Blue/Grey Futuristic) ------------------
def apply_theme():
    # Primary theme colors (Futuristic Blue/Grey)
    DARK_BG = "#0A0A10"  # Darkest background
    LIGHT_BG = "#1A1B26" # Card background (Grey-Blue)
    PRIMARY_BLUE = "#00BFFF" # Deep Sky Blue for accent
    TEXT_COLOR = "#E0E0FF" # Light text

    st.markdown(f"""
    <style>
        /* General Background */
        body {{
            background-color: {DARK_BG};
            color: {TEXT_COLOR};
        }}
        
        /* Streamlit components style overrides for a consistent dark/blue-grey look */
        .stApp {{
            background-color: {DARK_BG};
        }}
        
        /* Custom card style */
        .card {{
            background-color: {LIGHT_BG};
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0, 191, 255, 0.15); /* Light blue glow */
            margin-bottom: 20px;
        }}
        
        /* Headings - use primary blue accent */
        h1, h2, h3, h4, h5, h6 {{
            color: {TEXT_COLOR};
        }}
        h1 {{
             text-shadow: 0 0 6px rgba(0, 191, 255, 0.6);
             text-align: center;
        }}

        /* Result Banners */
        .result-normal {{
            background: rgba(0, 255, 127, 0.15); /* Spring Green */
            border-left: 5px solid #00FF7F;
            padding: 12px;
            border-radius: 6px;
            font-weight: 700;
            text-align: center;
            color: #00FF7F;
        }}
        .result-pneumonia {{
            background: rgba(255, 69, 0, 0.15); /* Orange Red */
            border-left: 5px solid #FF4500;
            padding: 12px;
            border-radius: 6px;
            font-weight: 700;
            text-align: center;
            color: #FF4500;
        }}
        
        /* Image Cards */
        .img-card {{
            text-align: center;
            margin-top: 15px;
        }}
        .img-thumb, .img-thumb-heat {{
            max-width: 280px;  /* Slightly bigger for better viewing */
            width: 100%;
            height: auto;
            border-radius: 8px; 
            margin-top: 10px;
            border: 2px solid {PRIMARY_BLUE};
        }}
        .img-thumb-heat {{
            border: 2px solid #FF4500; /* Red border for heatmap */
        }}
        
        /* Muted Text */
        .small-muted {{
            font-size: 13px;
            opacity: 0.7;
            color: {TEXT_COLOR};
        }}

        /* Streamlit primary button */
        .stButton>button {{
            background-color: {PRIMARY_BLUE};
            border-color: {PRIMARY_BLUE};
            color: {DARK_BG};
        }}
        .stButton>button:hover {{
            background-color: #008CBA; 
            border-color: #008CBA;
        }}

    </style>
    """, unsafe_allow_html=True)

apply_theme()

# -------------------- SIDEBAR ------------------------
with st.sidebar:
    st.markdown("## ⚙ Controls")
    
    # Theme toggle (simplified)
    if st.button("🎨 Toggle Theme (Light/Dark)"):
        # Note: True Streamlit theming is done via config.toml, but we mimic it here
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.experimental_rerun() # Rerun to apply potential changes (though not strictly necessary for inline CSS changes)
        
    enable_gradcam = st.checkbox("Grad-CAM heatmap", value=True, help="Explainable AI overlay to show activation regions")

    st.markdown("---")
    st.markdown("### 📧 Email Configuration")
    sender_email = st.text_input("Sender Gmail", placeholder="yourname@gmail.com")
    sender_app_password = st.text_input("Gmail App Password", type="password", help="Use a Gmail App Password, not your regular password.")
    default_recipient = st.text_input("Default Recipient (optional)", placeholder="recipient@example.com")
    st.markdown("---")
    st.markdown("*Model:* `pneumonia_final_Dibyendu.h5`")
    st.markdown("*Input:* 300×300 RGB (EfficientNet preprocess)")


# -------------------- HELPERS ------------------------
@st.cache_resource(show_spinner=True)
def load_model_cached():
    """Load the Keras model with error handling and warm-up."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at: {MODEL_PATH}")
        st.stop()
    
    # Explicitly compile=False, as we don't need the optimizer for inference
    model = load_model(MODEL_PATH, compile=False) 
    # Warm-up a dummy forward pass to catch load issues early
    try:
        _ = model.predict(np.zeros((1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32), verbose=0)
    except Exception as e:
         st.error(f"Model warm-up failed. Check model file integrity. Error: {e}")
         st.stop()
    return model

def prepare_image(pil_img: Image.Image) -> np.ndarray:
    """Resize -> RGB -> EfficientNet preprocess -> add batch dimension."""
    # Ensure correct size and color depth
    img = pil_img.convert("RGB").resize(IMAGE_SIZE)
    arr = np.asarray(img, dtype=np.float32)
    # Apply EfficientNet preprocessing
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def npimg_to_b64(np_img_bgr: np.ndarray) -> str:
    """Convert OpenCV BGR image to base64 PNG string."""
    _, im_png = cv2.imencode(".png", np_img_bgr)
    return base64.b64encode(im_png.tobytes()).decode()

def render_gauge(percent: float, label: str):
    """Draw a clean semicircle gauge using inline SVG."""
    p = max(0, min(100, percent))
    # Map 0..100 to -90..90 degrees
    angle = -90 + (p * 180.0 / 100.0)
    
    # Define colors based on prediction (using a simple threshold for visualization)
    gauge_color = "#FF4500" if p > 50 else "#00FF7F" # Red for high Pneumonia confidence, Green for Normal
    
    st.markdown(f"""
    <div style="text-align:center; margin-top: 10px;">
      <svg viewBox="0 0 100 60" style="width: 180px; max-width: 90%; margin: 6px auto 0 auto;">
        <path d="M10,50 A40,40 0 0,1 90,50" fill="none" stroke="rgba(255,255,255,0.25)" stroke-width="6"/>
        <path d="M10,50 A40,40 0 0,1 90,50" fill="none" stroke="{gauge_color}" stroke-width="6"
              stroke-dasharray="{p*1.256},{125.6}"/>
        <line x1="50" y1="50" x2="{50 + 38 * np.cos(np.radians(angle))}" y2="{50 + 38 * np.sin(np.radians(angle))}"
              stroke="{gauge_color}" stroke-width="2"/>
        <text x="50" y="58" text-anchor="middle" font-size="8" fill="currentColor">{label}: {p:.1f}%</text>
      </svg>
    </div>
    """, unsafe_allow_html=True)

def generate_pdf_report(image_name: str, result: str, confidence: float, notes: str, heatmap_b64: str | None) -> bytes:
    """Create a PDF report in memory including notes and optional heatmap."""
    # Note: fpdf requires temporary file access for embedding images from b64 string
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 12, "Pneumonia Detection Report (AI Assist)", ln=True, align="C")

    pdf.set_font("Arial", "", 13)
    pdf.ln(6)
    pdf.cell(0, 8, f"Image: {image_name}", ln=True)
    pdf.cell(0, 8, f"Prediction: {result}", ln=True)
    pdf.cell(0, 8, f"Confidence: {confidence:.2f}%", ln=True)
    pdf.cell(0, 8, f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(4)
    
    pdf.set_font("Arial", "I", 11)
    pdf.set_text_color(150, 150, 150) # Grey disclaimer text
    disclaimer = "Disclaimer: This is an assistive AI tool for educational and research use only. It is NOT a medical diagnosis. Please consult a qualified clinician for interpretation."
    pdf.multi_cell(0, 7, disclaimer)
    pdf.set_text_color(0, 0, 0) # Reset text color to black
    pdf.ln(4)
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 6, "Doctor Notes:", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 6, f"{notes if notes else 'N/A'}")
    pdf.ln(3)

    # Embed heatmap if available
    if heatmap_b64:
        temp_file_name = f"temp_gradcam_{os.getpid()}.png"
        try:
            img_bytes = base64.b64decode(heatmap_b64)
            with open(temp_file_name, "wb") as f:
                f.write(img_bytes)
                
            pdf.ln(3)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "Grad-CAM Heatmap:", ln=True)
            # Center the image
            pdf.image(temp_file_name, x=(pdf.w - 120) / 2, w=120) 
            
        except Exception as e:
            # st.warning(f"Failed to embed heatmap in PDF: {e}") # Log error but don't stop
            pass
        finally:
            if os.path.exists(temp_file_name):
                os.remove(temp_file_name)

    # Output to memory
    out = pdf.output(dest="S").encode("latin1")
    return out

def find_last_conv_layer(keras_model: tf.keras.Model):
    """Robustly get the last 4D layer (conv/activation) for Grad-CAM."""
    for layer in reversed(keras_model.layers):
        try:
            # Check if the output shape is 4D (batch, height, width, channels)
            if len(layer.output.shape) == 4:
                return layer.name
        except Exception:
            continue
    return None

def gradcam_overlay(pil_img: Image.Image, model: tf.keras.Model, processed_batch: np.ndarray, target_class_idx: int) -> np.ndarray:
    """Compute Grad-CAM heatmap and overlay on the original image."""
    layer_name = find_last_conv_layer(model)
    
    if layer_name is None:
        # Fallback: return the original image as BGR (OpenCV format)
        return np.asarray(pil_img.convert("RGB").resize(IMAGE_SIZE))[:, :, ::-1]

    # 1. Create a model that outputs the activation map and the final prediction
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    # 2. Compute the gradient of the predicted class with respect to the activation map
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(processed_batch)
        # Loss is the output score for the predicted class
        loss = preds[:, target_class_idx] 

    # 3. Get the gradient
    grads = tape.gradient(loss, conv_out)
    if grads is None:
         return np.asarray(pil_img.convert("RGB").resize(IMAGE_SIZE))[:, :, ::-1]

    # 4. Average the gradient over all spatial locations (Global Average Pooling)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)) # (C,)

    # 5. Multiply each channel in the activation map by the average gradient (weights)
    conv_out = conv_out[0] # H x W x C (remove batch dim)
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_out), axis=-1)

    # 6. Apply ReLU to keep only positive contributions
    heatmap = tf.nn.relu(heatmap)
    # Normalize to 0-1 range
    denom = tf.reduce_max(heatmap)
    heatmap = heatmap / (denom + 1e-10)
    heatmap = heatmap.numpy()

    # 7. Resize heatmap to original image size
    heatmap = cv2.resize(heatmap, IMAGE_SIZE[::-1])
    heatmap = np.uint8(255 * heatmap)
    
    # 8. Apply color map (JET is standard for Grad-CAM)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # 9. Overlay the heatmap on the original image
    original_rgb = np.asarray(pil_img.convert("RGB").resize(IMAGE_SIZE))
    original_bgr = original_rgb[:, :, ::-1] # Convert RGB to BGR for OpenCV blend
    # Blend the original image (60% weight) with the heatmap (40% weight)
    overlay = cv2.addWeighted(original_bgr, 0.6, heatmap_colored, 0.4, 0)
    
    return overlay

def send_email_with_attachment(sender_email: str, app_password: str, recipient: str, subject: str, body: str, attachment_bytes: bytes, filename: str) -> bool:
    """Send email via Gmail SMTP with an attachment; safe fallbacks."""
    try:
        if not (sender_email and app_password and recipient):
            st.warning("Please fill Sender Gmail, App Password and Recipient in the sidebar.", icon="⚠️")
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
        # Use a higher timeout for potentially slow SMTP connection
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context, timeout=10) as server:
            server.login(sender_email, app_password)
            server.send_message(msg)
        return True
        
    except smtplib.SMTPAuthenticationError:
        st.error("Email sending failed: Authentication Error. Check your Gmail App Password and email address.", icon="❌")
        return False
    except Exception:
        st.toast("Email sending failed. Check configurations and internet connection.", icon="❌")
        return False

# -------------------- HEADER -------------------------
st.markdown("<h1 class='title'>🩺 Pneumonia Detection AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;' class='small-muted'>Upload a chest X-ray • AI predicts • View Grad-CAM heatmap • Generate PDF report</p>", unsafe_allow_html=True)
st.markdown("---")

# -------------------- MODEL LOAD ---------------------
model = load_model_cached()


# -------------------- UPLOAD & PREDICT ---------------
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    uploaded = st.file_uploader("📤 Upload Chest X-ray (JPG/PNG)", type=["jpg", "jpeg", "png"])
    
    pil_img = None
    if uploaded:
        try:
            # Read the file
            pil_img = Image.open(uploaded)
            
            # Store original as base64 for consistent display sizing
            st.session_state.last_original_b64 = base64.b64encode(uploaded.getvalue()).decode()
            
            # Display original image
            st.markdown(f"""
                <div class='img-card'>
                    <h4>Uploaded X-ray: {uploaded.name}</h4>
                    <img class='img-thumb' src='data:image/png;base64,{st.session_state.last_original_b64}'/>
                    <div class='small-muted'>Image: {uploaded.name}</div>
                </div>
            """, unsafe_allow_html=True)
            
        except Exception:
            st.error("Could not read this image file. Ensure it's a valid JPG/PNG.", icon="❌")
            pil_img = None
            
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🔍 Prediction & Notes")

    # Doctor notes input
    st.session_state.doctor_notes = st.text_area("Doctor notes (optional)", value=st.session_state.doctor_notes, help="These notes will be saved inside the PDF report.")

    predict_btn = st.button("Run AI Prediction", use_container_width=True, type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

# Perform prediction
if predict_btn:
    if pil_img is None:
        st.warning("Please upload an image first.", icon="⚠️")
    else:
        # Clear previous prediction data
        st.session_state.last_overlay_b64 = None
        st.session_state.last_pdf_bytes = None
        
        with st.spinner("Analyzing X-ray with EfficientNet…"):
            try:
                # 1. Prediction
                batch = prepare_image(pil_img)
                # [Normal, Pneumonia] probabilities
                probs = model.predict(batch, verbose=0)[0]  
                
                top_idx = int(np.argmax(probs))
                label = CLASS_NAMES[top_idx]
                conf = probs[top_idx] * 100.0

                # 2. Show result
                result_col = col_right.container() # Display results in the right column
                style = "result-pneumonia" if label == "Pneumonia" else "result-normal"
                result_col.markdown(f"<div class='{style}' style='margin-bottom: 10px;'>"
                                    f"{'⚠️' if label=='Pneumonia' else '✅'} **{label}**<br>"
                                    f"Confidence: {conf:.2f}%</div>", unsafe_allow_html=True)

                # 3. Confidence gauge
                render_gauge(conf, "Confidence")

                # 4. Grad-CAM (if enabled)
                if enable_gradcam:
                    try:
                        overlay_bgr = gradcam_overlay(pil_img, model, batch, top_idx)
                        # Save overlay to session b64
                        st.session_state.last_overlay_b64 = npimg_to_b64(overlay_bgr)

                        # Display Heatmap results in a new section under the prediction
                        col_results_heat, col_results_dl = st.columns([2, 1])
                        
                        with col_results_heat:
                            st.markdown(f"""
                                <div class='img-card'>
                                    <h4>🔥 Grad-CAM Overlay</h4>
                                    <img class='img-thumb-heat' src='data:image/png;base64,{st.session_state.last_overlay_b64}'/>
                                    <div class='small-muted'>Highlighted risk regions</div>
                                </div>
                            """, unsafe_allow_html=True)

                        with col_results_dl:
                            st.markdown("### Downloads")
                            st.download_button(
                                "⬇ Download Heatmap (PNG)",
                                data=base64.b64decode(st.session_state.last_overlay_b64),
                                file_name=f"heatmap_{os.path.splitext(uploaded.name)[0]}.png",
                                mime="image/png",
                                use_container_width=True
                            )
                        
                    except Exception:
                        st.toast("Grad-CAM failed on this model/TF build.", icon="⚠️")
                        st.session_state.last_overlay_b64 = None
                        
                # 5. Save to history
                st.session_state.history.append({
                    "Image": uploaded.name,
                    "Result": label,
                    "Confidence": f"{conf:.2f}%",
                    "Time": time.strftime("%Y-%m-%d %H:%M:%S")
                })

                # 6. PDF download
                pdf_bytes = generate_pdf_report(
                    uploaded.name, label, conf, st.session_state.doctor_notes, st.session_state.last_overlay_b64
                )
                st.session_state.last_pdf_bytes = pdf_bytes
                
                # Use the right column for the report download button
                with col_right:
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

st.markdown("---")

# -------------------- EMAIL REPORT -------------------
st.markdown("## 📧 Email Report")
colE1, colE2, colE3 = st.columns([2, 2, 1])
with colE1:
    recipient_email = st.text_input("Recipient email", value=default_recipient, key="recipient_email_input")
with colE2:
    subj = st.text_input("Subject", value="Pneumonia AI Report")

with colE3:
    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True) # Spacer for alignment
    send_btn = st.button("Send Last Report", use_container_width=True, key="send_email_button")
    
if send_btn:
    recipient = (recipient_email or default_recipient or "").strip()
    if not recipient:
        st.warning("Please provide a recipient email address.", icon="⚠️")
    elif st.session_state.last_pdf_bytes is None:
        st.warning("Generate a report first (run a prediction).", icon="⚠️")
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
            st.success("Email sent successfully! ✅")

st.markdown("---")

# -------------------- BATCH PREDICTION ----------------
st.markdown("## 📦 Batch Prediction")
batch_files = st.file_uploader("Upload multiple X-rays", type=["jpg","jpeg","png"], accept_multiple_files=True, key="batch_uploader")
if batch_files:
    if st.button("Run Batch Prediction", use_container_width=True, key="run_batch_button", type="secondary"):
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
            st.success(f"Batch prediction finished for {len(batch_files)} files.")

if st.session_state.batch_results is not None:
    st.dataframe(st.session_state.batch_results, use_container_width=True)
    st.download_button(
        "⬇ Download Batch CSV",
        data=st.session_state.batch_results.to_csv(index=False).encode("utf-8"),
        file_name="batch_results.csv",
        mime="text/csv",
        use_container_width=True
    )

st.markdown("---")

# -------------------- HISTORY / EXPORT ----------------
st.markdown("## 📜 Prediction History")
if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download History (CSV)", data=csv_bytes, file_name="prediction_history.csv", mime="text/csv", use_container_width=True)
else:
    st.info("No predictions yet. Your results will appear here after the first run.")

st.markdown("---")

# -------------------- MINI DASHBOARD ------------------
st.markdown("## 📊 Mini Dashboard (Session)")
if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    counts = df["Result"].value_counts().reindex(CLASS_NAMES, fill_value=0)
    
    st.subheader("Count of Diagnoses")
    st.bar_chart(counts, use_container_width=True)
    
    # Calculate average confidence
    df['Confidence_Val'] = df['Confidence'].str.replace('%', '').astype(float)
    avg_conf = df['Confidence_Val'].mean()
    st.metric(label="Average Prediction Confidence", value=f"{avg_conf:.2f}%")
else:
    st.write("Run a few predictions to see session analytics.")

st.markdown("---")

# -------------------- DISCLAIMER ----------------------
st.markdown("<p class='small-muted' style='text-align:center;'>⚠ This tool is for educational/research use only and is not a medical device. "
             "Always consult a qualified clinician for diagnosis and treatment.</p>", unsafe_allow_html=True)
