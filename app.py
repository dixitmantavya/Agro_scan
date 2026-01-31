import streamlit as st
import os
import tempfile
import cv2
import numpy as np

from model.predict import predict_disease
from utils.disease_info import DISEASE_INFO
from utils.tomato_disease_guide import TOMATO_DISEASE_GUIDE
from utils.weather import get_weather

# ================= CONFIG =================

st.set_page_config(
    page_title="Agro-Scan | Plant Disease Detection",
    layout="centered"
)

VIRAL_KEYWORDS = ["virus", "mosaic", "curl"]
FUNGAL_KEYWORDS = ["blight", "rust", "mold", "spot"]
BACTERIAL_KEYWORDS = ["bacterial"]

# ================= LEAF EXTRACTION =================

def extract_leaf_mask(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return img_bgr, mask

    largest = max(contours, key=cv2.contourArea)
    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, [largest], -1, 255, -1)

    leaf_only = cv2.bitwise_and(img_bgr, img_bgr, mask=clean_mask)
    return leaf_only, clean_mask

# ================= REAL SEVERITY =================

def compute_severity_from_pixels(leaf_bgr, leaf_mask):
    hsv = cv2.cvtColor(leaf_bgr, cv2.COLOR_BGR2HSV)

    lower_disease = np.array([5, 50, 50])
    upper_disease = np.array([35, 255, 255])

    disease_mask = cv2.inRange(hsv, lower_disease, upper_disease)
    disease_mask = cv2.bitwise_and(disease_mask, disease_mask, mask=leaf_mask)

    infected_pixels = np.count_nonzero(disease_mask)
    total_leaf_pixels = np.count_nonzero(leaf_mask)

    if total_leaf_pixels == 0:
        return 0.0, "Unknown", disease_mask

    severity_percent = (infected_pixels / total_leaf_pixels) * 100

    if severity_percent < 5:
        level = "Very Low"
    elif severity_percent < 20:
        level = "Mild"
    elif severity_percent < 50:
        level = "Moderate"
    else:
        level = "Severe"

    return severity_percent, level, disease_mask

# ================= HELPERS =================

def resize_with_padding(img, target_size=224):
    h, w, _ = img.shape
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_w, pad_h = target_size - new_w, target_size - new_h
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2

    return cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

def parse_class_name(class_name):
    if "___" in class_name:
        crop, disease = class_name.split("___", 1)
    elif "_" in class_name:
        crop, disease = class_name.split("_", 1)
    else:
        crop, disease = class_name, "Unknown"
    return crop, disease.replace("_", " ").strip()

def weather_risk_advice(disease_name, weather_info):
    advice = []
    disease = disease_name.lower()

    try:
        temp = float(weather_info.split("Temperature:")[1].split("°")[0].strip())
        humidity = int(weather_info.split("Humidity:")[1].split("%")[0].strip())
    except Exception:
        return []

    if any(k in disease for k in FUNGAL_KEYWORDS) and humidity > 70:
        advice.append("⚠️ High humidity favors fungal spread.")
    if any(k in disease for k in BACTERIAL_KEYWORDS) and humidity > 65 and temp > 25:
        advice.append("⚠️ Warm & humid conditions favor bacterial diseases.")
    if any(k in disease for k in VIRAL_KEYWORDS) and temp > 25:
        advice.append("⚠️ Warm weather increases viral disease vectors.")

    return advice

# ================= UI =================

st.title("🌿 Agro-Scan – AI Plant Disease Detector")
st.write("Upload a plant leaf image to detect disease and true severity.")

uploaded_file = st.file_uploader("📷 Upload a leaf image", type=["jpg", "jpeg", "png"])
city = st.text_input("🌦️ Enter your city (optional – weather insights)")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        image_path = tmp.name

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        st.error("❌ Failed to read image.")
        st.stop()

    leaf_bgr, leaf_mask = extract_leaf_mask(img_bgr)
    leaf_rgb = cv2.cvtColor(leaf_bgr, cv2.COLOR_BGR2RGB)

    st.image(leaf_rgb, caption="Detected Leaf Region", use_container_width=True)

    img_resized = resize_with_padding(leaf_rgb)
    img_input = np.expand_dims(img_resized / 255.0, axis=0)

    with st.spinner("🔍 Analyzing image..."):
        results = predict_disease(img_input, top_k=3)

    st.subheader("🔍 Top Predictions")
    for i, (cls, conf) in enumerate(results, start=1):
        crop, disease = parse_class_name(cls)
        st.write(f"**{i}. 🌱 {crop} – 🦠 {disease}** ({conf*100:.2f}%)")

    best_class, best_confidence = results[0]
    crop, disease = parse_class_name(best_class)

    severity_percent, severity_level, disease_mask = compute_severity_from_pixels(
        leaf_bgr, leaf_mask
    )

    st.subheader("📊 Disease Severity (real area based)")
    st.write(f"**Severity:** {severity_level}")
    st.write(f"Infected leaf area: **{severity_percent:.2f}%**")

    # 🔥 NEW: severity → action
    st.subheader("🧭 Recommended Action")

    if severity_level == "Very Low":
        st.success("Monitor the plant. Remove a few infected leaves. No spray needed yet.")

    elif severity_level == "Mild":
        st.info("Remove infected leaves and do a preventive spray.")
        st.write("Repeat preventive spray after **7 days** if symptoms remain.")

    elif severity_level == "Moderate":
        st.warning("Spray recommended pesticide/fungicide now.")
        st.write("Repeat the spray after **7 days**.")

    else:  # Severe
        st.error("Immediate chemical control required.")
        st.write("Remove heavily infected parts and spray now.")
        st.write("Repeat the spray after **5 days** until severity reduces.")

    red_overlay = leaf_rgb.copy()
    red_overlay[disease_mask > 0] = [255, 0, 0]
    blended = cv2.addWeighted(leaf_rgb, 0.7, red_overlay, 0.3, 0)
    st.image(blended, caption="Detected Diseased Regions", use_container_width=True)

    if city:
        weather_info = get_weather(city)
        st.subheader("🌦️ Local Weather")
        st.write(weather_info)

        for r in weather_risk_advice(disease, weather_info):
            st.warning(r)

    st.subheader("📘 Detailed Farmer Guide")
    info = TOMATO_DISEASE_GUIDE.get(best_class)

    if info:
        st.markdown("### 🔍 Symptoms")
        for s in info["symptoms"]:
            st.write(f"- {s}")

        st.markdown("### 🧪 Cause")
        st.write(info["cause"])

        st.markdown("### 🚫 Prevention")
        for p in info["prevention"]:
            st.write(f"- {p}")

        st.markdown("### 💉 Chemical Treatment")
        for chem in info["treatment_chemicals"]:
            st.markdown(f"- [{chem['name']}]({chem['link']})")
    else:
        st.write(DISEASE_INFO.get(best_class, "ℹ️ No detailed data available."))

    os.remove(image_path)

else:
    st.info("⬆️ Upload an image to begin.")
