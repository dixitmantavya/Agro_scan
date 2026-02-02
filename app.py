import streamlit as st
import os
import tempfile
import cv2
import numpy as np

from model.predict import predict_disease
from utils.disease_info import DISEASE_INFO
from utils.tomato_disease_guide import TOMATO_DISEASE_GUIDE
from utils.weather import get_weather

from utils.fertilizer import recommend_fertilizer
from utils.fertilizer_amount import calculate_fertilizer_amount


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


# ================= SEVERITY =================

def compute_severity_from_pixels(leaf_bgr, leaf_mask):
    hsv = cv2.cvtColor(leaf_bgr, cv2.COLOR_BGR2HSV)

    lower_disease = np.array([5, 50, 50])
    upper_disease = np.array([35, 255, 255])

    disease_mask = cv2.inRange(hsv, lower_disease, upper_disease)
    disease_mask = cv2.bitwise_and(disease_mask, disease_mask, mask=leaf_mask)

    infected = np.count_nonzero(disease_mask)
    total = np.count_nonzero(leaf_mask)

    if total == 0:
        return 0, "Unknown", disease_mask

    percent = (infected / total) * 100

    if percent < 5:
        level = "Very Low"
    elif percent < 20:
        level = "Mild"
    elif percent < 50:
        level = "Moderate"
    else:
        level = "Severe"

    return percent, level, disease_mask


# ================= HELPERS =================

def resize_with_padding(img, target=224):
    h, w, _ = img.shape

    scale = target / max(h, w)
    nw, nh = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (nw, nh))

    pad_w, pad_h = target - nw, target - nh

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    return cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )


def parse_class_name(class_name):
    if "___" in class_name:
        crop, disease = class_name.split("___", 1)
    else:
        crop, disease = class_name, "Unknown"

    return crop, disease.replace("_", " ")


def get_guide_info(best_class):
    for key in TOMATO_DISEASE_GUIDE:
        if key.lower() == best_class.lower():
            return TOMATO_DISEASE_GUIDE[key]
    return None


def weather_risk_advice(disease_name, weather_info):
    advice = []

    try:
        temp = float(weather_info.split("Temperature:")[1].split("°")[0])
        humidity = int(weather_info.split("Humidity:")[1].split("%")[0])
    except:
        return advice

    disease = disease_name.lower()

    if any(k in disease for k in FUNGAL_KEYWORDS) and humidity > 70:
        advice.append("⚠️ High humidity favors fungal diseases")

    if any(k in disease for k in BACTERIAL_KEYWORDS) and humidity > 65:
        advice.append("⚠️ Warm humid weather favors bacterial diseases")

    if any(k in disease for k in VIRAL_KEYWORDS) and temp > 25:
        advice.append("⚠️ Warm weather boosts viral vectors")

    return advice


# ================= UI =================

st.title("🌿 Agro-Scan – AI Plant Disease Detector")
st.write("Upload a leaf image to detect disease & real severity.")

uploaded_file = st.file_uploader("📷 Upload leaf image", type=["jpg","jpeg","png"])
city = st.text_input("🌦️ Enter city (optional)")


if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    img_bgr = cv2.imread(path)

    leaf_bgr, leaf_mask = extract_leaf_mask(img_bgr)
    leaf_rgb = cv2.cvtColor(leaf_bgr, cv2.COLOR_BGR2RGB)

    st.image(leaf_rgb, caption="Leaf Region")

    resized = resize_with_padding(leaf_rgb)
    inp = np.expand_dims(resized/255.0, 0)

    with st.spinner("Analyzing..."):
        results = predict_disease(inp, top_k=3)

    st.subheader("🔍 Predictions")

    for i,(cls,conf) in enumerate(results,1):
        crop,disease = parse_class_name(cls)
        st.write(f"{i}. {crop} - {disease} ({conf*100:.2f}%)")

    best_class,best_conf = results[0]
    crop,disease = parse_class_name(best_class)

    sev_percent, sev_level, dmask = compute_severity_from_pixels(
        leaf_bgr, leaf_mask
    )

    st.subheader("📊 Severity")
    st.write(f"Level: **{sev_level}**")
    st.write(f"Infected: **{sev_percent:.2f}%**")

    overlay = leaf_rgb.copy()
    overlay[dmask>0] = [255,0,0]
    blended = cv2.addWeighted(leaf_rgb,0.7,overlay,0.3,0)

    st.image(blended, caption="Diseased Area")

    if city:
        weather = get_weather(city)
        st.subheader("🌦️ Weather")
        st.write(weather)

        for a in weather_risk_advice(disease, weather):
            st.warning(a)

    # ===== FULL FARMER GUIDE =====

    st.subheader("📘 Farmer Guide")

    info = get_guide_info(best_class)

    if info:

        with st.expander("View Details", expanded=True):

            if "symptoms" in info:
                st.markdown("### 🩺 Symptoms")
                for s in info["symptoms"]:
                    st.write(f"- {s}")

            if "cause" in info:
                st.markdown("### 🔬 Cause")
                st.write(info["cause"])

            if "favourable_conditions" in info:
                st.markdown("### 🌦️ Conditions")
                st.write(info["favourable_conditions"])

            if "spread" in info:
                st.markdown("### 🌱 Spread")
                st.write(info["spread"])

            if "prevention" in info:
                st.markdown("### 🛡️ Prevention")
                for p in info["prevention"]:
                    st.write(f"- {p}")

            if "treatment_chemicals" in info:
                st.markdown("### 💊 Chemicals")
                for c in info["treatment_chemicals"]:
                    st.markdown(f"- [{c['name']}]({c['link']})")

            if "organic_options" in info:
                st.markdown("### 🌿 Organic")
                for o in info["organic_options"]:
                    st.write(f"- {o}")

            if "fertilizer_support" in info:
                st.markdown("### 🌾 Fertilizer")
                for f in info["fertilizer_support"]:
                    st.write(f"- {f}")

    os.remove(path)

else:
    st.info("Upload an image to begin.")


# ================= FERTILIZER =================

st.markdown("---")
if st.toggle("🌾 Fertilizer Calculator"):

    st.header("Fertilizer")

    crop = st.text_input("Crop")
    N = st.number_input("Soil N",0.0, value=50.0)
    P = st.number_input("Soil P",0.0, value=30.0)
    K = st.number_input("Soil K",0.0, value=40.0)
    ph = st.number_input("Soil pH",0.0,14.0,6.5)

    area = st.number_input("Area (ha)",0.01,1.0)
    fert = st.selectbox("Fertilizer",["Urea","DAP","MOP","NPK_20_20_20"])
    rain = st.number_input("Rainfall",0.0,100.0)

    if st.button("Calculate"):

        text,_ = recommend_fertilizer(crop,N,P,K,ph)
        st.write(text)

        amt = calculate_fertilizer_amount(
            crop,N,P,K,area,fert,rain
        )

        st.success(f"Apply {amt} kg")


# ================= PESTICIDE =================

st.markdown("---")
if st.toggle("🛡️ Pesticide Recommendation"):

    st.header("Pesticides")

    crop = st.text_input("Crop name")

    if st.button("Get Pesticides"):
        _,pests = recommend_fertilizer(crop,0,0,0,7)

        if pests:
            for p in pests:
                st.write(f"- {p}")
        else:
            st.warning("No data found")
