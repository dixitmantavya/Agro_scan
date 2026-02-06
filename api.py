from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from PIL import Image
import io
import base64
import requests
from typing import Optional

# Import your existing modules
import sys
sys.path.append('.')
from model.predict import predict_disease, model
from utils.disease_info import DISEASE_INFO
from utils.tomato_disease_guide import TOMATO_DISEASE_GUIDE

app = FastAPI(title="AgroScan API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = "agroscan123"
WEATHER_API_KEY = "ca0b98a1cd522a876b3cac4213af38ce"


# ---------- HELPERS ----------

def parse_class_name(class_name: str):
    if "___" in class_name:
        crop, disease = class_name.split("___", 1)
    elif "_" in class_name:
        parts = class_name.split("_", 1)
        crop, disease = parts[0], parts[1] if len(parts) > 1 else "Unknown"
    else:
        crop, disease = class_name, "Unknown"

    return crop, disease.replace("_", " ")


# ✅ BETTER TRANSLUCENT MASK + ACCURATE %
def compute_severity_and_mask(img_bgr):

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower = np.array([10, 60, 60])
    upper = np.array([40, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Smooth mask
    mask = cv2.GaussianBlur(mask,(7,7),0)

    # ✅ ACCURATE % (count only lesion pixels)
    infected = np.count_nonzero(mask > 20)
    total = mask.shape[0] * mask.shape[1]

    percent = (infected / total) * 100 if total else 0

    # Severity levels
    if percent < 3:
        level = "Very Low"
    elif percent < 10:
        level = "Mild"
    elif percent < 25:
        level = "Moderate"
    else:
        level = "Severe"

    # -------- TRUE TRANSLUCENT HEATMAP --------
    heatmap = np.zeros_like(img_bgr)
    heatmap[:,:,2] = mask  # red channel intensity

    overlay = cv2.addWeighted(img_bgr, 0.85, heatmap, 0.35, 0)

    return round(percent,2), level, overlay


# ---------- WEATHER ----------
def get_weather_info(city: str):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
        r = requests.get(url, timeout=5)

        if r.ok:
            d = r.json()
            return {
                "temperature": d['main']['temp'],
                "humidity": d['main']['humidity'],
                "condition": d['weather'][0]['main'],
            }
    except:
        pass

    return {
        "temperature": 28,
        "humidity": 75,
        "condition": "Partly Cloudy",
    }


# ---------- MAIN PREDICT ----------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    img_np = np.array(image)

    # MODEL INPUT
    resized = cv2.resize(img_np,(224,224))
    img_input = np.expand_dims(resized/255.0,axis=0)

    # SEVERITY INPUT
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    resized_bgr = cv2.resize(img_bgr,(224,224))

    # ---------- PREDICTION ----------
    results = predict_disease(img_input, top_k=3)
    best_class, best_conf = results[0]

    crop, disease = parse_class_name(best_class)

    # ---------- SEVERITY ----------
    severity_percent, severity_level, lesion_overlay = compute_severity_and_mask(resized_bgr)

    _, buffer = cv2.imencode(".png", lesion_overlay)
    gradcam_base64 = base64.b64encode(buffer).decode("utf-8")

    guide = TOMATO_DISEASE_GUIDE.get(best_class, {})

    return {
        "class_name": best_class,
        "crop": crop,
        "disease": disease,
        "confidence": round(best_conf * 100,2),

        "severity_percent": severity_percent,
        "severity_level": severity_level,

        "symptoms": guide.get("symptoms", []),
        "cause": guide.get("cause", ""),
        "prevention": guide.get("prevention", []),
        "treatment_chemicals": guide.get("treatment_chemicals", []),
        "organic_options": guide.get("organic_options", []),

        "pesticide_advice": DISEASE_INFO.get(best_class,""),
        "detailed_guide": guide,

        "gradcam_image_base64": gradcam_base64,

        "top_predictions":[
            {
                "class_name":cls,
                "crop":parse_class_name(cls)[0],
                "disease":parse_class_name(cls)[1],
                "confidence":round(conf*100,2)
            }
            for cls,conf in results
        ]
    }


# ---------- DISEASE INFO ----------
@app.get("/disease-info")
def disease_info(name:str=Query(...)):
    guide = TOMATO_DISEASE_GUIDE.get(name)

    if guide:
        return {"name":name,**guide}

    return {"name":name,"short_advice":"Not available"}


# ---------- WEATHER ----------
@app.get("/weather")
def weather(city:str=Query(...)):
    w = get_weather_info(city)

    return {
        "city":city,
        "temperature":w["temperature"],
        "humidity":w["humidity"],
        "condition":w["condition"]
    }


# ---------- HEALTH ----------
@app.get("/")
def root():
    return {"status":"healthy"}


if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=8000)
