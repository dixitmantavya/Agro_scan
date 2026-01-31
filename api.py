from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Query
import numpy as np
import cv2
from PIL import Image
import io
import base64

from model.predict import predict_disease, model
from utils.disease_info import DISEASE_INFO
from utils.tomato_disease_guide import TOMATO_DISEASE_GUIDE
from utils.gradcam import make_gradcam_heatmap, overlay_heatmap
from utils.weather import get_weather
from utils.fertilizer import recommend_fertilizer

app = FastAPI(title="AgroScan API")

API_KEY = "agroscan123"


# ---------- helpers ----------

def check_api_key(x_api_key: str | None):
    if x_api_key is None:
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def parse_class_name(class_name: str):
    if "___" in class_name:
        crop, disease = class_name.split("___", 1)
    else:
        crop, disease = class_name, "Unknown"
    return crop, disease.replace("_", " ")


def compute_severity(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_disease = np.array([5, 50, 50])
    upper_disease = np.array([35, 255, 255])

    mask = cv2.inRange(hsv, lower_disease, upper_disease)
    infected = np.count_nonzero(mask)
    total = mask.size
    pct = (infected / total) * 100 if total else 0

    if pct < 5:
        level = "Very Low"
    elif pct < 20:
        level = "Mild"
    elif pct < 50:
        level = "Moderate"
    else:
        level = "Severe"

    return round(pct, 2), level


# ---------- 1) main prediction ----------

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    x_api_key: str | None = Header(default=None)
):
    check_api_key(x_api_key)

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    resized = cv2.resize(img_bgr, (224, 224))
    img_input = np.expand_dims(resized / 255.0, axis=0)

    results = predict_disease(img_input, top_k=3)
    best_class, best_conf = results[0]

    crop, disease = parse_class_name(best_class)
    severity_percent, severity_level = compute_severity(resized)

    heatmap = make_gradcam_heatmap(img_input, model, "Conv_1")
    overlay = overlay_heatmap(
        cv2.cvtColor(resized, cv2.COLOR_BGR2RGB),
        heatmap
    )

    _, buffer = cv2.imencode(".png", overlay)
    gradcam_base64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

    return {
        "class_name": best_class,
        "crop": crop,
        "disease": disease,
        "confidence": round(best_conf * 100, 2),
        "severity_percent": severity_percent,
        "severity_level": severity_level,
        "pesticide_advice": DISEASE_INFO.get(best_class),
        "detailed_guide": TOMATO_DISEASE_GUIDE.get(best_class),
        "gradcam_image_base64": gradcam_base64
    }


# ---------- 2) disease info ----------

@app.get("/disease-info")
def disease_info(name: str = Query(...)):
    return {
        "name": name,
        "short_advice": DISEASE_INFO.get(name),
        "detailed_guide": TOMATO_DISEASE_GUIDE.get(name)
    }


# ---------- 3) weather ----------

@app.get("/weather")
def weather(city: str = Query(...)):
    # uses your existing weather util
    info = get_weather(city)
    return {
        "city": city,
        "weather_info": info
    }


# ---------- 4) fertilizer recommendation ----------

@app.post("/fertilizer/recommend")
def fertilizer_recommend(data: dict):
    """
    expects JSON body:
    {
      "crop": "Tomato",
      "soil_N": 40,
      "soil_P": 20,
      "soil_K": 30,
      "soil_ph": 6.5
    }
    """
    crop = data["crop"]
    soil_N = data["soil_N"]
    soil_P = data["soil_P"]
    soil_K = data["soil_K"]
    soil_ph = data["soil_ph"]

    rec, pests = recommend_fertilizer(crop, soil_N, soil_P, soil_K, soil_ph)

    return {
        "recommendation": rec,
        "related_pesticides": pests
    }


# ---------- 5) simple pesticide endpoint ----------

@app.post("/pesticide/recommend")
def pesticide_recommend(data: dict):
    """
    expects JSON body:
    { "class_name": "Tomato___Early_blight" }
    """
    class_name = data["class_name"]
    return {
        "pesticide_advice": DISEASE_INFO.get(class_name),
        "detailed_guide": TOMATO_DISEASE_GUIDE.get(class_name)
    }
