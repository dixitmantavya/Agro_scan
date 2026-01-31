from fastapi import FastAPI, UploadFile, File, Header, HTTPException
import numpy as np
import cv2
from PIL import Image
import io
import base64

from model.predict import predict_disease, model
from utils.disease_info import DISEASE_INFO
from utils.tomato_disease_guide import TOMATO_DISEASE_GUIDE
from utils.gradcam import make_gradcam_heatmap, overlay_heatmap

app = FastAPI(title="AgroScan API")

# optional API key (only checked if provided)
API_KEY = "agroscan123"


# ---------- helpers ----------

def check_api_key(x_api_key: str | None):
    """
    If client sends x-api-key header, validate it.
    If no key is sent, allow request (useful for testing/UI dev).
    """
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

    # simple yellow/brown infection detection
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


# ---------- main prediction endpoint ----------

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    x_api_key: str | None = Header(default=None)
):
    # validate key only if user provided one
    check_api_key(x_api_key)

    contents = await file.read()

    # read image
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # resize for model
    resized = cv2.resize(img_bgr, (224, 224))
    img_input = np.expand_dims(resized / 255.0, axis=0)

    # prediction
    results = predict_disease(img_input, top_k=3)
    best_class, best_conf = results[0]

    crop, disease = parse_class_name(best_class)

    # severity estimation
    severity_percent, severity_level = compute_severity(resized)

    # Grad-CAM
    last_conv_layer = "Conv_1"  # MobileNetV2 last conv layer name
    heatmap = make_gradcam_heatmap(img_input, model, last_conv_layer)
    overlay = overlay_heatmap(
        cv2.cvtColor(resized, cv2.COLOR_BGR2RGB),
        heatmap
    )

    # convert Grad-CAM image to base64
    _, buffer = cv2.imencode(".png", overlay)
    gradcam_base64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

    return {
        "crop": crop,
        "disease": disease,
        "confidence": round(best_conf * 100, 2),

        "top_predictions": [
            {"class": c, "confidence": round(conf * 100, 2)}
            for c, conf in results
        ],

        "severity_percent": severity_percent,
        "severity_level": severity_level,

        "pesticide_advice": DISEASE_INFO.get(best_class),
        "detailed_guide": TOMATO_DISEASE_GUIDE.get(best_class),

        # show in frontend as:
        # <img src="data:image/png;base64,{{gradcam_image_base64}}">
        "gradcam_image_base64": gradcam_base64
    }
