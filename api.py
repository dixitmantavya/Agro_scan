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
from utils.gradcam import make_gradcam_heatmap, overlay_heatmap

app = FastAPI(title="AgroScan API")

# ============ FIX: PROPER CORS CONFIGURATION ============
# Remove duplicate CORS middleware - keep only one!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (or specify ["http://localhost:3000", "https://yourdomain.com"])
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],  # Expose all headers to browser
)

# Optional API key (keep if you need it, remove if not)
API_KEY = "agroscan123"

# OpenWeatherMap API key
WEATHER_API_KEY = "your_openweathermap_api_key"


# ---------- HELPERS ----------

def check_api_key(x_api_key: str | None):
    """Optional API key validation"""
    if x_api_key is None:
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def parse_class_name(class_name: str):
    """Parse class name to extract crop and disease"""
    if "___" in class_name:
        crop, disease = class_name.split("___", 1)
    elif "_" in class_name:
        parts = class_name.split("_", 1)
        crop, disease = parts[0], parts[1] if len(parts) > 1 else "Unknown"
    else:
        crop, disease = class_name, "Unknown"
    return crop, disease.replace("_", " ")


def compute_severity(img_bgr):
    """Compute disease severity from image"""
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


def get_weather_info(city: str):
    """Get weather information from OpenWeatherMap API"""
    if not WEATHER_API_KEY or WEATHER_API_KEY == "your_openweathermap_api_key":
        # Return mock data if no API key
        return {
            "temperature": 28,
            "humidity": 75,
            "condition": "Partly Cloudy",
            "description": "Temperature: 28°C, Humidity: 75%, Condition: Partly Cloudy"
        }
    
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
        response = requests.get(url, timeout=5)
        
        if response.ok:
            data = response.json()
            temp = data['main']['temp']
            humidity = data['main']['humidity']
            condition = data['weather'][0]['main']
            
            return {
                "temperature": temp,
                "humidity": humidity,
                "condition": condition,
                "description": f"Temperature: {temp}°C, Humidity: {humidity}%, Condition: {condition}"
            }
    except Exception as e:
        print(f"Weather API error: {e}")
    
    # Fallback to mock data
    return {
        "temperature": 28,
        "humidity": 75,
        "condition": "Partly Cloudy",
        "description": "Temperature: 28°C, Humidity: 75%, Condition: Partly Cloudy"
    }


# ---------- 1) MAIN PREDICTION ENDPOINT ----------

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    x_api_key: str | None = Header(default=None)
):
    try:
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

        # GradCAM
        try:
            heatmap = make_gradcam_heatmap(img_input, model, "Conv_1")
            overlay = overlay_heatmap(
                cv2.cvtColor(resized, cv2.COLOR_BGR2RGB),
                heatmap
            )
            _, buffer = cv2.imencode(".png", overlay)
            gradcam_base64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
        except Exception as e:
            print("GradCAM failed:", e)
            gradcam_base64 = ""

        # ✅ ALWAYS GET GUIDE SAFELY
        guide = TOMATO_DISEASE_GUIDE.get(best_class, {})

        return {
            "class_name": best_class,
            "crop": crop,
            "disease": disease,
            "confidence": round(best_conf * 100, 2),

            "severity_percent": severity_percent,
            "severity_level": severity_level,

            # ✅ UI SAFE DATA
            "symptoms": guide.get("symptoms", []),
            "cause": guide.get("cause", ""),
            "prevention": guide.get("prevention", []),
            "treatment_chemicals": guide.get("treatment_chemicals", []),
            "organic_options": guide.get("organic_options", []),
            "fertilizer_support": guide.get("fertilizer_support", []),

            "pesticide_advice": DISEASE_INFO.get(best_class, ""),
            "detailed_guide": guide,

            "gradcam_image_base64": gradcam_base64,

            "top_predictions": [
                {
                    "class_name": cls,
                    "crop": parse_class_name(cls)[0],
                    "disease": parse_class_name(cls)[1],
                    "confidence": round(conf * 100, 2)
                }
                for cls, conf in results
            ]
        }

    except Exception as e:
        print(f"Error in /predict: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



# ---------- 2) DISEASE INFO ENDPOINT ----------

@app.get("/disease-info")
def disease_info(name: str = Query(..., description="Disease class name")):
    """
    Get detailed information about a specific disease
    """
    guide = TOMATO_DISEASE_GUIDE.get(name)
    
    if guide:
        return {
            "name": name,
            "symptoms": guide.get("symptoms", []),
            "cause": guide.get("cause", ""),
            "prevention": guide.get("prevention", []),
            "treatment_chemicals": guide.get("treatment_chemicals", []),
            "organic_options": guide.get("organic_options", []),
            "fertilizer_support": guide.get("fertilizer_support", []),
            "short_advice": DISEASE_INFO.get(name, ""),
            "detailed_guide": guide
        }
    else:
        return {
            "name": name,
            "short_advice": DISEASE_INFO.get(name, "Information not available"),
            "detailed_guide": None
        }


# ---------- 3) WEATHER ENDPOINT ----------

@app.get("/weather")
def weather(city: str = Query(..., description="City name")):
    """
    Get weather information for disease risk assessment
    """
    weather_data = get_weather_info(city)
    
    return {
        "city": city,
        "weather_info": weather_data["description"],
        "temperature": weather_data["temperature"],
        "humidity": weather_data["humidity"],
        "condition": weather_data["condition"]
    }


# ---------- 4) FERTILIZER RECOMMENDATION ----------

@app.post("/fertilizer/recommend")
def fertilizer_recommend(data: dict):
    """
    Fertilizer recommendation based on soil parameters
    Expected JSON body:
    {
      "crop": "Tomato",
      "soil_N": 40,
      "soil_P": 20,
      "soil_K": 30,
      "soil_ph": 6.5,
      "area": 1.0,  # hectares
      "fertilizerType": "NPK"
    }
    """
    try:
        crop = data.get("crop", "Unknown")
        soil_N = data.get("soil_N", 0)
        soil_P = data.get("soil_P", 0)
        soil_K = data.get("soil_K", 0)
        soil_ph = data.get("soil_ph", 7.0)
        area = data.get("area", 1.0)
        fert_type = data.get("fertilizerType", "NPK")

        # Simple recommendation logic
        recommendations = []
        
        if soil_N < 40:
            recommendations.append("Apply Urea or Ammonium Sulfate for Nitrogen")
        if soil_P < 20:
            recommendations.append("Apply DAP or Single Super Phosphate for Phosphorus")
        if soil_K < 30:
            recommendations.append("Apply MOP (Muriate of Potash) for Potassium")
        
        if not recommendations:
            message = f"Soil nutrient levels are adequate for {crop}. Apply balanced NPK as maintenance."
        else:
            message = f"For {crop}: " + "; ".join(recommendations)
        
        # pH adjustment
        ph_note = ""
        if soil_ph < 6.0:
            ph_note = "Soil is acidic. Add agricultural lime to raise pH."
        elif soil_ph > 7.5:
            ph_note = "Soil is alkaline. Add sulfur or organic matter to lower pH."
        else:
            ph_note = "Soil pH is optimal."
        
        # Calculate approximate amount
        amount = round(area * 45, 2)  # kg per hectare
        
        return {
            "recommendation": message,
            "message": message,
            "amount": amount,
            "phNote": ph_note,
            "suggestedPesticides": [
                "Imidacloprid (for aphids)",
                "Neem extract (organic)",
                "Spinosad (for caterpillars)"
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")


# ---------- 5) PESTICIDE RECOMMENDATION ----------

@app.post("/pesticide/recommend")
def pesticide_recommend(data: dict):
    """
    Pesticide recommendation based on disease and severity
    Expected JSON body:
    {
      "crop": "Tomato",
      "disease": "Early Blight",
      "severity": "Moderate",
      "area": 1.0,
      "applicationStage": "vegetative"
    }
    """
    try:
        crop = data.get("crop", "Unknown")
        disease = data.get("disease", "Unknown")
        severity = data.get("severity", "Moderate")
        area = data.get("area", 1.0)
        stage = data.get("applicationStage", "vegetative")
        
        # Construct class name for lookup
        class_name = data.get("class_name")
        if not class_name:
            # Try to construct from crop and disease
            disease_key = disease.replace(" ", "_")
            class_name = f"{crop}___{disease_key}"
        
        # Get advice from disease info
        advice = DISEASE_INFO.get(class_name, "")
        guide = TOMATO_DISEASE_GUIDE.get(class_name, {})
        
        # Default pesticide recommendation
        pesticide = "Mancozeb 75% WP"
        chemicals = guide.get("treatment_chemicals", [])
        if chemicals:
            pesticide = chemicals[0].get("name", pesticide)
        
        # Calculate dosage
        dosage = f"{round(area * 2.5, 2)} kg for {area} hectares"
        
        # Frequency based on severity
        if severity == "Severe":
            frequency = "Apply every 5-7 days until symptoms reduce"
        elif severity == "Moderate":
            frequency = "Apply every 7-10 days"
        else:
            frequency = "Apply every 10-14 days as preventive measure"
        
        return {
            "pesticide": pesticide,
            "dosage": dosage,
            "applicationMethod": "Foliar spray with 500-600 liters of water per hectare",
            "frequency": frequency,
            "precautions": [
                "Wear protective clothing and gloves during application",
                "Do not spray during flowering to protect pollinators",
                "Maintain pre-harvest interval of 7-14 days",
                "Avoid spraying during rain or high wind",
                "Store in cool, dry place away from food"
            ],
            "alternatives": [
                {"name": chemical.get("name", ""), "type": "Chemical"}
                for chemical in chemicals[1:4]
            ] if len(chemicals) > 1 else [
                {"name": "Neem Oil", "type": "Organic"},
                {"name": "Copper Hydroxide", "type": "Chemical"}
            ],
            "pesticideAdvice": advice,
            "detailedGuide": guide
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")


# ---------- 6) HEALTH CHECK ----------

@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "AgroScan API is running",
        "version": "2.0",
        "cors_enabled": True
    }


@app.get("/health")
def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "endpoints": {
            "predict": "/predict",
            "disease_info": "/disease-info",
            "weather": "/weather",
            "fertilizer": "/fertilizer/recommend",
            "pesticide": "/pesticide/recommend"
        },
        "cors_configured": True
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
