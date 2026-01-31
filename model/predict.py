import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "plant_disease_model.h5")
CLASS_PATH = os.path.join(BASE_DIR, "class_indices.json")

model = tf.keras.models.load_model(MODEL_PATH) #type:ignore

with open(CLASS_PATH, "r") as f:
    class_indices = json.load(f)

INDEX_TO_CLASS = {v: k for k, v in class_indices.items()}

def predict_disease(input_data, top_k=3):
    """
    input_data can be:
    - image path (str)
    - preprocessed numpy array (1, 224, 224, 3)
    """

    if isinstance(input_data, str):
        img = Image.open(input_data).convert("RGB").resize((224, 224))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)
    else:
        img = input_data

    preds = model.predict(img)[0]
    top_indices = preds.argsort()[-top_k:][::-1]

    return [(INDEX_TO_CLASS[i], float(preds[i])) for i in top_indices]
