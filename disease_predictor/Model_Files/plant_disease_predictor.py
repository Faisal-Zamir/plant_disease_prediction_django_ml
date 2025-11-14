import os
import cv2
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "plant_diseases_model.pkl")
PCA_PATH = os.path.join(BASE_DIR, "pca_object.pkl")

model = joblib.load(MODEL_PATH)
pca = joblib.load(PCA_PATH)

def predict_plant_disease(image_path):
    """Return predicted class + confidence score."""
    
    # Step 1
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Step 2
    img_resized = cv2.resize(img, (128, 128))

    # Step 3
    img_flat = img_resized.reshape(1, -1) / 255.0

    # Step 4
    img_pca = pca.transform(img_flat)

    # Step 5
    prediction = model.predict(img_pca)[0]

    # Step 6 — Confidence score
    proba = model.predict_proba(img_pca)[0]
    
    confidence = float(np.max(proba) * 100)  # % score

    # Step 7 — label
    label = "Healthy" if prediction == 0 else "Diseased"

    return {
        "prediction": label,
        "confidence": round(confidence, 2),
    }
