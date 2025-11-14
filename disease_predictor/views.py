from django.shortcuts import render
from disease_predictor.Model_Files.plant_disease_predictor import predict_plant_disease
import tempfile
import numpy as np
import cv2
from disease_predictor.Model_Files.plant_disease_predictor import model, pca
from django.core.files.storage import FileSystemStorage


def homepage(request):
    prediction_result = None
    confidence = None

    if request.method == "POST" and request.FILES.get("leaf_image"):
        leaf_image = request.FILES["leaf_image"]

        # Save uploaded image to MEDIA folder
        fs = FileSystemStorage()
        saved_name = fs.save(leaf_image.name, leaf_image)
        uploaded_url = fs.url(saved_name)

        # Save uploaded image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            for chunk in leaf_image.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name

        # Predict using your function
        prediction_result = predict_plant_disease(tmp_path)

        # Compute confidence score CORRECTLY
        img = cv2.imread(tmp_path)
        img_resized = cv2.resize(img, (128, 128))
        img_flat = img_resized.reshape(1, -1) / 255.0
        img_pca = pca.transform(img_flat)

        proba = model.predict_proba(img_pca)[0]
        confidence = float(np.max(proba) * 100)

        print("[DEBUG] Prediction:", prediction_result)
        if prediction_result is None:
            print("[DEBUG] Prediction result is None")
        else:
            prediction_result = prediction_result['prediction']
            
        print("[DEBUG] Confidence:", confidence)

    context = {
        "prediction_result": prediction_result,
        "confidence": round(confidence, 2) if confidence else None,
        "uploaded_url":uploaded_url,
    }

    return render(request, 'disease_predictor/homepage.html', context)
