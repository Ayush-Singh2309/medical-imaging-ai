import os
import base64
import io
from io import BytesIO
import requests
import pydicom
import numpy as np
import tensorflow as tf
import keras
from pathlib import Path
from PIL import Image
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from dotenv import load_dotenv
from keras.applications.densenet import preprocess_input

from medical_imaging_ai.constants import CLASSES, IMAGE_SIZE
from medical_imaging_ai.paths import PROJECT_ROOT
from medical_imaging_ai.explainability import generate_gradcam

from medical_imaging_ai.storage import save_study_images

from medical_imaging_ai.database import (
    init_db,
    save_analysis,
    update_study_image_paths,
    get_all_studies,
    get_study_by_id
)
load_dotenv()

ORTHANC_URL = os.getenv("ORTHANC_URL", "http://localhost:8042")
ORTHANC_USERNAME = os.getenv("ORTHANC_USERNAME", "orthanc")
ORTHANC_PASSWORD = os.getenv("ORTHANC_PASSWORD")

if not ORTHANC_PASSWORD:
    raise RuntimeError("ORTHANC_PASSWORD environment variable is missing")

init_db()

app = FastAPI(
    title="Chest X-ray AI API",
    description="Educational multi-label chest X-ray classification API",
    version="1.0.0"
)


MODEL_PATH = PROJECT_ROOT / "models" / "best_model.keras"

model = None


@app.on_event("startup")
def load_trained_model():
    global model

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found: {MODEL_PATH}")

    model = keras.models.load_model(MODEL_PATH)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


def preprocess_uploaded_image(image_bytes):
    """
    Decode and preprocess an uploaded X-ray.
    Uses the same preprocessing as model training.
    """

    try:
        image = tf.io.decode_image(
            image_bytes,
            channels=1,
            expand_animations=False
        )

        image = tf.image.resize(
            image,
            IMAGE_SIZE
        )

        image = tf.image.grayscale_to_rgb(image)

        image = tf.cast(image, tf.float32)

        original_image = image.numpy().astype("uint8")

        model_image = preprocess_input(image)

        model_tensor = tf.expand_dims(
            model_image,
            axis=0
        )

        return model_tensor, original_image

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to decode image: {exc}"
        )


def create_gradcam_overlay(original_image, heatmap, alpha=0.4):
    """
    Create a PNG overlay and return it as base64.
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.imshow(original_image, cmap="gray")
    ax.imshow(heatmap, cmap="jet", alpha=alpha)
    ax.axis("off")

    buffer = io.BytesIO()

    fig.savefig(
        buffer,
        format="png",
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close(fig)

    buffer.seek(0)

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


@app.post("/analyze")
async def analyze_xray(
    file: UploadFile = File(...),
    disease: str | None = Form(default=None)
):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded."
        )

    if file.content_type not in {
        "image/png",
        "image/jpeg"
    }:
        raise HTTPException(
            status_code=400,
            detail="Only PNG and JPEG images are supported."
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    try:
        # Decode and preprocess
        image_tensor, original_image = preprocess_uploaded_image(
            image_bytes
        )

        # Predict all 14 diseases
        predictions = model.predict(
            image_tensor,
            verbose=0
        )[0]

        prediction_results = [
            {
                "disease": name,
                "probability": float(probability)
            }
            for name, probability in zip(CLASSES, predictions)
        ]

        prediction_results.sort(
            key=lambda item: item["probability"],
            reverse=True
        )

        # Select disease for Grad-CAM
        if disease is None:
            class_index = int(np.argmax(predictions))
        else:
            if disease not in CLASSES:
                raise HTTPException(
                    status_code=400,
                    detail="Unknown disease name."
                )

            class_index = CLASSES.index(disease)

        heatmap = generate_gradcam(
            model,
            image_tensor,
            class_index
        )

        overlay = create_gradcam_overlay(
            original_image,
            heatmap
        )

        
        # ----------------------------------------------
        # Save study metadata and predictions
        # ----------------------------------------------

        study_id = save_analysis(
            image_filename=file.filename or "unknown",
            predictions=prediction_results,
            explained_disease=CLASSES[class_index],
            explained_probability=float(
                predictions[class_index]
            )
        )

        # ----------------------------------------------
        # Save original X-ray and Grad-CAM
        # ----------------------------------------------

        original_path, gradcam_path = save_study_images(
            study_id=study_id,
            original_image_bytes=image_bytes,
            gradcam_base64=overlay,
            original_filename=file.filename or "image.png"
        )

        # Store image locations in SQLite
        update_study_image_paths(
            study_id=study_id,
            image_path=str(original_path),
            gradcam_path=str(gradcam_path)
        )

        # ----------------------------------------------
        # Return API response
        # ----------------------------------------------

        return JSONResponse({
            "study_id": study_id,
            "predictions": prediction_results,
            "explained_disease": CLASSES[class_index],
            "explained_probability": float(
                predictions[class_index]
            ),
            "gradcam_overlay": overlay,
            "disclaimer": (
                "For educational and research purposes only. "
                "Not for clinical diagnosis or treatment decisions."
            )
        })


    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {exc}"
        )



# --------------------------------------------------
# Retrieve all studies
# --------------------------------------------------

@app.get("/studies")
def list_studies():

    try:
        studies = get_all_studies()

        return {
            "total": len(studies),
            "studies": studies
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to retrieve studies: {exc}"
        )


# --------------------------------------------------
# Retrieve one study
# --------------------------------------------------

@app.get("/studies/{study_id}")
def retrieve_study(study_id: int):

    try:
        study = get_study_by_id(study_id)

        if study is None:
            raise HTTPException(
                status_code=404,
                detail=f"Study {study_id} not found."
            )

        return study

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to retrieve study: {exc}"
        )

# --------------------------------------------------
# Retrieve original X-ray
# --------------------------------------------------

@app.get("/studies/{study_id}/image")
def get_study_image(study_id: int):

    study = get_study_by_id(study_id)

    if study is None:
        raise HTTPException(
            status_code=404,
            detail="Study not found."
        )

    image_path = study.get("image_path")

    if not image_path:
        raise HTTPException(
            status_code=404,
            detail="Original image is not available."
        )

    path = Path(image_path)

    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Image file not found."
        )

    return FileResponse(path)


# --------------------------------------------------
# Retrieve Grad-CAM overlay
# --------------------------------------------------

@app.get("/studies/{study_id}/gradcam")
def get_study_gradcam(study_id: int):

    study = get_study_by_id(study_id)

    if study is None:
        raise HTTPException(
            status_code=404,
            detail="Study not found."
        )

    gradcam_path = study.get("gradcam_path")

    if not gradcam_path:
        raise HTTPException(
            status_code=404,
            detail="Grad-CAM image is not available."
        )

    path = Path(gradcam_path)

    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Grad-CAM file not found."
        )

    return FileResponse(
        path,
        media_type="image/png"
    )


def retrieve_dicom_instance(instance_id: str):
    url = f"{ORTHANC_URL}/instances/{instance_id}/file"

    response = requests.get(
        url,
        auth=(ORTHANC_USERNAME, ORTHANC_PASSWORD),
        timeout=30,
    )

    if response.status_code == 404:
        raise HTTPException(status_code=404, detail="DICOM instance not found")

    response.raise_for_status()

    dicom = pydicom.dcmread(BytesIO(response.content))

    return dicom

@app.get("/dicom/instances/{instance_id}")
def get_dicom_instance(instance_id: str):
    dicom = retrieve_dicom_instance(instance_id)

    return {
        "patient_id": str(dicom.PatientID),
        "modality": str(dicom.Modality),
        "study_uid": str(dicom.StudyInstanceUID),
        "rows": int(dicom.Rows),
        "columns": int(dicom.Columns),
        "pixel_shape": list(dicom.pixel_array.shape),
    }

def preprocess_dicom_image(dicom):
    # Extract pixel data
    image = dicom.pixel_array

    # This first version supports our synthetic 8-bit grayscale DICOM
    if image.ndim != 2 or image.dtype != np.uint8:
        raise HTTPException(
            status_code=400,
            detail="Expected an 8-bit grayscale DICOM image."
        )

    # Convert to TensorFlow tensor
    image = tf.convert_to_tensor(image, dtype=tf.float32)

    # Resize to model input dimensions
    image = tf.image.resize(image[..., tf.newaxis], IMAGE_SIZE)

    # Convert grayscale to RGB
    image = tf.image.grayscale_to_rgb(image)

    # Apply DenseNet preprocessing
    image = preprocess_input(image)

    # Add batch dimension
    image = tf.expand_dims(image, axis=0)

    return image

@app.post("/dicom/analyze/{instance_id}")
def analyze_dicom(instance_id: str):
    # Retrieve DICOM from Orthanc
    dicom = retrieve_dicom_instance(instance_id)

    # Preprocess image
    image = preprocess_dicom_image(dicom)

    # Run model prediction
    predictions = model.predict(image, verbose=0)[0]

    # Format predictions
    prediction_results = [
        {
            "disease": disease,
            "probability": float(score)
        }
        for disease, score in zip(CLASSES, predictions)
    ]

    # Sort by score
    prediction_results.sort(
        key=lambda x: x["probability"],
        reverse=True
    )

    return {
        "patient_id": str(dicom.PatientID),
        "study_uid": str(dicom.StudyInstanceUID),
        "predictions": prediction_results,
        "disclaimer": "For educational and research purposes only. Not for clinical use."
    }


@app.post("/dicom/upload")
def upload_dicom(file: UploadFile):
    file_bytes = file.file.read()

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    # Validate that the uploaded file is readable as DICOM
    try:
        pydicom.dcmread(BytesIO(file_bytes))
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid DICOM file"
        )

    try:
        response = requests.post(
            f"{ORTHANC_URL}/instances",
            auth=(ORTHANC_USERNAME, ORTHANC_PASSWORD),
            headers={"Content-Type": "application/dicom"},
            data=file_bytes,
            timeout=60,
        )
        response.raise_for_status()

    except requests.RequestException:
        raise HTTPException(
            status_code=502,
            detail="Failed to upload DICOM to Orthanc"
        )

    result = response.json()

    return {
        "instance_id": result["ID"],
        "message": "DICOM uploaded successfully"
    }