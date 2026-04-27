import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.applications.efficientnet import preprocess_input
import tensorflow as tf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    MODEL = tf.keras.models.load_model(
        'efficientnetb0_final.keras',
        compile=False
    )
    CLASSES = ['Lung_Opacity', 'Normal', 'Viral_Pneumonia']
    print(f"Model loaded. Classes: {CLASSES}")
    print(f"Input shape: {MODEL.input_shape}")
except Exception as e:
    print(f"FATAL: Model failed to load — {e}")
    MODEL = None


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Replicates training preprocessing exactly:
    - Convert to RGB
    - Resize to 224x224
    - Normalize pixels to [0, 1]
    - Add batch dimension
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224), Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)


@app.get("/")
def root():
    return {"status": "running", "model_loaded": MODEL is not None}


@app.get("/health")
def health():
    return {
        "status"      : "healthy" if MODEL is not None else "model_not_loaded",
        "classes"     : CLASSES if MODEL is not None else [],
        "input_shape" : str(MODEL.input_shape) if MODEL is not None else None
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # in case model is not loading at startup 
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check server logs."
        )

    # Check extension as fallback.
    allowed_types = {"image/jpeg", "image/png", "image/jpg"}
    allowed_exts  = {".jpg", ".jpeg", ".png"}
    filename_ext  = "." + file.filename.rsplit(".", 1)[-1].lower() \
                    if "." in file.filename else ""

    if (file.content_type not in allowed_types
            and filename_ext not in allowed_exts):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}"
        )

    contents = await file.read()
    
    # corrupt or non-image files 
    try:
        processed_img = preprocess_image(contents)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Could not process image: {str(e)}"
        )

    predictions = MODEL(processed_img, training=False).numpy()
    pred_idx    = int(np.argmax(predictions[0]))
    confidence  = float(predictions[0][pred_idx])

    # Confidence threshold — flag uncertain predictions
    THRESHOLD   = 0.70
    is_uncertain = confidence < THRESHOLD

    return {
        "prediction"  : CLASSES[pred_idx],
        "confidence"  : round(confidence, 4),
        "uncertain"   : is_uncertain,
        "all_scores"  : {
            CLASSES[i]: round(float(predictions[0][i]), 4)
            for i in range(len(CLASSES))
        }
    }


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )




