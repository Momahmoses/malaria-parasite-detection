"""
FastAPI inference server for malaria detection.
Accepts blood smear image → returns diagnosis in <50ms.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import io
import logging
import time

logger = logging.getLogger(__name__)
app = FastAPI(title="MalariaScope AI", version="1.0.0",
              description="Automated malaria parasite detection from blood smear images")

detector = None


@app.on_event("startup")
async def load_model():
    global detector
    from src.models.detector import MalariaDetector
    detector = MalariaDetector(model_path=os.getenv("MODEL_PATH"))
    logger.info("Malaria detector initialized")


import os


@app.post("/analyze")
async def analyze_slide(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        pil_image = pil_image.resize((640, 640))
        image_array = np.array(pil_image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot process image: {e}")

    t0 = time.time()
    if detector:
        result = detector.analyze_slide(image_array)
    else:
        from src.models.detector import MalariaDetector
        d = MalariaDetector()
        result = d.analyze_slide(image_array)

    result["processing_time_ms"] = round((time.time() - t0) * 1000, 1)
    return JSONResponse(content=result)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": detector is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
