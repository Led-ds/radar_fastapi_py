from fastapi import APIRouter, UploadFile, HTTPException
from PIL import Image
import numpy as np
from app.models.model_loader import get_model

router = APIRouter()
model = get_model()

@router.post("/analyze/")
async def analyze(file: UploadFile):
    try:
        # Processar a imagem enviada
        img = Image.open(file.file).convert("L").resize((64, 64))
        img_array = np.array(img).reshape(1, 64, 64, 1) / 255.0

        # Fazer a predição
        prediction = model.predict(img_array)
        result = "Stable" if prediction[0][0] > 0.5 else "Unstable"

        return {"result": result, "confidence": float(prediction[0][0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
