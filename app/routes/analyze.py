from fastapi import APIRouter, UploadFile, HTTPException, status
from PIL import Image
import numpy as np
import logging
import time
from app.models.model_loader import get_model

# Configuração do logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
model = get_model()

def preprocess_image(file: UploadFile, target_size=(64, 64), grayscale=False):
    """ Processa a imagem e converte para o formato adequado ao modelo """
    try:
        img = Image.open(file.file)

        # Redimensiona a imagem para o tamanho desejado
        img = img.resize(target_size)

        if grayscale:
            img = img.convert("L")  # Converte para escala de cinza
            img_array = np.array(img).reshape(1, target_size[0], target_size[1], 1)
        else:
            img = img.convert("RGB")  # Mantém a imagem colorida
            img_array = np.array(img).reshape(1, target_size[0], target_size[1], 3)

        img_array = img_array / 255.0  # Normalização
        return img_array
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Erro no processamento da imagem: {str(e)}")


@router.post("/analyze/")
async def analyze(file: UploadFile, grayscale: bool = False, size: int = 64):
    start_time = time.time()  # Tempo inicial para métricas de desempenho
    try:
        # Garantir que a entrada seja válida
        if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Formato de arquivo não suportado. Use PNG, JPG ou JPEG.")

        # Ajuste dinâmico de tamanho e conversão de cor
        img_array = preprocess_image(file, target_size=(size, size), grayscale=grayscale)

        # Fazer a predição
        prediction = model.predict(img_array)
        result = "Stable" if prediction[0][0] > 0.5 else "Unstable"
        confidence = float(prediction[0][0])

        # Log das métricas
        elapsed_time = time.time() - start_time
        logger.info(f"Imagem analisada: {file.filename}, Resultado: {result}, Confiança: {confidence:.4f}, Tempo de execução: {elapsed_time:.4f}s")

        return {"result": result, "confidence": confidence, "processing_time": elapsed_time}
    except HTTPException as http_err:
        raise http_err  # Erros já tratados anteriormente
    except Exception as e:
        logger.error(f"Erro interno: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Erro interno no processamento da imagem.")

@router.get("/metrics/")
def get_model_metrics():
    """ Retorna métricas de desempenho da rede neural """
    try:
        metrics = model.evaluate()  # Supondo que o modelo tenha essa função
        return {"loss": float(metrics[0]), "accuracy": float(metrics[1])}
    except Exception as e:
        logger.error(f"Erro ao obter métricas: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Erro ao obter métricas do modelo.")
