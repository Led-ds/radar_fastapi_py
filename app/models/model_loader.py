import os
from keras._tf_keras.keras.models import load_model

def get_model(model_path="app/models/models/meu_modelo.h5"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ O arquivo do modelo não foi encontrado: {model_path}")

    print(f"✅ Carregando modelo: {model_path}")
    return load_model(model_path)