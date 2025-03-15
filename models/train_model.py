import os
import numpy as np
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras._tf_keras.keras.optimizers import Adam

# Definir parâmetros configuráveis
IMG_SIZE = 64  # Tamanho dinâmico da imagem
COLOR_MODE = "rgb"  # "grayscale" para 1 canal, "rgb" para 3 canais
CHANNELS = 1 if COLOR_MODE == "grayscale" else 3
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, CHANNELS)

# Criar modelo CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=INPUT_SHAPE),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compilar o modelo
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Gerar dados fictícios de imagens com a configuração de canais correta
X_train = np.random.random((100, IMG_SIZE, IMG_SIZE, CHANNELS))  # 100 imagens de treino
y_train = np.random.randint(2, size=(100, 1))  # Labels binárias (Stable/Unstable)

# Treinar o modelo
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Criar diretório se não existir
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "meu_modelo.h5")

# Salvar modelo treinado
model.save(MODEL_PATH)
print(f"Modelo salvo como '{MODEL_PATH}'")
