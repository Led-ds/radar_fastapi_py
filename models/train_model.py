from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np

# Definir tamanho da imagem esperado
img_size = (64, 64, 1)

# Criando um modelo de CNN para processar imagens
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=img_size),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compilar e treinar
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Gerar dados fictícios de imagens com tamanho correto
X_train = np.random.random((100, 64, 64, 1))  # 100 imagens, 64x64, escala de cinza
y_train = np.random.randint(2, size=(100, 1))

model.fit(X_train, y_train, epochs=10, batch_size=32)

# Salvar o modelo treinado
model.save('models/meu_modelo.h5')

print("Modelo salvo como 'models/meu_modelo.h5'")
