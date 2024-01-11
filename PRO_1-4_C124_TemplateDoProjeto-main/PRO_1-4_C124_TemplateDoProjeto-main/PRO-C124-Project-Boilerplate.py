import cv2
import numpy as np
import tensorflow as tf

# Carregando o modelo
model = tf.keras.models.load_model('modelo.h5')

# Anexando a câmera indexada como 0, com o software da aplicação
camera = cv2.VideoCapture(0)

# Loop infinito
while True:
    # Lendo / requisitando um quadro da câmera
    status, frame = camera.read()

    # Se tivemos sucesso ao ler o quadro
    if status:
        # Inverta o quadro
        frame = cv2.flip(frame, 1)

        # Redimensione o quadro
        resized_frame = cv2.resize(frame, (224, 224))

        # Expanda a dimensão do array junto com o eixo 0
        expanded_frame = np.expand_dims(resized_frame, axis=0)

        # Normalize para facilitar o processamento
        normalized_frame = expanded_frame / 255.0

        # Obtenha previsões do modelo
        predictions = model.predict(normalized_frame)

        # Obter rótulo da classe com maior probabilidade
        predicted_class = np.argmax(predictions)

        # Exibindo os quadros capturados
        cv2.imshow('feed', frame)

        # Aguardando 1ms
        code = cv2.waitKey(1)

        # Se a barra de espaço foi pressionada, interrompa o loop
        if code == 32:
            break

# Libere a câmera do software da aplicação
camera.release()

# Feche a janela aberta
cv2.destroyAllWindows()
