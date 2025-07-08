
import json
with open('frases.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    frases = data['frases']


import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts(frases)

palabra_a_indice = tokenizer.word_index
indice_a_palabra = {i: w for w, i in palabra_a_indice.items()}

print("Vocabulario:", palabra_a_indice)

secuencias = []
for frase in frases:
    tokens = tokenizer.texts_to_sequences([frase])[0]
    for i in range(1, len(tokens)):
        secuencia = tokens[:i+1]
        secuencias.append(secuencia)

print("Secuencias sin padding:")
for s in secuencias:
    print(s)

from tensorflow.keras.utils import to_categorical

max_len = max([len(seq) for seq in secuencias])
secuencias_padded = pad_sequences(secuencias, maxlen=max_len, padding='pre')

secuencias_padded = np.array(secuencias_padded)

X = secuencias_padded[:, :-1]
y = secuencias_padded[:, -1]

total_palabras = len(tokenizer.word_index) + 1
y = to_categorical(y, num_classes=total_palabras)

print("Forma de X:", X.shape)
print(X)
print("Forma de y:", y.shape)
print(y)

#definimos el modelo
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense


input_length = X.shape[1]               
vocab_size = len(tokenizer.word_index) + 1  

#construimos el modelo
model = Sequential()
# usamos embedding para tener un mini word2vec, básicamente genera un array de 10
# palabras que se usan en contextos parecidos deben tener vectores parecidos.
# Cuanto más largo sea el vector, más preciso será pero tardará más.
model.add(Embedding(input_dim=vocab_size, output_dim=20, input_length=input_length))  # 10 = tamaño del embedding
# Red recurrente que recuerda el contexto de lo anterior (long-short term memory)
# Cuantas más neuronas más capacidad para aprender patrones complejos pero si hay
# pocos datos se sobreentrenará
model.add(LSTM(256))  # puedes probar con 64 o 128 también
# necesito una neurona por cada palabra del vocabulario
# softmax se usa para que cada neurona devuelva una probabilidad
model.add(Dense(vocab_size, activation='softmax'))  # predice una palabra del vocabulario

# 3. Compilamos
# categorical_crossentropy sirve para que compare del vector Dense con el real,
# es decir si el nuestro es [0.2, 0.83, 0.1, 0] y el real es [0, 1, 0, 0, 0] será bajo
# y adam se encarga de optimizar y ajustar los pesos
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. Entrenamos
model.fit(X, y, epochs=300, verbose=1)

# BLOQUE 4 — Predicción
def predecir_palabra_siguiente(frase_parcial, tokenizer, model, max_len):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    import numpy as np

    # 1. Convertir frase a tokens
    secuencia = tokenizer.texts_to_sequences([frase_parcial])[0]

    # 2. Rellenar con ceros para que tenga la longitud adecuada
    secuencia_padded = pad_sequences([secuencia], maxlen=max_len - 1, padding='pre')

    # 3. Predecir la palabra siguiente
    pred = model.predict(secuencia_padded, verbose=0)
    palabra_index = np.argmax(pred)

    # 4. Buscar la palabra correspondiente
    for palabra, index in tokenizer.word_index.items():
        if index == palabra_index:
            return palabra

    return "Palabra no encontrada"

frase = "me gusta"
palabra = predecir_palabra_siguiente(frase, tokenizer, model, max_len)
print(f"La siguiente palabra predicha para '{frase}' es: {palabra}")

# BLOQUE 5 — Guardar modelo y tokenizer
model.save("modelo_prediccion_pictogramas.h5")
print("✅ Modelo guarda
