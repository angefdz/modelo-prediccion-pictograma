from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pickle

model = tf.keras.models.load_model("modelo.h5")
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

max_len = 5

app = FastAPI()

class FraseInput(BaseModel):
    frase: str

@app.post("/predecir")
def predecir_siguiente_palabra(input: FraseInput):
    tokens = tokenizer.texts_to_sequences([input.frase])[0]
    padded = tf.keras.preprocessing.sequence.pad_sequences([tokens], maxlen=max_len, padding="pre")
    prediction = model.predict(padded, verbose=0)
    predicted_index = np.argmax(prediction)
    
    indice_a_palabra = {v: k for k, v in tokenizer.word_index.items()}
    siguiente_palabra = indice_a_palabra.get(predicted_index, "[desconocido]")

    return {"sugerencia": siguiente_palabra}