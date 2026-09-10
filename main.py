import json, unicodedata
from collections import defaultdict
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
from lemmatization import normalize_spanish_phrase

try:
    import torch
    from transformers import AutoTokenizer
    from neural_model import BilingualPictogramPredictor
except ImportError:
    torch=AutoTokenizer=BilingualPictogramPredictor=None

BASE_DIR=Path(__file__).resolve().parent
def normalize(value): return unicodedata.normalize("NFC",value.strip().lower())
def load_model(language):
    path=BASE_DIR/f"modelo_{language}.json"; return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
models={language:load_model(language) for language in ("es","en")}
neural_model=neural_tokenizer=neural_config=None
artifact=BASE_DIR/"transformer_model"
if torch is not None and (artifact/"model.pt").exists():
    neural_config=json.loads((artifact/"config.json").read_text(encoding="utf-8"))
    neural_tokenizer=AutoTokenizer.from_pretrained(artifact/"tokenizer")
    neural_model=BilingualPictogramPredictor(neural_config["base_model"],len(neural_config["pictogram_ids"]),str(artifact/"encoder_config"))
    neural_model.load_state_dict(torch.load(artifact/"model.pt",map_location="cpu",weights_only=True))
    neural_model.eval()
app=FastAPI(title="Predictor AAC bilingüe",version="5.0.0")

class PhraseInput(BaseModel):
    pictograma_ids:list[int]|None=Field(default=None,max_length=100)
    lemas:list[str]|None=Field(default=None,max_length=100)
    frase:str|None=Field(default=None,max_length=500)
    texto:str|None=Field(default=None,max_length=500)
    idioma:str=Field(default="es",pattern="^(es|en)$")
    @model_validator(mode="after")
    def require_content(self):
        if not self.pictograma_ids and not self.lemas and not (self.frase and self.frase.strip()) and not (self.texto and self.texto.strip()): raise ValueError("Debe enviarse al menos un pictograma")
        return self

def resolve_sequence(data,model):
    valid=set(model["labels"])
    if data.pictograma_ids: return [value for value in data.pictograma_ids if str(value) in valid]
    lemmas=data.lemas
    if not lemmas and data.frase: lemmas=normalize_spanish_phrase(data.frase) if data.idioma=="es" else data.frase.split()
    return [model["label_to_id"][key] for lemma in (lemmas or []) if (key:=normalize(lemma)) in model["label_to_id"]]

@app.get("/health")
def health():
    keys=("version","type","catalog_size","training_phrases","context_size")
    return {"status":"ok" if all(models.values()) else "degraded","neural_model":neural_model is not None,"models":{lang:({key:model[key] for key in keys} if model else None) for lang,model in models.items()}}

def predict_neural(text,language,model):
    if neural_model is None or not text.strip(): return None
    # La cabeza elegida ya codifica el idioma; el texto debe conservar exactamente
    # el mismo formato usado durante el entrenamiento.
    tokens=neural_tokenizer(text,return_tensors="pt",truncation=True,max_length=32)
    with torch.no_grad(): probabilities=torch.softmax(neural_model(**tokens,language=language),dim=-1)[0]
    values,indices=probabilities.topk(3)
    alternatives=[]
    for value,index in zip(values.tolist(),indices.tolist()):
        identifier=neural_config["pictogram_ids"][index]
        alternatives.append({"pictograma_id":identifier,"pictograma":model["labels"][str(identifier)],"confianza":value})
    return alternatives

@app.post("/predecir")
def predict(data:PhraseInput):
    model=models.get(data.idioma)
    if model is None: raise HTTPException(503,f"No hay modelo disponible para '{data.idioma}'")
    sequence=resolve_sequence(data,model)
    if not sequence: raise HTTPException(422,"Ningún pictograma pertenece al catálogo del modelo")
    text=data.texto or data.frase or " ".join(data.lemas or [])
    neural_alternatives=predict_neural(text,data.idioma,model)
    if neural_alternatives and neural_alternatives[0]["confianza"] >= .18:
        first=neural_alternatives[0]
        return {"pictograma_id":first["pictograma_id"],"sugerencia":first["pictograma"],"confianza":first["confianza"],"alternativas":neural_alternatives,"idioma":data.idioma,"contexto_utilizado":len(sequence),"motor":"transformer"}
    scores=defaultdict(float); matched=0; matching=[]
    for length in range(min(model["context_size"],len(sequence)),0,-1):
        counts=model["transitions"].get(",".join(map(str,sequence[-length:])))
        if counts:
            if not matching: matched=length
            matching.append((length,counts))
    # Se usa el contexto más específico disponible. Solo se retrocede a uno
    # más corto cuando el largo no existe, evitando sugerencias globales que
    # contaminen una frase concreta.
    if matching:
        _,counts=matching[0]; total=sum(counts.values())
        for identifier,count in counts.items(): scores[int(identifier)]=count/total
    if not scores: raise HTTPException(404,"No hay una continuación válida para este contexto")
    ranked=sorted(scores.items(),key=lambda item:(-item[1],item[0]))[:3]; total=sum(score for _,score in ranked)
    alternatives=[{"pictograma_id":identifier,"pictograma":model["labels"][str(identifier)],"confianza":score/total} for identifier,score in ranked]
    first=alternatives[0]
    return {"pictograma_id":first["pictograma_id"],"sugerencia":first["pictograma"],"confianza":first["confianza"],"alternativas":alternatives,"idioma":data.idioma,"contexto_utilizado":matched,"motor":"ngram_fallback"}
