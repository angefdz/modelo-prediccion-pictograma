"""API del predictor AAC: GRU global sobre IDs con fallback N-gram."""

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

try:
    import torch
    from neural_model import GRUPictogramPredictor
except ImportError:
    torch = None
    GRUPictogramPredictor = None

BASE_DIR = Path(__file__).resolve().parent
CATALOG = {int(item["id"]): item for item in json.loads(
    (BASE_DIR / "pictogram_catalog.json").read_text(encoding="utf-8"))["pictograms"]}


def load_ngram():
    path = BASE_DIR / "modelo_es.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


ngram_model = load_ngram()
gru_model = gru_config = None
artifact = BASE_DIR / "gru_model"
if torch is not None and (artifact / "model.pt").exists():
    gru_config = json.loads((artifact / "config.json").read_text(encoding="utf-8"))
    gru_model = GRUPictogramPredictor(
        gru_config["vocabulary_size"], len(gru_config["output_ids"]),
        gru_config["embedding_dim"], gru_config["hidden_dim"],
        gru_config["layers"], gru_config["dropout"])
    gru_model.load_state_dict(torch.load(artifact / "model.pt", map_location="cpu", weights_only=True))
    gru_model.eval()

app = FastAPI(title="Predictor AAC por IDs", version="6.0.0")


class PhraseInput(BaseModel):
    pictograma_ids: list[int] = Field(default_factory=list, max_length=100)
    lemas: list[str] | None = Field(default=None, max_length=100)
    frase: str | None = Field(default=None, max_length=500)
    texto: str | None = Field(default=None, max_length=500)
    idioma: str = Field(default="es", pattern="^(es|en)$")

    @model_validator(mode="after")
    def valid_ids(self):
        if any(value <= 0 for value in self.pictograma_ids):
            raise ValueError("Los IDs de pictograma deben ser positivos")
        return self


@app.get("/health")
def health():
    return {"status": "ok" if gru_model is not None else "degraded",
            "engine": "gru" if gru_model is not None else "ngram_fallback",
            "catalog_size": len(CATALOG),
            "model": gru_config.get("metrics") if gru_config else None}


def localized_name(identifier: int, language: str) -> str:
    names = CATALOG[identifier]["names"]
    return names.get(language) or names["es"]


def predict_gru(sequence: list[int], language: str):
    if gru_model is None or gru_config is None:
        return None
    token_by_id = {int(key): value for key, value in gru_config["token_by_id"].items()}
    # Los IDs personalizados todavía no forman parte del vocabulario global;
    # se omiten aquí y podrán entrar por la rama personal del predictor híbrido.
    tokens = [gru_config["start_token"]] + [token_by_id[value] for value in sequence if value in token_by_id]
    tokens = tokens[-gru_config["max_context"]:]
    with torch.no_grad():
        probabilities = torch.softmax(gru_model(
            torch.tensor([tokens], dtype=torch.long),
            torch.tensor([len(tokens)], dtype=torch.long)), dim=-1)[0]
    values, indices = probabilities.topk(min(10, len(gru_config["output_ids"])))
    alternatives = []
    for probability, index in zip(values.tolist(), indices.tolist()):
        identifier = gru_config["output_ids"][index]
        if identifier == gru_config["end_id"] or identifier not in CATALOG:
            continue
        alternatives.append({"pictograma_id": identifier,
                             "pictograma": localized_name(identifier, language),
                             "confianza": round(float(probability), 6)})
        if len(alternatives) == 3:
            break
    return alternatives


def predict_ngram(sequence: list[int], language: str):
    if not ngram_model or not sequence:
        return []
    counts = None
    for length in range(min(ngram_model["context_size"], len(sequence)), 0, -1):
        counts = ngram_model["transitions"].get(",".join(map(str, sequence[-length:])))
        if counts:
            break
    if not counts:
        return []
    total = sum(counts.values())
    ranked = sorted(counts.items(), key=lambda item: (-item[1], int(item[0])))[:3]
    return [{"pictograma_id": int(identifier),
             "pictograma": localized_name(int(identifier), language),
             "confianza": round(count / total, 6)}
            for identifier, count in ranked if int(identifier) in CATALOG]


@app.post("/predecir")
def predict(data: PhraseInput):
    alternatives = predict_gru(data.pictograma_ids, data.idioma)
    engine = "gru"
    if not alternatives:
        alternatives = predict_ngram(data.pictograma_ids, data.idioma)
        engine = "ngram_fallback"
    if not alternatives:
        raise HTTPException(404, "No hay una continuación válida para este contexto")
    return {**alternatives[0], "alternativas": alternatives, "idioma": data.idioma,
            "contexto_utilizado": len(data.pictograma_ids), "motor": engine}
