"""Normalización de formas verbales españolas al lema usado por pictogramas."""

import re
import unicodedata

import json
from pathlib import Path


IRREGULAR = {
    "jugar": ["juego", "juegas", "juega", "jugamos", "jugáis", "juegan", "jugué", "jugó", "jugaré"],
    "querer": ["quiero", "quieres", "quiere", "queremos", "queréis", "quieren", "quise", "quiso", "querré"],
    "dormir": ["duermo", "duermes", "duerme", "dormimos", "dormís", "duermen", "durmió"],
    "pensar": ["pienso", "piensas", "piensa", "pensamos", "pensáis", "piensan"],
    "sentir": ["siento", "sientes", "siente", "sentimos", "sentís", "sienten", "sintió"],
    "cerrar": ["cierro", "cierras", "cierra", "cerramos", "cerráis", "cierran"],
    "encontrar": ["encuentro", "encuentras", "encuentra", "encontramos", "encontráis", "encuentran"],
}


def _regular_forms(infinitive):
    stem, ending = infinitive[:-2], infinitive[-2:]
    present = {
        "ar": ["o", "as", "a", "amos", "áis", "an"],
        "er": ["o", "es", "e", "emos", "éis", "en"],
        "ir": ["o", "es", "e", "imos", "ís", "en"],
    }[ending]
    past = {
        "ar": ["é", "aste", "ó", "amos", "asteis", "aron"],
        "er": ["í", "iste", "ió", "imos", "isteis", "ieron"],
        "ir": ["í", "iste", "ió", "imos", "isteis", "ieron"],
    }[ending]
    future = ["é", "ás", "á", "emos", "éis", "án"]
    return [stem + suffix for suffix in present + past] + [infinitive + suffix for suffix in future]


def _build_form_index():
    catalog = json.loads((Path(__file__).resolve().parent / "pictogram_catalog.json").read_text(encoding="utf-8"))
    infinitives = {item["names"]["es"].lower() for item in catalog["pictograms"] if item["type"] == "verbo"}
    result = {infinitive: infinitive for infinitive in infinitives}
    for infinitive in infinitives:
        if infinitive[-2:] in {"ar", "er", "ir"}:
            result.update({form: infinitive for form in _regular_forms(infinitive)})
    for infinitive, forms in IRREGULAR.items():
        result.update({form: infinitive for form in forms})
    return result


FORM_TO_LEMMA = _build_form_index()


def normalize_spanish_phrase(text):
    normalized = unicodedata.normalize("NFC", text.strip().lower())
    tokens = re.findall(r"[a-záéíóúüñ]+", normalized)
    return [FORM_TO_LEMMA.get(token, token) for token in tokens]
