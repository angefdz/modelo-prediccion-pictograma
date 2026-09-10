#!/usr/bin/env python3
"""Genera un corpus universal de secuencias de pictogramas (4-8 tokens).

Los dos TXT de salida tienen exactamente el mismo orden de registros:
  - dataset_frases_ids.txt: secuencias que consumira el modelo.
  - dataset_frases_palabras.txt: equivalencia legible de cada ID.

Los nombres son los lemas exactos del catalogo; por tanto cada palabra de una
linea corresponde uno a uno con el ID que ocupa la misma posicion.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parent
CATALOG_PATH = ROOT / "pictogram_catalog.json"
DEFAULT_OUTPUT = ROOT / "dataset"


def load_catalog() -> tuple[dict[int, dict], dict[str, int]]:
    raw = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))["pictograms"]
    by_id = {int(item["id"]): item for item in raw}
    by_es = {item["names"]["es"].casefold(): int(item["id"]) for item in raw}
    return by_id, by_es


CATALOG, ES_ID = load_catalog()


def pid(name: str) -> int:
    try:
        return ES_ID[name.casefold()]
    except KeyError as exc:
        raise KeyError(f"No existe el pictograma {name!r}") from exc


def ids(*names: str) -> list[int]:
    return [pid(name) for name in names]


def category(number: int) -> list[int]:
    return [value for value, item in CATALOG.items() if number in item.get("categories", [])]


SUBJECTS = ids("Yo", "Tú", "Él", "Ella", "Nosotros", "Ellos", "Ellas", "Usted")
PEOPLE = ids("Mamá", "Papá", "Hermano", "Hermana", "Amigo", "Niño", "Niña", "Abuelo", "Abuela", "Primo", "Prima")
PLACES = ids("Casa", "Escuela", "Parque", "Playa", "Bosque", "Ciudad", "Biblioteca", "Museo", "Mercado", "Restaurante")
FOOD = ids("Manzana", "Pera", "Plátano", "Fresa", "Pollo", "Pescado", "Huevo", "Queso", "Pan", "Arroz", "Pasta", "Sopa", "Ensalada", "Hamburguesa", "Pizza", "Tarta", "Helado", "Chocolate", "Galleta", "Yogur")
DRINKS = ids("Agua", "Zumo", "Café", "Té", "Leche", "Batido", "Refresco", "Limonada", "Chocolate caliente", "Infusión")
TOYS = ids("Muñeca", "Muñeco", "Peluche", "Coche", "Tren", "Rompecabezas", "Puzzle", "Cometa", "Patinete", "Bicicleta", "Balón", "Lego")
SCHOOL = ids("Deberes", "Lápiz", "Regla", "Borrador", "Pizarra", "Clase", "Escritorio", "Calculadora", "Agenda", "Examen", "Libro", "Cuaderno", "Mochila")
CLOTHES = ids("Camisa", "Pantalón", "Falda", "Zapatos", "Calcetines", "Chaqueta", "Guantes", "Bufanda", "Abrigo", "Vestido", "Botas", "Pijama", "Zapatillas", "Chaleco")
COLORS = ids("Rojo", "Azul", "Verde", "Amarillo", "Negro", "Blanco", "Morado", "Rosa", "Gris", "Marrón", "Naranja")
EMOTIONS = ids("Feliz", "Triste", "Cansado", "Enfadado", "Asustado", "Sorprendido", "Orgulloso", "Nervioso", "Preocupado", "Contento", "Aburrido")
BODY = ids("Cabeza", "Brazo", "Mano", "Pierna", "Pie", "Ojo", "Oreja", "Boca", "Nariz", "Cuello", "Hombro", "Espalda", "Estómago", "Diente", "Rodilla")
TRANSPORT = ids("Autobús", "Barco", "Moto", "Taxi", "Metro", "Furgoneta", "Tranvía", "Ferry", "Bicicleta", "Tren", "Avión")
TECH = ids("Teléfono", "Ordenador", "Cámara", "Wifi", "Consola", "Móvil", "Tableta", "Impresora", "Altavoz", "Auriculares", "Televisor")
ANIMALS = ids("Perro", "Gato", "Pájaro", "Pez", "Caballo", "Conejo", "Tortuga", "Rana")
SPORTS = ids("Fútbol", "Baloncesto", "Tenis", "Natación", "Ciclismo", "Gimnasia", "Voleibol", "Judo", "Karate", "Golf", "Ping pong", "Esquí", "Surf", "Patinaje", "Senderismo")
INSTRUMENTS = category(29)
PROFESSIONS = category(22)
DAYS = ids("Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo")
MONTHS = category(24)
NUMBERS = category(17)
WEATHER = category(20)
NATURE = category(21)
CELEBRATIONS = category(26)
TOOLS = category(27)
HEALTH = ids("Medicina", "Pastilla", "Termómetro", "Consulta", "Vacuna", "Descanso", "Cuidado")
HOME_OBJECTS = ids("Mesa", "Silla", "Puerta", "Ventana", "Lámpara", "Espejo", "Almohada", "Cama", "Toalla", "Cocina", "Sofá", "Horno", "Lavadora", "Cortina", "Alfombra")
OBJECTS = list(dict.fromkeys(SCHOOL + TOYS + TECH + HOME_OBJECTS + CLOTHES + TOOLS + HEALTH + INSTRUMENTS))

REFLEXIVE = {
    pid("Yo"): pid("Me"), pid("Tú"): pid("te"), pid("Él"): pid("se"),
    pid("Ella"): pid("se"), pid("Nosotros"): pid("nos"),
    pid("Ellos"): pid("se"), pid("Ellas"): pid("se"), pid("Usted"): pid("se"),
}

# Verbo -> objetos y lugares semanticamente compatibles.
HOME = ids("Casa", "Cocina", "Salón", "Baño", "Dormitorio")
LEARNING_PLACES = ids("Casa", "Escuela", "Biblioteca")
LEISURE_PLACES = ids("Casa", "Escuela", "Parque", "Playa")
FOOD_PLACES = ids("Casa", "Escuela", "Parque", "Restaurante")
SHOPPING_PLACES = ids("Tienda", "Mercado")
FRAMES = [
    (pid("Comer"), FOOD, FOOD_PLACES), (pid("Beber"), DRINKS, FOOD_PLACES),
    (pid("Leer"), ids("Libro", "Cuaderno", "Deberes"), LEARNING_PLACES),
    (pid("Escribir"), ids("Cuaderno", "Papel", "Deberes", "Agenda"), LEARNING_PLACES),
    (pid("Dibujar"), ids("Papel", "Cuaderno", "Pizarra", "Pintura"), LEARNING_PLACES),
    (pid("Jugar"), TOYS + SPORTS, LEISURE_PLACES),
    (pid("Comprar"), FOOD + DRINKS + CLOTHES + TOYS, SHOPPING_PLACES),
    (pid("Buscar"), OBJECTS, PLACES), (pid("Encontrar"), OBJECTS, PLACES),
    (pid("Abrir"), ids("Puerta", "Ventana", "Caja", "Bolsa", "Mochila"), HOME + ids("Escuela")),
    (pid("Cerrar"), ids("Puerta", "Ventana", "Caja", "Bolsa", "Mochila"), HOME + ids("Escuela")),
    (pid("Lavar"), CLOTHES + ids("Mano", "Cabello", "Plato", "Vaso"), HOME),
    (pid("Limpiar"), HOME_OBJECTS + ids("Plato", "Vaso", "Pizarra"), HOME + ids("Escuela")),
    (pid("Mirar"), TECH + ANIMALS + ids("Mapa", "Pizarra"), PLACES),
    (pid("Escuchar"), INSTRUMENTS + ids("Música", "Radio"), HOME + ids("Escuela", "Parque")),
    (pid("Tocar"), INSTRUMENTS, HOME + ids("Escuela")),
    (pid("Pintar"), ids("Papel", "Pizarra", "Cuadro", "Puerta"), HOME + ids("Escuela")),
    (pid("Guardar"), OBJECTS, HOME + ids("Escuela", "Oficina")),
    (pid("Coger"), OBJECTS + FOOD + DRINKS, PLACES),
]


def pick(rng: random.Random, values: list[int]) -> int:
    return rng.choice(values)


def action(rng: random.Random) -> tuple[int, int, int]:
    verb, objects, places = rng.choice(FRAMES)
    return verb, pick(rng, objects), pick(rng, places)


def es_candidate(length: int, rng: random.Random) -> tuple[int, ...]:
    s, person, place = pick(rng, SUBJECTS), pick(rng, PEOPLE), pick(rng, PLACES)
    verb, obj, action_place = action(rng)
    choices: dict[int, list[Callable[[], list[int]]]] = {
        4: [
            lambda: [pid("Querer"), pid("Jugar"), pid("A"), pick(rng, SPORTS)],
            lambda: [s, pid("Querer"), verb, obj],
            lambda: [s, verb, pid("Un"), obj],
            lambda: [pick(rng, DAYS), s, verb, obj],
            lambda: [pid("Mi"), pick(rng, CLOTHES), pid("Estar"), pick(rng, COLORS)],
            lambda: [s, pid("Ser"), pick(rng, PROFESSIONS), pick(rng, EMOTIONS)],
            lambda: [pick(rng, DAYS), pick(rng, WEATHER), pid("En"), place],
            lambda: [s, pid("Querer"), pick(rng, NUMBERS), pick(rng, OBJECTS + FOOD + DRINKS)],
        ],
        5: [
            lambda: [s, verb, obj, pid("En"), action_place],
            lambda: [s, pid("Jugar"), pick(rng, TOYS), pid("Con"), person],
            lambda: [s, pid("Estar"), pick(rng, EMOTIONS), pid("En"), place],
            lambda: [pid("Mi"), pick(rng, BODY), pid("Doler"), pid("En"), place],
            lambda: [s, pid("Querer"), pid("Un"), pick(rng, OBJECTS), pid("Por favor")],
            lambda: [s, pid("Mirar"), pick(rng, NATURE), pid("En"), place],
            lambda: [s, pid("Buscar"), pick(rng, TOOLS), pid("En"), place],
        ],
        6: [
            lambda: [s, pid("Querer"), verb, obj, pid("En"), action_place],
            lambda: [s, verb, pid("Un"), obj, pid("En"), action_place],
            lambda: [pick(rng, DAYS), s, verb, obj, pid("En"), action_place],
            lambda: [s, pid("Hablar"), pid("Con"), person, pid("En"), place],
            lambda: [s, pid("Viajar"), pid("A"), place, pid("En"), pick(rng, TRANSPORT)],
            lambda: [s, REFLEXIVE[s], pid("Sentir"), pick(rng, EMOTIONS), pid("En"), place],
            lambda: [pick(rng, MONTHS), pick(rng, CELEBRATIONS), pid("En"), place, pid("Con"), person],
            lambda: [s, pid("Jugar"), pid("A"), pick(rng, SPORTS), pid("Con"), person],
        ],
        7: [
            lambda: [s, pid("Querer"), verb, pid("Un"), obj, pid("En"), action_place],
            lambda: [pick(rng, DAYS), s, verb, pid("Un"), obj, pid("En"), action_place],
            lambda: [pick(rng, DAYS), s, pid("Comer"), pick(rng, FOOD), pid("Y"), pid("Beber"), pick(rng, DRINKS)],
            lambda: [s, pid("Jugar"), pick(rng, SPORTS), pid("Con"), person, pid("En"), place],
            lambda: [s, pid("Querer"), pid("Comprar"), pick(rng, CLOTHES), pick(rng, COLORS), pid("En"), place],
        ],
        8: [
            lambda: [pick(rng, DAYS), s, verb, obj, pid("En"), action_place, pid("Con"), person],
            lambda: [s, pid("Querer"), verb, obj, pid("Y"), pid("Beber"), pick(rng, DRINKS), pid("Por favor")],
            lambda: [s, pid("Comer"), pick(rng, FOOD), pid("Y"), pid("Beber"), pick(rng, DRINKS), pid("En"), place],
            lambda: [s, pid("Viajar"), pid("A"), place, pid("Con"), person, pid("En"), pick(rng, TRANSPORT)],
            lambda: [pick(rng, DAYS), s, pid("Estudiar"), pick(rng, SCHOOL), pid("En"), pid("Escuela"), pid("Con"), person],
        ],
    }
    return tuple(rng.choice(choices[length])())


def generate_dataset(sentence_count: int, seed: int) -> list[tuple[int, ...]]:
    rng = random.Random(seed)
    base, remainder = divmod(sentence_count, 5)
    quotas = {length: base + (1 if length - 4 < remainder else 0) for length in range(4, 9)}
    result: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for length in range(4, 9):
        attempts = 0
        while sum(1 for row in result if len(row) == length) < quotas[length]:
            row = es_candidate(length, rng)
            attempts += 1
            if row not in seen:
                seen.add(row)
                result.append(row)
            if attempts > quotas[length] * 300:
                raise RuntimeError(f"No se alcanzan {quotas[length]} frases unicas de longitud {length}")
    rng.shuffle(result)
    return result


def write_dataset(output: Path, sentence_count: int, seed: int) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    ids_path = output / "dataset_frases_ids.txt"
    words_path = output / "dataset_frases_palabras.txt"
    rows = generate_dataset(sentence_count, seed)
    used: set[int] = set()

    with ids_path.open("w", encoding="utf-8") as ids_file, words_path.open("w", encoding="utf-8") as words_file:
        for row in rows:
            names = [CATALOG[value]["names"]["es"] for value in row]
            ids_file.write(" ".join(map(str, row)) + "\n")
            words_file.write(" ".join(names) + "\n")
            used.update(row)

    validate(ids_path, words_path, sentence_count)
    return {
        "ids": str(ids_path), "words": str(words_path),
        "sentences": len(rows),
        "by_length": dict(sorted(Counter(map(len, rows)).items())),
        "different_pictograms": len(used),
    }


def validate(ids_path: Path, words_path: Path, expected: int) -> None:
    seen: set[tuple[int, ...]] = set()
    count = 0
    with ids_path.open(encoding="utf-8") as left, words_path.open(encoding="utf-8") as right:
        for line_number, (ids_line, words_line) in enumerate(zip(left, right, strict=True), 1):
            sequence = tuple(map(int, ids_line.split()))
            assert 4 <= len(sequence) <= 8
            expected_words = " ".join(CATALOG[value]["names"]["es"] for value in sequence)
            assert expected_words == words_line.rstrip("\n"), (line_number, expected_words, words_line)
            assert sequence not in seen, f"Frase duplicada en linea {line_number}"
            seen.add(sequence)
            count += 1
    assert count == expected, (count, expected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frases", type=int, default=25_000)
    parser.add_argument("--semilla", type=int, default=20260910)
    parser.add_argument("--salida", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.frases < 5:
        parser.error("--frases debe ser al menos 5")
    print(json.dumps(write_dataset(args.salida, args.frases, args.semilla), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
