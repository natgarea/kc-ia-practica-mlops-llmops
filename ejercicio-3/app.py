from fastapi import FastAPI, HTTPException

app = FastAPI()

POKEMONS = {
    "pikachu": {"name": "Pikachu", "type": "electric", "hp": 35, "attack": 55, "speed": 90},
    "bulbasaur": {"name": "Bulbasaur", "type": "grass/poison", "hp": 45, "attack": 49, "speed": 45},
    "charmander": {"name": "Charmander", "type": "fire", "hp": 39, "attack": 52, "speed": 65},
}

ALLOWED_FIELDS = {"hp", "attack", "speed"}

@app.get("/pokemon")
def pokemon(name: str):
    key = name.lower()
    if key not in POKEMONS:
        raise HTTPException(404, "Pokémon not found")
    return POKEMONS[key]

@app.get("/pokemon/list")
def pokemon_list():
    return {"pokemons": [p["name"] for p in POKEMONS.values()]}

@app.get("/pokemon/compare")
def pokemon_compare(name1: str, name2: str, field: str):
    field = field.lower()
    if field not in ALLOWED_FIELDS:
        raise HTTPException(400, f"Field must be one of {sorted(ALLOWED_FIELDS)}")

    p1 = pokemon(name1)
    p2 = pokemon(name2)
    v1, v2 = p1[field], p2[field]

    return {
        "field": field,
        "pokemon_1": {"name": p1["name"], "value": v1},
        "pokemon_2": {"name": p2["name"], "value": v2},
        "winner": p1["name"] if v1 > v2 else (p2["name"] if v2 > v1 else "tie"),
    }

from transformers import pipeline

sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)

@app.get("/pokemon-sentiment")
def pokemon_sentiment(text: str):
    return sentiment_pipe(text)[0]

qa_pipe = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad",
)

@app.get("/pokemon-qa")
def pokemon_qa(question: str, context: str):
    return qa_pipe(question=question, context=context)