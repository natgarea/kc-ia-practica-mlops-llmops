import requests

BASE_URL = "http://localhost:8000"

def call_pokemon():
    print("\n--- /pokemon ---")
    resp = requests.get(
        f"{BASE_URL}/pokemon",
        params={"name": "Pikachu"},
        timeout=10,
    )
    resp.raise_for_status()
    print(resp.json())


def call_pokemon_list():
    print("\n--- /pokemon/list ---")
    resp = requests.get(
        f"{BASE_URL}/pokemon/list",
        timeout=10,
    )
    resp.raise_for_status()
    print(resp.json())


def call_pokemon_compare():
    print("\n--- /pokemon/compare ---")
    resp = requests.get(
        f"{BASE_URL}/pokemon/compare",
        params={
            "name1": "Pikachu",
            "name2": "Charmander",
            "field": "speed",
        },
        timeout=10,
    )
    resp.raise_for_status()
    print(resp.json())


def call_pokemon_sentiment():
    print("\n--- /pokemon-sentiment ---")
    resp = requests.get(
        f"{BASE_URL}/pokemon-sentiment",
        params={"text": "Charmander is the best pokemon ever"},
        timeout=10,
    )
    resp.raise_for_status()
    print(resp.json())


def call_pokemon_qa():
    print("\n--- /pokemon-qa ---")
    resp = requests.get(
        f"{BASE_URL}/pokemon-qa",
        params={
            "question": "What type of pokemon is Bulbasaur?",
            "context": "Bulbasaur is a grass and poison type Pokémon that uses plant-based abilities to fight.",
        },
        timeout=10,
    )
    resp.raise_for_status()
    print(resp.json())


if __name__ == "__main__":
    call_pokemon()
    call_pokemon_list()
    call_pokemon_compare()
    call_pokemon_sentiment()
    call_pokemon_qa()
