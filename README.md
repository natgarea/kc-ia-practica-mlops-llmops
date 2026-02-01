# Práctica MLops / LLMops

## Ejercicio 1
Modelo de clasificación con Scikit-learn y MLflow que incluye EDA, preprocesamiento de texto mediante tokenización y vectorización, entrenamiento con algoritmos de clasificación de Scikit-learn sobre datos preprocesados, evaluación mediante métricas estándar, registro en MLflow de métricas, hiperparámetros y artifacts durante el entrenamiento para comparar experimentos, explicación de las métricas y selección del modelo final con su correspondiente justificación.

### Ficheros
* [Notebook](ejercicio-1/modelo-sms-spam.ipynb)
* [Requirements](requirements.txt)
* [Screenshots](ejercicio-1/screenshots)

### Cómo ejecutar el entorno

Crear y activar un entorno virtual:

```bash
python3 -m venv practica
source practica/bin/activate
```

Actualizar `pip` e instalar dependencias:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Lanzar el notebook:

```bash
jupyter notebook
```

## Ejercicio 2
Generar dos ficheros .py: main y funciones, con al menos dos argumentos de entrada. He extraído la lógica del notebook del ejercicio 1 a un script.

### Ficheros
* [funciones.py](ejercicio-2/funciones.py)
* [main.py](ejercicio-2/main.py)
* [Requirements](requirements.txt)

### Cómo ejecutar

```bash
python ejercicio-2/main.py --model nb --experiment-name 'sms spam classification'
python ejercicio-2/main.py --model logreg --experiment-name 'sms spam classification'
python ejercicio-2/main.py --model svc --experiment-name 'sms spam classification'
```

## Ejercicio 3
Generar un script con al menos 5 modulos app.get y dos de ellos tienen que ser pipelines de HF. He elegido una temática de Pokemon para hacerlo entretenido. Intenté usar un pipeline de generación de texto para que me diese una línea sobre batallas, pero daba texto sin sentido así que me he quedado con dos pipelines sencillas: sentiment y qa.

No he hecho la parte opcional de Google Cloud.

### Ficheros
* [app.py](ejercicio-3/app.py)
* [request_pokemon.py](ejercicio-3/request_pokemon.py)
* [Requirements](requirements.txt)
* [Screenshots](ejercicio-3/screenshots)

### Cómo ejecutar

#### app.py

Arrancar fastapi:
```bash
python -m fastapi run ejercicio-3/app.py
```

#### requests.py

En otra terminal:
```bash
python ejercicio-3/request_pokemon.py
```