# Práctica MLops / LLMops

## Ejercicio 1
Modelo de clasificación con Scikit-learn y MLflow que incluye EDA, preprocesamiento de texto mediante tokenización y vectorización, entrenamiento con algoritmos de clasificación de Scikit-learn sobre datos preprocesados, evaluación mediante métricas estándar, registro en MLflow de métricas, hiperparámetros y artifacts durante el entrenamiento para comparar experimentos, explicación de las métricas y selección del modelo final con su correspondiente justificación.

### Ficheros
* [Notebook](ejercicio-1/modelo-sms-spam.ipynb)
* [Screenshots](ejercicio-1/screenshots)
* [Requirements](requirements.txt)

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
Generar dos ficheros .py: main y funciones, con al menos dos argumentos de entrada. He extraído la lógica del notebook del ejercicio 1 a un script, el fichero de requirements es el mismo.

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