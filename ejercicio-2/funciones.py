from datasets import load_dataset
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import mlflow
import mlflow.sklearn


VECTORIZER_DEFAULTS = {
    "lowercase": True,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
}

def load_data():
    ds = load_dataset("sms_spam")
    df = ds["train"].to_pandas()
    return df


def split_data(df, test_size=0.2, random_state=42):
    X = df["sms"].astype(str)
    y = df["label"]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_pipeline(model_type):
    """
    Builds a sklearn pipeline with fixed hyperparameters.

    Args:
        model_type (str): "nb", "logreg", "svc"

    Returns:
        sklearn.pipeline.Pipeline: pipeline with vectorizer and classifier.
    """
    if model_type == "nb":
        return Pipeline(
            [
                ("vect", CountVectorizer(**VECTORIZER_DEFAULTS)),
                ("clf", MultinomialNB(alpha=1.0)),
            ]
        )

    if model_type == "logreg":
        return Pipeline(
            [
                ("vect", TfidfVectorizer(**VECTORIZER_DEFAULTS)),
                ("clf", LogisticRegression(C=1.0, max_iter=1000, solver="liblinear")),
            ]
        )

    if model_type == "svc":
        return Pipeline(
            [
                ("vect", TfidfVectorizer(**VECTORIZER_DEFAULTS)),
                ("clf", LinearSVC(C=1.0)),
            ]
        )

    raise ValueError(f"Unknown model_type: {model_type}")


def train_and_evaluate(pipeline, X_train, X_test, y_train, y_test):
    """
    Trains a scikit-learn pipeline and evaluates it on a test set.

    Returns a dictionary of evaluation metrics (accuracy, precision, recall,
    F1-score for the spam class), the main evaluation artifacts, and the trained
    pipeline.
    """

    # Train
    pipeline.fit(X_train, y_train)
    # Predict
    y_pred = pipeline.predict(X_test)

    # Set evaluation metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "spam_precision": precision_score(y_test, y_pred, pos_label=1),
        "spam_recall": recall_score(y_test, y_pred, pos_label=1),
        "spam_f1": f1_score(y_test, y_pred, pos_label=1),
    }

    # Set artifacts
    artifacts = {
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
    }

    return metrics, artifacts, pipeline


def log_to_mlflow(
    experiment_name,
    run_name,
    model_type,
    pipeline,
    metrics,
    artifacts,
):
    """
    Logs model parameters, metrics, artifacts, and the trained pipeline to MLflow.

    This function creates or selects an MLflow experiment, starts a new run,
    and logs model type, vectorizer and model parameters, evaluation metrics and artifacts.
    """
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model", model_type)

        mlflow.log_param("vectorizer_lowercase", VECTORIZER_DEFAULTS["lowercase"])
        mlflow.log_param("vectorizer_ngram_range", str(VECTORIZER_DEFAULTS["ngram_range"]))
        mlflow.log_param("vectorizer_min_df", VECTORIZER_DEFAULTS["min_df"])
        mlflow.log_param("vectorizer_max_df", VECTORIZER_DEFAULTS["max_df"])

        if model_type == "nb":
            mlflow.log_param("alpha", 1.0)
        elif model_type == "logreg":
            mlflow.log_param("C", 1.0)
            mlflow.log_param("max_iter", 1000)
            mlflow.log_param("solver", "liblinear")
        elif model_type == "svc":
            mlflow.log_param("C", 1.0)

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        with open("classification_report.txt", "w", encoding="utf-8") as f:
            f.write(artifacts["classification_report"])
        mlflow.log_artifact("classification_report.txt")

        pd.DataFrame(
            artifacts["confusion_matrix"],
            index=["true_ham", "true_spam"],
            columns=["pred_ham", "pred_spam"],
        ).to_csv("confusion_matrix.csv")
        mlflow.log_artifact("confusion_matrix.csv")

        mlflow.sklearn.log_model(pipeline, model_type)
