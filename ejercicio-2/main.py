import argparse
from funciones import load_data, split_data, build_pipeline, train_and_evaluate, log_to_mlflow

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["nb", "logreg", "svc"])
    parser.add_argument("--experiment-name", required=True)
    args = parser.parse_args()

    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    pipeline = build_pipeline(args.model)  # hiperparams fijos dentro
    metrics, artifacts, trained_pipeline = train_and_evaluate(pipeline, X_train, X_test, y_train, y_test)

    log_to_mlflow(
        experiment_name=args.experiment_name,
        run_name=f"sms-{args.model}",
        model_type=args.model,
        pipeline=trained_pipeline,
        metrics=metrics,
        artifacts=artifacts,
    )

if __name__ == "__main__":
    main()