import json

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from predictive_maintenance.common import (
    ALIAS,
    CANDIDATE_PATH,
    EVALUATION_PATH,
    MODEL_NAME,
    PARAMS_PATH,
    TEST_PATH,
    load_params,
    split_xy,
)


def evaluation(y_true, y_pred):
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    matrice = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = matrice.ravel()
    metrics = {
        "f1": f1, 
        "precision": precision, 
        "recall": recall, 
        "false_alarms": int(fp),
        "missed_failures": int(fn),
        "detected_failures": int(tp),
        }
    return metrics


def cost(metrics, cost_missed_failure, cost_false_alarm):
    return (metrics["missed_failures"] * cost_missed_failure) + (metrics["false_alarms"] * cost_false_alarm)


def get_champion_version(client):
    try:
        return client.get_model_version_by_alias(MODEL_NAME, ALIAS).version
    except MlflowException as e:
        return None


if __name__ == "__main__":
    params = load_params(PARAMS_PATH)["evaluate"]
    client = MlflowClient()

    test_df = pd.read_csv(TEST_PATH)
    X_test, y_test = split_xy(test_df)

    with open(CANDIDATE_PATH) as f:
        candidate = json.load(f)

    candidate_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{candidate['version']}")
    y_pred = candidate_model.predict(X_test)
    candidate_metrics = evaluation(y_test, y_pred)
    candidate_cost = cost(candidate_metrics, params["cost_missed_failure"], params["cost_false_alarm"])
    candidate_metrics.update({"cost" : candidate_cost})

    champion_version = get_champion_version(client)

    champion_cost = None
    promote = False

    if champion_version is None:
        promote = True
    else:
        champion_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{ALIAS}")
        champion_metrics = evaluation(y_test, champion_model.predict(X_test))
        champion_cost = cost(champion_metrics, params["cost_missed_failure"], params["cost_false_alarm"])
        promote = candidate_cost < champion_cost

    with mlflow.start_run(run_id=candidate["run_id"]):
        mlflow.log_metrics(candidate_metrics)

    if promote:
        client.set_registered_model_alias(MODEL_NAME, ALIAS, candidate["version"])

    result = {
        "candidate_version": int(candidate["version"]),
        "candidate_cost": candidate_cost,
        "champion_version": int(champion_version) if champion_version is not None else None,
        "champion_cost": champion_cost,
        "promoted": promote,
    }
    with open(EVALUATION_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(result)