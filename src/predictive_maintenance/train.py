import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature

CSV_PATH = "data/raw/data.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100 
MAX_DEPTH = None
EXPERIMENT_NAME = "predictive-maintenance"
MODEL_NAME = "predictive-maintenance"
CLASS_WEIGHT = "balanced"

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.drop(columns = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"])

    X = df.drop(columns = ["Machine failure"])
    y = df["Machine failure"]

    return X, y


def split(X, y): 
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    return X_train, X_test, y_train, y_test

def build_pipeline():
    t = [("type", OneHotEncoder(), ["Type"])]
    preprocessing = ColumnTransformer(transformers = t, remainder="passthrough")
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
        class_weight=CLASS_WEIGHT,
    )
    pipeline = Pipeline(steps=[("preprocessing", preprocessing), ("model", model)])

    return pipeline


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
        "fausse alerte": int(fp),
        "panne non detectée": int(fn),
        "panne detectée": int(tp),
        }
    return metrics


if __name__ == "__main__":
    X, y = load_data(CSV_PATH)
    X_train, X_test, y_train, y_test = split(X, y)

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        pipeline = build_pipeline()
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        metrics = evaluation(y_test, y_pred)
        print(metrics)

        mlflow.log_params({"n_estimators": N_ESTIMATORS,  
                           "max_depth": MAX_DEPTH, 
                           "random_state": RANDOM_STATE, 
                           "test_size": TEST_SIZE,
                           "class_weight": CLASS_WEIGHT})

        mlflow.log_metrics(metrics)

        signature = infer_signature(X_train, pipeline.predict(X_train))

        model_info = mlflow.sklearn.log_model(sk_model=pipeline, 
                                 name="model", 
                                 signature=signature, 
                                 input_example=X_train.head(5), 
                                 registered_model_name=MODEL_NAME,
                                 skops_trusted_types=["sklearn.tree._tree.Tree"])

    version = model_info.registered_model_version
    client = MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, "champion", version)